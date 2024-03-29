import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.aggregator import (
    AttachmentAggregator,
    MutationAttachmentAggregator,
    PhyloAttachmentAggregator,
    TreeAggregator,
)
from pigglet.tree_likelihood_mover import (
    MutationTreeLikelihoodMover,
    PhyloTreeLikelihoodMover,
    TreeLikelihoodMover,
)

logger = logging.getLogger(__name__)


def rearrange_gl_axes_for_performance(gls):
    glstmp = np.zeros((gls.shape[1], gls.shape[2], gls.shape[0]))
    for genotype_idx in range(gls.shape[2]):
        for site_idx in range(gls.shape[0]):
            glstmp[:, genotype_idx, site_idx] = gls[site_idx, :, genotype_idx]
    return np.moveaxis(glstmp, 2, 0)


@dataclass
class MCMCRunner:
    map_g: nx.DiGraph
    like_mover: TreeLikelihoodMover
    agg: AttachmentAggregator
    prng: Any
    num_sampling_iter: int = 1
    num_burnin_iter: int = 1
    reporting_interval: int = 1
    new_like: float = 0.0
    current_like: float = 0.0
    map_like: float = 0.0
    tree_aggregator: Optional[TreeAggregator] = None

    def __post_init__(self):
        self.current_like = self.like_mover.log_likelihood()
        self.map_like = self.current_like

    @classmethod
    def mutation_tree_from_gls(
        cls, gls, prng=random, num_cpus=1, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_mutation_tree(gls.shape[0])
        like_mover = MutationTreeLikelihoodMover(graph, gls, prng=prng)
        return cls(
            map_g=graph.copy(),
            like_mover=like_mover,
            agg=MutationAttachmentAggregator(),
            prng=prng,
            **kwargs,
        )

    @classmethod
    def phylogenetic_tree_from_gls(
        cls, gls, prng=random, num_actors=1, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_phylogenetic_tree(
            num_samples=gls.shape[1], seed=prng.randrange(1, 2 ^ 32)
        )
        gls = rearrange_gl_axes_for_performance(gls)
        if num_actors > 1:
            import ray

            if not ray.is_initialized():
                ray.init(num_cpus=num_actors)

            from pigglet.tree_likelihood_mover_ray import (
                PhyloTreeLikelihoodMoverDirector,
            )

            like_mover = PhyloTreeLikelihoodMoverDirector(
                graph, gls, prng, num_actors=num_actors
            )
        else:
            like_mover = PhyloTreeLikelihoodMover(graph, gls, prng)
        if "double_check_ll_calculation" in kwargs:
            like_mover.double_check_ll_calculations = kwargs.pop(
                "double_check_ll_calculation"
            )
        return cls(
            map_g=graph.copy(),
            like_mover=like_mover,
            agg=PhyloAttachmentAggregator(),
            prng=prng,
            **kwargs,
        )

    @property
    def gls(self):
        return self.like_mover.calc.gls

    def run(self):
        iteration = 0
        pbar = self._get_progress_bar(type="burnin")
        while iteration < self.num_burnin_iter + self.num_sampling_iter:
            if iteration == self.num_burnin_iter:
                logger.info("Entering sampling iterations")
                pbar = self._get_progress_bar(type="sampling")
            accepted = self._mh_step()
            if not accepted:
                continue
            if iteration >= self.num_burnin_iter:
                self._update_map(iteration)
                self.agg.add_attachment_log_likes(self.like_mover.calc)
                if self.tree_aggregator:
                    self.tree_aggregator.store_tree(self.g)
            if iteration % self.reporting_interval == 0 and iteration != 0:
                if logger.isEnabledFor(logging.INFO):
                    self._iteration_logging(iteration)
            iteration += 1
            pbar.update()

    def _iteration_logging(self, iteration):
        """Logging for a reporting interval"""
        n_tries = self.like_mover.get_tracker_n_tries()
        acceptance_ratios = self.like_mover.get_tracker_acceptance_ratios()
        move_times = (
            self.like_mover.get_tracker_successful_proposal_time_proportions()
        )
        update_list = list(self.like_mover.get_calc_n_node_update_list())
        self.like_mover.clear_calc_n_node_update_list()

        logger.info(
            f"Iteration {iteration} "
            f"| current like: {self.current_like} "
            f"| MAP like: {self.map_like}"
        )
        percentiles = np.percentile(
            update_list, [5, 25, 50, 75, 95], interpolation="higher",
        )
        logger.info(
            f"Iteration {iteration} "
            f"| {len(update_list)} updates "
            f"| 5, 25, 50, 75, 95 percentile nodes updated: {percentiles}"
        )
        logger.info(
            f"Iteration {iteration} "
            f"| acceptance rate: "
            f"{self.reporting_interval / n_tries:.1%}"
        )
        for move_idx, move in enumerate(
            self.like_mover.get_available_move_names()
        ):
            logger.info(
                f"Iteration {iteration} "
                f"| function: {move} "
                "| acceptance rate:"
                f" {acceptance_ratios[move_idx]:.1%} "
                "| proportion time per acceptance:"
                f" {move_times[move_idx]:.1%} "
            )
        self.like_mover.flush_tracker()

    @property
    def g(self):
        return self.like_mover.g

    def set_move_weight(self, move_name: str, weight: float):
        assert weight >= 0
        available_moves = self.like_mover.get_available_move_names()
        move_index = available_moves.index(move_name)
        self.like_mover.set_move_weight(move_index, weight)

    @property
    def move_weights(self):
        available_moves = self.like_mover.get_available_move_names()
        weights = self.like_mover.move_weights
        return {k: v for k, v in zip(available_moves, weights)}

    def _update_map(self, iteration):
        if self.new_like > self.map_like:
            self.map_g = self.g.copy()
            self.map_like = self.new_like
            logging.debug(
                "Iteration %s: new MAP tree with likelihood %s",
                iteration,
                self.map_like,
            )

    def _mh_step(self):
        """Propose tree and MH reject proposal"""
        self.new_like = self.like_mover.random_move_and_get_like()
        if not self.like_mover.move_changed_tree:
            assert np.allclose(
                self.current_like, self.new_like, atol=1.0e-1
            ), (self.current_like, self.new_like)
        accepted = self._mh_acceptance()
        if not accepted:
            self.like_mover.undo()
            self.like_mover.log_likelihood()
        else:
            self.current_like = self.new_like
        return accepted

    def _mh_acceptance(self):
        """Perform Metropolis Hastings rejection step. Return if proposal was
        accepted"""
        if self.new_like >= self.current_like:
            return True
        ratio = (
            math.exp(self.new_like - self.current_like)
            * self.like_mover.mh_correction
        )

        rand_val = self.prng.random()
        if rand_val < ratio:
            return True
        return False

    def _get_progress_bar(self, type):
        if type == "burnin":
            return tqdm(
                total=self.num_burnin_iter,
                desc="Burnin iterations",
                unit="iterations",
                mininterval=5.0,
            )
        elif type == "sampling":
            return tqdm(
                total=self.num_sampling_iter,
                desc="Sampling iterations",
                unit="iterations",
                mininterval=5.0,
            )
        raise ValueError


def build_random_phylogenetic_tree(num_samples, seed):
    assert num_samples > 1
    import msprime

    ts = msprime.simulate(
        sample_size=num_samples,
        Ne=100 * num_samples,
        recombination_rate=0,
        random_seed=seed,
    )
    tree = ts.first()
    g = nx.DiGraph(as_dict_of_dicts(tree))

    return g


def build_random_mutation_tree(num_sites):
    dag = nx.gnr_graph(num_sites + 1, 0).reverse()
    nx.relabel_nodes(dag, {n: n - 1 for n in dag.nodes}, copy=False)
    assert max(dag.nodes) + 1 == num_sites
    assert min(dag.nodes) == -1
    return dag


def as_dict_of_dicts(tree):
    """
    Convert tree to dict of dicts for conversion to a
    `networkx graph <https://networkx.github.io/documentation/stable/
    reference/classes/digraph.html>`_.
    For example::
        >>> import networkx as nx
        >>> nx.DiGraph(tree.as_dict_of_dicts())
        >>> # undirected graphs work as well
        >>> nx.Graph(tree.as_dict_of_dicts())
    :return: Dictionary of dictionaries of dictionaries where the first key
        is the source, the second key is the target of an edge, and the
        third key is an edge annotation. At this point the only annotation
        is "branch_length", the length of the branch (in generations).
    """
    dod = {}
    for parent in tree.nodes():
        dod[parent] = {}
        for child in tree.children(parent):
            dod[parent][child] = {}
    return dod
