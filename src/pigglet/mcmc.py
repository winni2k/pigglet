import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.aggregator import (
    AttachmentAggregator,
    MutationAttachmentAggregator,
    PhyloAttachmentAggregator,
    TreeAggregator,
)
from pigglet.tree_interactor import (
    MutationTreeInteractor,
    PhyloTreeInteractor,
    TreeInteractor,
)
from pigglet.tree_likelihood_mover import (
    MutationTreeLikelihoodMover,
    PhyloTreeLikelihoodMover,
    TreeLikelihoodMover,
)

NUM_MCMC_MOVES = 3

logger = logging.getLogger(__name__)


@dataclass
class MCMCRunner:
    gls: np.ndarray
    map_g: nx.DiGraph
    tree_move_weights: List[float]
    tree_interactor: TreeInteractor
    mover: TreeLikelihoodMover
    agg: AttachmentAggregator
    prng: Any
    num_sampling_iter: int = 1
    num_burnin_iter: int = 1
    reporting_interval: int = 1
    new_like: float = 0.0
    current_like: float = 0.0
    map_like: float = 0.0
    mcmc_moves: List[int] = field(
        default_factory=lambda: list(range(NUM_MCMC_MOVES))
    )
    tree_aggregator: Optional[TreeAggregator] = None

    def __post_init__(self):
        self.current_like = self.mover.calc.log_likelihood()
        self.map_like = self.current_like

    @classmethod
    def mutation_tree_from_gls(
        cls, gls, prng=random, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_mutation_tree(gls.shape[0])
        mover = MutationTreeLikelihoodMover(graph, gls, prng=prng)
        return cls(
            gls=gls,
            map_g=graph.copy(),
            tree_move_weights=[1] * len(mover.mover.available_moves),
            tree_interactor=MutationTreeInteractor(graph, prng=prng),
            mover=mover,
            agg=MutationAttachmentAggregator(),
            prng=prng,
            **kwargs,
        )

    @classmethod
    def phylogenetic_tree_from_gls(
        cls, gls, prng=random, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_phylogenetic_tree(
            num_samples=gls.shape[1], seed=prng.randrange(1, 2 ^ 32)
        )
        mover = PhyloTreeLikelihoodMover(graph, gls, prng)
        if "tree_move_weights" not in kwargs:
            kwargs["tree_move_weights"] = [1] * len(
                mover.mover.available_moves
            )
        if "double_check_ll_calculation" in kwargs:
            mover.calc.double_check_ll_calculations = kwargs.pop(
                "double_check_ll_calculation"
            )
        return cls(
            gls=gls,
            map_g=graph.copy(),
            tree_interactor=PhyloTreeInteractor(graph),
            mover=mover,
            agg=PhyloAttachmentAggregator(),
            prng=prng,
            **kwargs,
        )

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
                self.agg.add_attachment_log_likes(self.mover.calc)
                if self.tree_aggregator:
                    self.tree_aggregator.store_tree(self.g)
            if iteration % self.reporting_interval == 0 and iteration != 0:
                if logger.isEnabledFor(logging.INFO):
                    self._iteration_logging(iteration)
            iteration += 1
            pbar.update()

    def _iteration_logging(self, iteration):
        """Logging for a reporting interval"""
        tracker = self.mover.mover.move_tracker
        logger.info(
            f"Iteration {iteration} "
            f"| current like: {self.current_like} "
            f"| MAP like: {self.map_like}"
        )
        percentiles = np.percentile(
            self.mover.calc.n_node_update_list,
            [5, 25, 50, 75, 95],
            interpolation="higher",
        )
        logger.info(
            f"Iteration {iteration} "
            f"| {len(self.mover.calc.n_node_update_list)} updates "
            f"| 5, 25, 50, 75, 95 percentile nodes updated: {percentiles}"
        )
        self.mover.calc.n_node_update_list.clear()
        logger.info(
            f"Iteration {iteration} "
            f"| acceptance rate: "
            f"{self.reporting_interval / tracker.n_tries:.1%}"
        )
        acceptance_ratios = tracker.get_acceptance_ratios()
        for move_idx, move in enumerate(self.mover.mover.available_moves):
            logger.info(
                f"Iteration {iteration} "
                f"| function: {move.__name__} "
                f"| acceptance rate:"
                f" {acceptance_ratios[move_idx]:.1%}"
            )
        tracker.flush()

    @property
    def g(self):
        return self.mover.mover.g

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
        self.mover.random_move(weights=self.tree_move_weights)
        self.new_like = self.mover.log_likelihood()
        accepted = self._mh_acceptance()
        self.mover.mover.register_mh_result(accepted)
        if not accepted:
            self.mover.undo()
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
            * self.mover.mh_correction
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
