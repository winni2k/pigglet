import logging
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.constants import TreeIsTooSmallError
from pigglet.likelihoods import AttachmentAggregator, TreeLikelihoodCalculator
from pigglet.tree import MutationTreeInteractor, PhyloTreeInteractor, TreeMoveMemento
from pigglet.tree_utils import roots_of_tree

NUM_MCMC_MOVES = 3

logger = logging.getLogger(__name__)


class TreeLikelihoodMover:
    def __init__(self, g, gls):
        self.mover = MoveExecutor(g)
        self.calc = TreeLikelihoodCalculator(g, gls)

    def random_move(self, weights=None):
        self.mover.random_move(weights=weights)
        self.calc.register_changed_nodes(*self.mover.changed_nodes)

    def undo(self):
        self.calc.register_changed_nodes(*self.mover.changed_nodes)
        self.mover.undo(memento=self.mover.memento)

    def sample_marginalized_log_likelihood(self):
        return self.calc.sample_marginalized_log_likelihood()

    @property
    def mh_correction(self):
        return self.mover.mh_correction

    @property
    def changed_nodes(self):
        return self.mover.changed_nodes

    @property
    def attachment_log_like(self):
        return self.calc.attachment_log_like

    @property
    def memento(self):
        return self.mover.memento


@dataclass
class MCMCRunner:
    gls: np.ndarray
    map_g: nx.DiGraph
    tree_move_weights: List[float]
    tree_interactor: MutationTreeInteractor
    mover: TreeLikelihoodMover
    num_sampling_iter: int = 1
    num_burnin_iter: int = 1
    reporting_interval: int = 1
    new_like: Optional[float] = None
    current_like: Optional[float] = None
    map_like: Optional[float] = None
    agg: AttachmentAggregator = field(default_factory=AttachmentAggregator)
    mcmc_moves: List[int] = field(default_factory=lambda: list(range(NUM_MCMC_MOVES)))

    def __post_init__(self):
        self.current_like = self.mover.calc.sample_marginalized_log_likelihood()
        self.map_like = self.current_like

    @classmethod
    def mutation_tree_from_gls(
        cls, gls, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_mutation_tree(gls.shape[0])
        tree_move_weights = [1] * NUM_MCMC_MOVES
        tree_interactor = MutationTreeInteractor(graph)
        mover = TreeLikelihoodMover(graph, gls)
        return cls(
            gls=gls,
            map_g=graph.copy(),
            tree_move_weights=tree_move_weights,
            tree_interactor=tree_interactor,
            mover=mover,
            **kwargs,
        )

    @classmethod
    def phylogenetic_tree_from_gls(
        cls, gls, **kwargs,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_mutation_tree(gls.shape[0])
        tree_move_weights = [1] * NUM_MCMC_MOVES
        tree_interactor = MutationTreeInteractor(graph)
        mover = TreeLikelihoodMover(graph, gls)
        return cls(
            gls=gls,
            map_g=graph.copy(),
            tree_move_weights=tree_move_weights,
            tree_interactor=tree_interactor,
            mover=mover,
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

            if iteration % self.reporting_interval == 0 and iteration != 0:
                tracker = self.mover.mover.move_tracker
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"Iteration {iteration} "
                        f"| current like: {self.current_like} "
                        f"| MAP like: {self.map_like}"
                    )
                    percentiles = np.percentile(
                        self.mover.calc.n_node_update_list,
                        [50, 95, 99],
                        interpolation="higher",
                    )
                    logger.info(
                        f"Iteration {iteration} "
                        f"| median, 95, 99 percentile of nodes updated per move:"
                        f" {percentiles}"
                    )
                    self.mover.calc.n_node_update_list.clear()
                    logger.info(
                        f"Iteration {iteration} "
                        f"| acceptance rate: {self.reporting_interval / tracker.n_tries:.1%}"
                    )
                    acceptance_ratios = tracker.get_acceptance_ratios()
                    for move_idx, move in enumerate(self.mover.mover.available_moves):
                        logger.info(
                            f"Iteration {iteration} "
                            f"| function: {move.__name__} "
                            f"| acceptance rate: {acceptance_ratios[move_idx]:.1%}"
                        )
                tracker.flush()
            iteration += 1
            if iteration > self.num_burnin_iter:
                self.agg.add_attachment_log_likes(self.mover.calc)
            pbar.update()

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
        self.new_like = self.mover.sample_marginalized_log_likelihood()
        accepted = self._mh_acceptance()
        self.mover.mover.move_tracker.register_mh_result(accepted)
        if not accepted:
            self.mover.undo()
        else:
            self.current_like = self.new_like
        return accepted

    def _mh_acceptance(self):
        """Perform Metropolis Hastings rejection step. Return if proposal was accepted"""
        if self.new_like >= self.current_like:
            return True
        ratio = math.exp(self.new_like - self.current_like) * self.mover.mh_correction

        rand_val = random.random()
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


@dataclass
class MoveTracker:
    n_moves: int
    _move_tries: List[int] = field(default_factory=list)
    _move_acceptances: List[int] = field(default_factory=list)
    _current_try: Optional[int] = None
    n_tries: int = 0

    def __post_init__(self):
        self.flush()

    def register_try(self, move_idx: int):
        assert self._current_try is None
        self._current_try = move_idx

    def register_mh_result(self, accepted: bool):
        assert self._current_try is not None
        self.n_tries += 1
        self._move_tries[self._current_try] += 1
        if accepted:
            self._move_acceptances[self._current_try] += 1
        self._current_try = None

    def get_acceptance_ratios(self) -> List[float]:
        return [
            a / t if t else np.nan
            for a, t in zip(self._move_acceptances, self._move_tries)
        ]

    def flush(self):
        self._move_tries = [0] * self.n_moves
        self._move_acceptances = [0] * self.n_moves
        self.n_tries = 0


class MoveExecutor:
    def __init__(self, g):
        self.g = g
        self.interactor = MutationTreeInteractor(self.g)
        self.memento = None
        self.available_moves = [
            # self.prune_and_reattach,
            self.extending_subtree_prune_and_regraft,
            self.swap_node,
            self.swap_subtree,
        ]
        self.move_tracker = MoveTracker(len(self.available_moves))
        self.changed_nodes = list(roots_of_tree(g))
        self.ext_choice_prob = 0.5

    @property
    def mh_correction(self):
        return self.interactor.mh_correction

    def undo(self, memento):
        self.interactor.undo(memento)

    def extending_subtree_prune_and_regraft(self):
        """AKA eSPR, as described in Lakner et al. 2008"""
        if len(self.g) < 2:
            raise TreeIsTooSmallError
        node = random.randrange(len(self.g) - 1)
        parent = parent_node_of(self.g, node)
        self.memento = self.interactor.prune(node)
        try:
            memento = self.interactor.extend_attach(node, parent, self.ext_choice_prob)
        except TreeIsTooSmallError:
            memento = self.interactor.attach(node, parent)
        self.memento.append(memento)
        self.changed_nodes = [node]

    def prune_and_reattach(self):
        if len(self.g) < 2:
            raise TreeIsTooSmallError
        node = random.randrange(len(self.g) - 1)
        self.memento = self.interactor.prune(node)
        self.memento.append(self.interactor.uniform_attach(node))
        self.changed_nodes = [node]

    def swap_node(self):
        if self._tree_is_too_small_for_advanced_moves():
            self.memento = TreeMoveMemento()
            return
        n1, n2 = self._get_two_distinct_nodes()
        self.memento = self.interactor.swap_labels(n1, n2)
        self.changed_nodes = [n1, n2]

    def swap_subtree(self):
        if self._tree_is_too_small_for_advanced_moves():
            self.memento = TreeMoveMemento()
            return
        n1, n2 = self._get_two_distinct_nodes()
        self.memento = self.interactor.swap_subtrees(n1, n2)
        self.changed_nodes = [n1, n2]

    def random_move(self, weights=None):
        if weights is None:
            weights = [1, 1, 1]
        choice = random.choices(range(len(self.available_moves)), weights=weights)[0]
        self.move_tracker.register_try(choice)
        self.available_moves[choice]()

    def _get_two_distinct_nodes(self):
        n1 = n2 = 0
        while n1 == n2:
            n1 = random.randrange(len(self.g) - 1)
            n2 = random.randrange(len(self.g) - 1)
        return n1, n2

    def _tree_is_too_small_for_advanced_moves(self):
        if len(self.g) < 3:
            return True
        return False


def build_random_mutation_tree(num_sites):
    dag = nx.gnr_graph(num_sites + 1, 0).reverse()
    nx.relabel_nodes(dag, {n: n - 1 for n in dag.nodes}, copy=False)
    assert max(dag.nodes) + 1 == num_sites
    assert min(dag.nodes) == -1
    return dag


def parent_node_of(g, n):
    parents = list(g.pred[n])
    assert len(parents) == 1
    return parents[0]
