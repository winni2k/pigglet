from abc import ABC
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional, Tuple
import logging
import numpy as np

from pigglet.constants import TreeIsTooSmallError
from pigglet.likelihoods import (
    MutationTreeLikelihoodCalculator,
    PhyloTreeLikelihoodCalculator,
)
from pigglet.tree import MutationTreeMoveMemento
from pigglet.tree_interactor import (
    MutationTreeInteractor,
    PhyloTreeInteractor,
    PhyloTreeMoveMemento,
)
from pigglet.tree_utils import parent_node_of, roots_of_tree

logger = logging.getLogger(__name__)


class TreeLikelihoodMover(ABC):
    def __init__(self):
        self.mover = None
        self.calc = None

    def make_and_register_random_move(self):
        self.mover.random_move()
        self.calc.register_changed_nodes(*self.mover.changed_nodes)

    def undo(self):
        self.mover.move_tracker.register_failure()
        self.mover.undo(memento=self.mover.memento)
        self.calc.register_changed_nodes(*self.mover.changed_nodes)

    def random_move_and_get_like(self):
        self.make_and_register_random_move()
        return self.log_likelihood()

    def log_likelihood(self):
        return self.calc.log_likelihood()

    @property
    def move_changed_tree(self):
        return len(self.mover.memento) != 0

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
    def prng(self):
        return self.mover.prng

    @property
    def has_changed_nodes(self):
        return self.calc.has_changed_nodes()

    @property
    def check_logsumexp_accuracy(self):
        return self.calc.summer.check_calc

    @check_logsumexp_accuracy.setter
    def check_logsumexp_accuracy(self, value):
        self.calc.summer.check_calc = value

    @property
    def logsumexp_refresh_rate(self):
        return self.calc.summer.check_calc

    @logsumexp_refresh_rate.setter
    def logsumexp_refresh_rate(self, value):
        self.calc.summer.max_diffs = value

    @property
    def g(self):
        return self.mover.g

    def get_tracker_n_tries(self):
        return self.mover.move_tracker.n_tries

    def get_tracker_acceptance_ratios(self):
        return self.mover.move_tracker.get_acceptance_ratios()

    def get_tracker_successful_proposal_time_proportions(self):
        return (
            self.mover.move_tracker.get_successful_proposal_time_proportions()
        )

    def flush_tracker(self):
        return self.mover.move_tracker.flush()

    def get_calc_n_node_update_list(self):
        return self.calc.n_node_update_list

    def clear_calc_n_node_update_list(self):
        self.calc.n_node_update_list.clear()

    def get_available_move_names(self):
        return [m.__name__ for m in self.mover.available_moves]


class PhyloTreeLikelihoodMover(TreeLikelihoodMover):
    """Make phylogenetic tree moves while keeping tree likelihoods updated"""

    def __init__(self, g, gls, prng):
        super().__init__()
        self.mover = PhyloTreeMoveCaretaker(g, prng=prng)
        self.calc = PhyloTreeLikelihoodCalculator(g, gls)

    @property
    def double_check_ll_calculations(self):
        return self.calc.double_check_ll_calculations

    @double_check_ll_calculations.setter
    def double_check_ll_calculations(self, value):
        self.calc.double_check_ll_calculations = value


class MutationTreeLikelihoodMover(TreeLikelihoodMover):
    """Make mutation tree moves while keeping tree likelihoods updated"""

    def __init__(self, g, gls, prng):
        super().__init__()
        self.mover = MutationTreeMoveCaretaker(g, prng=prng)
        self.calc = MutationTreeLikelihoodCalculator(g, gls)


@dataclass
class MoveTracker:
    n_moves: int
    _move_tries: List[int] = field(default_factory=list)
    _move_acceptances: List[int] = field(default_factory=list)
    _move_times: List[float] = field(default_factory=list)
    _current_try: Optional[int] = None
    _timer_start: Optional[float] = None

    n_tries: int = 0

    def __post_init__(self):
        self.flush()

    def register_try(self, move_idx: int) -> None:
        if self._current_try is not None:
            self._register_mh_result(True)
        assert self._current_try is None
        self._current_try = move_idx
        self._timer_start = perf_counter()

    def register_failure(self) -> None:
        self._register_mh_result(False)

    def _register_mh_result(self, accepted: bool) -> None:
        assert self._current_try is not None
        assert self._timer_start is not None
        try_time = perf_counter() - self._timer_start
        self._timer_start = None
        self.n_tries += 1

        self._move_times[self._current_try] += try_time
        self._move_tries[self._current_try] += 1
        if accepted:
            self._move_acceptances[self._current_try] += 1
        self._current_try = None

    def get_acceptance_ratios(self) -> List[float]:
        return [
            a / t if t else np.nan
            for a, t in zip(self._move_acceptances, self._move_tries)
        ]

    def get_successful_proposal_time_proportions(self) -> List[float]:
        prop_succ_time = [
            t / r if r else np.nan
            for t, r in zip(self._move_times, self.get_acceptance_ratios())
        ]
        total_time = sum(prop_succ_time)
        return [t / total_time for t in prop_succ_time]

    def flush(self):
        self._move_tries = [0] * self.n_moves
        self._move_acceptances = [0] * self.n_moves
        self._move_times = [0] * self.n_moves
        self.n_tries = 0


class UnchangedTopologyError(Exception):
    """Raised when a valid MCMC move would yield no change in topology"""

    pass


class TreeMoveCaretaker(ABC):
    def __init__(self):
        self.move_tracker = None
        self.interactor = None

    @property
    def mh_correction(self):
        return self.interactor.mh_correction

    def undo(self, memento):
        self.interactor.undo(memento)


class PhyloTreeMoveCaretaker(TreeMoveCaretaker):
    """In charge of undoing moves"""

    def __init__(self, g, prng):
        self.g = g
        self.prng = prng
        self.interactor = PhyloTreeInteractor(self.g)
        self.memento = PhyloTreeMoveMemento()
        self.available_moves = [
            self.extending_subtree_prune_and_regraft,
            self.swap_leaf,
        ]
        self.weights = [1] * len(self.available_moves)
        self.move_tracker = MoveTracker(len(self.available_moves))
        self.ext_choice_prob = 0.33

    @property
    def changed_nodes(self):
        return self.interactor.changed_nodes

    def extending_subtree_prune_and_regraft(
        self, node: int = None
    ) -> Tuple[int, Tuple[int, int]]:
        """AKA eSPR, as described in Lakner et al. 2008"""
        # if len(self.g) < 6:
        #     raise TreeIsTooSmallError("Tree contains less than four leaves")
        choices = self._get_choices(node)
        node = self.prng.choice(choices)
        self.memento, edge = self.interactor.extend_prune_and_regraft(
            node, prop_attach=self.ext_choice_prob
        )
        assert node is not None
        return node, edge

    def _get_choices(self, node: Optional[int] = None) -> List[int]:
        choices = [u for u in self.g.nodes if self.g.in_degree(u) != 0]
        if node is not None:
            assert node in choices
            choices = [node]
        return choices

    def swap_leaf(self) -> Tuple[int, int]:
        if len(self.interactor.leaf_nodes) < 3:
            raise TreeIsTooSmallError(
                "Tree contains less than three leaf nodes"
            )
        n1, n2 = self._get_two_leaves()
        self.memento = self.interactor.swap_leaves(n1, n2)
        return n1, n2

    def random_move(self):
        choice = self.prng.choices(
            range(len(self.available_moves)), weights=self.weights
        )[0]
        self.available_moves[choice]()
        # logger.debug(
        #     "Random Move -- choice, memento: %s, %s", choice, self.memento
        # )
        self.move_tracker.register_try(choice)

    def _get_two_leaves(self):
        leaf_nodes = self.interactor.leaf_node_list
        n1 = self.prng.choice(leaf_nodes)
        n2 = self.prng.choice(leaf_nodes)
        return n1, n2


class MutationTreeMoveCaretaker(TreeMoveCaretaker):
    def __init__(self, g, prng):
        self.g = g
        self.prng = prng
        self.interactor = MutationTreeInteractor(self.g, prng=prng)
        self.memento = MutationTreeMoveMemento()
        self.available_moves = [
            # self.prune_and_reattach,
            self.extending_subtree_prune_and_regraft,
            self.swap_node,
            self.swap_subtree,
        ]
        self.weights = [1] * len(self.available_moves)
        self.move_tracker = MoveTracker(len(self.available_moves))
        self.changed_nodes = list(roots_of_tree(g))
        self.ext_choice_prob = 0.5

    def extending_subtree_prune_and_regraft(self):
        """AKA eSPR, as described in Lakner et al. 2008"""
        if len(self.g) < 2:
            raise TreeIsTooSmallError
        node = self.prng.randrange(len(self.g) - 1)
        parent = parent_node_of(self.g, node)
        self.memento = self.interactor.prune(node)
        try:
            memento = self.interactor.extend_attach(
                node, parent, self.ext_choice_prob
            )
        except TreeIsTooSmallError:
            memento = self.interactor.attach(node, parent)
        self.memento.append(memento)
        self.changed_nodes = [node]

    def prune_and_regraft(self):
        if len(self.g) < 2:
            raise TreeIsTooSmallError
        node = self.prng.randrange(len(self.g) - 1)
        self.memento = self.interactor.prune(node)
        self.memento.append(self.interactor.uniform_attach(node))
        self.changed_nodes = [node]

    def swap_node(self):
        if self._tree_is_too_small_for_advanced_moves():
            self.memento = MutationTreeMoveMemento()
            return
        n1, n2 = self._get_two_distinct_nodes()
        self.memento = self.interactor.swap_labels(n1, n2)
        self.changed_nodes = [n1, n2]

    def swap_subtree(self):
        if self._tree_is_too_small_for_advanced_moves():
            self.memento = MutationTreeMoveMemento()
            return
        n1, n2 = self._get_two_distinct_nodes()
        self.memento = self.interactor.swap_subtrees(n1, n2)
        self.changed_nodes = [n1, n2]

    def random_move(self):
        choice = self.prng.choices(
            range(len(self.available_moves)), weights=self.weights
        )[0]
        self.move_tracker.register_try(choice)
        self.available_moves[choice]()

    def _get_two_distinct_nodes(self):
        n1 = n2 = 0
        while n1 == n2:
            n1 = self.prng.randrange(len(self.g) - 1)
            n2 = self.prng.randrange(len(self.g) - 1)
        return n1, n2

    def _tree_is_too_small_for_advanced_moves(self):
        if len(self.g) < 3:
            return True
        return False
