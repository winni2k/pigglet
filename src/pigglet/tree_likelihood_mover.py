import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pigglet.constants import TreeIsTooSmallError
from pigglet.likelihoods import MutationTreeLikelihoodCalculator
from pigglet.tree import TreeMoveMemento
from pigglet.tree_interactor import MutationTreeInteractor
from pigglet.tree_utils import roots_of_tree, parent_node_of


class MutationTreeLikelihoodMover:
    """Make mutation tree moves while keeping tree likelihoods updated"""

    def __init__(self, g, gls):
        self.mover = MutationTreeMoveExecutor(g)
        self.calc = MutationTreeLikelihoodCalculator(g, gls)

    def random_move(self, weights=None):
        self.mover.random_move(weights=weights)
        self.calc.register_changed_nodes(*self.mover.changed_nodes)

    def undo(self):
        self.calc.register_changed_nodes(*self.mover.changed_nodes)
        self.mover.undo(memento=self.mover.memento)

    def sample_marginalized_log_likelihood(self):
        return self.calc.log_likelihood()

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


class MutationTreeMoveExecutor:
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
        choice = random.choices(
            range(len(self.available_moves)), weights=weights
        )[0]
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
