import logging
import math
import random

import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.constants import TreeIsTooSmallError
from pigglet.likelihoods import AttachmentAggregator, TreeLikelihoodCalculator
from pigglet.tree import TreeInteractor, TreeMoveMemento
from pigglet.tree_utils import roots_of_tree

NUM_MCMC_MOVES = 3


class MCMCRunner:
    def __init__(
        self,
        gls,
        graph,
        num_sampling_iter,
        num_burnin_iter,
        tree_move_weights,
        tree_interactor,
        mover,
        current_like,
        reporting_interval,
    ):
        self.g = graph
        self.map_g = graph.copy()
        self.gls = gls
        self.num_sampling_iter = num_sampling_iter
        self.num_burnin_iter = num_burnin_iter
        self.tree_move_weights = tree_move_weights
        self.tree_interactor = tree_interactor
        self.current_like = current_like
        self.new_like = None
        self.map_like = current_like
        self.agg = AttachmentAggregator()
        self.reporting_interval = reporting_interval
        self.mcmc_moves = list(range(NUM_MCMC_MOVES))
        self.mover = mover

    @classmethod
    def from_gls(
        cls,
        gls,
        num_sampling_iter=10,
        num_burnin_iter=10,
        prune_and_reattach_weight=1,
        swap_node_weight=1,
        swap_subtree_weight=1,
        reporting_interval=1,
    ):
        assert np.alltrue(gls <= 0), gls
        graph = build_random_mutation_tree(gls.shape[0])
        tree_move_weights = [
            prune_and_reattach_weight,
            swap_node_weight,
            swap_subtree_weight,
        ]
        assert len(tree_move_weights) == NUM_MCMC_MOVES
        tree_interactor = TreeInteractor(graph)
        mover = TreeLikelihoodMover(graph, gls)
        return cls(
            gls,
            graph,
            num_sampling_iter=num_sampling_iter,
            num_burnin_iter=num_burnin_iter,
            tree_move_weights=tree_move_weights,
            tree_interactor=tree_interactor,
            mover=mover,
            current_like=mover.calc.sample_marginalized_log_likelihood(),
            reporting_interval=reporting_interval,
        )

    def run(self):
        iteration = 0
        tries = 0
        pbar = self._get_progress_bar(type="burnin")
        while iteration < self.num_burnin_iter + self.num_sampling_iter:
            if iteration == self.num_burnin_iter:
                logging.info("Entering sampling iterations")
                pbar = self._get_progress_bar(type="sampling")
            accepted = self._mh_step()
            tries += 1
            if not accepted:
                continue
            if iteration >= self.num_burnin_iter:
                self._update_map(iteration)
            if iteration % self.reporting_interval == 0 and iteration != 0:
                logging.info(
                    "Iteration %s: acceptance rate: %s\tcurrent like: %s"
                    "\tMAP like: %s",
                    iteration,
                    self.reporting_interval / tries,
                    self.current_like,
                    self.map_like,
                )
                tries = 0
            iteration += 1
            if iteration > self.num_burnin_iter:
                self.agg.add_attachment_log_likes(self.mover.calc)
            pbar.update()

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


class MoveExecutor:
    def __init__(self, g):
        self.g = g
        self.interactor = TreeInteractor(self.g)
        self.memento = None
        self.available_moves = [
            # self.prune_and_reattach,
            self.extending_subtree_prune_and_regraft,
            self.swap_node,
            self.swap_subtree,
        ]
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
        random.choices(self.available_moves, weights=weights)[0]()

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


def parent_node_of(g, n):
    parents = list(g.pred[n])
    assert len(parents) == 1
    return parents[0]
