import logging
import random

import networkx as nx
from tqdm import tqdm

from pigglet.likelihoods import TreeLikelihoodCalculator, AttachmentAggregator
from pigglet.tree import TreeInteractor, TreeMoveMemento
from pigglet.tree_utils import roots_of_tree

NUM_MCMC_MOVES = 3


class MCMCRunner:

    def __init__(self, gls, graph, num_sampling_iter, num_burnin_iter,
                 tree_move_weights, tree_interactor, likelihood_calculator,
                 current_like, reporting_interval):
        self.g = graph
        self.map_g = graph
        self.gls = gls
        self.num_sampling_iter = num_sampling_iter
        self.num_burnin_iter = num_burnin_iter
        self.tree_move_weights = tree_move_weights
        self.tree_interactor = tree_interactor
        self.calc = likelihood_calculator
        self.current_like = current_like
        self.new_like = None
        self.map_like = current_like
        self.agg = AttachmentAggregator()
        self.reporting_interval = reporting_interval
        self.mcmc_moves = list(range(NUM_MCMC_MOVES))
        self.mover = MoveExecutor(self.g)

    @classmethod
    def from_gls(
            cls, gls,
            num_sampling_iter=10,
            num_burnin_iter=10,
            prune_and_reattach_weight=1,
            swap_node_weight=1,
            swap_subtree_weight=1,
            reporting_interval=1,
    ):
        graph = build_random_mutation_tree(gls.shape[0])
        tree_move_weights = [
            prune_and_reattach_weight,
            swap_node_weight,
            swap_subtree_weight
        ]
        assert len(tree_move_weights) == NUM_MCMC_MOVES
        tree_interactor = TreeInteractor(graph)
        like_calc = TreeLikelihoodCalculator(graph, gls)
        return cls(
            gls,
            graph,
            num_sampling_iter=num_sampling_iter,
            num_burnin_iter=num_burnin_iter,
            tree_move_weights=tree_move_weights,
            tree_interactor=tree_interactor,
            likelihood_calculator=like_calc,
            current_like=like_calc.sample_marginalized_log_likelihood(),
            reporting_interval=reporting_interval,
        )

    def run(self):
        iteration = 0
        tries = 0
        pbar = self._get_progress_bar(type='burnin')
        while iteration < self.num_burnin_iter + self.num_sampling_iter:
            if iteration == self.num_burnin_iter:
                logging.info('Entering sampling iterations')
                pbar = self._get_progress_bar(type='sampling')
            accepted = self._mh_step()
            tries += 1
            if not accepted:
                continue
            self._update_map(iteration)
            if iteration % self.reporting_interval == 0 and iteration != 0:
                logging.info(
                    'Iteration %s: acceptance rate: %s\tcurrent like: %s'
                    '\tMAP like: %s',
                    iteration,
                    self.reporting_interval / tries,
                    self.current_like,
                    self.map_like
                )
                tries = 0
            iteration += 1
            if iteration > self.num_burnin_iter:
                self.agg.add_attachment_log_likes(self.calc)
            pbar.update()

    def _update_map(self, iteration):
        if self.new_like > self.map_like:
            self.map_g = self.g.copy()
            self.map_like = self.new_like
            logging.debug('Iteration %s: new MAP tree with likelihood %s',
                          iteration,
                          self.map_like)

    def _mh_step(self):
        """Propose tree and MH reject proposal"""
        self.mover.random_move(weights=self.tree_move_weights)
        self.calc.recalculate_attachment_log_like_from_nodes(*self.mover.changed_nodes)
        self.new_like = self.calc.sample_marginalized_log_likelihood()
        accepted = self._mh_acceptance()
        if not accepted:
            self.mover.undo(self.mover.memento)
        else:
            self.current_like = self.new_like
        return accepted

    def _mh_acceptance(self):
        """Perform Metropolis Hastings rejection step. Return if proposal was accepted"""
        accept = False
        if self.new_like >= self.current_like:
            accept = True
        elif random.random() > self.current_like / self.new_like * self.mover.mh_correction:
            accept = True
        return accept

    def _get_progress_bar(self, type):
        if type == 'burnin':
            return tqdm(
                total=self.num_burnin_iter,
                desc='Burnin iterations',
                unit='iterations',
                mininterval=5.0,
            )
        elif type == 'sampling':
            return tqdm(
                total=self.num_sampling_iter,
                desc='Sampling iterations',
                unit='iterations',
                mininterval=5.0,
            )
        raise ValueError


class MoveExecutor:
    def __init__(self, g):
        self.g = g
        self.interactor = TreeInteractor(self.g)
        self.memento = None
        self.available_moves = [
            self.prune_and_reattach,
            self.swap_node,
            self.swap_subtree
        ]
        self.changed_nodes = list(roots_of_tree(g))

    @property
    def mh_correction(self):
        return self.interactor.mh_correction

    def undo(self, memento):
        self.interactor.undo(memento)

    def prune_and_reattach(self):
        if self._is_tree_too_small():
            self.memento = TreeMoveMemento()
            return
        node = random.randrange(len(self.g) - 1)
        self.memento = self.interactor.prune(node)
        self.memento.append(self.interactor.uniform_attach(node))
        self.changed_nodes = [node]

    def swap_node(self):
        if self._is_tree_too_small():
            self.memento = TreeMoveMemento()
            return
        n1, n2 = self._get_two_distinct_nodes()
        self.memento = self.interactor.swap_labels(n1, n2)
        self.changed_nodes = [n1, n2]

    def swap_subtree(self):
        if self._is_tree_too_small():
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
        assert len(self.g) > 2
        n1 = n2 = 0
        while n1 == n2:
            n1 = random.randrange(len(self.g) - 1)
            n2 = random.randrange(len(self.g) - 1)
        return n1, n2

    def _is_tree_too_small(self):
        if len(self.g) < 3:
            return True
        return False


def build_random_mutation_tree(num_sites):
    dag = nx.gnr_graph(num_sites + 1, 0).reverse()
    nx.relabel_nodes(dag, {n: n - 1 for n in dag.nodes}, copy=False)
    assert max(dag.nodes) + 1 == num_sites
    assert min(dag.nodes) == -1
    return dag
