import random

import networkx as nx

from pigglet.likelihoods import TreeLikelihoodCalculator
from pigglet.tree import TreeInteractor

NUM_MCMC_MOVES = 3


class MCMCRunner:

    def __init__(self, gls, graph, num_sampling_iter, num_burnin_iter,
                 tree_move_weights, tree_interactor, likelihood_calculator):
        self.g = graph
        self.gls = gls
        self.num_sampling_iter = num_sampling_iter
        self.num_burnin_iter = num_burnin_iter
        self.tree_move_weights = tree_move_weights
        self.tree_interactor = tree_interactor
        self.calc = likelihood_calculator
        self.current_like = self.calc.sample_marginalized_likelihood()

    @classmethod
    def from_gls(cls, gls,
                 num_sampling_iter=10,
                 num_burnin_iter=10,
                 prune_and_reattach_weight=1,
                 swap_node_weight=1,
                 swap_subtree_weight=1):
        graph = build_random_mutation_tree(gls.shape[0])
        tree_move_weights = [
            prune_and_reattach_weight,
            swap_node_weight,
            swap_subtree_weight
        ]
        assert len(tree_move_weights) == NUM_MCMC_MOVES
        tree_interactor = TreeInteractor(graph)
        like_calc = TreeLikelihoodCalculator(graph, gls)
        return cls(gls, graph,
                   num_sampling_iter=num_sampling_iter,
                   num_burnin_iter=num_burnin_iter,
                   tree_move_weights=tree_move_weights,
                   tree_interactor=tree_interactor,
                   likelihood_calculator=like_calc)

    def run(self):
        iteration = 0
        mcmc_moves = list(range(NUM_MCMC_MOVES))
        mover = Mover(self.g)
        moves = [mover.prune_and_reattach,
                 mover.swap_node,
                 mover.swap_subtree]
        while iteration < self.num_burnin_iter + self.num_sampling_iter:
            mover.set_g(self.g.copy())
            move = random.choices(mcmc_moves, weights=self.tree_move_weights)[0]
            mh_correction = moves[move]()
            new_g = mover.g
            self.calc.set_g(new_g)
            new_like = self.calc.sample_marginalized_likelihood()
            accepted = self._choose_g(new_g, new_like, mh_correction)
            if accepted:
                iteration += 1

    def _choose_g(self, new_g, new_like, mh_correction):
        """Perform Metropolis Hastings rejection step. Return if proposal was accepted"""
        accept = False
        if new_like >= self.current_like:
            accept = True
        elif random.random() > self.current_like / new_like * mh_correction:
            accept = True
        if accept:
            self.g = new_g
            self.current_like = new_like
        return accept


class Mover:
    def __init__(self, g):
        self.g = None
        self.interactor = None
        self.set_g(g)

    def set_g(self, g):
        self.g = g
        self.interactor = TreeInteractor(self.g)

    def prune_and_reattach(self):
        if self._is_tree_too_small():
            return 1
        node = random.randrange(len(self.g) - 1)
        self.interactor.prune(node)
        return self.interactor.uniform_attach(node)

    def swap_node(self):
        if self._is_tree_too_small():
            return 1
        n1, n2 = self._get_two_distinct_nodes()
        return self.interactor.swap_labels(n1, n2)

    def swap_subtree(self):
        if self._is_tree_too_small():
            return 1
        n1, n2 = self._get_two_distinct_nodes()
        return self.interactor.swap_subtrees(n1, n2)

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