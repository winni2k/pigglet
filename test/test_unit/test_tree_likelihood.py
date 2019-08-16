import networkx as nx
import numpy as np

NUM_GLS = 2


class TreeLikelihoodCalculator:
    """Calculates likelihood of tree (g) from gls"""

    def __init__(self, g, gls, sample_nodes):
        self.g = g
        self.gls = gls
        self.sample_nodes = sample_nodes
        self.sample_ids = [g.nodes[n]['sample_id'] for n in self.sample_nodes]

    def calculate_likelihood(self):
        genotypes = 0
        return np.sum(self.gls[:, self.sample_ids, genotypes])


class BalancedTreeLikelihoodBuilder:

    def __init__(self):
        self.num_sites = 3
        self.height = 2

    def build(self):
        g = nx.balanced_tree(2, self.height, nx.DiGraph())
        sample_nodes = sample_nodes_of_tree(g)
        for idx, sample_node in enumerate(sample_nodes):
            g.nodes[sample_node]['sample_id'] = idx
        gls = np.zeros((self.num_sites, len(sample_nodes), NUM_GLS))
        return g, gls


def sample_nodes_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.out_degree)]


class TestLikelihoodOfBalancedTreeHeightTwo:

    def test_without_mutations(self):
        # given
        g, gls = BalancedTreeLikelihoodBuilder().build()
        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 0

    def test_without_mutations_and_likelihood_one(self):
        # given
        g, gls = BalancedTreeLikelihoodBuilder().build()
        gls[1, :, 0] = 1
        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == len(sample_nodes_of_tree(g))
