import networkx as nx
import numpy as np
import pytest

NUM_GLS = 2


class TreeLikelihoodCalculator:
    """Calculates likelihood of tree (g) from gls"""

    def __init__(self, g, gls, sample_nodes):
        self.g = g
        self.gls = gls
        self.sample_nodes = sample_nodes
        assert self.gls.shape[1] == len(self.sample_nodes)

        self.sample_ids = [g.nodes[n]['sample_id'] for n in self.sample_nodes]
        assert max(self.sample_ids) + 1 == len(self.sample_nodes)

        self.mutation_matrix_mask = np.zeros_like(gls, np.bool_).reshape(-1)
        self.mutation_matrix_mask[::NUM_GLS] = True
        self.mutation_matrix_mask = self.mutation_matrix_mask.reshape(gls.shape)

        self.root = roots_of_tree(self.g)
        assert len(self.root) == 1
        self.root = self.root[0]


    def calculate_likelihood(self):
        # self._update_mutation_matrix_mask()
        return np.sum(self.gls[self.mutation_matrix_mask].reshape(self.gls.shape[0], self.gls.shape[1]))

    def _update_mutation_matrix_mask(self):
        for path in enumerate(nx.all_simple_paths(self.g, self.root, self.sample_nodes)):
            for node in path:
                try:
                    self.g.nodes[node]['mutations']
                except KeyError:
                    pass


class BalancedTreeLikelihoodBuilder:
    """All methods and attributes are private, except for build() and methods prefixed with "with_"
    """

    def __init__(self):
        self.num_sites = 3
        self.height = 2
        self.likelihood_peaks = set()
        self.mutated_nodes = set()
        self.sample_id_to_node = {}
        self.mutation_gl = 1
        self.gls = None
        self.g = None

    def _mutate_nodes(self):
        # mutate nodes
        for sample_id, site_idx in self.mutated_nodes:
            try:
                self.g.nodes[self.sample_id_to_node[sample_id]]['mutations'].append(site_idx)
            except KeyError:
                self.g.nodes[self.sample_id_to_node[sample_id]]['mutations'] = [site_idx]
            self.gls[site_idx, sample_id, 1] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self):
        self.g = nx.balanced_tree(2, self.height, nx.DiGraph())
        sample_nodes = sample_nodes_of_tree(self.g)
        for idx, sample_node in enumerate(sample_nodes):
            self.g.nodes[sample_node]['sample_id'] = idx
            self.sample_id_to_node[idx] = sample_node
        self.gls = np.zeros((self.num_sites, len(sample_nodes), NUM_GLS))

        self._add_likelihood_peaks()
        self._mutate_nodes()

        return self.g, self.gls

    def with_likelihood_peak_at_all_hom_ref(self):
        self.likelihood_peaks.add(0)

    def with_mutated_sample_id(self, sample_id, site_idx):
        self.mutated_nodes.add((sample_id, site_idx))


def sample_nodes_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.out_degree)]

def roots_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.in_degree)]

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
        b = BalancedTreeLikelihoodBuilder()
        b.with_likelihood_peak_at_all_hom_ref()
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 4 * 3

    # todo: return sample nodes in random order
    @pytest.mark.xfail(reason='Still needs implementing')
    def test_with_mutation_before_fourth_node(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_sample_id(3, 1)
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 1

