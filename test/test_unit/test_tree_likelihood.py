import networkx as nx
import numpy as np
import pytest
import itertools

from pigglet.constants import NUM_GLS
from pigglet.likelihoods import TreeLikelihoodCalculator


def append_to_or_create_list_attribute(node_selection, attribute, val):
    try:
        node_selection['mutations'].add(val)
    except KeyError:
        node_selection['mutations'] = {val}


class BalancedTreeLikelihoodBuilder:
    """All methods and attributes are private, except for build() and methods prefixed with "with_"
    """

    def __init__(self):
        self.num_sites = 3
        self.height = 2
        self.likelihood_peaks = set()
        self.mutated_sample_ids = set()
        self.mutated_nodes = set()
        self.mutated_gls = set()
        self.sample_id_to_node = {}
        self.mutation_gl = 1
        self.sample_ids = None
        self.gls = None
        self.g = None

    def _mutate_nodes(self):
        for sample_id, site_idx in self.mutated_sample_ids:
            append_to_or_create_list_attribute(
                self.g.nodes[self.sample_id_to_node[sample_id]],
                'mutations',
                site_idx
            )
        for node_id, site_idx in self.mutated_nodes:
            append_to_or_create_list_attribute(
                self.g.nodes[node_id],
                'mutations',
                site_idx
            )

    def _mutate_gls(self):
        for sample_id, site_idx in itertools.chain(self.mutated_gls,
                                                   self.mutated_sample_ids):
            self.gls[site_idx, sample_id, 1] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self):
        self.g = nx.balanced_tree(2, self.height, nx.DiGraph())
        sample_nodes = sample_nodes_of_tree(self.g)
        if self.sample_ids is None:
            self.sample_ids = list(range(len(sample_nodes)))
        for sample_id, sample_node in zip(self.sample_ids, sample_nodes):
            self.g.nodes[sample_node]['sample_id'] = sample_id
            self.sample_id_to_node[sample_id] = sample_node
        self.gls = np.zeros((self.num_sites, len(sample_nodes), NUM_GLS))

        self._add_likelihood_peaks()
        self._mutate_nodes()
        self._mutate_gls()

        return self.g, self.gls

    def with_likelihood_peak_at_all_hom_ref(self):
        self.likelihood_peaks.add(0)
        return self

    def with_mutated_sample_id_at_site(self, sample_id, site_idx):
        self.mutated_sample_ids.add((sample_id, site_idx))
        return self

    def with_mutated_node_at(self, node, site_idx):
        self.mutated_nodes.add((node, site_idx))
        return self

    def with_mutated_gl_at(self, node, site_idx):
        self.mutated_gls.add((node, site_idx))

    def with_sample_ids(self, *ids):
        self.sample_ids = ids
        return self


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
        b = BalancedTreeLikelihoodBuilder()
        b.with_likelihood_peak_at_all_hom_ref()
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 4 * 3

    def test_with_mutation_before_fourth_node(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_sample_id_at_site(3, 1)
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 1

    def test_with_mutation_before_fourth_node_and_scrambled_sample_ids(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_sample_id_at_site(3, 1)
        b.with_sample_ids(3, 0, 1, 2)
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 1

    def test_with_mutation_before_fourth_node_and_non_sequential_sample_ids(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_sample_id_at_site(3, 1)
        b.with_sample_ids(3, 0, 4, 2)
        g, gls = b.build()

        # when/then
        with pytest.raises(AssertionError):
            calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

    def test_with_two_private_mutations(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_sample_id_at_site(3, 1)
        b.with_mutated_sample_id_at_site(2, 0)
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2

    def test_with_doubleton_mutation(self):
        # given
        b = BalancedTreeLikelihoodBuilder()
        b.with_mutated_node_at(2, 1)
        b.with_mutated_gl_at(2, 1)
        b.with_mutated_gl_at(3, 1)
        g, gls = b.build()

        calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2
