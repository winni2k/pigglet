"""
Regarding builder classes
=========================

All builder classes end in "Builder"
All attributes are private.
All methods prefixed with "_" are private.
Call build() to obtain the constructed object.
"""

import networkx as nx
import numpy as np
import pytest
import itertools

from pigglet.constants import NUM_GLS
from pigglet.likelihoods import TreeLikelihoodCalculator


def sample_nodes_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.out_degree)]


class TreeBuilder:
    def __init__(self):
        self.g = nx.DiGraph()
        self.sample_ids = []

    def with_balanced_tree(self, height=2, n_branches=2):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        nx.relabel_nodes(self.g, {n: n - 1 for n in self.g.nodes}, copy=False)
        return self

    def with_mutation_at(self, attachment_node, new_node_id):
        self.g.add_edge(attachment_node, new_node_id)
        return self

    def with_sample_at(self, attachment_node, new_sample_name):
        self.g.add_node(new_sample_name, sample_id=len(self.sample_ids))
        self.g.add_edge(attachment_node, new_sample_name)
        return self

    def with_sample_ids(self, *ids):
        self.sample_ids = ids
        return self

    def build(self):
        if len(self.g.nodes()) == 0:
            self.g.add_edge(-1, 0)
        sample_nodes = sample_nodes_of_tree(self.g)
        if len(self.sample_ids) == 0:
            self.sample_ids = list(range(len(sample_nodes)))
        for sample_id, sample_node in zip(self.sample_ids, sample_nodes):
            self.g.nodes[sample_node]['sample_id'] = sample_id

        return self.g


class TreeLikelihoodBuilder:

    def __init__(self):
        self.likelihood_peaks = set()
        self.mutated_sample_ids = set()
        self.mutated_gls = set()
        self.mutation_gl = 1
        self.gls = None
        self.tree_builder = TreeBuilder()

    def _mutate_gls(self):
        for sample_id, site_idx in itertools.chain(self.mutated_gls,
                                                   self.mutated_sample_ids):
            self.gls[site_idx, sample_id, 1] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self):
        tree = self.tree_builder.build()
        sample_nodes = sample_nodes_of_tree(tree)

        num_sites = len(tree.nodes()) - 1 - len(sample_nodes)
        self.gls = np.zeros((num_sites, len(sample_nodes), NUM_GLS))

        self._add_likelihood_peaks()
        self._mutate_gls()

        return tree, self.gls

    def with_balanced_tree(self, height=2, n_branches=2):
        self.tree_builder.with_balanced_tree(height=height, n_branches=n_branches)
        return self

    def with_likelihood_peak_at_all_hom_ref(self):
        self.likelihood_peaks.add(0)
        return self

    def with_mutated_sample_id_at(self, sample_id, site_idx):
        self.mutated_sample_ids.add((sample_id, site_idx))
        return self

    def with_mutated_gl_at(self, sample_id, site_idx):
        self.mutated_gls.add((sample_id, site_idx))

    def with_sample_ids(self, *ids):
        self.tree_builder.with_sample_ids(*ids)
        return self

    def with_mutation_at(self, attachment_node, new_node_id):
        self.tree_builder.with_mutation_at(attachment_node, new_node_id)
        return self

    def with_sample_at(self, attachment_node, new_sample_name):
        self.tree_builder.with_sample_at(attachment_node, new_sample_name)
        return self


class TreeLikelihoodCalculatorBuilder(TreeLikelihoodBuilder):

    def build(self):
        g, gls = super().build()
        return TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))


class TestLikelihoodOfBalancedTreeHeightTwo:

    def test_one_sample_no_mutation(self):
        # given
        calc = TreeLikelihoodCalculatorBuilder().build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 0

    def test_one_sample_one_private_mutation(self):
        # given

        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_sample_at(0, 'samp_1')
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 1

    def test_one_sample_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_sample_at(1, 'samp_1')
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2

    def test_without_mutations_and_likelihood_one(self):
        # given
        b = TreeLikelihoodBuilder()
        b.with_balanced_tree(1, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        g, gls = b.build()

        sample_nodes = sample_nodes_of_tree(g)
        calc = TreeLikelihoodCalculator(g, gls, sample_nodes)

        # when
        like = calc.calculate_likelihood()

        # then
        assert len(sample_nodes) == 2
        assert like == 0

    def test_with_two_mutations_four_samples_and_likelihood_one(self):
        # given
        b = TreeLikelihoodBuilder()
        b.with_balanced_tree(2)
        b.with_likelihood_peak_at_all_hom_ref()
        g, gls = b.build()

        sample_nodes = sample_nodes_of_tree(g)
        calc = TreeLikelihoodCalculator(g, gls, sample_nodes)

        # when
        like = calc.calculate_likelihood()

        # then
        assert len(sample_nodes) == 4
        assert like == 4

    @pytest.mark.parametrize('sample_id_to_mutate,exp_like', [(0, 0), (1, 1)])
    def test_with_two_samples_and_private_mutation(self, sample_id_to_mutate, exp_like):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_sample_at(-1, 'samp_1')
        b.with_sample_at(0, 'samp_2')

        b.with_mutated_sample_id_at(sample_id_to_mutate, 0)
        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == exp_like

    def test_with_doubleton_and_scrambled_sample_ids(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree()
        b.with_sample_ids(3, 0, 1, 2)
        b.with_mutated_sample_id_at(3, 0)
        b.with_mutated_sample_id_at(0, 0)
        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2

    def test_raises_with_mutation_before_fourth_node_and_non_sequential_sample_ids(self):
        # given
        b = TreeLikelihoodBuilder()
        b.with_balanced_tree()
        b.with_mutated_sample_id_at(3, 1)
        b.with_sample_ids(3, 0, 4, 2)
        g, gls = b.build()

        # when/then
        with pytest.raises(AssertionError):
            calc = TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))

    def test_with_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(-1, 1)
        b.with_sample_at(0, 'samp_1')
        b.with_sample_at(1, 'samp_2')

        b.with_mutated_sample_id_at(0, 0)
        b.with_mutated_sample_id_at(1, 1)

        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2

    def test_with_doubleton_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree()
        b.with_mutated_gl_at(2, 1)
        b.with_mutated_gl_at(3, 1)
        calc = b.build()

        # when
        like = calc.calculate_likelihood()

        # then
        assert like == 2
