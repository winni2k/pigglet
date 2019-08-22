"""
Regarding builder classes
=========================

All builder classes end in "Builder"
All attributes are private.
All methods prefixed with "_" are private.
Call build() to obtain the constructed object.
"""

import pytest

from pigglet.likelihoods import TreeLikelihoodCalculator
from pigglet_testing.builders.tree_likelihood import TreeLikelihoodBuilder, \
    TreeLikelihoodCalculatorBuilder
from pigglet_testing.utils import sample_nodes_of_tree


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
