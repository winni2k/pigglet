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

    def test_one_sample_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 1)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like = calc.calculate_likelihood([-1])

        # then
        assert like == 1

    def test_two_samples_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like = calc.calculate_likelihood([-1, -1])

        # then
        assert like == 2

    def test_one_sample_one_private_mutation(self):
        # given

        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when/then
        assert calc.calculate_likelihood([0]) == 1
        assert calc.calculate_likelihood([-1]) == 0

    def test_one_sample_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when
        like = calc.calculate_likelihood([1])

        # then
        assert like == 2

    def test_four_samples_two_mutations_and_likelihood_one(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(1)
        b.with_gl_dimensions(2, 4)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when/then
        assert calc.calculate_likelihood([1, 1, -1, -1]) == 6
        assert calc.calculate_likelihood([1, 1, 0, 0]) == 4
        assert calc.calculate_likelihood([1, 1, 1, 1]) == 4

    @pytest.mark.parametrize('sample_id_to_mutate,exp_like', [(0, 0), (1, 1)])
    def test_with_two_samples_and_private_mutation(self, sample_id_to_mutate, exp_like):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_gl_dimensions(1, 2)

        b.with_mutated_gl_at(sample_id_to_mutate, 0)
        calc = b.build()

        # when
        like = calc.calculate_likelihood([-1, 0])

        # then
        assert like == exp_like

    def test_raises_on_invalid_attachment_idx(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree()
        calc = b.build()

        # when/then
        with pytest.raises(AssertionError):
            calc.calculate_likelihood([4])

    def test_with_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(-1, 1)

        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 1)

        calc = b.build()

        # when
        like = calc.calculate_likelihood([0, 1])

        # then
        assert like == 2

    def test_with_doubleton_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(1)
        b.with_gl_dimensions(2, 4)
        b.with_mutated_gl_at(2, 1)
        b.with_mutated_gl_at(3, 1)
        calc = b.build()

        # when
        like = calc.calculate_likelihood([0, 0, 1, 1])

        # then
        assert like == 2
