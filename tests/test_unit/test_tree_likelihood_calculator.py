"""
Regarding builder classes
=========================

All builder classes end in "Builder"
All attributes are private.
All methods prefixed with "_" are private.
Call build() to obtain the constructed object.
"""
import math

import numpy as np
import pytest
from pytest import approx

from pigglet_testing.builders.tree_likelihood import TreeLikelihoodCalculatorBuilder


def get_marginal_likelihood(calc, idx):
    return np.exp(calc.attachment_marginaziled_sample_log_likelihoods()[idx])


def sum_of_exp_of(*log_likelihoods):
    return sum(math.exp(x) for x in log_likelihoods)


def log_sum_of_exp_of(*log_likelihoods):
    return math.log(sum_of_exp_of(*log_likelihoods))


class TestSampleLikelihood:

    def test_one_sample_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 1)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like = get_marginal_likelihood(calc, 0)

        # then
        assert like == approx(sum_of_exp_of(0, 1))

    def test_two_samples_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like1 = get_marginal_likelihood(calc, 0)
        like2 = get_marginal_likelihood(calc, 1)

        # then
        assert like1 == approx(sum_of_exp_of(0, 1))
        assert like2 == approx(sum_of_exp_of(0, 1))

    def test_one_sample_one_private_mutation(self):
        # given

        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 1))

    def test_one_sample_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 1, 2))

    def test_four_samples_two_mutations_and_likelihood_one(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(1)
        b.with_gl_dimensions(2, 4)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when/then
        for sample in range(4):
            assert get_marginal_likelihood(calc, sample) == approx(sum_of_exp_of(2, 1, 1))

    @pytest.mark.parametrize('sample_id_to_mutate', [0, 1])
    def test_with_two_samples_and_private_mutation(self, sample_id_to_mutate):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_gl_dimensions(1, 2)

        b.with_mutated_gl_at(sample_id_to_mutate, 0)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, sample_id_to_mutate) == approx(
            sum_of_exp_of(0, 1))
        assert get_marginal_likelihood(calc, abs(sample_id_to_mutate - 1)) == approx(
            sum_of_exp_of(0, 0))

    @pytest.mark.parametrize('phylogenetic_tree', [True, False])
    def test_raises_on_invalid_attachment_idx(self, phylogenetic_tree):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        if phylogenetic_tree:
            b.with_phylogenetic_tree()
        b.with_balanced_tree()
        calc = b.build()

        # when/then
        with pytest.raises(IndexError):
            get_marginal_likelihood(calc, 1)

    @pytest.mark.parametrize('phylogenetic_tree', [True, False])
    def test_with_two_private_mutations(self, phylogenetic_tree):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        if phylogenetic_tree:
            b.with_phylogenetic_tree()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(-1, 1)

        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 1)

        calc = b.build()

        # when
        for sample in range(2):
            assert get_marginal_likelihood(calc, sample) == approx(sum_of_exp_of(0, 0, 1))

    def test_with_doubleton_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(1)
        b.with_gl_dimensions(2, 4)
        b.with_mutated_gl_at(2, 1)
        b.with_mutated_gl_at(3, 1)
        calc = b.build()

        # when
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 0, 0))
        assert get_marginal_likelihood(calc, 1) == approx(sum_of_exp_of(0, 0, 0))
        assert get_marginal_likelihood(calc, 2) == approx(sum_of_exp_of(0, 0, 1))
        assert get_marginal_likelihood(calc, 3) == approx(sum_of_exp_of(0, 0, 1))


class TestSampleMarginalizedLikelihood:

    def test_single_mutation_one_sample(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when
        like = calc.sample_marginalized_log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(0, 1))

    def test_single_mutation_two_samples(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 0)
        calc = b.build()

        # when
        like = calc.sample_marginalized_log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(0, 1, 1, 2))

    def test_two_mutations_one_sample_balanced_tree(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when
        like = calc.sample_marginalized_log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(0, 1, 1))

    def test_two_mutations_one_sample(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when
        like = calc.sample_marginalized_log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(0, 1, 2))

    def test_two_mutations_two_samples(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        b.with_mutated_gl_at(1, 0)
        b.with_mutated_gl_at(1, 1)
        calc = b.build()

        # when
        like = calc.sample_marginalized_log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(0, 1, 2, 1, 2, 3, 2, 3, 4))


class TestMLAttachments:

    def test_single_mutation_one_sample(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when
        attachments = calc.ml_sample_attachments()

        # then
        assert list(attachments) == [1]

    def test_single_mutation_two_samples(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 0)
        calc = b.build()

        # when
        attachments = calc.ml_sample_attachments()

        # then
        assert list(attachments) == [1, 1]

    def test_two_mutations_one_sample(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when
        attachments = calc.ml_sample_attachments()

        # then
        assert list(attachments) == [2]

    def test_two_mutations_two_samples(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        b.with_mutated_gl_at(1, 0)
        calc = b.build()

        # when
        attachments = calc.ml_sample_attachments()

        # then
        assert list(attachments) == [2, 1]


class TestSiteLikelihood:

    def test_one_sample_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_gl_dimensions(1, 1)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like = get_marginal_likelihood(calc, 0)

        # then
        assert like == approx(sum_of_exp_of(0, 1))

    def test_two_sites_one_sample_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_unmutated_gl_at(0, 0)
        b.with_unmutated_gl_at(0, 1)
        b.with_tree_edge_between(-1, 0)
        calc = b.build()

        # when
        like1 = get_marginal_likelihood(calc, 0)
        like2 = get_marginal_likelihood(calc, 1)

        # then
        assert like1 == approx(sum_of_exp_of(1, 0))
        assert like2 == approx(sum_of_exp_of(1, 0))

    def test_one_sample_one_private_mutation(self):
        # given

        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 1))

    def test_one_sample_two_private_mutations(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_tree_edge_between(-1, 0)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 1))
        assert get_marginal_likelihood(calc, 1) == approx(sum_of_exp_of(0, 1))

    def test_four_samples_two_mutations_and_likelihood_one(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_balanced_tree(1)
        b.with_gl_dimensions(4, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when/then
        for site in range(2):
            assert get_marginal_likelihood(calc, site) == approx(sum_of_exp_of(2, 1, 1))

    @pytest.mark.parametrize('attachment_id_to_mutate', [0, 1])
    def test_with_two_samples_and_private_mutation(self, attachment_id_to_mutate):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_tree_edge_between(-1, 0)
        b.with_gl_dimensions(2, 1)

        b.with_mutated_gl_at(0, attachment_id_to_mutate)
        calc = b.build()

        # when/then
        assert get_marginal_likelihood(calc, attachment_id_to_mutate) == approx(
            sum_of_exp_of(0, 1))
        assert get_marginal_likelihood(calc, abs(attachment_id_to_mutate - 1)) == approx(
            sum_of_exp_of(0, 0))

    def test_with_doubleton_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_phylogenetic_tree()
        b.with_balanced_tree(1, n_branches=4)
        b.with_gl_dimensions(2, 4)
        b.with_mutated_gl_at(2, 1)
        b.with_mutated_gl_at(3, 1)
        calc = b.build()

        # when
        assert get_marginal_likelihood(calc, 0) == approx(sum_of_exp_of(0, 0, 1, 1))
        assert get_marginal_likelihood(calc, 1) == approx(sum_of_exp_of(0, 0, 1, 1))


# class TestSampleMarginalizedLikelihood:
#
#     def test_single_mutation_one_sample(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_mutated_gl_at(0, 0)
#         calc = b.build()
#
#         # when
#         like = calc.sample_marginalized_log_likelihood()
#
#         # then
#         assert like == approx(log_sum_of_exp_of(0, 1))
#
#     def test_single_mutation_two_samples(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(1, 0)
#         calc = b.build()
#
#         # when
#         like = calc.sample_marginalized_log_likelihood()
#
#         # then
#         assert like == approx(log_sum_of_exp_of(0, 1, 1, 2))
#
#     def test_two_mutations_one_sample_balanced_tree(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_balanced_tree(1)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(0, 1)
#         calc = b.build()
#
#         # when
#         like = calc.sample_marginalized_log_likelihood()
#
#         # then
#         assert like == approx(log_sum_of_exp_of(0, 1, 1))
#
#     def test_two_mutations_one_sample(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_tree_edge_between(0, 1)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(0, 1)
#         calc = b.build()
#
#         # when
#         like = calc.sample_marginalized_log_likelihood()
#
#         # then
#         assert like == approx(log_sum_of_exp_of(0, 1, 2))
#
#     def test_two_mutations_two_samples(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_tree_edge_between(0, 1)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(0, 1)
#         b.with_mutated_gl_at(1, 0)
#         b.with_mutated_gl_at(1, 1)
#         calc = b.build()
#
#         # when
#         like = calc.sample_marginalized_log_likelihood()
#
#         # then
#         assert like == approx(log_sum_of_exp_of(0, 1, 2, 1, 2, 3, 2, 3, 4))
#
#
# class TestMLAttachments:
#
#     def test_single_mutation_one_sample(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_mutated_gl_at(0, 0)
#         calc = b.build()
#
#         # when
#         attachments = calc.ml_sample_attachments()
#
#         # then
#         assert list(attachments) == [1]
#
#     def test_single_mutation_two_samples(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(1, 0)
#         calc = b.build()
#
#         # when
#         attachments = calc.ml_sample_attachments()
#
#         # then
#         assert list(attachments) == [1, 1]
#
#     def test_two_mutations_one_sample(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_tree_edge_between(0, 1)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(0, 1)
#         calc = b.build()
#
#         # when
#         attachments = calc.ml_sample_attachments()
#
#         # then
#         assert list(attachments) == [2]
#
#     def test_two_mutations_two_samples(self):
#         # given
#         b = TreeLikelihoodCalculatorBuilder()
#         b.with_tree_edge_between(-1, 0)
#         b.with_tree_edge_between(0, 1)
#         b.with_mutated_gl_at(0, 0)
#         b.with_mutated_gl_at(0, 1)
#         b.with_mutated_gl_at(1, 0)
#         calc = b.build()
#
#         # when
#         attachments = calc.ml_sample_attachments()
#
#         # then
#         assert list(attachments) == [2, 1]
