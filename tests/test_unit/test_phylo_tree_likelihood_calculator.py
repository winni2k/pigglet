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
from pigglet_testing.builders.tree_likelihood import (
    PhyloTreeLikelihoodCalculatorBuilder,
)
from pytest import approx


def get_mutation_likelihood(calc, site_idx):
    return np.exp(calc.attachment_marginalized_log_likelihoods()[site_idx])


def sum_of_exp_of(*log_likelihoods):
    return sum(math.exp(x) for x in log_likelihoods)


def log_sum_of_exp_of(*log_likelihoods):
    return math.log(sum_of_exp_of(*log_likelihoods))


class TestSampleLikelihood:
    def test_two_samples_one_site_no_mutation(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()

        # when
        like = get_mutation_likelihood(calc, 0)

        # then
        assert like == approx(sum_of_exp_of(-2, -1, -1))

    def test_two_sample_one_private_mutation(self):
        # given

        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        calc = b.build()

        # when
        like = get_mutation_likelihood(calc, 0)

        # then
        assert like == approx(sum_of_exp_of(-1, 0, -2))

    def test_two_samples_two_private_mutations_for_first_sample(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        b.with_unmutated_gl_at(1, 0)
        b.with_unmutated_gl_at(1, 1)
        calc = b.build()

        # when
        like1 = get_mutation_likelihood(calc, 0)
        like2 = get_mutation_likelihood(calc, 1)

        # then
        assert like1 == approx(sum_of_exp_of(-1, 0, -2))
        assert like2 == approx(sum_of_exp_of(-1, 0, -2))

    def test_three_samples_one_private_mutation_for_first_sample(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        b.with_unmutated_gl_at(2, 0)
        b.with_path(0, 1, 2)
        b.with_branch(1, 3)
        b.with_branch(0, 4)
        calc = b.build()

        # when
        like1 = get_mutation_likelihood(calc, 0)

        # then
        assert like1 == approx(sum_of_exp_of(-2, -1, 0, -2, -2))

    def test_three_samples_one_private_mutation_for_two_samples(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(0, 1)
        b.with_unmutated_gl_at(1, 0)
        b.with_unmutated_gl_at(1, 1)
        b.with_unmutated_gl_at(2, 0)
        b.with_mutated_gl_at(2, 1)

        b.with_path(0, 1, 2)
        b.with_branch(1, 3)
        b.with_branch(0, 4)

        calc = b.build()

        # when
        like1 = get_mutation_likelihood(calc, 0)
        like2 = get_mutation_likelihood(calc, 1)

        # then
        assert like1 == approx(sum_of_exp_of(-2, -1, 0, -2, -2))
        assert like2 == approx(sum_of_exp_of(-2, -3, -2, -2, 0))


class TestSampleMarginalizedLikelihood:
    def test_two_samples_one_site_no_mutation(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_unmutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        calc = b.build()

        # when
        like = calc.log_likelihood()

        # then
        assert like == approx(log_sum_of_exp_of(-2, -1, -1))
