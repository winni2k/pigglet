import math

import numpy as np
import pytest

from pigglet_testing.builders.tree_likelihood import TreeLikelihoodCalculatorBuilder


class AttachmentAggregator:

    def __init__(self):
        self.attachment_scores = None
        self.num_additions = 0

    def add_attachment_log_likes(self, calc):
        sum_likes = calc.attachment_marginaziled_sample_log_likelihoods()
        log_likes = calc.attachment_log_like - sum_likes
        if self.attachment_scores is None:
            self.attachment_scores = log_likes
        else:
            self.attachment_scores = np.logaddexp(self.attachment_scores, log_likes)
        self.num_additions += 1


class TestAttachmentAggregatorAddAttachmentlogLikes:

    def test_no_attachment_scores(self):
        # given
        agg = AttachmentAggregator()

        # then
        assert agg.attachment_scores is None

    def test_one_sample_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 1)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()
        agg = AttachmentAggregator()

        # when
        agg.add_attachment_log_likes(calc)

        # then
        assert list(np.exp(agg.attachment_scores.reshape(-1))) == pytest.approx(
            [math.e / (math.e + 1), 1 / (math.e + 1)]
        )

    def test_one_sample_one_site_one_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutated_gl_at(0, 0)
        calc = b.build()
        agg = AttachmentAggregator()

        # when
        agg.add_attachment_log_likes(calc)

        # then
        assert list(np.exp(agg.attachment_scores.reshape(-1))) == pytest.approx(
            [1 / (math.e + 1), math.e / (math.e + 1)]
        )

    def test_two_samples_one_site_no_mutation(self):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_gl_dimensions(1, 2)
        b.with_likelihood_peak_at_all_hom_ref()
        calc = b.build()
        agg = AttachmentAggregator()

        # when
        agg.add_attachment_log_likes(calc)

        # then
        likes = np.exp(agg.attachment_scores)
        sum_like = math.e + 1
        for samp_idx in [0, 1]:
            assert list(likes[:, samp_idx]) == pytest.approx(
                [math.e / sum_like, 1 / sum_like])

    @pytest.mark.parametrize('num_additions', [1, 2, 3])
    def test_one_sample_two_private_mutations(self, num_additions):
        # given
        b = TreeLikelihoodCalculatorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(0, 1)
        calc = b.build()
        agg = AttachmentAggregator()

        # when
        for _ in range(num_additions):
            agg.add_attachment_log_likes(calc)

        # then
        likes = np.exp(agg.attachment_scores)
        raw_likes = [1, math.e, math.e ** 2]
        expect_likes = np.zeros(len(raw_likes))
        for _ in range(num_additions):
            expect_likes += np.array(raw_likes) / sum(raw_likes)
        assert list(likes[:, 0]) == pytest.approx(list(expect_likes))
        assert agg.num_additions == num_additions
