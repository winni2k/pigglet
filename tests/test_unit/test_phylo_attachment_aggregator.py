import numpy as np
import pytest
from pigglet_testing.builders.tree_likelihood import (
    PhyloTreeLikelihoodCalculatorBuilder,
)
from scipy.special import logsumexp

from pigglet.aggregator import PhyloAttachmentAggregator


class TestAddAttachmentlogLikes:
    def test_no_attachment_scores(self):
        # given
        agg = PhyloAttachmentAggregator()

        # then
        assert agg.attachment_scores is None

    def test_two_samples_one_site_no_mutation(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_balanced_tree(height=1, rev=True)
        b.with_unmutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        calc = b.build()
        agg = PhyloAttachmentAggregator()

        # when
        agg.add_attachment_log_likes(calc)

        # then
        expected_log_likes = np.array([[-2], [-1], [-1]])
        site_sum_like = logsumexp(expected_log_likes, axis=0)
        log_likes = agg.attachment_scores
        log_likes += site_sum_like[0]
        assert log_likes[0, 0] == pytest.approx(logsumexp([-2, -1]))
        assert log_likes[0, 1] == pytest.approx(logsumexp([-2, -1]))

    def test_two_samples_three_mutations(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        b.with_mutated_gl_at(1, 1)
        b.with_unmutated_gl_at(0, 1)
        b.with_mutated_gl_at(1, 2)
        b.with_unmutated_gl_at(0, 2)
        b.with_balanced_tree(1, rev=True)
        calc = b.build()
        agg = PhyloAttachmentAggregator()

        # when
        agg.add_attachment_log_likes(calc)

        # then
        expected_log_likes = np.array([[-1, -1, -1], [-2, 0, 0], [0, -2, -2]])
        site_sum_like = logsumexp(expected_log_likes, axis=0)
        log_likes = agg.attachment_scores
        assert log_likes[0, 0] == logsumexp([-1, 0]) - site_sum_like[0]
        assert log_likes[1, 0] == logsumexp([-1, -2]) - site_sum_like[1]
        assert log_likes[2, 0] == logsumexp([-1, -2]) - site_sum_like[2]
        assert log_likes[0, 1] == logsumexp([-1, -2]) - site_sum_like[0]
        assert log_likes[1, 1] == logsumexp([-1, 0]) - site_sum_like[1]
        assert log_likes[2, 1] == logsumexp([-1, 0]) - site_sum_like[2]

    @pytest.mark.parametrize("num_additions", [1, 2, 3])
    def test_two_samples_two_private_mutations_repeat_add(self, num_additions):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(0, 1)
        b.with_mutated_gl_at(1, 1)
        b.with_unmutated_gl_at(1, 0)
        b.with_balanced_tree(1, rev=True)
        calc = b.build()
        agg = PhyloAttachmentAggregator()

        # when
        for _ in range(num_additions):
            agg.add_attachment_log_likes(calc)

        # then
        expected_log_likes = np.array([[-1, -1], [0, -2], [-2, 0]])
        site_sum_like = logsumexp(expected_log_likes, axis=0)
        site_sample_likes = np.array(
            [
                [logsumexp([-1, 0]), logsumexp([-1, -2])],
                [logsumexp([-1, -2]), logsumexp([-1, -0])],
            ]
        )
        site_sample_likes -= site_sum_like
        like = agg.attachment_scores
        expect_likes = site_sample_likes.copy()
        for _ in range(num_additions - 1):
            expect_likes = np.logaddexp(site_sample_likes, expect_likes)
        assert agg.num_additions == num_additions
        assert like == pytest.approx(expect_likes)
        assert agg.averaged_mutation_probabilities() == pytest.approx(
            site_sample_likes
        )
