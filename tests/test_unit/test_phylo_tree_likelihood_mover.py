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
from hypothesis import given
from hypothesis import strategies as st

from pigglet_testing.builders.tree import PhyloTreeBuilder
from pigglet_testing.builders.tree_likelihood import (
    MCMCBuilder,
    PhyloMoveExecutorBuilder,
    PhyloTreeLikelihoodCalculatorBuilder,
)
from pytest import approx

from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.tree_interactor import PhyloTreeInteractor
from pigglet.tree_likelihood_mover import PhyloTreeLikelihoodMover


def get_mutation_likelihood(calc, site_idx):
    return np.exp(calc.attachment_marginalized_log_likelihoods()[site_idx])


def sum_of_exp_of(*log_likelihoods):
    return sum(math.exp(x) for x in log_likelihoods)


def log_sum_of_exp_of(*log_likelihoods):
    return math.log(sum_of_exp_of(*log_likelihoods))


class TestPruneAndRegraft:
    def test_prune_regraft_to_root_edge_does_not_change_likelihoods(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_path(0, 1, 2)
        b.with_branch(1, 3)
        b.with_branch(0, 4)
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 0)
        b.with_unmutated_gl_at(2, 0)
        calc = b.build()
        like = calc.log_likelihood()
        inter = PhyloTreeInteractor(calc.g)

        # when
        inter.prune_and_regraft(1, (0, 4))

        # then
        assert calc.log_likelihood() == like

    def test_three_samples_one_private_mutation_for_two_samples(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        b.with_mutated_gl_at(2, 0)

        b.with_path(0, 1, 2)
        b.with_branch(1, 3)
        b.with_branch(0, 4)

        calc = b.build()
        inter = PhyloTreeInteractor(calc.g)

        # when
        inter.prune_and_regraft(4, (1, 2))

        like1 = get_mutation_likelihood(calc, 0)

        # then
        assert like1 == approx(sum_of_exp_of(-1, -3, 0, -1, -1))


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


class TestRecalculateAttachmentLogLikeFromNodes:
    @given(
        st.integers(min_value=4, max_value=10),
        st.randoms(use_true_random=False),
    )
    def test_arbitrary_trees(self, n_samples, prng):
        # given
        b = MCMCBuilder()
        b.with_prng(prng)
        b.with_phylogenetic_tree()

        for sample in range(n_samples):
            b.with_mutated_gl_at(sample, sample)

        mcmc = b.build()
        mover = mcmc.mover
        calc = mover.calc

        # when
        mover.random_move()

        like = calc.register_changed_nodes(
            *mover.changed_nodes
        ).attachment_log_like.copy()

        recalc_like = PhyloTreeLikelihoodCalculator(
            calc.g, calc.gls
        ).attachment_log_like

        # then
        for idx in range(recalc_like.shape[0]):
            assert np.allclose(recalc_like[idx], like[idx]), idx


class TestChangedNodes:
    @given(st.randoms())
    def test_swap_leaf(self, prng):
        # given
        b = PhyloMoveExecutorBuilder(prng=prng)
        b.with_path(6, 5, 0)
        b.with_branch(5, 1)
        b.with_branch(6, 4)
        b.with_branch(4, 2)
        b.with_branch(4, 3)

        caretaker = b.build()

        # when
        n1, n2 = caretaker.swap_leaf()

        # then
        if {n1, n2} in ({0, 1}, {2, 3}):
            assert caretaker.changed_nodes == set()
        else:
            assert set(caretaker.changed_nodes) == {4, 5}
            # assert caretaker.changed_nodes == {4: {2, 3}, 5: {0, 1}}

    @given(st.randoms(use_true_random=False))
    def test_espr(self, prng):
        # given
        b = PhyloMoveExecutorBuilder(prng=prng)
        b.with_path(6, 5, 0)
        b.with_branch(5, 1)
        b.with_branch(6, 4)
        b.with_branch(4, 2)
        b.with_branch(4, 3)

        caretaker = b.build()

        # when
        node, edge = caretaker.extending_subtree_prune_and_regraft()

        # then
        assert set(caretaker.changed_nodes).isdisjoint(set(range(4)))
        assert node not in caretaker.changed_nodes
        if node == 4 and edge in ((5, 0), (5, 1)):
            assert caretaker.changed_nodes == {6}
            # assert caretaker.changed_nodes == {5: {0, 1}, 6: set()}
        elif node == 5 and edge in ((4, 2), (4, 3)):
            assert caretaker.changed_nodes == {6}
            # assert caretaker.changed_nodes == {4: {2, 3}, 6: set()}

    @given(st.randoms(use_true_random=False))
    def test_espr_bigger(self, prng):
        # given
        b = PhyloTreeBuilder(prng=prng)
        b.with_balanced_tree(height=4)

        inter = PhyloTreeInteractor(b.build(), prng=prng)

        # when
        node = 7
        memento, edge = inter.extend_prune_and_regraft(node, 0.0001)

        # then
        assert set(inter.changed_nodes).isdisjoint(set(range(7, 15)))
        assert node not in inter.changed_nodes

        if edge[0] in (2, 5, 6, 11, 12, 13, 14):
            assert inter.changed_nodes == {3, 1}


@given(
    st.integers(min_value=4, max_value=10), st.randoms(),
)
def test_arbitrary_trees_and_moves_undo_ok(n_samples, prng):
    # given
    b = MCMCBuilder()
    b.with_prng(prng)
    b.with_phylogenetic_tree()

    for sample in range(n_samples):
        b.with_mutated_gl_at(sample, sample)

    mcmc = b.build()
    mover = PhyloTreeLikelihoodMover(g=mcmc.g, gls=mcmc.gls, prng=prng)
    like = mover.attachment_log_like.copy()

    # when/then
    mover.random_move()
    assert mover.calc.has_changed_nodes()
    mover.attachment_log_like
    assert not mover.calc.has_changed_nodes()
    mover.undo()
    assert mover.calc.has_changed_nodes()

    # then
    assert like is not None
    for u in range(like.shape[0]):
        assert np.allclose(like[u], mover.attachment_log_like[u])
