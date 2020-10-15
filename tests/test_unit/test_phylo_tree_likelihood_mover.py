"""
Regarding builder classes
=========================

All builder classes end in "Builder"
All attributes are private.
All methods prefixed with "_" are private.
Call build() to obtain the constructed object.
"""
import math
import random

import networkx as nx
import numpy as np
import pytest
from hypothesis import given, assume
from hypothesis import strategies as st

from pigglet_testing.builders.tree import PhyloTreeBuilder
from pigglet_testing.builders.tree_likelihood import (
    MCMCBuilder,
    PhyloMoveCaretakerBuilder,
    PhyloTreeLikelihoodCalculatorBuilder,
)
from pytest import approx

from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.tree_interactor import PhyloTreeInteractor
from pigglet.tree_likelihood_mover import (
    PhyloTreeLikelihoodMover,
    PhyloTreeMoveCaretaker,
)


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
        b.with_path(4, 3, 0)
        b.with_branch(3, 1)
        b.with_branch(4, 2)

        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 0)
        b.with_unmutated_gl_at(2, 0)
        calc = b.build()
        like = calc.log_likelihood()
        inter = PhyloTreeInteractor(calc.g)

        # when
        inter.prune_and_regraft(3, (4, 2))
        calc.register_changed_nodes(*list(inter.changed_nodes))

        # then
        assert calc.log_likelihood() == like

    def test_three_samples_one_private_mutation_for_two_samples(self):
        # given
        b = PhyloTreeLikelihoodCalculatorBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_unmutated_gl_at(1, 0)
        b.with_mutated_gl_at(2, 0)

        b.with_path(4, 3, 0)
        b.with_branch(3, 1)
        b.with_branch(4, 2)
        # b.with_path(0, 1, 2)
        # b.with_branch(1, 3)
        # b.with_branch(0, 4)

        calc = b.build()
        inter = PhyloTreeInteractor(calc.g)

        # when
        inter.prune_and_regraft(2, (3, 0))
        calc.register_changed_nodes(*list(inter.changed_nodes))

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
        mover = mcmc.like_mover
        calc = mover.calc

        # when
        mover.make_and_register_random_move()

        like = calc.register_changed_nodes(
            *mover.changed_nodes
        ).attachment_log_like.copy()

        recalc_like = PhyloTreeLikelihoodCalculator(
            calc.g, calc.gls
        ).attachment_log_like

        # then
        for idx in range(recalc_like.shape[0]):
            assert np.allclose(recalc_like[idx], like[idx]), idx


class TestTreeChanges:
    @given(st.randoms(use_true_random=True))
    def test_swap_leaf(self, prng):
        # given
        b = PhyloMoveCaretakerBuilder(prng=prng)
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

    @pytest.mark.parametrize("node", list(range(5)))
    def test_espr_accepts_all_non_root_nodes(self, node):
        # given
        c = PhyloTreeMoveCaretaker(
            nx.DiGraph([[6, 5], [6, 4], [5, 0], [5, 1], [4, 2], [4, 3]]),
            prng=random,
        )

        # when/then
        ret_node, edge = c.extending_subtree_prune_and_regraft(node=node)

        # then
        assert ret_node == node

    @given(st.randoms(use_true_random=False))
    def test_espr(self, prng):
        # given
        b = PhyloMoveCaretakerBuilder(prng=prng)
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
        elif node == 5 and edge in ((4, 2), (4, 3)):
            assert caretaker.changed_nodes == {6}

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

    @given(st.randoms(use_true_random=False))
    def test_espr_also_returns_original_tree_with_mh_correction_1(self, prng):
        # given
        b = PhyloTreeBuilder(prng=prng)
        b.with_balanced_tree(height=3)

        inter = PhyloTreeInteractor(b.build(), prng=prng)

        # when
        node = 7
        memento, edge = inter.extend_prune_and_regraft(node, 0.99999)

        # then
        assert set(inter.changed_nodes) == set()
        assert inter.mh_correction == 1.0
        assert edge in {(1, 3), (3, 8)}

    def test_double_change_prune_and_regraft(self):
        # given
        b = PhyloTreeBuilder(prng=None)
        b.with_balanced_tree(height=4)

        inter = PhyloTreeInteractor(b.build(), prng=None)

        # when/then
        inter.prune_and_regraft(7, (2, 5))
        assert inter.changed_nodes == {1, 3}
        inter.prune_and_regraft(14, (4, 9))
        assert inter.changed_nodes == {6, 2}

    def test_double_change_rooted_prune_and_regraft(self):
        # given
        b = PhyloTreeBuilder(prng=None)
        b.with_balanced_tree(height=4)

        inter = PhyloTreeInteractor(b.build(), prng=None)

        # when/then
        inter.rooted_prune_and_regraft(7)
        assert inter.changed_nodes == {1, 3}
        inter.rooted_prune_and_regraft(14)
        assert inter.changed_nodes == {6, 2}

    def test_double_change_swap_leaves(self):
        # given
        b = PhyloTreeBuilder(prng=None)
        b.with_balanced_tree(height=4)

        inter = PhyloTreeInteractor(b.build(), prng=None)

        # when/then
        inter.swap_leaves(15, 17)
        assert inter.changed_nodes == {7, 8}
        inter.swap_leaves(18, 19)
        assert inter.changed_nodes == {8, 9}


@given(
    st.integers(min_value=4, max_value=10), st.randoms(use_true_random=False),
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
    old_edges = list(mover.g.edges)
    mover.make_and_register_random_move()
    assume(sorted(old_edges) != sorted(mover.g.edges))
    assert mover.calc.has_changed_nodes()
    mover.attachment_log_like
    assert not mover.calc.has_changed_nodes()
    mover.undo()
    assert mover.calc.has_changed_nodes()

    # then
    assert like is not None
    for u in range(like.shape[0]):
        assert np.allclose(like[u], mover.attachment_log_like[u])


@given(
    st.randoms(use_true_random=False), st.integers(min_value=1, max_value=200)
)
def test_arbitrary_moves_with_high_certainty_deliver_real_likelihood(
    prng, certainty
):
    # given
    n_samples = 4
    b = MCMCBuilder()
    b.with_prng(prng)
    b.with_phylogenetic_tree()
    b.with_msprime_tree(
        sample_size=n_samples,
        Ne=1e6,
        mutation_rate=1e-4,
        random_seed=prng.randrange(1, 2 ^ 32),
    )
    b.with_certainty(certainty)
    mcmc = b.build()
    mover = PhyloTreeLikelihoodMover(g=mcmc.g, gls=mcmc.gls, prng=prng)

    # when/then
    like = mover.random_move_and_get_like()
    assert like != -np.inf
    new_like = mover.random_move_and_get_like()
    assert new_like != -np.inf
    mover.undo()

    # then
    assert pytest.approx(like, mover.log_likelihood())
