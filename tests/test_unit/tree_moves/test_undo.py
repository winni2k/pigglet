import itertools as it
import random

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from builders.tree_interactor import (
    MutationTreeInteractorBuilder,
    PhyloTreeInteractorBuilder,
)


class TestMutationTreeInteractor:
    def test_prune_of_single_mutation(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        move = interactor.prune(0)
        interactor.undo(move)

        # then
        assert list(interactor.g.edges()) == [(-1, 0)]

    def test_attach_of_single_mutation(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)
        memento = interactor.attach(0, -1)
        interactor.undo(memento)

        # then
        assert list(interactor.g.edges()) == []

    @given(st.randoms())
    def test_uniform_attach_to_root_connected_nodes(self, prng):
        # given
        b = MutationTreeInteractorBuilder(prng=prng)
        b.with_balanced_tree(2)
        inter = b.build()

        inter.prune(0)

        # when
        memento = inter.uniform_attach(0)
        inter.undo(memento)

        # then
        assert nx.ancestors(inter.g, 0) <= set()

    def test_swap_label_of_two_node_labels_in_balanced_tree(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when
        memento = inter.swap_labels(0, 4)
        inter.undo(memento)

        # then
        assert set(nx.ancestors(inter.g, 0)) == {-1}
        assert set(nx.ancestors(inter.g, 4)) == {-1, 1}

    @pytest.mark.parametrize("swap_nodes", [(2, 4), (4, 2)])
    def test_swap_subtree_of_two_nodes_in_different_lineages(self, swap_nodes):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        memento = inter.swap_subtrees(*swap_nodes)
        inter.undo(memento)

        # then
        assert set(nx.ancestors(inter.g, 2)) == {-1, 0}
        assert set(nx.descendants(inter.g, 2)) == {6, 7}
        assert set(nx.ancestors(inter.g, 4)) == {-1, 1}
        assert set(nx.descendants(inter.g, 4)) == {10, 11}

    @pytest.mark.parametrize("swap_nodes", [(0, 2), (2, 0)])
    def test_swap_subtree_of_two_nodes_in_line(self, swap_nodes):
        # given
        b = MutationTreeInteractorBuilder(prng=random)
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        memento = inter.swap_subtrees(*swap_nodes)
        inter.undo(memento)

        # then
        assert nx.ancestors(inter.g, 0) == {-1}
        assert nx.descendants(inter.g, 0) == {2, 3, 6, 7, 8, 9}
        assert nx.ancestors(inter.g, 2) == {-1, 0}
        assert nx.descendants(inter.g, 2) == {6, 7}


class TestPhyloTreeInteractor:
    @pytest.mark.parametrize("u,v", it.permutations(range(3, 7), 2))
    def test_prune_of_samples_on_four_sample_tree(self, u, v):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        interactor = b.build()

        # when
        move = interactor.swap_leaves(u, v)
        interactor.undo(move)

        # then
        assert set(interactor.g.edges()) == {
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
        }
        assert interactor.g.nodes[0]["leaves"] == set(range(3, 7))

    @pytest.mark.parametrize("u,e", [(3, (2, 5)), (1, (2, 5)), (1, (0, 2))])
    def test_prune_and_regraft_on_four_sample_tree(self, u, e):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=3)
        inter = b.build()
        edges = set(inter.g.edges)

        # when
        move = inter.prune_and_regraft(u, e)
        inter.undo(move)

        # then
        assert set(inter.g.edges()) == edges
        assert inter.g.nodes[0]["leaves"] == set(range(7, 15))
