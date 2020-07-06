import random

import networkx as nx
import pytest

from builders.tree_interactor import TreeInteractorBuilder


class TestUndo:
    def test_prune_of_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        move = interactor.prune(0)
        interactor.undo(move)

        # then
        assert list(interactor.g.edges()) == [(-1, 0)]

    def test_attach_of_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)
        memento = interactor.attach(0, -1)
        interactor.undo(memento)

        # then
        assert list(interactor.g.edges()) == []

    @pytest.mark.parametrize("seed", range(4))
    def test_uniform_attach_to_root_connected_nodes(self, seed):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        inter.prune(0)
        random.seed(seed)

        # when
        memento = inter.uniform_attach(0)
        inter.undo(memento)

        # then
        assert nx.ancestors(inter.g, 0) <= set()

    def test_swap_label_of_two_node_labels_in_balanced_tree(self):
        # given
        b = TreeInteractorBuilder()
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
        b = TreeInteractorBuilder()
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
        b = TreeInteractorBuilder()
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
