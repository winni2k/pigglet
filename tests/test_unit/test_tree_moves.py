import random

import networkx as nx
import pytest

from pigglet.tree import TreeInteractor
from pigglet_testing.builders.tree import TreeBuilder


class TreeInteractorBuilder(TreeBuilder):

    def build(self):
        g = super().build()
        return TreeInteractor(g)


class TestPrune:

    def test_removes_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)

        # then
        assert interactor.g.in_degree[0] == 0
        assert interactor.g.out_degree[-1] == 0


class TestAttach:
    def test_reattaches_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)
        interactor.attach(0, -1)

        # then
        assert interactor.g.in_degree[-1] == 0
        assert interactor.g.in_degree[0] == 1
        assert interactor.g.out_degree[-1] == 1
        assert interactor.g.out_degree[0] == 0


class TestUniformAttach:

    @pytest.mark.parametrize('seed', range(4))
    def test_only_reattaches_to_root_connected_nodes(self, seed):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        inter.prune(0)
        random.seed(seed)

        # when
        inter.uniform_attach(0)

        # then
        assert nx.ancestors(inter.g, 0) <= {-1, 1, 4, 5}

    def test_also_reattaches_to_root(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        inter = b.build()

        inter.prune(0)

        # when
        inter.uniform_attach(0)

        # then
        assert list(nx.ancestors(inter.g, 0)) == [-1]


class TestSwapNodeLabels:

    def test_swaps_two_node_labels_in_balanced_tree(self):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when
        inter.swap_labels(0, 4)

        # then
        assert set(nx.ancestors(inter.g, 0)) == {-1, 1}
        assert set(nx.ancestors(inter.g, 4)) == {-1}

    def test_raises_if_node_label_is_identical(self):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when/then
        with pytest.raises(ValueError):
            inter.swap_labels(0, 0)


class TestSwapSubtrees:
    @pytest.mark.parametrize('swap_nodes', [(2, 4), (4, 2)])
    def test_swaps_two_nodes_in_different_lineages(self, swap_nodes):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 2)) == {-1, 1}
        assert set(nx.descendants(inter.g, 2)) == {6, 7}
        assert set(nx.ancestors(inter.g, 4)) == {-1, 0}
        assert set(nx.descendants(inter.g, 4)) == {10, 11}
        assert inter.mh_correction == 1

    @pytest.mark.parametrize('swap_nodes', [(0, 2), (2, 0)])
    def test_swaps_two_nodes_in_line(self, swap_nodes):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 2}, {-1, 2, 6}, {-1, 2, 7}]
        assert set(nx.descendants(inter.g, 0)) == {3, 8, 9}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 3, 6, 7, 8, 9}
        assert inter.mh_correction == 3 / 4

    @pytest.mark.parametrize('swap_nodes', [(0, 1), (1, 0)])
    def test_swaps_two_nodes_in_line_from_unbalanced_tree(self, swap_nodes):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_tree_edge_between(1, 2)
        b.with_tree_edge_between(1, 3)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 1}, {-1, 1, 2}, {-1, 1, 3}]
        assert set(nx.descendants(inter.g, 0)) == set()
        assert set(nx.ancestors(inter.g, 1)) == {-1}
        assert set(nx.descendants(inter.g, 1)) == {0, 2, 3}
        assert inter.mh_correction == 3 / 1

    @pytest.mark.parametrize('swap_nodes', [(0, 2), (2, 0)])
    def test_swaps_two_nodes_two_jumps_away_in_line_from_unbalanced_tree(self,
                                                                         swap_nodes):
        """
        -1->0->1->2->3
                   |>4

        -1->2->3   ... 0->1?
            |>4
        """
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        b.with_tree_edge_between(0, 1)
        b.with_tree_edge_between(1, 2)
        b.with_tree_edge_between(2, 3)
        b.with_tree_edge_between(2, 4)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 2}, {-1, 2, 3}, {-1, 2, 4}]
        assert set(nx.descendants(inter.g, 0)) == {1}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 1, 3, 4}
        assert inter.mh_correction == 3 / 2

    def test_raises_when_swapping_with_root(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        inter = b.build()

        # when/then
        with pytest.raises(ValueError):
            inter.swap_subtrees(-1, 0)


class TestUndo:
    def test_prune_of_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        interactor = b.build()

        # when
        move = interactor.prune(0)
        interactor.undo(move)

        # then
        assert list(interactor.g.edges()) == [(-1, 0)]

    def test_attach_of_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_tree_edge_between(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)
        memento = interactor.attach(0, -1)
        interactor.undo(memento)

        # then
        assert list(interactor.g.edges()) == []

    @pytest.mark.parametrize('seed', range(4))
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

    @pytest.mark.parametrize('swap_nodes', [(2, 4), (4, 2)])
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

    @pytest.mark.parametrize('swap_nodes', [(0, 2), (2, 0)])
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
