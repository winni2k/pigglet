import random

import networkx as nx
import pytest

from pigglet.constants import TMP_LABEL
from pigglet.tree_utils import roots_of_tree
from pigglet_testing.builders.tree import TreeBuilder


class TreeInteractor:

    def __init__(self, g):
        self.g = g
        self.root = roots_of_tree(g)
        assert len(self.root) == 1
        self.root = self.root[0]

    def prune(self, node):
        for edge in list(self.g.in_edges(node)):
            self.g.remove_edge(*edge)

    def attach(self, node, target):
        self.g.add_edge(target, node)

    def uniform_attach(self, node):
        valid_attachment_points = list(nx.descendants(self.g, self.root))
        return self._uniform_attach_to_nodes(node, valid_attachment_points)

    def swap_labels(self, n1, n2):
        if n1 == n2:
            raise ValueError
        nx.relabel_nodes(self.g, {n1: TMP_LABEL}, copy=False)
        nx.relabel_nodes(self.g, {n2: n1, TMP_LABEL: n2}, copy=False)

    def swap_subtrees(self, n1, n2):
        if n1 == self.root or n2 == self.root:
            raise ValueError
        if n2 in nx.ancestors(self.g, n1):
            return self._uniform_subtree_swap(n2, n1)
        elif n2 in nx.descendants(self.g, n1):
            return self._uniform_subtree_swap(n1, n2)
        n1_parent = self._parent_of(n1)
        n2_parent = self._parent_of(n2)
        self.prune(n1)
        self.prune(n2)
        self.attach(n1, n2_parent)
        self.attach(n2, n1_parent)
        return 1

    def _parent_of(self, n):
        return next(self.g.predecessors(n))

    def _uniform_subtree_swap(self, ancestor, descendant):
        anc_parent = self._parent_of(ancestor)
        dec_descendants = nx.descendants(self.g, descendant)

        self.prune(descendant)
        self.attach(descendant, anc_parent)
        anc_descendants = nx.descendants(self.g, ancestor)

        self.prune(ancestor)
        self._uniform_attach_to_nodes(ancestor, dec_descendants)

        return (len(dec_descendants) + 1) / (len(anc_descendants) + 1)

    def _uniform_attach_to_nodes(self, node, target_nodes):
        target_nodes = list(target_nodes)
        attach_idx = random.randrange(len(target_nodes))
        self.attach(node, target_nodes[attach_idx])


class TreeInteractorBuilder(TreeBuilder):

    def build(self):
        g = super().build()
        return TreeInteractor(g)


class TestPrune:

    def test_removes_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
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
        b.with_mutation_at(-1, 0)
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
        edges = list(inter.g.in_edges(0))

        # then
        assert len(edges) == 1
        assert edges[0][0] in {-1, 1, 4, 5}


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
        mh_correction = inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 2)) == {-1, 1}
        assert set(nx.descendants(inter.g, 2)) == {6, 7}
        assert set(nx.ancestors(inter.g, 4)) == {-1, 0}
        assert set(nx.descendants(inter.g, 4)) == {10, 11}
        assert mh_correction == 1

    @pytest.mark.parametrize('swap_nodes', [(0, 2), (2, 0)])
    def test_swaps_two_nodes_in_line(self, swap_nodes):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        mh_correction = inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 2}, {-1, 2, 6}, {-1, 2, 7}]
        assert set(nx.descendants(inter.g, 0)) == {3, 8, 9}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 3, 6, 7, 8, 9}
        assert mh_correction == 3 / 4

    @pytest.mark.parametrize('swap_nodes', [(0, 1), (1, 0)])
    def test_swaps_two_nodes_in_line_from_unbalanced_tree(self, swap_nodes):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutation_at(1, 2)
        b.with_mutation_at(1, 3)
        inter = b.build()

        # when
        mh_correction = inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 1}, {-1, 1, 2}, {-1, 1, 3}]
        assert set(nx.descendants(inter.g, 0)) == set()
        assert set(nx.ancestors(inter.g, 1)) == {-1}
        assert set(nx.descendants(inter.g, 1)) == {0, 2, 3}
        assert mh_correction == 3 / 1

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
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutation_at(1, 2)
        b.with_mutation_at(2, 3)
        b.with_mutation_at(2, 4)
        inter = b.build()

        # when
        mh_correction = inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [{-1, 2}, {-1, 2, 3}, {-1, 2, 4}]
        assert set(nx.descendants(inter.g, 0)) == {1}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 1, 3, 4}
        assert mh_correction == 3 / 2

    def test_raises_when_swapping_with_root(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        inter = b.build()

        # when/then
        with pytest.raises(ValueError):
            inter.swap_subtrees(-1, 0)
