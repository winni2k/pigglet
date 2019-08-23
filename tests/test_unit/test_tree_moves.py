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
            self.g.remove_edge(edge[0], edge[1])
        return self

    def attach(self, node, target):
        self.g.add_edge(target, node)
        return self

    def uniform_attach(self, node):
        valid_attachment_points = list(nx.descendants(self.g, self.root))
        attach_idx = random.randrange(len(valid_attachment_points))
        self.attach(node, valid_attachment_points[attach_idx])
        return self

    def swap_labels(self, n1, n2):
        if n1 == n2:
            raise ValueError
        nx.relabel_nodes(self.g, {n1: TMP_LABEL}, copy=False)
        nx.relabel_nodes(self.g, {n2: n1, TMP_LABEL: n2}, copy=False)
        return self

    def swap_subtrees(self, n1, n2):
        if n2 in nx.ancestors(self.g, n1) or n2 in nx.descendants(self.g, n1):
            return self
        n1_parent = list(self.g.in_edges(n1))[0][0]
        n2_parent = list(self.g.in_edges(n2))[0][0]
        self.prune(n1)
        self.prune(n2)
        self.attach(n1, n2_parent)
        self.attach(n2, n1_parent)
        return self


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

    def test_swaps_two_indirectly_related_nodes(self):
        # given
        b = TreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        inter.swap_subtrees(2, 4)

        # then
        assert set(nx.ancestors(inter.g, 2)) == {-1, 1}
        assert set(nx.descendants(inter.g, 2)) == {6, 7}
        assert set(nx.ancestors(inter.g, 4)) == {-1, 0}
        assert set(nx.descendants(inter.g, 4)) == {10, 11}
