import networkx as nx
import pytest

from builders.tree_interactor import (
    MutationTreeInteractorBuilder,
    PhyloTreeInteractorBuilder,
)


class TestSwapNodeLabels:
    def test_swaps_two_node_labels_in_balanced_tree(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when
        inter.swap_labels(0, 4)

        # then
        assert set(nx.ancestors(inter.g, 0)) == {-1, 1}
        assert set(nx.ancestors(inter.g, 4)) == {-1}

    def test_raises_if_node_label_is_identical(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when/then
        with pytest.raises(ValueError):
            inter.swap_labels(0, 0)


class TestSwapLeaves:
    def test_swaps_two_node_labels_in_balanced_tree(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when
        inter.swap_leaves(4, 5)

        # then
        assert inter.g.nodes[0]["leaves"] == set(range(3, 7))
        assert inter.g.nodes[1]["leaves"] == {3, 5}
        assert inter.g.nodes[2]["leaves"] == {4, 6}

    def test_raises_if_node_label_is_identical(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when/then
        with pytest.raises(AssertionError):
            inter.swap_leaves(0, 0)

    def test_raises_if_node_label_is_not_leaf(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        # when/then
        with pytest.raises(AssertionError):
            inter.swap_leaves(1, 3)
