import networkx as nx
import pytest
from builders.tree_interactor import MutationTreeInteractorBuilder


class TestSwapSubtrees:
    @pytest.mark.parametrize("swap_nodes", [(2, 4), (4, 2)])
    def test_swaps_two_nodes_in_different_lineages(self, swap_nodes):
        # given
        b = MutationTreeInteractorBuilder()
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

    @pytest.mark.parametrize("swap_nodes", [(0, 2), (2, 0)])
    def test_swaps_two_nodes_in_line(self, swap_nodes):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(3)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [
            {-1, 2},
            {-1, 2, 6},
            {-1, 2, 7},
        ]
        assert set(nx.descendants(inter.g, 0)) == {3, 8, 9}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 3, 6, 7, 8, 9}
        assert inter.mh_correction == 3 / 4

    @pytest.mark.parametrize("swap_nodes", [(0, 1), (1, 0)])
    def test_swaps_two_nodes_in_line_from_unbalanced_tree(self, swap_nodes):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutation_at(1, 2)
        b.with_mutation_at(1, 3)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [
            {-1, 1},
            {-1, 1, 2},
            {-1, 1, 3},
        ]
        assert set(nx.descendants(inter.g, 0)) == set()
        assert set(nx.ancestors(inter.g, 1)) == {-1}
        assert set(nx.descendants(inter.g, 1)) == {0, 2, 3}
        assert inter.mh_correction == 3 / 1

    @pytest.mark.parametrize("swap_nodes", [(0, 2), (2, 0)])
    def test_swaps_two_nodes_two_jumps_away_in_line_from_unbalanced_tree(
        self, swap_nodes
    ):
        """
        -1->0->1->2->3
                   |>4

        -1->2->3   ... 0->1?
            |>4
        """
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        b.with_mutation_at(1, 2)
        b.with_mutation_at(2, 3)
        b.with_mutation_at(2, 4)
        inter = b.build()

        # when
        inter.swap_subtrees(*swap_nodes)

        # then
        assert set(nx.ancestors(inter.g, 0)) in [
            {-1, 2},
            {-1, 2, 3},
            {-1, 2, 4},
        ]
        assert set(nx.descendants(inter.g, 0)) == {1}
        assert set(nx.ancestors(inter.g, 2)) == {-1}
        assert set(nx.descendants(inter.g, 2)) == {0, 1, 3, 4}
        assert inter.mh_correction == 3 / 2

    def test_raises_when_swapping_with_root(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        inter = b.build()

        # when/then
        with pytest.raises(ValueError):
            inter.swap_subtrees(-1, 0)
