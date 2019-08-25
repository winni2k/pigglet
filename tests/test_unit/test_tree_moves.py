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

        # then
        assert nx.ancestors(inter.g, 0) <= {-1, 1, 4, 5}

    def test_also__reattaches_to_root(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
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
        assert set(nx.ancestors(inter.g, 0)) <= {-1, 2, 6, 7}
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
        assert set(nx.ancestors(inter.g, 0)) <= {-1, 1, 2, 3}
        assert set(nx.descendants(inter.g, 0)) == set()
        assert set(nx.ancestors(inter.g, 1)) == {-1}
        assert set(nx.descendants(inter.g, 1)) == {0, 2, 3}
        assert mh_correction == 3 / 1

    @pytest.mark.parametrize('swap_nodes', [(0, 2), (2, 0)])
    def test_swaps_two_nodes_two_jumps_away_in_line_from_unbalanced_tree(self,
                                                                         swap_nodes):
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
        assert set(nx.ancestors(inter.g, 0)) <= {-1, 2, 3, 4}
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
