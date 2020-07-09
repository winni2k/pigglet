import random

import networkx as nx
import pytest

from builders.tree_interactor import (
    MutationTreeInteractorBuilder,
    PhyloTreeInteractorBuilder,
)
from pigglet.constants import TreeIsTooSmallError


class TestPrune:
    def test_removes_single_mutation(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)

        # then
        assert interactor.g.in_degree[0] == 0
        assert interactor.g.out_degree[-1] == 0


class TestPruneEdgePhyloTree:
    def test_raises_with_two_leaf_nodes(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_branch(0, 1)
        b.with_branch(0, 2)
        interactor = b.build()

        # when/then
        with pytest.raises(ValueError):
            interactor.prune_edge(0, 1)

    def test_detaches_node(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        interactor = b.build()
        in_degree = interactor.g.in_degree

        # when
        assert in_degree[0] == 0
        assert in_degree[1] == 1
        interactor.prune_edge(0, 1)

        # then
        assert in_degree[1] == 0

    def test_removes_root_of_balanced_tree(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        interactor = b.build()

        # when
        interactor.prune_edge(0, 1)

        # then
        assert 0 not in interactor.g
        assert interactor.g.in_degree[2] == 0

    def test_removes_redundant_internal_node(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=3)
        interactor = b.build()

        # when
        interactor.prune_edge(1, 3)

        # then
        assert 1 not in interactor.g
        assert (0, 4) in interactor.g.edges


class TestAttach:
    def test_reattaches_single_mutation(self):
        # given
        b = MutationTreeInteractorBuilder()
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


class TestAttachNodeToEdgePhyloTree:
    def test_creates_new_node_to_accomodate_edge_attachment(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=3)
        inter = b.build()

        # when
        inter.prune_edge(1, 3)
        n_nodes_in_g = len(inter.g)
        inter.attach_node_to_edge(3, (0, 4))

        # then
        assert n_nodes_in_g + 1 == len(inter.g)
        new_parent = list(inter.g.pred[3])[0]
        assert new_parent == list(inter.g.pred[4])[0]
        assert (0, 4) not in inter.g.edges


class TestUniformAttach:
    @pytest.mark.parametrize("seed", range(4))
    def test_only_reattaches_to_root_connected_nodes(self, seed):
        # given
        b = MutationTreeInteractorBuilder()
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
        b = MutationTreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        inter = b.build()

        inter.prune(0)

        # when
        inter.uniform_attach(0)

        # then
        assert list(nx.ancestors(inter.g, 0)) == [-1]


class TestExtendAttach:
    @pytest.mark.parametrize("seed", range(4))
    def test_only_reattaches_to_root_connected_nodes_with_appropriate_mh_correction(
        self, seed
    ):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_balanced_tree(2)
        inter = b.build()

        inter.prune(0)
        random.seed(seed)

        # when
        inter.extend_attach(0, -1, prop_attach=0.45)

        # then
        assert nx.ancestors(inter.g, 0) <= {-1, 1, 4, 5}
        preds = set(inter.g.pred[0])
        if preds < {-1, 1}:
            assert inter.mh_correction == 1 - 0.45
        elif preds < {4, 5}:
            assert inter.mh_correction == 1
        else:
            assert False

    def test_raises_if_no_move_possible(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_path(1)
        inter = b.build()

        inter.prune(0)

        # when
        with pytest.raises(TreeIsTooSmallError):
            inter.extend_attach(0, -1, prop_attach=0.45)

    def test_double_constrained_move_mh_correction_is_one(self):
        # given
        b = MutationTreeInteractorBuilder()
        b.with_path(2)
        inter = b.build()

        inter.prune(1)

        # when
        inter.extend_attach(1, -1, 0)

        # then
        assert inter.mh_correction == 1

    def test_start_constrained_move_mh_correction(self):
        # given
        random.seed(1)
        b = MutationTreeInteractorBuilder()
        b.with_path(3)
        inter = b.build()

        inter.prune(2)

        # when
        inter.extend_attach(2, -1, 0.999)

        # then
        print(inter.g.edges)
        assert inter.mh_correction == 1 - 0.999

    def test_end_constrained_move_mh_correction(self):
        # given
        random.seed(1)
        b = MutationTreeInteractorBuilder()
        b.with_path(3)
        inter = b.build()

        inter.prune(2)

        # when
        inter.extend_attach(2, 0, 0.05)

        # then
        print(inter.g.edges)
        assert inter.mh_correction == 1 / (1 - 0.05)
