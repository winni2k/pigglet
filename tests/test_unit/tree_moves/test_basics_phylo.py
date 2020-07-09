import pytest

from builders.tree_interactor import PhyloTreeInteractorBuilder
from pigglet.tree import PhyloTreeInteractor


class TestPruneEdge:
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


class TestAttachNodeToEdge:
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

    def test_raises_when_asked_to_attach_non_existant_node(self):
        # given
        inter = PhyloTreeInteractor()

        # when
        with pytest.raises(ValueError):
            inter.attach_node_to_edge(3, (0, 1))


class TestCreateSampleOnEdge:
    def test_creates_new_node_on_minimal_tree(self):
        # given
        inter = PhyloTreeInteractor()

        # when
        inter.create_sample_on_edge(3, (0, 1))

        # then
        assert 5 == len(inter.g)
        new_parent = list(inter.g.pred[3])[0]
        assert new_parent == list(inter.g.pred[1])[0]
        assert (0, 1) not in inter.g.edges
