from builders.tree_interactor import PhyloTreeInteractorBuilder
from pigglet.tree_interactor import PhyloTreeInteractor


class TestPruneAndRegraft:
    def test_leaves_tree_unchanged_with_two_leaf_nodes(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_branch(0, 1)
        b.with_branch(0, 2)
        interactor = b.build()

        # when
        interactor.prune_and_regraft(1, (0, 2))

        assert set(interactor.g.edges) == {(0, 1), (0, 2)}
        assert interactor.g.nodes[0]["leaves"] == {1, 2}

    def test_leaves_tree_unchanged_with_four_leaf_nodes(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        interactor = b.build()
        previous_edges = set(interactor.g.edges)

        # when
        interactor.prune_and_regraft(1, (0, 2))

        assert set(interactor.g.edges) == previous_edges
        assert interactor.g.nodes[0]["leaves"] == set(range(3, 7))

    def test_moves_sub_tree(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        inter = b.build()

        # when
        inter.prune_and_regraft(1, (2, 5))

        # then
        assert inter.g.edges == {
            (2, 0),
            (2, 6),
            (0, 1),
            (0, 5),
            (1, 3),
            (1, 4),
        }
        assert inter.g.nodes[2]["leaves"] == set(range(3, 7))
        assert inter.g.nodes[0]["leaves"] == set(range(3, 6))

    def test_bug1(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=3)
        inter = b.build()

        # when
        inter.prune_and_regraft(6, (0, 1))

        # then
        assert inter.g.edges >= {
            (0, 2),
            (0, 5),
            (2, 1),
            (2, 6),
            (1, 3),
            (1, 4),
        }
        assert inter.g.nodes[0]["leaves"] == set(range(7, 15))
        assert inter.g.nodes[2]["leaves"] == set(range(7, 15)) - {11, 12}

    def test_bug2_regraft_edge_pointing_to_parent(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=3)
        inter = b.build()

        # when
        inter.prune_and_regraft(6, (0, 2))

        # then
        assert inter.g.edges >= {
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
        }
        assert inter.g.nodes[0]["leaves"] == set(range(7, 15))
        assert inter.g.nodes[2]["leaves"] == set(range(11, 15))

    def test_bug3_regraft_root_onto_distal_edge(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_path(6, 5, 1)
        b.with_path(5, 4, 0)
        b.with_branch(4, 2)
        b.with_branch(6, 3)
        inter = b.build()

        # when
        inter.rooted_prune_and_regraft(4)

        # then
        assert inter.g.edges >= {
            (5, 6),
            (5, 4),
            (4, 0),
            (4, 2),
            (6, 3),
            (6, 1),
        }
        assert inter.g.nodes[5]["leaves"] == set(range(4))
        assert inter.g.nodes[4]["leaves"] == {0, 2}
        assert inter.g.nodes[6]["leaves"] == {1, 3}


class TestCalculateDescendantLeavesOf:
    def test_two_leaves(self):
        # given
        inter = PhyloTreeInteractor()

        # then
        assert inter.g.nodes[0]["leaves"] == {1, 2}

    def test_four_leaves(self):
        # given
        b = PhyloTreeInteractorBuilder()
        b.with_balanced_tree(height=2)
        inter = b.build()

        # when
        inter.swap_leaves(3, 5)

        # then
        assert inter.g.nodes[0]["leaves"] == {3, 4, 5, 6}
        assert inter.g.nodes[1]["leaves"] == {4, 5}
        assert inter.g.nodes[2]["leaves"] == {3, 6}
