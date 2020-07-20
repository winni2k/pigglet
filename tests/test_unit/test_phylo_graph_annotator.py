import networkx as nx
from hypothesis import given
from hypothesis import strategies as st

from pigglet.tree_interactor import postorder_nodes_from_frontier


class TestPostorderNodesFromFrontier:
    def test_two_leaves(self):
        # given
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)

        # when
        nodes = list(postorder_nodes_from_frontier(g, 0))

        # then
        assert nodes == [0]

    def test_two_leaves_all_nodes(self):
        # given
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)

        # when
        nodes = list(postorder_nodes_from_frontier(g, [0, 1, 2]))

        # then
        assert nodes in ([2, 1, 0], [1, 2, 0])

    def test_three_leaves(self):
        # given
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)

        # when
        nodes = list(postorder_nodes_from_frontier(g, range(5)))

        # then
        assert len(set(nodes)) == 5
        assert all(u in g for u in g)
        assert nodes[4] == 0
        assert set(nodes[0:3]) == {1, 3, 4} or nodes[0] == 2

    @given(st.permutations([0, 1, 4, 6]))
    def test_four_leaves_pathological_situation(self, frontier):
        # given
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)
        g.add_edge(4, 5)
        g.add_edge(4, 6)

        # when
        nodes = list(postorder_nodes_from_frontier(g, frontier))

        # then
        assert nodes == [6, 4, 1, 0]

    @given(
        st.tuples(
            st.lists(st.sampled_from([6, 8, 9]), max_size=3), st.just((7, 10))
        ).map(lambda x: list(x[0]) + list(x[1])),
        st.randoms(use_true_random=False),
    )
    def test_bug1_in_run_of_nodes(self, frontier, prng):
        prng.shuffle(frontier)
        # given
        g = nx.DiGraph(
            [
                (9, 8),
                (9, 10),
                (8, 0),
                (8, 6),
                (6, 7),
                (6, 5),
                (7, 4),
                (7, 2),
                (10, 1),
                (10, 3),
            ]
        )

        # when
        nodes = list(postorder_nodes_from_frontier(g, frontier))

        # then
        assert nodes in ([10, 7, 6, 8, 9], [7, 6, 8, 10, 9])

    def test_bug1_in_run_of_nodes_again(self):
        # given
        frontier = [7, 10]
        print(frontier)
        g = nx.DiGraph(
            [
                (9, 8),
                (9, 10),
                (8, 0),
                (8, 6),
                (6, 7),
                (6, 5),
                (7, 4),
                (7, 2),
                (10, 1),
                (10, 3),
            ]
        )

        # when
        nodes = list(postorder_nodes_from_frontier(g, frontier))

        # then
        assert nodes in ([10, 7, 6, 8, 9], [7, 6, 8, 10, 9])
