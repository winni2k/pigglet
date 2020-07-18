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
        assert nodes == [2, 1, 0]

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
        assert nodes == [4, 3, 2, 1, 0]

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

    @given(st.permutations([1, 2, 4, 6]))
    def test_four_leaves_pathological_situation2(self, frontier):
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
        assert nodes in ([6, 4, 2, 1, 0], [6, 4, 1, 2, 0])
