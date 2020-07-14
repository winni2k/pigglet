def roots_of_tree(g):
    """Return list of the roots of the tree g"""
    return [n for n, d in g.in_degree if d == 0]


def parent_node_of(g, n):
    parents = list(g.pred[n])
    assert len(parents) == 1
    return parents[0]
