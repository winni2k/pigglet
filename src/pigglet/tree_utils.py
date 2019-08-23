def roots_of_tree(g):
    """Return list of the roots of the tree g"""
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.in_degree)]
