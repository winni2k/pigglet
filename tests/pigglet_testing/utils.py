def sample_nodes_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.out_degree)]
