import networkx as nx

from pigglet_testing.builders.tree_likelihood import MCMCBuilder


def test_arbitrary_trees(n_burnin_iter):
    # given
    n_mutations = 10
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    leaf_nodes = [x for x in rand_g.nodes() if rand_g.out_degree(x) == 0]
    print('leaf nodes:', leaf_nodes)
    print(dict(rand_g.out_degree()))
    print(rand_g.edges())

    b = MCMCBuilder()
    b.with_n_burnin_iter(n_burnin_iter)
    b.with_n_sampling_iter(0)

    # set GLs to het for every mutation of the sample and to hom ref for all other mutations
    for sample, attachment_point in enumerate(filter(lambda n: n != -1, rand_g)):
        print(sample)
        mutations = set(nx.ancestors(rand_g, attachment_point))
        mutations.add(attachment_point)
        mutations.remove(-1)
        for mutation in mutations:
            b.with_mutated_gl_at(sample, mutation)
        for non_mutation in set(rand_g) - mutations:
            if non_mutation != -1:
                b.with_unmutated_gl_at(sample, non_mutation)

    mcmc = b.build()

    # when
    mcmc.run()


if __name__ == '__main__':
    import timeit

    test_arbitrary_trees(1)
    timeit.timeit('test_arbitrary_trees(1)', number=1)
