from statistics import mean, stdev

import networkx as nx

from pigglet_testing.builders.tree_likelihood import MCMCBuilder


def test_arbitrary_trees(n_burnin_iter=10, n_mutations=10):
    # given
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    # leaf_nodes = [x for x in rand_g.nodes() if rand_g.out_degree(x) == 0]
    # print('leaf nodes:', leaf_nodes)
    # print(dict(rand_g.out_degree()))
    # print(rand_g.edges())

    b = MCMCBuilder()
    b.with_n_burnin_iter(n_burnin_iter)
    b.with_n_sampling_iter(0)

    # set GLs to het for every mutation of the sample and to hom ref for all other mutations
    for sample, attachment_point in enumerate(filter(lambda n: n != -1, rand_g)):
        mutations = set(nx.ancestors(rand_g, attachment_point))
        mutations.add(attachment_point)
        mutations.remove(-1)
        for mutation in mutations:
            b.with_mutated_gl_at(sample, mutation)
        for non_mutation in set(rand_g) - mutations:
            if non_mutation != -1:
                b.with_unmutated_gl_at(sample, non_mutation)

    mcmc = b.build()

    return mcmc


if __name__ == '__main__':
    import timeit

    # mcmc = test_arbitrary_trees(10)
    repeats=3
    for number, mutations in [(10, 100), (10, 300), (1000, 10)]:
        print(f'number={number}, repeats={repeats}, mutations={mutations}')
        timings = timeit.repeat('mcmc.calc.sample_marginalized_log_likelihood()',
                                number=number,
                                repeat=repeats,
                                globals=globals(),
                                setup=f'mcmc = test_arbitrary_trees(n_mutations={mutations})')
        print(mean(timings), stdev(timings))
