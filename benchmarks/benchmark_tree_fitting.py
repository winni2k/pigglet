from statistics import mean, stdev

import networkx as nx

from pigglet_testing.builders.tree_likelihood import MCMCBuilder, \
    add_gl_at_ancestor_mutations_for


def test_arbitrary_trees_all_mutations_have_sample(n_burnin_iter=10, n_mutations=10):
    # given
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    # leaf_nodes = [x for x in rand_g.nodes() if rand_g.out_degree(x) == 0]
    # print('leaf nodes:', leaf_nodes)
    # print(dict(rand_g.out_degree()))
    # print(rand_g.edges())

    b = MCMCBuilder()
    b.with_normalized_gls()
    b.with_n_burnin_iter(n_burnin_iter)
    b.with_n_sampling_iter(0)

    # set GLs to het for every mutation of the sample and to hom ref for all other mutations
    for sample, attachment_point in enumerate(filter(lambda n: n != -1, rand_g)):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()

    return mcmc


def test_arbitrary_trees_leaves_have_sample(n_burnin_iter=10, n_mutations=10):
    # given
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    leaf_nodes = [x for x in rand_g.nodes() if rand_g.out_degree(x) == 0]

    b = MCMCBuilder()
    b.with_normalized_gls()
    b.with_n_burnin_iter(n_burnin_iter)
    b.with_n_sampling_iter(0)

    # set GLs to het for every mutation of the sample and to hom ref for all other mutations
    for sample, attachment_point in enumerate(leaf_nodes):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()

    return mcmc


def test_arbitrary_trees_leaves_have_sample_after_move(n_burnin_iter=10, n_mutations=10):
    # given
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    leaf_nodes = [x for x in rand_g.nodes() if rand_g.out_degree(x) == 0]

    b = MCMCBuilder()
    b.with_normalized_gls()
    b.with_n_burnin_iter(n_burnin_iter)
    b.with_n_sampling_iter(0)

    # set GLs to het for every mutation of the sample and to hom ref for all other mutations
    for sample, attachment_point in enumerate(leaf_nodes):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()
    mcmc.mover.sample_marginalized_log_likelihood()
    mcmc.mover.random_move()

    return mcmc


if __name__ == '__main__':
    import timeit

    # m (samples) == n (sites)
    repeats = 3
    for setup_string in [
        'test_arbitrary_trees_leaves_have_sample_after_move(n_mutations={mutations})',
        'test_arbitrary_trees_leaves_have_sample(n_mutations={mutations})',
        'test_arbitrary_trees_all_mutations_have_sample(n_mutations={mutations})',
    ]:
        print(f'Testing {setup_string}')
        for number, mutations in [(10, 100), (10, 300), (10, 1000), (10, 3000),
                                  (1000, 10)]:
            print(f'number={number}, repeats={repeats}, mutations={mutations}')
            timings = timeit.repeat('mcmc.mover.sample_marginalized_log_likelihood()',
                                    number=number,
                                    repeat=repeats,
                                    globals=globals(),
                                    setup='mcmc = ' + setup_string.format(
                                        mutations=mutations))
            print(mean(timings), stdev(timings))
