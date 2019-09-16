import networkx as nx
import pytest

from pigglet_testing.builders.tree_likelihood import MCMCBuilder, \
    add_gl_at_ancestor_mutations_for


def test_finds_one_sample_one_site():
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == {(-1, 0)}


def test_finds_two_samples_two_sites():
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    b.with_unmutated_gl_at(1, 0)
    b.with_mutated_gl_at(1, 1)
    b.with_unmutated_gl_at(0, 1)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == {(-1, 0), (-1, 1)}


def test_finds_two_samples_two_sites_in_line():
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    b.with_mutated_gl_at(1, 0)
    b.with_mutated_gl_at(1, 1)
    b.with_unmutated_gl_at(0, 1)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == {(-1, 0), (0, 1)}


@pytest.mark.parametrize('n_mutations', [3, 4])
def test_arbitrary_trees(n_mutations):
    # given
    # n_mutations = 4
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    b = MCMCBuilder()
    b.with_n_burnin_iter(10 * 2 ** n_mutations)

    for sample, attachment_point in enumerate(filter(lambda n: n != -1, rand_g)):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == set(rand_g.edges())


@pytest.mark.parametrize('burnin,sampling', [(0, 3), (3, 0), (3, 3)])
def test_aggregates_the_correct_number_of_runs(burnin, sampling):
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    b.with_n_burnin_iter(burnin)
    b.with_n_sampling_iter(sampling)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert mcmc.agg.num_additions == sampling
