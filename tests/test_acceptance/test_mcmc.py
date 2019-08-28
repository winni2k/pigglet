import networkx as nx
import pytest

from pigglet_testing.builders.tree_likelihood import MCMCBuilder


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

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == set(rand_g.edges())
