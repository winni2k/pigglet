import logging
import random

import msprime
import networkx as nx
import pytest
from hypothesis import given, strategies

from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.mcmc import as_dict_of_dicts
from pigglet_testing.builders.tree_likelihood import (
    MCMCBuilder,
    MutationMoveExecutorBuilder,
    add_gl_at_ancestor_mutations_for,
    add_gl_at_each_ancestor_node_for_nodes,
    PhyloMoveExecutorBuilder,
)

logging.basicConfig(level=logging.INFO)


def test_finds_one_sample_one_site():
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == {(-1, 0)}


# @pytest.mark.parametrize("mutation_tree", (True, False))
def test_finds_two_samples_two_sites():
    mutation_tree = True
    b = MCMCBuilder()
    b.mutation_tree = mutation_tree
    b.with_mutated_gl_at(0, 0)
    b.with_mutated_gl_at(1, 1)
    b.with_unmutated_gl_at(1, 0)
    b.with_unmutated_gl_at(0, 1)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    if mutation_tree:
        assert set(mcmc.map_g.edges()) == {(-1, 0), (-1, 1)}
    else:
        assert set(mcmc.map_g.edges()) == {(2, 0), (2, 1)}


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


@pytest.mark.parametrize("n_mutations", [3, 4])
def test_arbitrary_trees(n_mutations):
    # given
    # n_mutations = 4
    rand_g = nx.gnr_graph(n_mutations, 0, seed=42).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    b = MCMCBuilder()
    b.with_n_burnin_iter(20 * 2 ** n_mutations)

    for sample, attachment_point in enumerate(
        filter(lambda n: n != -1, rand_g)
    ):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == set(rand_g.edges())


@pytest.mark.parametrize("n_samples", [3, 4])
def test_arbitrary_phylo_trees(n_samples):
    # given
    ts = msprime.simulate(
        sample_size=n_samples, recombination_rate=0, random_seed=42
    )
    rand_g = nx.DiGraph(as_dict_of_dicts(ts.first()))

    b = MCMCBuilder()
    b.certainty = 10
    b.reporting_interval = 100
    b.with_phylogenetic_tree()
    b.with_n_burnin_iter(10)
    b.with_n_sampling_iter(10 * 2 ** n_samples)
    add_gl_at_each_ancestor_node_for_nodes(b, rand_g)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    print(rand_g.edges)
    print(mcmc.map_g.edges)
    gls = mcmc.gls

    rand_g_simple_path_lengths = [
        len(path)
        for path in sorted(
            nx.all_simple_paths(nx.Graph(rand_g), 0, range(1, n_samples))
        )
    ]
    map_g_simple_path_lengths = [
        len(path)
        for path in sorted(
            nx.all_simple_paths(nx.Graph(mcmc.map_g), 0, range(1, n_samples))
        )
    ]
    assert rand_g_simple_path_lengths == map_g_simple_path_lengths
    assert (
        PhyloTreeLikelihoodCalculator(rand_g, gls).log_likelihood()
        == PhyloTreeLikelihoodCalculator(mcmc.map_g, gls).log_likelihood()
    )


@pytest.mark.parametrize("burnin,sampling", [(0, 3), (3, 0), (3, 3)])
def test_aggregates_the_correct_number_of_runs(burnin, sampling):
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    b.with_mutated_gl_at(0, 1)
    b.with_n_burnin_iter(burnin)
    b.with_n_sampling_iter(sampling)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert mcmc.agg.num_additions == sampling


class TestMutationMoveExecutor:
    @given(
        strategies.data(),
        strategies.integers(min_value=1, max_value=3),
        strategies.integers(min_value=2, max_value=10),
    )
    def test_undoes_any_move(self, data, num_moves, n_mutations):
        # given
        b = MutationMoveExecutorBuilder()
        b.with_random_tree(n_mutations)
        exe = b.build()
        original_tree = exe.g.copy()

        for idx in range(num_moves):
            exe.available_moves[
                data.draw(strategies.integers(min_value=0, max_value=2))
            ]()
            if idx == 0:
                memento = exe.memento
            else:
                memento.append(exe.memento)

        # when
        exe.undo(memento)

        # then
        assert set(exe.g.edges) == set(original_tree.edges)


class TestPhyloMoveExecutor:
    @pytest.mark.parametrize("seed", range(4))
    def test_a_bunch_of_moves(self, seed):
        # given
        b = PhyloMoveExecutorBuilder()
        b.with_balanced_tree(height=3)
        exe = b.build()
        random.seed(seed)

        for i in range(20):
            old_g = exe.g.copy()
            try:
                exe.random_move()
            except Exception:
                logging.error(f"Original tree on iteration {i}:")
                logging.error(old_g.nodes(data=True))
                logging.error(old_g.edges)
                raise
            exe.register_mh_result(True)
