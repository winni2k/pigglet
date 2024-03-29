import logging
import random

import msprime
import networkx as nx
import pytest
from hypothesis import given, strategies
from hypothesis import strategies as st


from pigglet_testing.builders.tree_likelihood import (
    MCMCBuilder,
    MutationMoveCaretakerBuilder,
    PhyloMoveCaretakerBuilder,
    add_gl_at_ancestor_mutations_for,
    add_gl_at_each_ancestor_node_for_nodes,
)

from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.mcmc import as_dict_of_dicts

logging.basicConfig(level=logging.INFO)


def test_finds_one_sample_one_site():
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    mcmc = b.build()

    # when
    mcmc.run()

    # then
    assert set(mcmc.map_g.edges()) == {(-1, 0)}


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


def test_arbitrary_phylo_trees():
    # given
    n_samples = 4
    ts = msprime.simulate(
        sample_size=n_samples, recombination_rate=0, random_seed=42
    )
    rand_g = nx.DiGraph(as_dict_of_dicts(ts.first()))

    b = MCMCBuilder()
    b.certainty = 10
    b.reporting_interval = 100
    b.with_phylogenetic_tree()
    ts = b.with_msprime_tree(n_samples, 42, Ne=1e6, mutation_rate=1e-3)
    b.with_n_burnin_iter(10)
    b.with_n_sampling_iter(10 * 2 ** n_samples)
    mcmc = b.build()
    rand_g = nx.DiGraph(ts.first().as_dict_of_dicts())

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
    assert PhyloTreeLikelihoodCalculator(
        rand_g, gls
    ).log_likelihood() == pytest.approx(
        PhyloTreeLikelihoodCalculator(mcmc.map_g, gls).log_likelihood()
    )


@pytest.mark.parametrize("n_samples", range(5, 11))
def test_arbitrary_phylo_trees_with_internal_checks(n_samples):
    # given
    prng = random
    random.seed(42)
    ts = msprime.simulate(
        sample_size=n_samples,
        recombination_rate=0,
        random_seed=prng.randrange(1, 2 ^ 32),
    )
    rand_g = nx.DiGraph(as_dict_of_dicts(ts.first()))

    b = MCMCBuilder()
    b.with_prng(prng)
    b.reporting_interval = 100
    b.with_phylogenetic_tree()
    b.with_n_sampling_iter(1)
    b.with_n_burnin_iter(100)
    b.with_internal_attach_like_double_checking()
    # b.with_n_sampling_iter(10 * 2 ** n_samples)
    add_gl_at_each_ancestor_node_for_nodes(b, rand_g)
    mcmc = b.build()

    # when/then
    mcmc.run()


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
        st.lists(
            st.integers(min_value=0, max_value=2), min_size=1, max_size=5
        ),
        strategies.integers(min_value=2, max_value=10),
        strategies.randoms(use_true_random=False),
    )
    def test_undoes_any_move(self, moves, n_mutations, prng):
        # given
        b = MutationMoveCaretakerBuilder()

        b.with_random_tree(n_mutations)
        exe = b.build(prng)
        original_tree = exe.g.copy()

        for move_idx, move in enumerate(moves):
            exe.available_moves[move]()
            if move_idx == 0:
                memento = exe.memento
            else:
                memento.append(exe.memento)

        # when
        exe.undo(memento)

        # then
        assert set(exe.g.edges) == set(original_tree.edges)


class TestPhyloMoveExecutor:
    @given(st.randoms(use_true_random=False))
    def test_a_bunch_of_moves(self, prng):
        # given
        b = PhyloMoveCaretakerBuilder(prng=prng)
        b.with_balanced_tree(height=4)
        ct = b.build()

        for i in range(20):
            old_g = ct.g.copy()
            try:
                ct.random_move()
            except Exception:
                logging.error(f"Original tree on iteration {i}:")
                logging.error(old_g.nodes(data=True))
                logging.error(old_g.edges)
                raise

    @given(
        st.lists(
            st.integers(min_value=0, max_value=1), min_size=1, max_size=5
        ),
        strategies.integers(min_value=4, max_value=10),
        strategies.randoms(use_true_random=False),
    )
    def test_undoes_any_move(self, moves, n_mutations, prng):
        # given
        b = PhyloMoveCaretakerBuilder(prng=prng)
        b.with_random_tree(n_mutations)
        ct = b.build()
        original_edges = sorted(ct.g.edges)

        for move_idx, move in enumerate(moves):
            ct.available_moves[move]()
            if move_idx == 0:
                memento = ct.memento
            else:
                memento.append(ct.memento)

        # when
        ct.undo(memento)

        # then
        assert sorted(ct.g.edges) == original_edges


@pytest.mark.parametrize(
    "mutation_tree,num_actors", [(True, 1), (False, 1), (False, 2)]
)
def test_sets_tree_move_weights(mutation_tree, num_actors):
    b = MCMCBuilder()
    b.with_mutated_gl_at(0, 0)
    b.with_mutated_gl_at(1, 1)
    b.with_num_actors(num_actors)
    if mutation_tree:
        move_name = "swap_node"
    else:
        move_name = "swap_leaf"
        b.with_phylogenetic_tree()
    mcmc = b.build()

    # when
    assert 1 == mcmc.move_weights[move_name]
    mcmc.set_move_weight(move_name, 0.5)

    # then
    assert 0.5 == mcmc.move_weights[move_name]


@pytest.mark.parametrize("is_phylogenetic", (True, False))
def test_raises_if_unknown_move_weight_is_changed(is_phylogenetic):
    # given
    b = MCMCBuilder()
    if is_phylogenetic:
        b.with_phylogenetic_tree()

    for sample in range(2):
        b.with_mutated_gl_at(sample, sample)

    mcmc = b.build()

    # when/then
    with pytest.raises(ValueError):
        mcmc.set_move_weight("bla", 0.5)
