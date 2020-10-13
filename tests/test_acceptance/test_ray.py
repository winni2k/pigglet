import random

import pytest
from pigglet_testing.builders.tree_likelihood import MCMCBuilder
from pigglet.tree_likelihood_mover_ray import PhyloTreeLikelihoodMoverDirector


@pytest.mark.parametrize("n_samples", range(4, 11))
def test_ray_arbitrary_trees_and_moves_undo_ok(n_samples):
    # given
    prng = random
    b = MCMCBuilder()
    b.with_prng(prng)
    b.with_phylogenetic_tree()

    for sample in range(n_samples):
        b.with_mutated_gl_at(sample, sample)

    mcmc = b.build()

    mover = PhyloTreeLikelihoodMoverDirector(
        g=mcmc.g, gls=mcmc.gls, prng=prng, testing=True
    )
    like = mover.log_likelihood()
    old_edges = sorted(mover.g.edges)
    new_edges = old_edges
    # when/then
    while new_edges == old_edges:
        mover.random_move()
        new_edges = sorted(mover.g.edges)

    assert mover.has_changed_nodes()
    mover.log_likelihood()
    assert not mover.has_changed_nodes()
    mover.undo()
    assert mover.has_changed_nodes()

    # then
    assert like is not None
    assert like == pytest.approx(mover.log_likelihood())
