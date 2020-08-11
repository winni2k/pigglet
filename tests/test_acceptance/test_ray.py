import random

import numpy as np
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

    mover = PhyloTreeLikelihoodMoverDirector(g=mcmc.g, gls=mcmc.gls, prng=prng)
    like = mover.attachment_log_like.copy()

    # when/then
    mover.random_move()
    assert mover.has_changed_nodes()
    mover.attachment_log_like
    assert not mover.has_changed_nodes()
    mover.undo()
    assert mover.has_changed_nodes()

    # then
    assert like is not None
    for u in range(like.shape[0]):
        assert np.allclose(like[u], mover.attachment_log_like[u])
