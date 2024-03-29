import networkx as nx
import numpy as np
from hypothesis import given, strategies
from pigglet_testing.builders.tree_likelihood import (
    MCMCBuilder,
    add_gl_at_ancestor_mutations_for,
)

from pigglet.tree_likelihood_mover import MutationTreeLikelihoodMover


class TestRecalculateAttachmentLogLikeFromNodes:
    @given(strategies.integers(min_value=3, max_value=10))
    def test_arbitrary_trees(self, n_mutations):
        # given
        rand_g = nx.gnr_graph(n_mutations, 0).reverse()
        nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

        b = MCMCBuilder()

        for sample, attachment_point in enumerate(
            filter(lambda n: n != -1, rand_g)
        ):
            add_gl_at_ancestor_mutations_for(
                attachment_point, b, rand_g, sample
            )

        mcmc = b.build()
        mover = mcmc.like_mover
        calc = mover.calc

        # when
        mover.make_and_register_random_move()
        like = calc.register_changed_nodes(
            *mover.changed_nodes
        ).attachment_log_like.copy()
        root_like = calc.register_changed_nodes(-1).attachment_log_like.copy()

        # then
        assert root_like is not None
        assert np.allclose(root_like, like)


@given(
    strategies.integers(min_value=2, max_value=10),
    strategies.randoms(note_method_calls=True, use_true_random=False),
)
def test_arbitrary_trees_and_moves_undo_ok(n_mutations, prng):
    # given
    rand_g = nx.gnr_graph(n_mutations, 0).reverse()
    nx.relabel_nodes(rand_g, {n: n - 1 for n in rand_g}, copy=False)

    b = MCMCBuilder()

    for sample, attachment_point in enumerate(
        filter(lambda n: n != -1, rand_g)
    ):
        add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample)

    mcmc = b.build()
    mover = MutationTreeLikelihoodMover(g=mcmc.g, gls=mcmc.gls, prng=prng)
    like = mover.attachment_log_like

    # when/then
    mover.make_and_register_random_move()
    assert mover.calc.has_changed_nodes()
    mover.attachment_log_like
    assert not mover.calc.has_changed_nodes()
    mover.undo()
    assert mover.calc.has_changed_nodes()

    # then
    assert like is not None
    assert np.allclose(like, mover.attachment_log_like)
