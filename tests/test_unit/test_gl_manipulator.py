import itertools

import numpy as np
from pigglet_testing.builders.tree_likelihood import LikelihoodBuilder
from pytest import approx

from pigglet.gl_manipulator import GLManipulator


class TestNormalizesGLs:
    def test_one_sample_one_site(self):
        # given
        b = LikelihoodBuilder()
        b.with_mutated_gl_at(0, 0)
        mani = GLManipulator(b.build())

        # when
        mani.normalize()

        # then
        assert 1 == approx(np.sum(np.exp(mani.gls[0, 0])))

    def test_two_samples_two_sites(self):
        # given
        b = LikelihoodBuilder()
        b.with_mutated_gl_at(0, 0)
        b.with_mutated_gl_at(1, 1)
        mani = GLManipulator(b.build())

        # when
        mani.normalize()

        # then
        for x, y in itertools.product(range(2), range(2)):
            assert 1 == approx(np.sum(np.exp(mani.gls[x, y])))
