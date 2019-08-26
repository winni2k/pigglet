import random

import numpy as np

from pigglet.constants import NUM_GLS
from pigglet.likelihoods import TreeLikelihoodCalculator
from pigglet.mcmc import MCMCRunner
from pigglet_testing.builders.tree import TreeBuilder


class LikelihoodBuilder:

    def __init__(self):
        self.mutated_gls = set()
        self.unmutated_gls = set()
        self.likelihood_peaks = set()
        self.mutated_gls = set()
        self.unmutated_gls = set()
        self.mutation_gl = 1
        self.num_sites = 0
        self.num_samples = 0
        self.gls = None

    def _mutate_gls(self):
        for sample_id, site_idx in self.mutated_gls:
            self.gls[site_idx, sample_id, 1] = self.mutation_gl
        for sample_id, site_idx in self.unmutated_gls:
            self.gls[site_idx, sample_id, 0] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self, num_sites=None):
        if num_sites is not None:
            assert num_sites >= self.num_sites
            self.num_sites = num_sites
        self.gls = np.zeros((self.num_sites, self.num_samples, NUM_GLS))
        self._add_likelihood_peaks()
        self._mutate_gls()
        return self.gls

    def with_likelihood_peak_at_all_hom_ref(self):
        self.likelihood_peaks.add(0)
        return self

    def with_mutated_gl_at(self, sample_idx, site_idx):
        self._bump_site_and_sample_num(sample_idx, site_idx)
        self.mutated_gls.add((sample_idx, site_idx))
        return self

    def with_unmutated_gl_at(self, sample_idx, site_idx):
        self._bump_site_and_sample_num(sample_idx, site_idx)
        self.unmutated_gls.add((sample_idx, site_idx))

    def _bump_site_and_sample_num(self, sample_idx, site_idx):
        self.num_sites = max(site_idx + 1, self.num_sites)
        self.num_samples = max(sample_idx + 1, self.num_samples)

    def with_gl_dimensions(self, n_sites, n_samples):
        self.num_sites = n_sites
        self.num_samples = n_samples
        return self


class TreeLikelihoodBuilder:

    def __init__(self):
        self.tree_builder = TreeBuilder()
        self.likelihood_builder = LikelihoodBuilder()

    def __getattr__(self, attr):
        return getattr(self.likelihood_builder, attr)

    def build(self):
        tree = self.tree_builder.build()
        num_sites = len(tree) - 1
        gls = self.likelihood_builder.build(num_sites)
        return tree, gls

    def with_balanced_tree(self, height=2, n_branches=2):
        self.tree_builder.with_balanced_tree(height=height, n_branches=n_branches)
        return self

    def with_mutation_at(self, attachment_node, new_node_id):
        self.tree_builder.with_mutation_at(attachment_node, new_node_id)
        return self


class TreeLikelihoodCalculatorBuilder(TreeLikelihoodBuilder):

    def build(self):
        g, gls = super().build()
        return TreeLikelihoodCalculator(g, gls)


class MCMCBuilder(LikelihoodBuilder):
    def __init__(self):
        super().__init__()
        self.seed = None
        self.n_burnin_iter = 10

    def with_n_burnin_iter(self, n_iter):
        self.n_burnin_iter = n_iter
        return self

    def build(self):
        if self.seed is None:
            random.seed(42)
        gls = super().build()
        return MCMCRunner.from_gls(gls, num_burnin_iter=self.n_burnin_iter)
