import itertools

import numpy as np

from pigglet.constants import NUM_GLS
from pigglet.likelihoods import TreeLikelihoodCalculator
from pigglet_testing.builders.tree import TreeBuilder
from pigglet_testing.utils import sample_nodes_of_tree


class TreeLikelihoodBuilder:

    def __init__(self):
        self.likelihood_peaks = set()
        self.mutated_sample_ids = set()
        self.mutated_gls = set()
        self.mutation_gl = 1
        self.num_sites = 0
        self.num_samples = 0
        self.gls = None
        self.tree_builder = TreeBuilder()

    def _mutate_gls(self):
        for sample_id, site_idx in itertools.chain(self.mutated_gls,
                                                   self.mutated_sample_ids):
            self.gls[site_idx, sample_id, 1] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self):
        tree = self.tree_builder.build()
        self.num_sites = max(len(tree) -1 , self.num_sites)
        self.gls = np.zeros((self.num_sites, self.num_samples, NUM_GLS))

        self._add_likelihood_peaks()
        self._mutate_gls()

        return tree, self.gls

    def with_balanced_tree(self, height=2, n_branches=2):
        self.tree_builder.with_balanced_tree(height=height, n_branches=n_branches)
        return self

    def with_likelihood_peak_at_all_hom_ref(self):
        self.likelihood_peaks.add(0)
        return self

    def with_mutated_gl_at(self, sample_idx, site_idx):
        self.num_sites = max(site_idx + 1, self.num_sites)
        self.num_samples = max(sample_idx + 1, self.num_samples)
        self.mutated_gls.add((sample_idx, site_idx))
        return self

    def with_mutation_at(self, attachment_node, new_node_id):
        self.tree_builder.with_mutation_at(attachment_node, new_node_id)
        return self

    def with_gl_dimensions(self, n_sites, n_samples):
        self.num_sites = n_sites
        self.num_samples = n_samples
        return self

class TreeLikelihoodCalculatorBuilder(TreeLikelihoodBuilder):

    def build(self):
        g, gls = super().build()
        return TreeLikelihoodCalculator(g, gls, sample_nodes_of_tree(g))
