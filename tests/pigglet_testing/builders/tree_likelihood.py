import random
from dataclasses import dataclass, field
from typing import Set, Dict
import msprime

import networkx as nx
import numpy as np
from pigglet_testing.builders.tree import MutationTreeBuilder, PhyloTreeBuilder

from pigglet.constants import NUM_GLS, ROOT_LABEL
from pigglet.gl_manipulator import GLManipulator
from pigglet.likelihoods import (
    MutationTreeLikelihoodCalculator,
    PhyloTreeLikelihoodCalculator,
)
from pigglet.mcmc import MCMCRunner
from pigglet.tree_likelihood_mover import (
    MutationTreeMoveCaretaker,
    PhyloTreeMoveCaretaker,
)


@dataclass
class GlBuilder:

    mutated_gls: Set = field(default_factory=set)
    unmutated_gls: Set = field(default_factory=set)
    likelihood_peaks: Set = field(default_factory=set)
    mutation_gl = 0
    num_sites = 0
    num_samples = 0
    certainty = 1
    gls = None

    def _mutate_gls(self):
        for sample_id, site_idx in self.mutated_gls:
            self.gls[site_idx, sample_id, 1] = self.mutation_gl
        for sample_id, site_idx in self.unmutated_gls:
            self.gls[site_idx, sample_id, 0] = self.mutation_gl

    def _add_likelihood_peaks(self):
        for peak in self.likelihood_peaks:
            self.gls[:, :, peak] = self.mutation_gl

    def build(self):
        self.gls = (
            np.ones((self.num_sites, self.num_samples, NUM_GLS))
            * -1
            * self.certainty
        )
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


@dataclass
class MspGlBuilder:

    msp_args: Dict = field(default_factory=dict)
    num_samples = None
    certainty = 1

    def build(self):
        ts = msprime.simulate(self.num_samples, **self.msp_args)
        gmat = ts.genotype_matrix()
        ll_cert = -1 * self.certainty
        assert NUM_GLS == 2
        gls = [
            [0, ll_cert] if value == 0 else [ll_cert, 0]
            for _list in gmat.tolist()
            for value in _list
        ]
        return np.array(gls).reshape(-1, self.num_samples, NUM_GLS)

    def with_msprime_tree(
        self, sample_size, random_seed, Ne=1e6, mutation_rate=1e-2
    ):
        self.num_samples = sample_size
        self.msp_args = {
            "Ne": Ne,
            "mutation_rate": mutation_rate,
            "recombination_rate": 0,
            "random_seed": random_seed,
        }
        return self

    def with_certainty(self, c):
        self.certainty = c
        return self


class MutationTreeLikelihoodBuilder:
    def __init__(self):
        self.tree_builder = MutationTreeBuilder()
        self.likelihood_builder = GlBuilder()

    def __getattr__(self, attr):
        return getattr(self.likelihood_builder, attr)

    def build(self):
        tree = self.tree_builder.build()
        num_sites = len(tree) - 1
        self.likelihood_builder.num_sites = num_sites
        gls = self.likelihood_builder.build()
        return tree, gls

    def with_balanced_tree(self, height=2, n_branches=2):
        self.tree_builder.with_balanced_tree(
            height=height, n_branches=n_branches
        )
        return self

    def with_mutation_site_at(self, attachment_node, new_node_id):
        self.tree_builder.with_mutation_at(attachment_node, new_node_id)
        return self


class PhyloTreeLikelihoodBuilder:
    def __init__(self):
        self.tree_builder = PhyloTreeBuilder()
        self.likelihood_builder = GlBuilder()

    def __getattr__(self, attr):
        try:
            return getattr(self.likelihood_builder, attr)
        except AttributeError:
            pass
        return getattr(self.tree_builder, attr)

    def build(self):
        tree = self.tree_builder.build()
        gls = self.likelihood_builder.build()
        return tree, gls


class PhyloTreeLikelihoodCalculatorBuilder(PhyloTreeLikelihoodBuilder):
    def build(self):
        g, gls = super().build()
        return PhyloTreeLikelihoodCalculator(g, gls)


class MutationTreeLikelihoodCalculatorBuilder(MutationTreeLikelihoodBuilder):
    def build(self):
        g, gls = super().build()
        return MutationTreeLikelihoodCalculator(g, gls)


class MutationMoveCaretakerBuilder(MutationTreeBuilder):
    def build(self, prng):
        g = super().build()
        return MutationTreeMoveCaretaker(g, prng=prng)


class PhyloMoveCaretakerBuilder(PhyloTreeBuilder):
    def build(self):
        g = super().build()
        return PhyloTreeMoveCaretaker(g, prng=self.prng)


class MCMCBuilder:
    def __init__(self):
        self.l_builder = GlBuilder()
        self.seed = None
        self.prng = random
        self.n_burnin_iter = 10
        self.n_sampling_iter = 10
        self.normalize_gls = False
        self.mutation_tree = True
        self.reporting_interval = 10
        self.internal_attach_like_double_checking = True

    def __getattr__(self, item):
        return getattr(self.l_builder, item)

    def with_msprime_tree(self, *args, **kwargs):
        self.l_builder = MspGlBuilder()
        return self.l_builder.with_msprime_tree(*args, **kwargs)

    def with_internal_attach_like_double_checking(self):
        self.internal_attach_like_double_checking = True

    def with_n_burnin_iter(self, n_iter):
        self.n_burnin_iter = n_iter
        return self

    def with_phylogenetic_tree(self):
        self.mutation_tree = False
        return self

    def with_n_sampling_iter(self, n_iter):
        self.n_sampling_iter = n_iter
        return self

    def with_normalized_gls(self):
        self.normalize_gls = True
        return self

    def with_prng(self, prng):
        self.prng = prng

    def build(self):
        if self.prng is random:
            if self.seed is None:
                self.seed = 42
            self.prng.seed(self.seed)
        gls = self.l_builder.build()
        if self.normalize_gls:
            gls = GLManipulator(gls).normalize().gls
        if self.mutation_tree:
            runner = MCMCRunner.mutation_tree_from_gls(gls, prng=self.prng)
        else:
            runner = MCMCRunner.phylogenetic_tree_from_gls(gls, prng=self.prng)
            if self.internal_attach_like_double_checking:
                runner.like_mover.double_check_ll_calculations = True
        runner.num_burnin_iter = self.n_burnin_iter
        runner.num_sampling_iter = self.n_sampling_iter
        runner.reporting_interval = self.reporting_interval
        return runner


def add_gl_at_ancestor_mutations_for(attachment_point, b, rand_g, sample):
    mutations = set(nx.ancestors(rand_g, attachment_point))
    mutations.add(attachment_point)
    mutations.remove(ROOT_LABEL)
    for mutation in mutations:
        b.with_mutated_gl_at(sample, mutation)
    for non_mutation in set(rand_g) - mutations:
        if non_mutation != ROOT_LABEL:
            b.with_unmutated_gl_at(sample, non_mutation)


def add_gl_at_each_ancestor_node_for_nodes(b, g):
    """Add a mutation for each inner node of g to all samples"""
    leaf_nodes = {u for u in g.nodes if g.out_degree(u) == 0}
    inner_nodes = {u for u in g.nodes if g.out_degree(u) != 0}
    for mutation_idx, u in enumerate(sorted(inner_nodes)):
        mutated_samples = set(nx.descendants(g, u)) & leaf_nodes
        for sample in leaf_nodes:
            if sample in mutated_samples:
                b.with_mutated_gl_at(sample, mutation_idx)
            else:
                b.with_unmutated_gl_at(sample, mutation_idx)
