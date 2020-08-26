import random
from dataclasses import dataclass, field

import msprime
import networkx as nx

from pigglet.mcmc import as_dict_of_dicts
from pigglet_testing.expectations.tree import PhyloTreeExpectation

from pigglet.tree_converter import MutationToPhylogeneticTreeConverter
from pigglet.tree_interactor import PhyloTreeInteractor


@dataclass
class PhyloTreeBuilder:
    g: nx.DiGraph = field(default_factory=nx.DiGraph)
    prng: random.Random = None

    def with_branch(self, u, v):
        self.g.add_edge(u, v)
        return self

    def with_path(self, *nodes):
        nx.add_path(self.g, nodes)
        return self

    def with_balanced_tree(self, height=2, n_branches=2, rev=False):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        if rev:
            self.g = nx.relabel_nodes(
                self.g,
                {a: b for a, b in zip(list(self.g), reversed(list(self.g)))},
            )
        return self

    def with_random_tree(self, n_samples):
        ts = msprime.simulate(
            n_samples,
            recombination_rate=0,
            random_seed=self.prng.randrange(1, 2 ^ 32),
        )
        self.g = nx.from_dict_of_dicts(
            as_dict_of_dicts(ts.first()), create_using=self.g
        )
        return self

    def build(self):
        if len(self.g) < 3:
            self.with_balanced_tree(1, rev=True)
        return PhyloTreeInteractor(self.g, prng=self.prng).g


class MutationTreeBuilder:
    def __init__(self):
        self.g = nx.DiGraph()

    def with_balanced_tree(self, height=2, n_branches=2):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        self._relabel_nodes()
        return self

    def with_random_tree(self, n_mutations):
        self.g = nx.gnr_graph(n_mutations + 1, 0).reverse()
        self._relabel_nodes()

    def _relabel_nodes(self):
        nx.relabel_nodes(self.g, {n: n - 1 for n in self.g.nodes}, copy=False)

    def with_mutation_at(self, attachment_node, new_node_id):
        self.g.add_edge(attachment_node, new_node_id)
        return self

    def with_path(self, n_muts):
        start = -1
        for mut in range(n_muts):
            self.with_mutation_at(start, mut)
            start = mut
        return self

    def build(self):
        if len(self.g.nodes()) == 0:
            self.g.add_node(-1)
        return self.g


class PhylogeneticTreeConverterBuilder(MutationTreeBuilder):
    def build(self, prng=None):
        if prng is None:
            prng = random
        return MutationToPhylogeneticTreeConverter(super().build(), prng=prng)


class PhylogeneticTreeConverterTestDriver(PhylogeneticTreeConverterBuilder):
    def __init__(self):
        super().__init__()
        self.sample_attachments = []

    def with_sample_attachments(self, *attachments):
        self.sample_attachments = list(attachments)
        return self

    def build(self, prng=None):
        converter = super().build(prng=prng)
        return PhyloTreeExpectation(
            converter.convert(sample_attachments=self.sample_attachments)
        )
