import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import networkx as nx
import numpy as np

from pigglet.likelihoods import (
    MutationTreeLikelihoodCalculator,
    PhyloTreeLikelihoodCalculator,
    TreeLikelihoodCalculator,
)
from pigglet.scipy_import import logsumexp


class AttachmentAggregator(ABC):
    @abstractmethod
    def add_attachment_log_likes(self, calc):
        pass


class NullAttachmentAggregator(AttachmentAggregator):
    def add_attachment_log_likes(self, calc: TreeLikelihoodCalculator):
        pass


class MutationAttachmentAggregator(AttachmentAggregator):
    """Aggregates attachment scores"""

    def __init__(self):
        self.attachment_scores = None
        self.num_additions = 0

    def add_attachment_log_likes(self, calc: MutationTreeLikelihoodCalculator):
        sum_likes = calc.attachment_marginalized_log_likelihoods()
        log_likes = calc.attachment_log_like - sum_likes
        if self.attachment_scores is None:
            self.attachment_scores = log_likes
        else:
            with np.errstate(under="ignore"):
                self.attachment_scores = np.logaddexp(
                    self.attachment_scores, log_likes
                )
        self.num_additions += 1

    def normalized_attachment_probabilities(self):
        return self.attachment_scores - math.log(self.num_additions)


@dataclass
class PhyloAttachmentAggregator(AttachmentAggregator):
    """Aggregates attachment scores for phylogenetic trees

    The results of aggregation are stored in attachment_scores, an m x n
    matrix of mutation probabilities
    """

    attachment_scores = None
    num_additions: int = 0

    def add_attachment_log_likes(self, calc: PhyloTreeLikelihoodCalculator):
        """Calculate normalized mutation probabilities and aggregate
        in attachment_scores"""
        site_like_total = calc.attachment_marginalized_log_likelihoods()
        # log_likes.shape = len(g) x m
        log_likes = calc.attachment_log_like - site_like_total
        scores = np.zeros((calc.n_sites, calc.n_samples))
        node_sample_indicator = np.zeros(
            (len(calc.g), calc.n_samples), dtype=np.bool
        )
        for node, leaves in calc.node_sample_ids():
            node_sample_indicator[node, leaves] = True
        for leaf in range(calc.n_samples):
            scores[:, leaf] = logsumexp(
                log_likes[node_sample_indicator[:, leaf], :], axis=0
            )
        if self.attachment_scores is None:
            self.attachment_scores = scores
        else:
            with np.errstate(under="ignore"):
                self.attachment_scores = np.logaddexp(
                    self.attachment_scores, scores
                )
        self.num_additions += 1

    def averaged_mutation_probabilities(self):
        return self.attachment_scores - math.log(self.num_additions)


@dataclass
class TreeAggregator:
    trees: List[nx.DiGraph] = field(default_factory=list)

    def store_tree(self, g: nx.DiGraph):
        self.trees.append(g.copy())

    def to_newick(self):
        """Yield trees as single-line Newick trees"""
        for g in self.trees:
            yield tree_to_newick(g)


def tree_to_newick(
    g,
    root=None,
    one_base=False,
    leaf_lookup=None,
    node_branch_length_lookup=None,
):
    first = False
    if root is None:
        first = True
        roots = [u for u, d in g.in_degree() if d == 0]
        assert 1 == len(roots)
        root = roots[0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(
                tree_to_newick(
                    g,
                    root=child,
                    one_base=one_base,
                    leaf_lookup=leaf_lookup,
                    node_branch_length_lookup=node_branch_length_lookup,
                )
            )
        else:
            if "label" in g.nodes[child]:
                name = g.nodes[child]["label"].split("\\n")[0]
            else:
                if leaf_lookup:
                    name = leaf_lookup[child]
                elif one_base:
                    name = int(child) + 1
                else:
                    name = child
            subgs.append(str(name))
        if node_branch_length_lookup:
            subgs[-1] += f":{node_branch_length_lookup[child]:f}"
    new_subg = subgs[0]
    for idx in range(1, len(subgs)):
        new_subg = f"({new_subg}, {subgs[idx]})"
    if first:
        if node_branch_length_lookup:
            new_subg += f":{node_branch_length_lookup[root]:f}"
    return new_subg
