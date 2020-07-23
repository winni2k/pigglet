import math
from abc import ABC
from typing import List

import networkx as nx
import numpy as np
from dataclasses import field, dataclass

from pigglet.likelihoods import (
    MutationTreeLikelihoodCalculator,
    PhyloTreeLikelihoodCalculator,
)
from pigglet.scipy_import import logsumexp


class AttachmentAggregator(ABC):
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


class PhyloAttachmentAggregator(AttachmentAggregator):
    """Aggregates attachment scores for phylogenetic trees

    The results of aggregation are stored in attachment_scores, an m x n
    matrix of mutation probabilities
    """

    def __init__(self):
        self.attachment_scores = None
        self.num_additions = 0

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


def tree_to_newick(g, root=None):
    if root is None:
        roots = [u for u, d in g.in_degree() if d == 0]
        assert 1 == len(roots)
        root = roots[0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(tree_to_newick(g, root=child))
        else:
            if "label" in g.nodes[child]:
                name = g.nodes[child]["label"].split("\\n")[0]
            else:
                name = child
            subgs.append(name)
    new_subg = subgs[0]
    for idx in range(1, len(subgs)):
        new_subg = f"({new_subg}, {subgs[idx]})"
    return new_subg
