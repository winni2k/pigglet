import logging
import math
from dataclasses import dataclass
import numexpr as ne

import networkx as nx
import numpy as np
from pigglet.scipy_import import logsumexp

from pigglet.constants import HET_NUM, HOM_REF_NUM, LOG_LIKE_DTYPE
from pigglet.tree_utils import roots_of_tree

logger = logging.getLogger(__name__)


np.seterr(all="raise")


@dataclass
class TreeLikelihoodSummer:
    """Calculates efficient summed attachment log likelihood updates"""

    n_nodes: int
    n_samples: int
    max_diffs = 100
    _n_diffs = 0
    _ll_sum = None
    _update_mask = None
    _last_ll = None
    check_calc = False
    _weights = None

    def __post_init__(self):
        self._weights = np.vstack(
            [
                np.ones(self.n_samples),
                np.ones(self.n_samples),
                np.ones(self.n_samples) * -1,
            ]
        )

    def calculate(self, attach_ll):
        """
        Update the log of the sum of exponentiated attachment likelihoods.

        Every self.n_diffs calculations, the complete summation is performed.
        In between, only differences are calculated for the nodes that have been
        updated in the tree.
        """
        assert attach_ll.shape[0] == self.n_nodes, (
            attach_ll.shape,
            self.n_nodes,
        )
        if self._ll_sum is None:
            self._ll_sum = self._calculate_gold(attach_ll)
            self._last_ll = attach_ll.copy()
            self._n_diffs = 0
        elif self._n_diffs == self.max_diffs:
            self._ll_sum = self._calculate_and_report_gold(attach_ll, logging.INFO)
            self._last_ll[:] = attach_ll[:]
            self._n_diffs = 0
        elif self._update_mask:
            mask = sorted([i + 1 for i in self._update_mask])
            try:
                last_sum_delta = logsumexp(self._last_ll[mask, :], axis=0)
                new_sum_delta = logsumexp(attach_ll[mask, :], axis=0)
                stacked = np.vstack([self._ll_sum, new_sum_delta, last_sum_delta])
                self._ll_sum = logsumexp(stacked, axis=0, b=self._weights)
                self._last_ll[mask, :] = attach_ll[mask, :]
                self._n_diffs += 1
            except FloatingPointError:
                self._ll_sum = self._calculate_gold(attach_ll)
                self._last_ll[:] = attach_ll[:]
                self._n_diffs = 0
            if self.check_calc:
                self._calculate_and_report_gold(attach_ll, logger.getEffectiveLevel())
            self._update_mask.clear()
        return self._ll_sum

    def _calculate_gold(self, attach_ll):
        """Gold standard calculation of log sum of attachment likelihoods"""
        return logsumexp(attach_ll, axis=0)

    def register_changed_node(self, n):
        if self._update_mask is None:
            self._update_mask = set()
        assert n < self.n_nodes
        self._update_mask.add(n)

    def _calculate_and_report_gold(self, attach_ll, log_level):
        gold = self._calculate_gold(attach_ll)
        gold_diff = np.abs(gold - self._ll_sum)
        sd = np.std(gold_diff)
        max_diff = np.max(gold_diff / np.abs(gold))
        logger.log(
            log_level,
            f"logsumexp drift|"
            f"mean(gold): {np.mean(gold)}|max(abs(diff)/abs(gold)): {max_diff}|SD: {sd}",
        )
        return gold


class TreeLikelihoodCalculator:
    """Calculates likelihood of mutation tree (self.g) and attachment points
    from gls for m sites and n samples

    self.gls should have shape (m, n, NUM_GLS)
    self.mutation_matrix_mask has shape (m, n, NUM_GLS)

    The likelihood tree is a rooted mutation tree with unattached samples.
    This means that every node, except for the root node, represents
    a single mutation. The mutation node IDs are also the index of the mutation into the
    mutation and GL matrices.
    """

    def __init__(self, g, gls):
        glstmp = np.zeros((gls.shape[0], gls.shape[2], gls.shape[1]))
        for genotype_idx in range(gls.shape[2]):
            glstmp[:, genotype_idx, :] = gls[:, :, genotype_idx]
        self.gls = glstmp
        self.n_sites = self.gls.shape[0]
        self.n_samples = self.gls.shape[2]
        self.paths = None
        self._attachment_log_like = None
        self.summer = TreeLikelihoodSummer(self.n_sites + 1, self.n_samples)
        self.g = g
        roots = roots_of_tree(g)
        assert len(roots) == 1
        self.root = roots[0]
        self._changed_nodes = {self.root}

    @property
    def attachment_log_like(self):
        """Calculate the likelihoods of all possible sample attachments

        :returns (m+1) x n numpy array
        the cell at row i and column j is the probability of sample j attaching to site
        i-1, where i=0 is the root node"""

        if self.has_changed_nodes():
            for node in self._changed_nodes:
                self._recalculate_attachment_log_like_from(node)
            self._changed_nodes.clear()
        return self._attachment_log_like

    def attachment_marginalized_sample_log_likelihoods(self):
        """Calculate the marginal likelihoods of all possible sample attachments"""
        return self.summer.calculate(self.attachment_log_like)

    def sample_marginalized_log_likelihood(self):
        """Calculate the sum of the log likelihoods of all possible sample attachments"""
        return np.sum(self.attachment_marginalized_sample_log_likelihoods())

    def mutation_probabilites(self, attach_prob):
        """Accepts an (m + 1) x n matrix of (ideally normalized) log attachment probabilities

        m is the number of sites and n is the number of samples

        returns an m x n matrix of mutation probabilities. Each cell is the probability
        that site i is mutated in sample j.
        """

        attach_prob = attach_prob[1:]
        with np.errstate(under="ignore"):
            attach_prob = np.exp(attach_prob)
        mut_probs = np.zeros_like(attach_prob)
        seen_muts = []
        for u, v, label in nx.dfs_labeled_edges(self.g, self.root):
            if u == v:
                continue
            if label == "forward":
                seen_muts.append(v)
                mut_probs[seen_muts] += attach_prob[v]
            elif label == "reverse":
                seen_muts.pop()
            else:
                raise ValueError(f"Unexpected label: {label}")
        return mut_probs

    def ml_sample_attachments(self):
        return np.argmax(self.attachment_log_like, axis=0)

    def register_changed_nodes(self, *nodes):
        """Marks these nodes and all descendants of these nodes to have changed position
        in the graph"""
        for node in nodes:
            self._changed_nodes.add(node)
        return self

    def has_changed_nodes(self):
        return len(self._changed_nodes) != 0

    def _recalculate_attachment_log_like_from(self, start):
        attachment_log_like = self._attachment_log_like
        self.summer.register_changed_node(start)
        if start == self.root:
            attachment_log_like = np.zeros(
                (self.n_sites + 1, self.n_samples), dtype=LOG_LIKE_DTYPE
            )
            attachment_log_like[start + 1] = np.sum(
                self.gls[:, 0, :].reshape((self.n_sites, self.n_samples)), 0
            )
        else:
            parent = list(self.g.pred[start])[0]
            a = attachment_log_like[parent + 1]
            b = self.gls[start, HET_NUM, :]
            c = self.gls[start, HOM_REF_NUM, :]
            attachment_log_like[start + 1] = ne.evaluate("a + b -c")
        for u, v in nx.dfs_edges(self.g, start):
            self.summer.register_changed_node(v)
            a = attachment_log_like[u + 1]  # noqa
            b = self.gls[v, HET_NUM, :]  # noqa
            c = self.gls[v, HOM_REF_NUM, :]  # noqa
            attachment_log_like[v + 1] = ne.evaluate("a + b -c")
        self._attachment_log_like = attachment_log_like


class AttachmentAggregator:
    """Aggregates attachment scores"""

    def __init__(self):
        self.attachment_scores = None
        self.num_additions = 0

    def add_attachment_log_likes(self, calc):
        sum_likes = calc.attachment_marginalized_sample_log_likelihoods()
        log_likes = calc.attachment_log_like - sum_likes
        if self.attachment_scores is None:
            self.attachment_scores = log_likes
        else:
            with np.errstate(under="ignore"):
                self.attachment_scores = np.logaddexp(self.attachment_scores, log_likes)
        self.num_additions += 1

    def normalized_attachment_probabilities(self):
        return self.attachment_scores - math.log(self.num_additions)
