import math

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from pigglet.constants import HET_NUM, LOG_LIKE_DTYPE, HOM_REF_NUM
from pigglet.tree_utils import roots_of_tree


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
        self._summed_attachment_log_like = None
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

    def attachment_marginaziled_sample_log_likelihoods(self):
        """Calculate the marginal likelihoods of all possible sample attachments"""
        self._summed_attachment_log_like = logsumexp(self.attachment_log_like, axis=0)
        return self._summed_attachment_log_like

    def sample_marginalized_log_likelihood(self):
        """Calculate the sum of the log likelihoods of all possible sample attachments"""
        return np.sum(self.attachment_marginaziled_sample_log_likelihoods())

    def mutation_probabilites(self, attach_prob):
        """Accepts an (m + 1) x n matrix of (ideally normalized) log attachment probabilities

        m is the number of sites and n is the number of samples

        returns an m x n matrix of mutation probabilities. Each cell is the probability
        that site i is mutated in sample j.
        """

        attach_prob = attach_prob[1:]
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
        """Marks these nodes and all ancestors of these nodes to have changed position
        in the graph"""
        for node in nodes:
            self._changed_nodes.add(node)
        return self

    def has_changed_nodes(self):
        return len(self._changed_nodes) != 0

    def _recalculate_attachment_log_like_from(self, start):
        attachment_log_like = self._attachment_log_like
        if start == self.root:
            attachment_log_like = np.zeros(
                (self.n_sites + 1, self.n_samples), dtype=LOG_LIKE_DTYPE
            )
            attachment_log_like[start + 1] = np.sum(
                self.gls[:, 0, :].reshape((self.n_sites, self.n_samples)), 0
            )
        else:
            parent = list(self.g.pred[start])[0]
            attachment_log_like[start + 1] = (
                attachment_log_like[parent + 1]
                + self.gls[start, HET_NUM, :]
                - self.gls[start, HOM_REF_NUM, :]
            )
        for u, v in nx.dfs_edges(self.g, start):
            attachment_log_like[v + 1] = (
                attachment_log_like[u + 1]
                + self.gls[v, HET_NUM, :]
                - self.gls[v, HOM_REF_NUM, :]
            )
        self._attachment_log_like = attachment_log_like


class AttachmentAggregator:
    """Aggregates attachment scores"""

    def __init__(self):
        self.attachment_scores = None
        self.num_additions = 0

    def add_attachment_log_likes(self, calc):
        sum_likes = calc.attachment_marginaziled_sample_log_likelihoods()
        log_likes = calc.attachment_log_like - sum_likes
        if self.attachment_scores is None:
            self.attachment_scores = log_likes
        else:
            self.attachment_scores = np.logaddexp(self.attachment_scores, log_likes)
        self.num_additions += 1

    def normalized_attachment_probabilities(self):
        return self.attachment_scores - math.log(self.num_additions)
