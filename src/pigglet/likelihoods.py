import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import itertools as it
import networkx as nx
import numpy as np

from pigglet.constants import HET_NUM, HOM_REF_NUM, LOG_LIKE_DTYPE
from pigglet.scipy_import import logsumexp
from pigglet.tree_interactor import GraphAnnotator
from pigglet.tree_utils import roots_of_tree

logger = logging.getLogger(__name__)


np.seterr(all="raise")


@dataclass
class MutationTreeLikelihoodSummer:
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
        In between, only differences are calculated for the nodes that have
        been updated in the tree.
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
            self._ll_sum = self._calculate_and_report_gold(
                attach_ll, logging.INFO
            )
            self._last_ll[:] = attach_ll[:]
            self._n_diffs = 0
        elif self._update_mask:
            mask = sorted([i + 1 for i in self._update_mask])
            try:
                last_sum_delta = logsumexp(self._last_ll[mask, :], axis=0)
                new_sum_delta = logsumexp(attach_ll[mask, :], axis=0)
                stacked = np.vstack(
                    [self._ll_sum, new_sum_delta, last_sum_delta]
                )
                self._ll_sum = logsumexp(stacked, axis=0, b=self._weights)
                self._last_ll[mask, :] = attach_ll[mask, :]
                self._n_diffs += 1
            except FloatingPointError:
                self._ll_sum = self._calculate_gold(attach_ll)
                self._last_ll[:] = attach_ll[:]
                self._n_diffs = 0
            if self.check_calc:
                self._calculate_and_report_gold(
                    attach_ll, logger.getEffectiveLevel()
                )
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
        try:
            gold_diff = np.abs(gold - self._ll_sum)
            sd = np.std(gold_diff)
            max_diff = np.max(gold_diff / np.abs(gold))
        except FloatingPointError:
            pass
        logger.log(
            log_level,
            f"logsumexp drift"
            f"|mean(gold): {np.mean(gold)}"
            f"|max(abs(diff)/abs(gold)): {max_diff}|SD: {sd}",
        )
        return gold


class TreeLikelihoodCalculator(ABC):
    @abstractmethod
    def attachment_marginalized_log_likelihoods(self):
        """Calculate likelihood after marginalizing over attachment points"""
        pass

    @abstractmethod
    def log_likelihood(self):
        """Calculate tree likelihood"""
        pass


@dataclass
class PhyloTreeLikelihoodCalculator(TreeLikelihoodCalculator):
    """Calculates likelihood of phylogenetic tree (self.g) and
    attachment points from gls for m sites and n samples

    self.gls should have shape (m, n, NUM_GLS)
    self.mutation_matrix_mask has shape (m, n, NUM_GLS)

    The tree is a binary rooted phylogenetic tree where all leaf nodes are
    samples and mutations are unattached. The leaf node IDs are also the
    index of the sample into the mutation and GL matrices.
    """

    g: nx.DiGraph
    gls: np.ndarray
    n_sites: int = 0
    n_samples: int = 0
    _changed_nodes: set = field(default_factory=set)
    _sample_lookup: dict = field(default_factory=dict)
    n_node_update_list: list = field(default_factory=list)
    _attachment_log_like: Optional[np.ndarray] = None
    leaf_nodes: frozenset = field(init=False)

    def __post_init__(self):
        self.n_sites = self.gls.shape[0]
        self.n_samples = self.gls.shape[1]
        self.leaf_nodes = frozenset(
            {u for u in self.g.nodes if self.g.out_degree(u) == 0}
        )
        self._sample_lookup = {
            u: i for i, u in enumerate(sorted(self.leaf_nodes))
        }
        GraphAnnotator(self.g).annotate_all_nodes_with_descendant_leaves(
            self.root
        )
        gls = self.gls
        glstmp = np.zeros((gls.shape[1], gls.shape[2], gls.shape[0]))
        for genotype_idx in range(gls.shape[2]):
            for site_idx in range(gls.shape[0]):
                glstmp[:, genotype_idx, site_idx] = gls[
                    site_idx, :, genotype_idx
                ]
        self.gls = np.moveaxis(glstmp, 2, 0)

    @property
    def root(self):
        roots = roots_of_tree(self.g)
        assert len(roots) == 1
        return roots[0]

    @property
    def attachment_log_like(self) -> np.ndarray:
        """Calculate the likelihoods of all possible mutation attachments

        :returns (len(self.g)) x m numpy array
        the cell at row i and column j is the probability of mutation j
        attaching to node i in the phylogenetic tree
        """
        if self._attachment_log_like is None:
            self._attachment_log_like_complete_recalculation()
        elif self._changed_nodes:
            self._attachment_log_like_partial_recalculation()
        return self._attachment_log_like

    def _attachment_log_like_complete_recalculation(self):
        """Completely recalculate attachment log likelihoods"""
        n_node_updates = 0
        attach_ll = np.zeros((len(self.g), self.n_sites), dtype=LOG_LIKE_DTYPE)

        # If a mutation attaches to the root,
        # then all samples have the mutation
        attach_ll[self.root] = np.sum(self.gls[:, :, HET_NUM], 1)
        for u, v in nx.dfs_edges(self.g, self.root):
            n_node_updates += 1
            update_idxs = [
                self._sample_lookup[s]
                for s in self.g.nodes[u]["leaves"] - self.g.nodes[v]["leaves"]
            ]
            attach_ll[v] = (
                attach_ll[u]
                + np.sum(self.gls[:, update_idxs, HOM_REF_NUM], 1)
                - np.sum(self.gls[:, update_idxs, HET_NUM], 1,)
            )
        self._changed_nodes.clear()
        self.n_node_update_list.append(n_node_updates)
        self._attachment_log_like = attach_ll

    def _attachment_log_like_partial_recalculation(self):
        """Recalculate attachment log likelihoods only for changed nodes
        and their ancestors"""
        n_node_updates = 0
        attach_ll = self._attachment_log_like
        seen_nodes = set()
        for start in self._changed_nodes:
            for node in it.chain([start], nx.ancestors(self.g, start)):
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                n_node_updates += 1
                leaves = self.g.nodes[node]["leaves"]
                other_leaves = self.leaf_nodes - leaves
                sample_idxs = [self._sample_lookup[u] for u in leaves]
                other_idxs = [self._sample_lookup[u] for u in other_leaves]
                attach_ll[node] = np.sum(
                    self.gls[:, sample_idxs, HET_NUM], 1
                ) + np.sum(self.gls[:, other_idxs, HOM_REF_NUM], 1)
        self._changed_nodes.clear()
        self.n_node_update_list.append(n_node_updates)
        self._attachment_log_like = attach_ll

    def node_sample_ids(self):
        """Iterate over node ids and associated samples (leaves)"""
        lookup = self._sample_lookup
        for node, leaves in self.g.nodes(data="leaves"):
            yield node, [lookup[leaf] for leaf in leaves]

    def attachment_marginalized_log_likelihoods(self) -> np.ndarray:
        """Calculate the marginal likelihoods of all possible mutation
        attachments for each of m mutations

        :returns: ndarray of shape len(g)"""
        return logsumexp(self.attachment_log_like, axis=0)

    def log_likelihood(self):
        """Calculate the tree likelihood"""
        return np.sum(self.attachment_marginalized_log_likelihoods())

    def register_changed_nodes(self, *nodes):
        """Marks these nodes and all ancestors of these nodes to have changed
        position in the graph"""
        for node in nodes:
            self._changed_nodes.add(node)
        return self

    def has_changed_nodes(self):
        return len(self._changed_nodes) != 0


class MutationTreeLikelihoodCalculator(TreeLikelihoodCalculator):
    """Calculates likelihood of mutation tree (self.g) and attachment points
    from gls for m sites and n samples

    self.gls should have shape (m, n, NUM_GLS)
    self.mutation_matrix_mask has shape (m, n, NUM_GLS)

    The tree is a rooted mutation tree with unattached samples.
    This means that every node, except for the root node, represents
    a single mutation. The mutation node IDs are also the index of the
    mutation into the mutation and GL matrices.
    """

    def __init__(self, g, gls):
        self.gls = gls
        self.n_sites = self.gls.shape[0]
        self.n_samples = self.gls.shape[1]
        self.paths = None
        self._attachment_log_like = None
        self.summer = MutationTreeLikelihoodSummer(
            self.n_sites + 1, self.n_samples
        )
        self.g = g
        roots = roots_of_tree(g)
        assert len(roots) == 1
        self.root = roots[0]
        self._changed_nodes = {self.root}
        self._n_node_updates = None
        self.n_node_update_list = []

    @property
    def attachment_log_like(self):
        """Calculate the likelihoods of all possible sample attachments

        :returns (m+1) x n numpy array
        the cell at row i and column j is the probability of sample j
        attaching to site i-1, where i=0 is the root node"""

        n_node_updates = 0
        if self.has_changed_nodes():
            for node in self._changed_nodes:
                self._recalculate_attachment_log_like_from(node)
                n_node_updates += self._n_node_updates
            self._changed_nodes.clear()
        self.n_node_update_list.append(n_node_updates)

        return self._attachment_log_like

    def attachment_marginalized_log_likelihoods(self):
        """Calculate the marginal likelihoods of all possible sample
        attachments"""
        return self.summer.calculate(self.attachment_log_like)

    def log_likelihood(self):
        """Calculate the tree likelihood"""
        return np.sum(self.attachment_marginalized_log_likelihoods())

    def mutation_probabilites(self, attach_prob):
        """Accepts an (m + 1) x n matrix of (ideally normalized) log attachment
        probabilities

        m is the number of sites and n is the number of samples

        returns an m x n matrix of mutation probabilities. Each cell is the
        probability that site i is mutated in sample j.
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
        """Marks these nodes and all descendants of these nodes to have changed
        position in the graph"""
        for node in nodes:
            self._changed_nodes.add(node)
        return self

    def has_changed_nodes(self):
        return len(self._changed_nodes) != 0

    def _recalculate_attachment_log_like_from(self, start):
        attachment_log_like = self._attachment_log_like
        self._n_node_updates = 1
        self.summer.register_changed_node(start)
        if start == self.root:
            attachment_log_like = np.zeros(
                (self.n_sites + 1, self.n_samples), dtype=LOG_LIKE_DTYPE
            )
            attachment_log_like[start + 1] = np.sum(
                self.gls[:, :, HOM_REF_NUM], 0
            )
        else:
            parent = list(self.g.pred[start])[0]
            attachment_log_like[start + 1] = (
                attachment_log_like[parent + 1]
                + self.gls[start, :, HET_NUM]
                - self.gls[start, :, HOM_REF_NUM]
            )
        for u, v in nx.dfs_edges(self.g, start):
            self._n_node_updates += 1
            self.summer.register_changed_node(v)
            attachment_log_like[v + 1] = (
                attachment_log_like[u + 1]
                + self.gls[v, :, HET_NUM]
                - self.gls[v, :, HOM_REF_NUM]
            )
        self._attachment_log_like = attachment_log_like


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
