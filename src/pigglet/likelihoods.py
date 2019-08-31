import math

import networkx as nx
import numpy as np

from pigglet.constants import NUM_GLS, HET_NUM
from pigglet.tree_utils import roots_of_tree


class TreeLikelihoodCalculator:
    """Calculates likelihood of mutation tree (g) and attachment points
    from gls for m sites and n samples

    self.gls should have shape (m, n, NUM_GLS)
    self.mutation_matrix_mask has shape (m, n, NUM_GLS)

    The likelihood tree is a rooted mutation tree with unattached samples.
    This means that every node, except for the root node, represents
    a single mutation. The mutation node IDs are also the index of the mutation into the
    mutation and GL matrices.
    """

    def __init__(self, g, gls):
        self.gls = gls
        self.n_samples = self.gls.shape[1]
        self.g = None
        self.root = None
        self.paths = None
        self.set_g(g)

    def set_g(self, g):
        roots = roots_of_tree(g)
        assert len(roots) == 1
        self.root = roots[0]
        self.g = g

    def sample_likelihood(self, sample_idx):
        """Calculate the likelihood of all attachment points for a sample index"""
        assert sample_idx < self.n_samples
        sample_gls = self.gls[:, sample_idx].reshape(-1)
        attachment_log_like = np.zeros(self.gls.shape[0] + 1)
        current_log_like = np.sum(sample_gls[::NUM_GLS])
        attachment_log_like[0] = current_log_like
        for u, v, label in nx.dfs_labeled_edges(self.g, self.root):
            if u == v:
                pass
            elif label == 'forward':
                current_log_like += sample_gls[NUM_GLS * v + HET_NUM] \
                                    - sample_gls[NUM_GLS * v]
                attachment_log_like[v + 1] = current_log_like
            elif label == 'reverse':
                current_log_like -= sample_gls[NUM_GLS * v + HET_NUM] \
                                    - sample_gls[NUM_GLS * v]
            else:
                raise ValueError(f'Unexpected label: {label}')
        return np.sum(np.exp(attachment_log_like))

    def sample_marginalized_log_likelihood(self):
        """Calculate the sum of the likelihoods of all possible sample attachments"""
        like = 0
        for sample in range(self.n_samples):
            like += math.log(self.sample_likelihood(sample))
        return like
