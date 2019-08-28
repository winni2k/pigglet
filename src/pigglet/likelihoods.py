import math

import networkx as nx
import numpy as np

from pigglet.constants import NUM_GLS, HET_TUP, HOM_TUP
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
        assert sample_idx < self.n_samples
        mutation_mask = np.zeros(self.gls.shape[0] * 2, np.bool_)
        mutation_mask[::NUM_GLS] = True
        sample_gls = self.gls[:, sample_idx].reshape(-1)
        attachment_log_like = np.zeros(self.gls.shape[0] + 1)
        attachment_log_like[0] = np.sum(sample_gls[mutation_mask])
        for u, v, label in nx.dfs_labeled_edges(self.g, self.root):
            if u == v:
                pass
            elif label == 'forward':
                mutation_mask[NUM_GLS * v: (NUM_GLS * v + NUM_GLS)] = HET_TUP
                attachment_log_like[v + 1] = np.sum(sample_gls[mutation_mask])
            elif label == 'reverse':
                mutation_mask[NUM_GLS * v: (NUM_GLS * v + NUM_GLS)] = HOM_TUP
            else:
                raise ValueError(f'Unexpected label: {label}')
        return np.sum(np.exp(attachment_log_like))

    def sample_marginalized_likelihood(self):
        """Calculate the sum of the likelihoods of all possible sample attachments"""
        like = 0
        for sample in range(self.n_samples):
            like += math.log(self.sample_likelihood(sample))
        return math.exp(like)
