import networkx as nx
import numpy as np

from pigglet.constants import HET_NUM
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
        glstmp = np.zeros((gls.shape[0], gls.shape[2], gls.shape[1]))
        for genotype_idx in range(gls.shape[2]):
            glstmp[:, genotype_idx, :] = gls[:, :, genotype_idx]
        self.gls = glstmp
        self.n_sites = self.gls.shape[0]
        self.n_samples = self.gls.shape[2]
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
        return self.sample_likelihoods()[sample_idx]

    def sample_likelihoods(self):
        """Calculate the likelihoods of all possible sample attachments"""
        attachment_log_like = np.zeros((self.n_sites + 1, self.n_samples),
                                       dtype=np.float128)
        current_log_like = np.sum(
            self.gls[:, 0, :].reshape((self.n_sites, self.n_samples)),
            0)
        attachment_log_like[0] = current_log_like
        for u, v, label in nx.dfs_labeled_edges(self.g, self.root):
            if u == v:
                pass
            elif label == 'forward':
                current_log_like += self.gls[v, HET_NUM, :] \
                                    - self.gls[v, 0, :]
                attachment_log_like[v + 1] = current_log_like
            elif label == 'reverse':
                current_log_like -= self.gls[v, HET_NUM, :] \
                                    - self.gls[v, 0, :]
            else:
                raise ValueError(f'Unexpected label: {label}')
        return np.sum(np.exp(attachment_log_like), 0)

    def sample_marginalized_log_likelihood(self):
        """Calculate the sum of the log likelihoods of all possible sample attachments"""
        return np.sum(np.log(self.sample_likelihoods()))
