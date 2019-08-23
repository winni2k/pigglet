import networkx as nx
import numpy as np

from pigglet.constants import NUM_GLS, HET_TUP


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

    def __init__(self, g, gls, sample_nodes):
        self.g = g
        self.gls = gls
        self.n_samples = self.gls.shape[1]

        self.root = roots_of_tree(self.g)
        assert len(self.root) == 1
        self.root = self.root[0]

        self.mutation_matrix_mask = np.zeros_like(self.gls, np.bool_)

    def _reset_mutation_matrix(self):
        self.mutation_matrix_mask = self.mutation_matrix_mask.reshape(-1)
        self.mutation_matrix_mask[::NUM_GLS] = True
        for start in range(1, NUM_GLS):
            self.mutation_matrix_mask[1::NUM_GLS] = False
        self.mutation_matrix_mask = self.mutation_matrix_mask.reshape(self.gls.shape)

    def calculate_likelihood(self, attachment_nodes):
        """Calculate the likelihood of the mutation tree with samples attached at
        `attachment_nodes`"""
        assert len(attachment_nodes) == self.n_samples
        self._reset_mutation_matrix()
        self._update_mutation_matrix_mask(attachment_nodes)
        return np.sum(self.gls[self.mutation_matrix_mask].reshape(self.gls.shape[0],
                                                                  self.gls.shape[1]))

    def _update_mutation_matrix_mask(self, attachment_nodes):
        attachment_node_set = set(attachment_nodes)
        attachment_node_set.discard(self.root)
        seen_paths = {p[-1]: p for p in
                      nx.all_simple_paths(self.g, self.root, attachment_node_set)}
        for sample_idx, node in enumerate(attachment_nodes):
            if node > self.root:
                path = seen_paths[node]
                self.mutation_matrix_mask[
                    np.array(path[1:]),
                    sample_idx
                ] = HET_TUP


def roots_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.in_degree)]
