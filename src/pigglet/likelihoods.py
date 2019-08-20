import networkx as nx
import numpy as np

from pigglet.constants import NUM_GLS, HET_TUP


class TreeLikelihoodCalculator:
    """Calculates likelihood of tree (g) from gls for m sites and n samples

    self.gls should have shape (m, n, NUM_GLS)
    self.mutation_matrix_mask has shape (m, n, NUM_GLS)
    """

    def __init__(self, g, gls, sample_nodes):
        self.g = g
        self.gls = gls
        self.sample_nodes = sample_nodes
        assert self.gls.shape[1] == len(self.sample_nodes)

        self.sample_ids = [int(g.nodes[n]['sample_id']) for n in self.sample_nodes]
        assert max(self.sample_ids) + 1 == len(self.sample_nodes)

        self.mutation_matrix_mask = np.zeros_like(gls, np.bool_).reshape(-1)
        self.mutation_matrix_mask[::NUM_GLS] = True
        self.mutation_matrix_mask = self.mutation_matrix_mask.reshape(gls.shape)

        self.root = roots_of_tree(self.g)
        assert len(self.root) == 1
        self.root = self.root[0]

    def calculate_likelihood(self):
        self._update_mutation_matrix_mask()
        return np.sum(self.gls[self.mutation_matrix_mask].reshape(self.gls.shape[0],
                                                                  self.gls.shape[1]))

    def _update_mutation_matrix_mask(self):
        for idx, path in enumerate(
                nx.all_simple_paths(self.g, self.root, self.sample_nodes)):
            sample_mutations = []
            assert len(path) > 0
            for node in path:
                try:
                    sample_mutations += self.g.nodes[node]['mutations']
                except KeyError:
                    pass
            if sample_mutations:
                self.mutation_matrix_mask[
                    np.array(sample_mutations),
                    self.g.nodes[node]['sample_id']
                ] = HET_TUP


def roots_of_tree(g):
    return [tup[0] for tup in filter(lambda tup: tup[1] == 0, g.in_degree)]