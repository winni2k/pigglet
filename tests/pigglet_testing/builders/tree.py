import networkx as nx


class TreeBuilder:
    def __init__(self):
        self.g = nx.DiGraph()
        self.sample_ids = []

    def with_balanced_tree(self, height=2, n_branches=2):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        self._relabel_nodes()
        return self

    def with_random_tree(self, n_mutations):
        self.g = nx.gnr_graph(n_mutations, 0).reverse()
        self._relabel_nodes()

    def _relabel_nodes(self):
        nx.relabel_nodes(self.g, {n: n - 1 for n in self.g.nodes}, copy=False)

    def with_mutation_at(self, attachment_node, new_node_id):
        self.g.add_edge(attachment_node, new_node_id)
        return self

    def build(self):
        if len(self.g.nodes()) == 0:
            self.g.add_edge(-1, 0)
        return self.g
