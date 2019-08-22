import networkx as nx

from pigglet_testing.utils import sample_nodes_of_tree


class TreeBuilder:
    def __init__(self):
        self.g = nx.DiGraph()
        self.sample_ids = []

    def with_balanced_tree(self, height=2, n_branches=2):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        nx.relabel_nodes(self.g, {n: n - 1 for n in self.g.nodes}, copy=False)
        return self

    def with_mutation_at(self, attachment_node, new_node_id):
        self.g.add_edge(attachment_node, new_node_id)
        return self

    def with_sample_at(self, attachment_node, new_sample_name):
        self.g.add_node(new_sample_name, sample_id=len(self.sample_ids))
        self.g.add_edge(attachment_node, new_sample_name)
        return self

    def with_sample_ids(self, *ids):
        self.sample_ids = ids
        return self

    def build(self):
        if len(self.g.nodes()) == 0:
            self.g.add_edge(-1, 0)
        sample_nodes = sample_nodes_of_tree(self.g)
        if len(self.sample_ids) == 0:
            self.sample_ids = list(range(len(sample_nodes)))
        for sample_id, sample_node in zip(self.sample_ids, sample_nodes):
            self.g.nodes[sample_node]['sample_id'] = sample_id

        return self.g