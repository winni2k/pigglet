class PhyloTreeExpectation:
    def __init__(self, g):
        self.g = g

    def consists_of_edges(self, *edges):
        assert sorted(self.g.edges()) == sorted(set(edges))
        return self

    def only_has_mutations(self, *mutations):
        assert self.g.graph["mutations"] == set(mutations)
        return self

    def has_mutation_at_node(self, mutation, node):
        assert mutation in self.g.nodes[node]["mutations"]
        return self

    def has_node_mutations(self, node, mutations):
        assert self.g.nodes[node]["mutations"] == set(mutations)
        return self

    def has_mutation_attachments(self, attachments):
        assert self.g.graph["mutation_attachments"] == attachments
        return self
