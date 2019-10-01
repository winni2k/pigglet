import networkx as nx

from pigglet.tree import TreeInteractor


class PhylogeneticTreeConverter:
    def __init__(self, g):
        self.g = g
        self.phylo_g = None
        self.sample_attachments = None
        self.mutation_ids = None
        self.sample_ids = set()
        self.mutation_attachments = {}
        self.root = -1

    def convert(self, sample_attachments):
        self.sample_attachments = sample_attachments
        self._relabel_nodes_and_move_mutations_into_attribute()
        self._merge_tree()

        self._find_mutation_attachments()
        redundant_nodes = set()
        for node in self.phylo_g.nodes():
            if node not in self.sample_ids \
                    and len(nx.descendants(self.phylo_g, node) & self.sample_ids) == 0:
                redundant_nodes.add(node)
        for node in redundant_nodes:
            for mutation in self.phylo_g.node[node]['mutations']:
                del self.mutation_attachments[mutation]
        self.phylo_g.remove_nodes_from(redundant_nodes)

        # mut_attach = self.mutation_attachments
        # first, last = min(mut_attach.keys()), max(mut_attach.keys())
        # attach = [None] * (last - first + 1)
        # for k, attach_node in mut_attach.items():
        #     if attach_node < first:
        #         attach[k-first] = attach_node
        #     elif self.phylo_g.out_degree(attach_node) == 0:
        #         self.phylo_g.graph['mutations'].remove(k)
        #         self.phylo_g.remove_node(k)
        # self.phylo_g.graph['mutation_attachment_points'] = attach
        self.phylo_g.graph['mutation_attachments'] = self.mutation_attachments.copy()
        return self.phylo_g

    def _relabel_nodes_and_move_mutations_into_attribute(self):
        first_mutation = len(self.sample_attachments)
        self.phylo_g = nx.relabel_nodes(self.g,
                                        {n: n + first_mutation for n in self.g.nodes() if
                                         n != self.root})

        self.mutation_ids = frozenset(n for n in self.phylo_g.nodes() if n != self.root)
        for node in self.mutation_ids:
            self.phylo_g.node[node]['mutations'] = {node}
        self.phylo_g.node[self.root]['mutations'] = set()
        self.phylo_g.graph['mutations'] = self.mutation_ids
        assert len(self.sample_ids) == 0
        for idx, attachment in enumerate(self.sample_attachments):
            self.sample_ids.add(idx)
            if attachment != self.root:
                attachment += first_mutation
            self.phylo_g.add_edge(attachment, idx)
        self.sample_ids = frozenset(self.sample_ids)

    def _merge_tree(self):
        inter = TreeInteractor(self.phylo_g)
        start_over = True
        while start_over:
            start_over = False
            for u, v in nx.edge_dfs(self.phylo_g):
                children = set(self.phylo_g.succ[u])
                if len(children) == 1 and children & self.mutation_ids:
                    inter.merge_mutation_nodes(u, next(iter(children)))
                    start_over = True
                    break

    def _find_mutation_attachments(self):
        assert len(self.mutation_attachments) == 0
        for node in self.phylo_g.nodes():
            if 'mutations' in self.phylo_g.node[node]:
                for mut in self.phylo_g.node[node]['mutations']:
                    self.mutation_attachments[mut] = node


class TreeBuilder:
    def __init__(self):
        self.g = nx.DiGraph()
        self.sample_ids = []

    def with_balanced_tree(self, height=2, n_branches=2):
        self.g = nx.balanced_tree(n_branches, height, nx.DiGraph())
        self._relabel_nodes()
        return self

    def with_random_tree(self, n_mutations):
        self.g = nx.gnr_graph(n_mutations + 1, 0).reverse()
        self._relabel_nodes()

    def _relabel_nodes(self):
        nx.relabel_nodes(self.g, {n: n - 1 for n in self.g.nodes}, copy=False)

    def with_mutation_at(self, attachment_node, new_node_id):
        self.g.add_edge(attachment_node, new_node_id)
        return self

    def with_path(self, n_muts):
        start = -1
        for mut in range(n_muts):
            self.with_mutation_at(start, mut)
            start = mut
        return self

    def build(self):
        if len(self.g.nodes()) == 0:
            self.g.add_node(-1)
        return self.g


class TreeConverterBuilder(TreeBuilder):

    def build(self):
        return PhylogeneticTreeConverter(super().build())
