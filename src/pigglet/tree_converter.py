import networkx as nx

from pigglet.tree_interactor import MutationTreeInteractor
from pigglet.tree_utils import roots_of_tree


class MutationToPhylogeneticTreeConverter:
    def __init__(self, g, prng):
        self.g = g
        self.prng = prng
        self.phylo_g = None
        self.sample_attachments = None
        self.mutation_ids = None
        self.sample_ids = set()
        self.mutation_attachments = {}
        roots = roots_of_tree(self.g)
        assert len(roots) == 1
        self.root = roots[0]

    def convert(self, sample_attachments):
        self.sample_attachments = sample_attachments
        self._test_prerequisites()
        self._relabel_nodes_and_move_mutations_into_attribute()
        self._merge_tree()

        self._find_mutation_attachments()
        redundant_nodes = set()
        for node in self.phylo_g.nodes():
            if (
                node not in self.sample_ids
                and len(nx.descendants(self.phylo_g, node) & self.sample_ids)
                == 0
            ):
                redundant_nodes.add(node)
        for node in redundant_nodes:
            for mutation in self.phylo_g.nodes[node]["mutations"]:
                del self.mutation_attachments[mutation]
        self.phylo_g.remove_nodes_from(redundant_nodes)
        self.phylo_g.graph[
            "mutation_attachments"
        ] = self.mutation_attachments.copy()
        return self.phylo_g

    def _relabel_nodes_and_move_mutations_into_attribute(self):
        first_mutation = len(self.sample_attachments)
        self.phylo_g = nx.relabel_nodes(
            self.g,
            {n: n + first_mutation for n in self.g.nodes() if n != self.root},
        )

        self.mutation_ids = frozenset(
            n for n in self.phylo_g.nodes() if n != self.root
        )
        for node in self.mutation_ids:
            self.phylo_g.nodes[node]["mutations"] = {node}
        self.phylo_g.nodes[self.root]["mutations"] = set()
        self.phylo_g.graph["mutations"] = self.mutation_ids
        assert len(self.sample_ids) == 0
        for idx, attachment in enumerate(self.sample_attachments):
            self.sample_ids.add(idx)
            if attachment != self.root:
                attachment += first_mutation
            self.phylo_g.add_edge(attachment, idx)
        self.sample_ids = frozenset(self.sample_ids)

    def _merge_tree(self):
        """"""
        inter = MutationTreeInteractor(self.phylo_g, self.prng)
        start_over = True
        while start_over:
            start_over = False
            for node in self.phylo_g.nodes():
                children = set(self.phylo_g.succ[node])
                if len(children) == 1 and children & self.mutation_ids:
                    inter.merge_mutation_nodes(node, children.pop())
                    start_over = True
                    break
                elif len(children) == 0 and node in self.mutation_ids:
                    self.phylo_g.remove_node(node)
                    start_over = True
                    break

    def _find_mutation_attachments(self):
        assert len(self.mutation_attachments) == 0
        for node in self.phylo_g.nodes():
            if "mutations" in self.phylo_g.nodes[node]:
                for mut in self.phylo_g.nodes[node]["mutations"]:
                    self.mutation_attachments[mut] = node

    def _test_prerequisites(self):
        if len(self.sample_attachments) == 0:
            raise ValueError("sample_attachments cannot be empty")
        for attach_point in self.sample_attachments:
            if attach_point not in self.g:
                raise ValueError(
                    f"Could not find sample attachment point"
                    f" {attach_point} in tree"
                )
