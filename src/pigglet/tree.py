import itertools
import random

import networkx as nx

from pigglet.constants import TMP_LABEL
from pigglet.tree_utils import roots_of_tree


class TreeMoveMemento:
    """This memento stores the information necessary to undo a TreeInteractor move"""

    def __init__(self, commands=None, args=None):
        if commands is None:
            commands = []
            args = []
        self.commands = commands
        self.args = args

    @classmethod
    def of_prune(cls, edge):
        return cls(commands=['attach'], args=[{'target': edge[0], 'node': edge[1]}])

    @classmethod
    def of_attach(cls, node, target):
        return cls(commands=['prune'], args=[{'node': node}])

    @classmethod
    def of_swap_labels(cls, n1, n2):
        return cls(commands=['swap_labels'], args=[{'n1': n1, 'n2': n2}])

    def append(self, other):
        self.commands += other.commands
        self.args += other.args


class TreeInteractor:
    """Manipulates a tree

    All public methods return a memento object that can be used to undo a move
    """

    def __init__(self, g):
        self.g = g
        self.root = roots_of_tree(g)
        assert len(self.root) == 1
        self.root = self.root[0]
        self.mh_correction = None

    def prune(self, node):
        """Remove the incoming link for a node"""
        self.mh_correction = 1
        edges = list(self.g.in_edges(node))
        self.g.remove_edge(*edges[0])
        return TreeMoveMemento.of_prune(edges[0])

    def attach(self, node, target):
        """Add a link pointing from target to node"""
        self.mh_correction = 1
        self.g.add_edge(target, node)
        return TreeMoveMemento.of_attach(node, target)

    def uniform_attach(self, node):
        self.mh_correction = 1
        valid_attachment_points = itertools.chain(
            nx.descendants(self.g, self.root),
            [self.root]
        )
        return self._uniform_attach_to_nodes(node, valid_attachment_points)

    def swap_labels(self, n1, n2):
        self.mh_correction = 1
        if n1 == n2 or n1 == self.root or n2 == self.root:
            raise ValueError
        nx.relabel_nodes(self.g, {n1: TMP_LABEL}, copy=False)
        nx.relabel_nodes(self.g, {n2: n1, TMP_LABEL: n2}, copy=False)
        return TreeMoveMemento.of_swap_labels(n1, n2)

    def swap_subtrees(self, n1, n2):
        if n1 == self.root or n2 == self.root:
            raise ValueError
        if n2 in nx.ancestors(self.g, n1):
            return self._uniform_subtree_swap(n2, n1)
        if n2 in nx.descendants(self.g, n1):
            return self._uniform_subtree_swap(n1, n2)
        self.mh_correction = 1
        n1_parent = self._parent_of(n1)
        n2_parent = self._parent_of(n2)
        memento = self.prune(n1)
        memento.append(self.prune(n2))
        memento.append(self.attach(n1, n2_parent))
        memento.append(self.attach(n2, n1_parent))
        return memento

    def undo(self, memento):
        for command, args in zip(reversed(memento.commands), reversed(memento.args)):
            getattr(self, command)(**args)

    def _parent_of(self, n):
        return next(self.g.predecessors(n))

    def _uniform_subtree_swap(self, ancestor, descendant):
        anc_parent = self._parent_of(ancestor)
        dec_descendants = set(nx.descendants(self.g, descendant))

        memento = self.prune(descendant)
        memento.append(self.attach(descendant, anc_parent))
        anc_descendants = set(nx.descendants(self.g, ancestor))

        memento.append(self.prune(ancestor))
        memento.append(self._uniform_attach_to_nodes(ancestor,
                                                     itertools.chain(dec_descendants,
                                                                     [descendant])))

        self.mh_correction = (len(dec_descendants) + 1) / (len(anc_descendants) + 1)
        return memento

    def _uniform_attach_to_nodes(self, node, target_nodes):
        target_nodes = list(target_nodes)
        attach_idx = random.randrange(len(target_nodes))
        return self.attach(node, target_nodes[attach_idx])

    def merge_mutation_nodes(self, keep, merge):
        self.g.node[keep]['mutations'] = self.g.node[keep]['mutations'] \
                                         | self.g.node[merge]['mutations']
        for merge_child in self.g.succ[merge]:
            self.g.add_edge(keep, merge_child)
        self.g.remove_node(merge)


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
        """"""
        inter = TreeInteractor(self.phylo_g)
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
            if 'mutations' in self.phylo_g.node[node]:
                for mut in self.phylo_g.node[node]['mutations']:
                    self.mutation_attachments[mut] = node
