import itertools
import random

import networkx as nx

from pigglet.constants import TMP_LABEL
from pigglet.tree_utils import roots_of_tree


class TreeInteractor:

    def __init__(self, g):
        self.g = g
        self.root = roots_of_tree(g)
        assert len(self.root) == 1
        self.root = self.root[0]

    def prune(self, node):
        for edge in list(self.g.in_edges(node)):
            self.g.remove_edge(edge[0], edge[1])

    def attach(self, node, target):
        self.g.add_edge(target, node)

    def uniform_attach(self, node):
        valid_attachment_points = itertools.chain(
            nx.descendants(self.g, self.root),
            [self.root]
        )
        self._uniform_attach_to_nodes(node, valid_attachment_points)
        return 1

    def swap_labels(self, n1, n2):
        if n1 == n2:
            raise ValueError
        nx.relabel_nodes(self.g, {n1: TMP_LABEL}, copy=False)
        nx.relabel_nodes(self.g, {n2: n1, TMP_LABEL: n2}, copy=False)
        return 1

    def swap_subtrees(self, n1, n2):
        if n1 == -1 or n2 == -1:
            raise ValueError
        if n2 in nx.ancestors(self.g, n1):
            return self._uniform_subtree_swap(n2, n1)
        elif n2 in nx.descendants(self.g, n1):
            return self._uniform_subtree_swap(n1, n2)
        n1_parent = next(self.g.predecessors(n1))
        n2_parent = next(self.g.predecessors(n2))
        self.prune(n1)
        self.prune(n2)
        self.attach(n1, n2_parent)
        self.attach(n2, n1_parent)
        return 1

    def _uniform_subtree_swap(self, ancestor, descendant):
        anc_parent = next(self.g.predecessors(ancestor))
        dec_descendants = nx.descendants(self.g, descendant)

        self.prune(descendant)
        self.attach(descendant, anc_parent)
        anc_descendants = nx.descendants(self.g, ancestor)

        self.prune(ancestor)
        self._uniform_attach_to_nodes(ancestor,
                                      itertools.chain(dec_descendants, [descendant]))

        return (len(dec_descendants) + 1) / (len(anc_descendants) + 1)

    def _uniform_attach_to_nodes(self, node, target_nodes):
        target_nodes = list(target_nodes)
        attach_idx = random.randrange(len(target_nodes))
        self.attach(node, target_nodes[attach_idx])
