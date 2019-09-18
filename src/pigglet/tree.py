import itertools
import random

import networkx as nx

from pigglet.constants import TMP_LABEL
from pigglet.tree_utils import roots_of_tree


class TreeMoveMemento:
    """This memento stores the information necessary to undo a TreeInteractor move"""

    def __init__(self, commands, args):
        self.commands = commands
        self.args = args

    @classmethod
    def of_prune(cls, edge):
        return cls(commands=['attach'], args=[{'target': edge[0], 'node': edge[1]}])

    @classmethod
    def of_attach(cls, node, target):
        return cls(commands=['prune'], args=[{'node': node}])


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
        self._uniform_attach_to_nodes(node, valid_attachment_points)

    def swap_labels(self, n1, n2):
        self.mh_correction = 1
        if n1 == n2:
            raise ValueError
        nx.relabel_nodes(self.g, {n1: TMP_LABEL}, copy=False)
        nx.relabel_nodes(self.g, {n2: n1, TMP_LABEL: n2}, copy=False)

    def swap_subtrees(self, n1, n2):
        if n1 == self.root or n2 == self.root:
            raise ValueError
        if n2 in nx.ancestors(self.g, n1):
            self.mh_correction = self._uniform_subtree_swap(n2, n1)
            return
        elif n2 in nx.descendants(self.g, n1):
            self.mh_correction = self._uniform_subtree_swap(n1, n2)
            return
        self.mh_correction = 1
        n1_parent = self._parent_of(n1)
        n2_parent = self._parent_of(n2)
        self.prune(n1)
        self.prune(n2)
        self.attach(n1, n2_parent)
        self.attach(n2, n1_parent)

    def undo(self, memento):
        for idx, command in enumerate(memento.commands):
            getattr(self, command)(**memento.args[idx])

    def _parent_of(self, n):
        return next(self.g.predecessors(n))

    def _uniform_subtree_swap(self, ancestor, descendant):
        anc_parent = self._parent_of(ancestor)
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
