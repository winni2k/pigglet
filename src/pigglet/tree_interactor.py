import itertools
import random
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Tuple

import networkx as nx

from pigglet.constants import TMP_LABEL, TreeIsTooSmallError
from pigglet.tree import RandomWalkStopType, TreeMoveMemento
from pigglet.tree_utils import roots_of_tree


def random_graph_walk_with_memory_from(g, start):
    """Walks the graph starting from start

    Does not yield start node.

    :yields: current node, number of unvisited neighbors of current node
    """
    current_node = start
    seen = {current_node}
    neighbors = [n for n in nx.all_neighbors(g, current_node) if n not in seen]
    current_node = random.choice(neighbors)
    while True:
        seen.add(current_node)
        neighbors = [
            n for n in nx.all_neighbors(g, current_node) if n not in seen
        ]
        yield current_node, neighbors
        if not neighbors:
            return
        current_node = random.choice(neighbors)


class TreeInteractor(ABC):
    pass


@dataclass
class PhyloTreeInteractor(TreeInteractor):
    g: nx.DiGraph = field(default_factory=nx.DiGraph)
    leaf_nodes: set = field(default_factory=set)
    root: int = 0
    mh_correction: float = 0
    _inner_g: nx.DiGraph = field(default_factory=nx.DiGraph)
    _last_node_id: int = 0

    def __post_init__(self):
        if len(self.g) == 0:
            self.g.add_edges_from([(0, 1), (0, 2)])
        self.leaf_nodes = {u for u in self.g if self.g.out_degree[u] == 0}
        roots = [u for u in self.g if self.g.in_degree[u] == 0]
        assert len(roots) == 1, roots
        self.root = roots[0]
        self._inner_g = self.g.subgraph(
            u for u in self.g if u not in self.leaf_nodes
        )
        self.check_binary_rooted_tree()
        if "leaves" not in self.g.nodes[self.root]:
            self.annotate_all_nodes_with_descendant_leaves()

    def attach_node_to_edge(self, node, edge: Tuple[Any, Any]):
        """Attach node to edge and generate new nodes as necessary"""
        if node not in self.g:
            raise ValueError(
                f"Node to attach ({node}) must already exist in tree"
            )
        new_node = self._generate_node_id()
        nx.add_path(self.g, [edge[0], new_node, edge[1]])
        self.g.add_edge(new_node, node)
        self.g.remove_edge(*edge)
        self._annotate_descendant_leaves_and_ancestors_of(new_node)
        return new_node

    def create_sample_on_edge(self, u: Any, v: Any) -> Tuple[Any, Any]:
        """
        Creates a new leaf node and attaches it to the edge (u, v)

        :returns: newly created edge (u, leaf_node)
        """
        sample_node = self._generate_node_id()
        self.g.add_node(sample_node)
        self.g.nodes[sample_node]["leaves"] = {sample_node}
        new_node = self.attach_node_to_edge(sample_node, (u, v))
        self.leaf_nodes.add(sample_node)
        return new_node, sample_node

    def prune_edge(self, u, v):
        """Prunes an edge and suppresses u if necessary"""
        if (u, v) not in self._inner_g.edges:
            raise ValueError(
                f"Edge {(u, v)} cannot be pruned because it is"
                f" not an inner edge"
            )
        g = self.g
        g.remove_edge(u, v)
        if g.in_degree[u] == 0:
            assert g.out_degree[u] == 1
            g.remove_node(u)
        else:
            assert g.degree[u] == 2
            parent = next(self.g.predecessors(u))
            g.add_edge(parent, next(self.g.successors(u)))
            g.remove_node(u)
            self._annotate_descendant_leaves_and_ancestors_of(parent)

    def random_edge(self):
        return random.choice(self.g.edges)

    def check_binary_rooted_tree(self):
        assert nx.is_directed_acyclic_graph(self.g)
        for node in self.g:
            if node == self.root:
                assert self.g.in_degree(node) == 0
                assert self.g.out_degree(node) == 2
            elif node in self.leaf_nodes:
                assert self.g.in_degree(node) == 1
                assert self.g.out_degree(node) == 0
            else:
                assert self.g.in_degree(node) == 1
                assert self.g.out_degree(node) == 2

    def _generate_node_id(self):
        while self._last_node_id in self.g:
            self._last_node_id += 1
        return self._last_node_id

    def _annotate_descendant_leaves_of(self, new_node):
        if self.g.out_degree(new_node) == 0:
            self.g.nodes[new_node]["leaves"] = {new_node}
            return
        children = list(self.g.successors(new_node))
        assert len(children) == 2
        self.g.nodes[new_node]["leaves"] = (
            self.g.nodes[children[0]]["leaves"]
            | self.g.nodes[children[1]]["leaves"]
        )

    def _annotate_descendant_leaves_and_ancestors_of(self, new_node):
        current = new_node
        while True:
            self._annotate_descendant_leaves_of(current)
            predecessors = list(self.g.predecessors(current))
            if not predecessors:
                break
            assert len(predecessors) == 1
            current = predecessors[0]

    def annotate_all_nodes_with_descendant_leaves(self):
        for node in nx.dfs_postorder_nodes(self.g, self.root):
            self._annotate_descendant_leaves_of(node)


class MutationTreeInteractor(TreeInteractor):
    """Manipulates a mutation tree

    All public methods return a memento object that can be used to undo a move
    """

    def __init__(self, g):
        self.g = g
        self.root = roots_of_tree(g)
        assert len(self.root) == 1, self.root
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
            nx.descendants(self.g, self.root), [self.root]
        )
        return self._uniform_attach_to_nodes(node, valid_attachment_points)

    def extend_attach(self, node, start, prop_attach):
        assert 0 <= prop_attach < 1
        assert node != start

        if len(list(nx.all_neighbors(self.g, start))) == 0:
            raise TreeIsTooSmallError
        if len(list(nx.all_neighbors(self.g, start))) == 1:
            start_constraint = RandomWalkStopType.CONSTRAINED
        else:
            start_constraint = RandomWalkStopType.UNCONSTRAINED
        for attach_node, neighbors in random_graph_walk_with_memory_from(
            self.g, start
        ):
            if random.random() < prop_attach:
                break

        constraint = RandomWalkStopType.UNCONSTRAINED
        if not neighbors:
            constraint = RandomWalkStopType.CONSTRAINED

        memento = self.attach(node, attach_node)
        assert self.mh_correction == 1
        if start_constraint == constraint:
            self.mh_correction = 1
        elif start_constraint == RandomWalkStopType.CONSTRAINED:
            self.mh_correction = 1 - prop_attach
        elif start_constraint == RandomWalkStopType.UNCONSTRAINED:
            self.mh_correction = 1 / (1 - prop_attach)
        else:
            raise Exception("Programmer error")
        return memento

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
        for command, args in zip(
            reversed(memento.commands), reversed(memento.args)
        ):
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
        memento.append(
            self._uniform_attach_to_nodes(
                ancestor, itertools.chain(dec_descendants, [descendant])
            )
        )

        self.mh_correction = (len(dec_descendants) + 1) / (
            len(anc_descendants) + 1
        )
        return memento

    def _uniform_attach_to_nodes(self, node, target_nodes):
        target_nodes = list(target_nodes)
        attach_idx = random.randrange(len(target_nodes))
        return self.attach(node, target_nodes[attach_idx])

    def merge_mutation_nodes(self, keep, merge):
        self.g.nodes[keep]["mutations"] = (
            self.g.nodes[keep]["mutations"] | self.g.nodes[merge]["mutations"]
        )
        for merge_child in self.g.succ[merge]:
            self.g.add_edge(keep, merge_child)
        self.g.remove_node(merge)
