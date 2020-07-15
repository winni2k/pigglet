import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict, Optional

import networkx as nx

from pigglet.constants import TMP_LABEL, TreeIsTooSmallError
from pigglet.tree import (
    RandomWalkStopType,
    MutationTreeMoveMemento,
)
from pigglet.tree_utils import roots_of_tree

Node = int


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


@dataclass
class PhyloTreeMoveMemento:
    """Stores the information necessary to undo a PhyloTreeInteractor move"""

    commands: List = field(default_factory=list)
    args: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, other):
        self.commands += other.commands
        self.args += other.args


@dataclass
class PhyloTreeMoveMementoBuilder:
    interactor: Any

    def of_prune_and_regraft(
        self, node, edge: Optional[Tuple[Node, Node]] = None
    ):
        if not edge:
            return PhyloTreeMoveMemento(
                commands=[self.interactor.rooted_prune_and_regraft],
                args=[{"node": node}],
            )
        return PhyloTreeMoveMemento(
            commands=[self.interactor.prune_and_regraft],
            args=[{"node": node, "edge": edge}],
        )

    def of_swap_leaves(self, u, v):
        return PhyloTreeMoveMemento(
            commands=[self.interactor.swap_leaves], args=[{"u": u, "v": v}]
        )


class TreeInteractor(ABC):
    @abstractmethod
    def undo(self, memento):
        pass


@dataclass
class PhyloTreeInteractor(TreeInteractor):
    g: nx.DiGraph = field(default_factory=nx.DiGraph)
    leaf_nodes: frozenset = field(default_factory=frozenset)
    mh_correction: float = 0
    _inner_g: nx.DiGraph = field(default_factory=nx.DiGraph)
    _last_node_id: int = 0
    _memento_builder: PhyloTreeMoveMementoBuilder = field(init=False)

    def __post_init__(self):
        self._memento_builder = PhyloTreeMoveMementoBuilder(self)
        if len(self.g) == 0:
            self.g.add_edges_from([(0, 1), (0, 2)])
        self.leaf_nodes = frozenset(
            u for u in self.g if self.g.out_degree[u] == 0
        )
        self._inner_g = self.g.subgraph(
            u for u in self.g if u not in self.leaf_nodes
        )
        self.check_binary_rooted_tree()
        if "leaves" not in self.g.nodes[self.root]:
            self.annotate_all_nodes_with_descendant_leaves()

    @property
    def root(self):
        roots = [u for u in self.g if self.g.in_degree[u] == 0]
        assert len(roots) == 1, roots
        return roots[0]

    def undo(self, memento):
        for command, args in zip(
            reversed(memento.commands), reversed(memento.args)
        ):
            command(**args)

    def prune_and_regraft(
        self, node: Node, edge: Tuple[Node, Node]
    ) -> PhyloTreeMoveMemento:
        g = self.g
        assert g.in_degree(node) == 1
        parent = next(g.predecessors(node))
        updated_nodes = {parent}
        if edge[0] == parent:
            return PhyloTreeMoveMemento()
        if g.in_degree(parent) == 0:
            g.remove_node(parent)
            memento = self._memento_builder.of_prune_and_regraft(node)
        else:
            parent_parent = next(g.predecessors(parent))
            parent_child = next(u for u in g.successors(parent) if u != node)
            g.add_edge(parent_parent, parent_child)
            g.remove_node(parent)
            updated_nodes.add(parent_parent)
            memento = self._memento_builder.of_prune_and_regraft(
                node, (parent_parent, parent_child)
            )
        nx.add_path(g, [edge[0], parent, edge[1]])
        g.add_edge(parent, node)
        g.remove_edge(*edge)
        for u in updated_nodes:
            self._annotate_leaves_of_node_and_its_ancestors(u)
        return memento

    def rooted_prune_and_regraft(self, node: Node):
        g = self.g
        root = self.root
        parent = next(g.predecessors(node))
        parent_parent = next(g.predecessors(parent))
        parent_child = next(u for u in g.successors(parent) if u != node)
        g.add_edge(parent_parent, parent_child)
        g.remove_node(parent)
        g.add_edge(parent, root)
        g.add_edge(parent, node)
        for u in [parent_parent]:
            self._annotate_leaves_of_node_and_its_ancestors(u)
        return self._memento_builder.of_prune_and_regraft(
            node, (parent_parent, parent_child)
        )

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

    def _annotate_leaves_of_node_and_its_ancestors(self, new_node):
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

    def swap_leaves(self, u, v) -> PhyloTreeMoveMemento:
        assert u != v
        assert u in self.leaf_nodes
        assert v in self.leaf_nodes
        parents = [next(self.g.predecessors(node)) for node in [u, v]]
        for node, parent in zip([u, v], parents):
            self.g.remove_edge(parent, node)
        for node, parent in zip([u, v], reversed(parents)):
            self.g.add_edge(parent, node)
        for parent in parents:
            self._annotate_leaves_of_node_and_its_ancestors(parent)
        return self._memento_builder.of_swap_leaves(u, v)

    # def extend_attach(self, node, start, prop_attach):
    #     assert 0 <= prop_attach < 1
    #     assert node != start
    #
    #     if len(list(nx.all_neighbors(self.g, start))) == 0:
    #         raise TreeIsTooSmallError
    #     if len(list(nx.all_neighbors(self.g, start))) == 1:
    #         start_constraint = RandomWalkStopType.CONSTRAINED
    #     else:
    #         start_constraint = RandomWalkStopType.UNCONSTRAINED
    #     previous_node = start
    #     for attach_node, neighbors in random_graph_walk_with_memory_from(
    #         self.g, start
    #     ):
    #         if random.random() < prop_attach:
    #             break
    #         previous_node = attach_node
    #     attach_edge = (previous_node, attach_node)
    #     if attach_edge not in self.g.edges:
    #         attach_edge = reversed(attach_edge)
    #     assert attach_edge in self.g.edges
    #     if attach_edge[0] != start:
    #         parent = list(self.g.predecessors(start))
    #         if len(parent) == 0:
    #             self.g.remove_node(start)
    #     constraint = RandomWalkStopType.UNCONSTRAINED
    #     if not neighbors:
    #         constraint = RandomWalkStopType.CONSTRAINED
    #
    #     memento = self.attach_node_to_edge(node, attach_edge)
    #     assert self.mh_correction == 1
    #     if start_constraint == constraint:
    #         self.mh_correction = 1
    #     elif start_constraint == RandomWalkStopType.CONSTRAINED:
    #         self.mh_correction = 1 - prop_attach
    #     elif start_constraint == RandomWalkStopType.UNCONSTRAINED:
    #         self.mh_correction = 1 / (1 - prop_attach)
    #     else:
    #         raise Exception("Programmer error")
    #     return memento
    #


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

    def undo(self, memento):
        for command, args in zip(
            reversed(memento.commands), reversed(memento.args)
        ):
            getattr(self, command)(**args)

    def prune(self, node):
        """Remove the incoming link for a node"""
        self.mh_correction = 1
        edges = list(self.g.in_edges(node))
        self.g.remove_edge(*edges[0])
        return MutationTreeMoveMemento.of_prune(edges[0])

    def attach(self, node, target):
        """Add a link pointing from target to node"""
        self.mh_correction = 1
        self.g.add_edge(target, node)
        return MutationTreeMoveMemento.of_attach(node, target)

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
        return MutationTreeMoveMemento.of_swap_labels(n1, n2)

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
