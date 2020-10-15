import itertools
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from pigglet.constants import TMP_LABEL, TreeIsTooSmallError
from pigglet.tree import MutationTreeMoveMemento, RandomWalkStopType
from pigglet.tree_utils import roots_of_tree

Node = int
logger = logging.getLogger(__name__)


@dataclass
class GraphAnnotator:
    """Annotate leaves of nodes on graph

    updated_nodes contains key=node, value=node leaves before update
    """

    g: nx.DiGraph
    _updated_nodes: Dict[int, frozenset] = field(default_factory=dict)

    def traversed_inner_nodes(self) -> Iterable:
        return self._updated_nodes.keys()

    # def updated_nodes(self) -> Iterable:
    #     return self._updated_nodes.keys()

    def annotate_all_nodes_with_descendant_leaves(self, start=None):
        if start is None:
            roots = roots_of_tree(self.g)
            assert len(roots) == 1
            start = roots[0]
        self._updated_nodes.clear()
        for node in nx.dfs_postorder_nodes(self.g, start):
            if self.g.out_degree(node) == 0:
                self._annotate_leaf(node)
            else:
                self._annotate_leaves_from_successors(node)

    def _annotate_leaves_from_successors(self, node):
        node_view = self.g.nodes[node]
        children = list(self.g.successors(node))
        assert len(children) == 2
        new_leaves = (
            self.g.nodes[children[0]]["leaves"]
            | self.g.nodes[children[1]]["leaves"]
        )
        if "leaves" not in node_view:
            assert node not in self._updated_nodes
            self._updated_nodes[node] = frozenset()
        elif new_leaves != node_view["leaves"]:
            if node not in self._updated_nodes:
                self._updated_nodes[node] = node_view["leaves"]
        else:
            return
        node_view["leaves"] = new_leaves

    def _annotate_leaf(self, node):
        node_view = self.g.nodes[node]
        if "leaves" not in self.g.nodes[node]:
            self._updated_nodes[node] = frozenset()
            node_view["leaves"] = {node}

    def annotate_leaves_of_nodes_and_their_ancestors(self, *nodes):
        self._updated_nodes.clear()
        for node in postorder_nodes_from_frontier(self.g, nodes):
            if self.g.out_degree(node) == 0:
                self._annotate_leaf(node)
            else:
                self._annotate_leaves_from_successors(node)


def postorder_nodes_from_frontier(g, frontier):
    if frontier in g:
        frontier = {frontier}
    else:
        frontier = set(frontier)
    ancestors = set()
    for u in frontier:
        for anc in nx.ancestors(g, u):
            ancestors.add(anc)
    target = ancestors | frontier
    roots = roots_of_tree(g)
    assert len(roots) == 1
    for u in nx.dfs_postorder_nodes(g, source=roots[0]):
        if u in target:
            yield u


@dataclass
class PhyloTreeMoveMemento:
    """Stores the information necessary to undo a PhyloTreeInteractor move"""

    commands: List = field(default_factory=list)
    args: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, other):
        self.commands += other.commands
        self.args += other.args


@dataclass
class PhyloTreePruneRegraftMemento(PhyloTreeMoveMemento):
    pass


@dataclass
class PhyloTreeRootedPruneRegraftMemento(PhyloTreeMoveMemento):
    pass


@dataclass
class PhyloTreeSwapLeavesMemento(PhyloTreeMoveMemento):
    pass


@dataclass
class PhyloTreeMoveMementoBuilder:
    interactor: Any

    def of_prune_and_regraft(
        self, node, edge: Optional[Tuple[Node, Node]] = None
    ):
        if not edge:
            return PhyloTreeRootedPruneRegraftMemento(
                commands=[self.interactor.rooted_prune_and_regraft],
                args=[{"node": node}],
            )
        return PhyloTreePruneRegraftMemento(
            commands=[self.interactor.prune_and_regraft],
            args=[{"node": node, "edge": edge}],
        )

    def of_swap_leaves(self, u, v):
        return PhyloTreeSwapLeavesMemento(
            commands=[self.interactor.swap_leaves], args=[{"u": u, "v": v}]
        )


class TreeInteractor(ABC):
    @abstractmethod
    def undo(self, memento):
        pass

    def random_graph_walk_with_memory_from(self, start, seen=None):
        """Walks the graph starting from start

        Does not yield start node.

        :yields: current node, number of unvisited neighbors of current node
        """
        if not seen:
            seen = set()
        seen.add(start)
        current_node = start
        neighbors = [
            n for n in nx.all_neighbors(self.g, current_node) if n not in seen
        ]
        current_node = self.prng.choice(neighbors)
        while True:
            seen.add(current_node)
            neighbors = [
                n
                for n in nx.all_neighbors(self.g, current_node)
                if n not in seen
            ]
            yield current_node, neighbors
            if not neighbors:
                return
            current_node = self.prng.choice(neighbors)


@dataclass
class PhyloTreeInteractor(TreeInteractor):
    g: nx.DiGraph = field(default_factory=nx.DiGraph)
    prng: Any = field(default_factory=lambda: random)
    leaf_nodes: frozenset = field(init=False)
    leaf_node_list: list = field(init=False)
    mh_correction: float = 0
    inner_g: nx.DiGraph = field(default_factory=nx.DiGraph)
    _last_node_id: int = 0
    _memento_builder: PhyloTreeMoveMementoBuilder = field(init=False)
    _annotator: GraphAnnotator = field(init=False)
    changed_nodes: Set[int] = field(default_factory=set)

    def __post_init__(self):
        self._memento_builder = PhyloTreeMoveMementoBuilder(self)
        if len(self.g) < 3:
            raise ValueError
        self.leaf_nodes = frozenset(
            u for u in self.g if self.g.out_degree[u] == 0
        )
        self.leaf_node_list = sorted(self.leaf_nodes)
        self.inner_g = self.g.subgraph(
            u for u in self.g if u not in self.leaf_nodes
        )
        self.check_binary_rooted_tree()
        self._annotator = GraphAnnotator(self.g)
        self._annotator.annotate_all_nodes_with_descendant_leaves(self.root)

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
        updated_nodes = [parent]
        if parent in edge:
            return PhyloTreeMoveMemento()
        if g.in_degree(parent) == 0:
            g.remove_node(parent)
            memento = self._memento_builder.of_prune_and_regraft(node)
        else:
            parent_parent = next(g.predecessors(parent))
            parent_child = next(u for u in g.successors(parent) if u != node)
            g.add_edge(parent_parent, parent_child)
            g.remove_node(parent)
            updated_nodes.append(parent_parent)
            memento = self._memento_builder.of_prune_and_regraft(
                node, (parent_parent, parent_child)
            )
        nx.add_path(g, [edge[0], parent, edge[1]])
        g.add_edge(parent, node)
        g.remove_edge(*edge)
        try:
            self._annotator.annotate_leaves_of_nodes_and_their_ancestors(
                *updated_nodes
            )
        except KeyError:
            logger.error(f"During: prune_and_regraft({node}, {edge})")
            logger.error(f"Updated nodes: {updated_nodes}")
            logger.error(self.g.nodes(data=True))
            logger.error(self.g.edges)
            raise

        self.changed_nodes = set(updated_nodes)
        return memento

    def rooted_prune_and_regraft(self, node: Node):
        """Create new root with old root and node as children"""
        g = self.g
        root = self.root
        parent = next(g.predecessors(node))
        if parent == root:
            return PhyloTreeMoveMemento()
        predecessors = list(g.predecessors(parent))
        parent_parent = predecessors[0]
        parent_child = next(u for u in g.successors(parent) if u != node)
        g.add_edge(parent_parent, parent_child)
        g.remove_node(parent)
        g.add_edge(parent, root)
        g.add_edge(parent, node)
        self.check_binary_rooted_tree()
        self._annotator.annotate_leaves_of_nodes_and_their_ancestors(
            parent_parent
        )
        self.changed_nodes = {parent, parent_parent}
        return self._memento_builder.of_prune_and_regraft(
            node, (parent_parent, parent_child)
        )

    def swap_leaves(self, u, v) -> PhyloTreeMoveMemento:
        assert u != v
        assert u in self.leaf_nodes
        assert v in self.leaf_nodes
        parents = [next(self.g.predecessors(node)) for node in [u, v]]
        if parents[0] == parents[1]:
            return PhyloTreeMoveMemento()
        for node, parent in zip([u, v], parents):
            self.g.remove_edge(parent, node)
        for node, parent in zip([u, v], reversed(parents)):
            self.g.add_edge(parent, node)
        self._annotator.annotate_leaves_of_nodes_and_their_ancestors(*parents)
        self.changed_nodes = set(parents)
        return self._memento_builder.of_swap_leaves(u, v)

    def extend_prune_and_regraft(
        self, node, prop_attach
    ) -> Tuple[PhyloTreeMoveMemento, Tuple[int, int]]:
        assert 0 <= prop_attach < 1
        assert self.g.in_degree(node) == 1
        edge = self._find_attach_edge(node, prop_attach)
        if edge[0] == self.root:
            return (
                self.rooted_prune_and_regraft(node),
                edge,
            )
        if edge not in self.g.edges:
            edge = edge[1], edge[0]
        assert edge in self.g.edges, edge
        return self.prune_and_regraft(node, edge), edge

    def _find_attach_edge(self, node, prop_attach):
        start = next(self.g.predecessors(node))
        n_neighbors = len(list(nx.all_neighbors(self.g, start)))
        if n_neighbors == 1:
            raise TreeIsTooSmallError
        if n_neighbors == 2:
            start_constraint = RandomWalkStopType.CONSTRAINED
        else:
            start_constraint = RandomWalkStopType.UNCONSTRAINED
        previous_node = start
        step = 0
        for attach_node, neighbors in self.random_graph_walk_with_memory_from(
            start, seen={node}
        ):
            step += 1
            if not neighbors:
                break
            if self.prng.random() < prop_attach:
                break
            previous_node = attach_node
        if step == 1:
            self.mh_correction = 1
        else:
            self.mh_correction = determine_espr_mh_correction(
                neighbors, prop_attach, start_constraint
            )

        return previous_node, attach_node

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


def determine_espr_mh_correction(neighbors, prop_attach, start_constraint):
    if neighbors:
        constraint = RandomWalkStopType.UNCONSTRAINED
    else:
        constraint = RandomWalkStopType.CONSTRAINED
    if start_constraint == constraint:
        return 1
    elif start_constraint == RandomWalkStopType.CONSTRAINED:
        return 1 - prop_attach
    elif start_constraint == RandomWalkStopType.UNCONSTRAINED:
        return 1 / (1 - prop_attach)
    else:
        raise Exception("Programmer error")


class MutationTreeInteractor(TreeInteractor):
    """Manipulates a mutation tree

    All public methods return a memento object that can be used to undo a move
    """

    def __init__(self, g, prng):
        self.g = g
        self.prng = prng
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
        assert 0 <= prop_attach <= 1
        assert node != start

        if len(list(nx.all_neighbors(self.g, start))) == 0:
            raise TreeIsTooSmallError
        if len(list(nx.all_neighbors(self.g, start))) == 1:
            start_constraint = RandomWalkStopType.CONSTRAINED
        else:
            start_constraint = RandomWalkStopType.UNCONSTRAINED
        for attach_node, neighbors in self.random_graph_walk_with_memory_from(
            start
        ):
            if self.prng.random() < prop_attach:
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
        attach_idx = self.prng.randrange(len(target_nodes))
        return self.attach(node, target_nodes[attach_idx])

    def merge_mutation_nodes(self, keep, merge):
        self.g.nodes[keep]["mutations"] = (
            self.g.nodes[keep]["mutations"] | self.g.nodes[merge]["mutations"]
        )
        for merge_child in self.g.succ[merge]:
            self.g.add_edge(keep, merge_child)
        self.g.remove_node(merge)
