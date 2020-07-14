from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any


class RandomWalkStopType(Enum):
    CONSTRAINED = 0
    UNCONSTRAINED = 1


@dataclass
class TreeMoveMemento(ABC):
    commands: List[str] = field(default_factory=list)
    args: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PhyloTreeMoveMemento(TreeMoveMemento):
    """Stores the information necessary to undo a PhyloTreeInteractor move"""

    @classmethod
    def of_add_semiconnected_root(cls, new_root):
        return cls(
            commands=["remove_semiconnected_root"], args=[{"root": new_root}]
        )

    @classmethod
    def of_remove_semiconnected_root(cls, root):
        return cls(
            commands=["add_semiconnected_root"], args=[{"new_root": root}]
        )

    @classmethod
    def of_prune_edge(cls, u, v, parent, child):
        return cls(
            commands=["attach_edge_to_edge"],
            args=[{"new_edge": (u, v), "target_edge": (parent, child)}],
        )

    @classmethod
    def of_attach_edge_to_edge(cls, new_edge):
        return cls(
            commands=["prune_edge"],
            args=[{"u": new_edge[0], "v": new_edge[1]}],
        )


@dataclass
class MutationTreeMoveMemento(TreeMoveMemento):
    """Stores the information necessary to undo a MutationTreeInteractor
    move"""

    @classmethod
    def of_prune(cls, edge):
        return cls(
            commands=["attach"], args=[{"target": edge[0], "node": edge[1]}]
        )

    @classmethod
    def of_attach(cls, node, target):
        return cls(commands=["prune"], args=[{"node": node}])

    @classmethod
    def of_swap_labels(cls, n1, n2):
        return cls(commands=["swap_labels"], args=[{"n1": n1, "n2": n2}])

    def append(self, other):
        self.commands += other.commands
        self.args += other.args


def strip_tree(g):
    g = g.copy()
    g.graph.clear()
    for node in g:
        g.nodes[node].clear()
    return g
