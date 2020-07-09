from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class RandomWalkStopType(Enum):
    CONSTRAINED = 0
    UNCONSTRAINED = 1


@dataclass
class TreeMoveMemento:
    """This memento stores the information necessary to undo a TreeInteractor move"""

    commands: List[str] = field(default_factory=list)
    args: List[Dict[str, int]] = field(default_factory=list)

    @classmethod
    def of_prune(cls, edge):
        return cls(commands=["attach"], args=[{"target": edge[0], "node": edge[1]}])

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
