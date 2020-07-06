from pigglet.tree import TreeInteractor
from pigglet_testing.builders.tree import TreeBuilder


class TreeInteractorBuilder(TreeBuilder):
    def build(self):
        g = super().build()
        return TreeInteractor(g)
