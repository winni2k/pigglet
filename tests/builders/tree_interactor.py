from pigglet.tree import MutationTreeInteractor, PhyloTreeInteractor
from pigglet_testing.builders.tree import MutationTreeBuilder, PhyloTreeBuilder


class MutationTreeInteractorBuilder(MutationTreeBuilder):
    def build(self):
        g = super().build()
        return MutationTreeInteractor(g)


class PhyloTreeInteractorBuilder(PhyloTreeBuilder):
    def build(self):
        g = super().build()
        return PhyloTreeInteractor(g)
