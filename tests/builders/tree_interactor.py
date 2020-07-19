from pigglet_testing.builders.tree import MutationTreeBuilder, PhyloTreeBuilder

from pigglet.tree_interactor import MutationTreeInteractor, PhyloTreeInteractor


class MutationTreeInteractorBuilder(MutationTreeBuilder):
    def __init__(self, prng=None):
        super().__init__()
        self.prng = prng

    def build(self):
        g = super().build()
        return MutationTreeInteractor(g, self.prng)


class PhyloTreeInteractorBuilder(PhyloTreeBuilder):
    def build(self):
        g = super().build()
        return PhyloTreeInteractor(g)
