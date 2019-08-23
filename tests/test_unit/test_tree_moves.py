from pigglet_testing.builders.tree import TreeBuilder


class TreeInteractor:

    def __init__(self, g):
        self.g = g

    def prune(self, node):
        for edge in list(self.g.in_edges(node)):
            self.g.remove_edge(edge[0], edge[1])
        return self

    def attach(self, node, target):
        self.g.add_edge(target, node)
        return self


class TreeInteractorBuilder(TreeBuilder):

    def build(self):
        g = super().build()
        return TreeInteractor(g)


class TestPrune:

    def test_removes_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)

        # then
        assert interactor.g.in_degree[0] == 0
        assert interactor.g.out_degree[-1] == 0


class TestAttach:
    def test_reattaches_single_mutation(self):
        # given
        b = TreeInteractorBuilder()
        b.with_mutation_at(-1, 0)
        interactor = b.build()

        # when
        interactor.prune(0)
        interactor.attach(0, -1)

        # then
        assert interactor.g.in_degree[-1] == 0
        assert interactor.g.in_degree[0] == 1
        assert interactor.g.out_degree[-1] == 1
        assert interactor.g.out_degree[0] == 0
