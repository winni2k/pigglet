import pytest

from pigglet_testing.builders.tree import TreeConverterBuilder


class PhyloTreeExpectation:
    def __init__(self, g):
        self.g = g

    def consists_of_edges(self, *edges):
        assert self.g.edges() == set(edges)
        return self

    def only_has_mutations(self, *mutations):
        assert self.g.graph['mutations'] == set(mutations)
        return self

    def has_mutation_at_node(self, mutation, node):
        assert mutation in self.g.node[node]['mutations']
        return self

    def has_node_mutations(self, node, mutations):
        assert self.g.node[node]['mutations'] == set(mutations)
        return self

    def has_mutation_attachments(self, attachments):
        assert self.g.graph['mutation_attachments'] == attachments
        return self


def test_leaves_single_private_mutation_unchanged():
    b = TreeConverterBuilder()
    b.with_mutation_at(-1, 0)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[0]))

    # then
    expect.consists_of_edges((-1, 0))
    expect.only_has_mutations(1)
    expect.has_mutation_at_node(1, -1)


def test_converts_single_shared_mutation():
    b = TreeConverterBuilder()
    b.with_mutation_at(-1, 0)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[0, 0]))

    # then
    expect.consists_of_edges((-1, 0), (-1, 1))
    expect.has_mutation_at_node(2, -1)
    expect.only_has_mutations(2)


def test_converts_two_shared_mutations():
    b = TreeConverterBuilder()
    b.with_mutation_at(-1, 0)
    b.with_mutation_at(0, 1)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[1, 1]))

    # then
    expect.consists_of_edges((-1, 0), (-1, 1))
    expect.has_node_mutations(-1, [2, 3])
    expect.only_has_mutations(2, 3)


def test_converts_one_private_and_one_shared_mutation_two_samples():
    b = TreeConverterBuilder()
    b.with_mutation_at(-1, 0)
    b.with_mutation_at(0, 1)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[0, 1]))

    # then
    expect.has_mutation_attachments({2: -1, 3: 3})
    expect.only_has_mutations(2, 3)
    expect.consists_of_edges((-1, 0), (-1, 3), (3, 1))


def test_converts_one_irrelevant_one_shared_mutation_two_samples():
    b = TreeConverterBuilder()
    b.with_mutation_at(-1, 0)
    b.with_mutation_at(0, 1)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[0, 0]))

    # then
    expect.has_mutation_attachments({2: -1})
    expect.only_has_mutations(2, 3)
    expect.consists_of_edges((-1, 0), (-1, 1))


@pytest.mark.parametrize('n_muts', list(range(4)))
def test_converts_two_samples_with_n_mutations(n_muts):
    # given
    b = TreeConverterBuilder()
    b.with_path(n_muts)
    converter = b.build()

    # when
    expect = PhyloTreeExpectation(converter.convert(sample_attachments=[-1, -1]))

    # then
    expected_mutations = list(range(2, n_muts + 2))
    expect.has_mutation_attachments({})
    expect.only_has_mutations(*expected_mutations)
    expect.consists_of_edges((-1, 0), (-1, 1))
