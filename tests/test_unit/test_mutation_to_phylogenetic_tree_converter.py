import pytest

from pigglet_testing.builders.tree import (
    PhylogeneticTreeConverterBuilder,
    PhylogeneticTreeConverterTestDriver,
)
from pigglet_testing.expectations.tree import PhyloTreeExpectation


def test_raises_without_attachments():
    # given
    b = PhylogeneticTreeConverterTestDriver()
    b.with_mutation_at(0, 1)
    b.with_sample_attachments()

    # when/ then
    with pytest.raises(ValueError):
        b.build()


def test_raises_without_valid_attachments():
    # given
    b = PhylogeneticTreeConverterTestDriver()
    b.with_mutation_at(0, 1)
    b.with_sample_attachments(-1)

    # when/ then
    with pytest.raises(ValueError):
        b.build()


class TestPathTree:
    def test_leaves_single_private_mutation_unchanged(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_mutation_at(-1, 0)
        b.with_sample_attachments(0)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 0))
        expect.only_has_mutations(1)
        expect.has_mutation_at_node(1, -1)

    def test_converts_single_shared_mutation(self):
        b = PhylogeneticTreeConverterBuilder()
        b.with_mutation_at(-1, 0)
        converter = b.build()

        # when
        expect = PhyloTreeExpectation(
            converter.convert(sample_attachments=[0, 0])
        )

        # then
        expect.consists_of_edges((-1, 0), (-1, 1))
        expect.has_mutation_at_node(2, -1)
        expect.only_has_mutations(2)

    def test_converts_two_shared_mutations(self):
        b = PhylogeneticTreeConverterBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        converter = b.build()

        # when
        expect = PhyloTreeExpectation(
            converter.convert(sample_attachments=[1, 1])
        )

        # then
        expect.consists_of_edges((-1, 0), (-1, 1))
        expect.has_node_mutations(-1, [2, 3])
        expect.only_has_mutations(2, 3)

    def test_converts_one_private_and_one_shared_mutation_two_samples(self):
        b = PhylogeneticTreeConverterBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        converter = b.build()

        # when
        expect = PhyloTreeExpectation(
            converter.convert(sample_attachments=[0, 1])
        )

        # then
        expect.has_mutation_attachments({2: -1, 3: 3})
        expect.only_has_mutations(2, 3)
        expect.consists_of_edges((-1, 0), (-1, 3), (3, 1))

    def test_converts_one_irrelevant_one_shared_mutation_two_samples(self):
        b = PhylogeneticTreeConverterBuilder()
        b.with_mutation_at(-1, 0)
        b.with_mutation_at(0, 1)
        converter = b.build()

        # when
        expect = PhyloTreeExpectation(
            converter.convert(sample_attachments=[0, 0])
        )

        # then
        expect.has_mutation_attachments({2: -1})
        expect.only_has_mutations(2, 3)
        expect.consists_of_edges((-1, 0), (-1, 1))

    @pytest.mark.parametrize("n_muts", list(range(4)))
    def test_converts_two_samples_with_n_mutations(self, n_muts):
        # given
        b = PhylogeneticTreeConverterBuilder()
        b.with_path(n_muts)
        converter = b.build()

        # when
        expect = PhyloTreeExpectation(
            converter.convert(sample_attachments=[-1, -1])
        )

        # then
        expected_mutations = list(range(2, n_muts + 2))
        expect.has_mutation_attachments({})
        expect.only_has_mutations(*expected_mutations)
        expect.consists_of_edges((-1, 0), (-1, 1))


class TestBalancedTreeHeightOne:
    def test_two_irrelevant_mutations(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(1)
        b.with_sample_attachments(-1, -1)

        # when
        expect = b.build()

        # then
        expect.has_mutation_attachments({})
        expect.only_has_mutations(2, 3)
        expect.consists_of_edges((-1, 0), (-1, 1))

    def test_one_irrelevant_one_private_mutation(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(1)
        b.with_sample_attachments(-1, 0)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 0), (-1, 2), (2, 1))
        expect.has_mutation_attachments({2: 2})
        expect.only_has_mutations(2, 3)

    def test_two_private_mutations(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(1)
        b.with_sample_attachments(1, 0)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 3), (3, 0), (-1, 2), (2, 1))
        expect.has_mutation_attachments({2: 2, 3: 3})
        expect.only_has_mutations(2, 3)

    def test_one_private_one_shared_mutation(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(1)
        b.with_sample_attachments(-1, 0, 0)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 0), (-1, 3), (3, 1), (3, 2))
        expect.has_mutation_attachments({3: 3})
        expect.only_has_mutations(3, 4)


class TestBalancedTreeHeightTwo:
    def test_six_irrelevant_mutations(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(2)
        b.with_sample_attachments(-1)

        # when
        expect = b.build()

        # then
        expect.has_mutation_attachments({})
        expect.only_has_mutations(*list(range(1, 7)))
        expect.consists_of_edges((-1, 0))

    def test_five_irrelevant_one_private_mutation(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(2)
        b.with_sample_attachments(0)

        # when
        expect = b.build()

        # then
        expect.has_mutation_attachments({1: -1})
        expect.only_has_mutations(*list(range(1, 7)))
        expect.consists_of_edges((-1, 0))

    def test_four_irrelevant_two_private_mutations(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(2)
        b.with_sample_attachments(2)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 0))
        expect.has_mutation_attachments({1: -1, 3: -1})
        expect.only_has_mutations(*list(range(1, 7)))

    def test_two_irrelevant_four_private_mutations(self):
        b = PhylogeneticTreeConverterTestDriver()
        b.with_balanced_tree(2)
        b.with_sample_attachments(2, 5)

        # when
        expect = b.build()

        # then
        expect.consists_of_edges((-1, 2), (2, 0), (-1, 3), (3, 1))
        expect.has_mutation_attachments({2: 2, 4: 2, 3: 3, 7: 3})
