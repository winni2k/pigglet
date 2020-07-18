import itertools as it
import math
from subprocess import run

import h5py
import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner
from pigglet_testing.builders.vcf import VCFBuilder

from pigglet import cli


@pytest.mark.parametrize(
    "gl_tag,mutation_tree", it.product(["GL", "PL"], [True, False])
)
def test_single_mutation_one_sample_creates_trivial_graph(
    tmpdir, gl_tag, mutation_tree
):
    # given
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_gml = str(prefix) + ".map_tree.gml"
    out_h5 = str(prefix) + ".h5"

    # when
    command = ["pigglet", "infer", str(vcf_file), str(prefix)]
    if mutation_tree:
        command.append("--mutation-tree")
    else:
        command.append("--no-mutation-tree")
    result = run(command)
    assert result.returncode == 0

    # then
    if mutation_tree:
        assert list(nx.read_gml(out_gml).edges) == [("-1", "0")]
        with h5py.File(out_h5, "r") as fh:
            assert list(
                fh["map_tree/mutation_probabilities"]
            ) == pytest.approx([math.e / (math.e + 1)])
            assert list(fh["map_tree/map_sample_attachments"]) == [0, 0, 0]
            assert list(fh["map_tree/edge_list"][0]) == [-1, 0]
    else:
        # all_node_attachment_likes = np.array([0, -1, -2, -2])
        # site_sum_like = logsumexp(all_node_attachment_likes)
        # sample_prob = logsumexp([0, -1]) - site_sum_like
        g = nx.read_gml(out_gml)
        assert nx.is_directed_acyclic_graph(g)
        assert len(g) == 5
        assert len([u for u, d in g.in_degree() if d == 0]) == 1
        assert len([u for u, d in g.in_degree() if d == 1]) == 4
        assert len([u for u, d in g.out_degree() if d == 0]) == 3
        assert len([u for u, d in g.out_degree() if d == 2]) == 2
        with h5py.File(out_h5, "r") as fh:
            mut_probs = fh["map_phylo_tree/mutation_probabilities"]
            assert mut_probs.shape == (1, 3)
            # assert mut_probs[0] == pytest.approx([sample_prob]*3)
            assert len([e for e in fh["map_phylo_tree/edge_list"]]) == 4


@pytest.mark.parametrize("invoke", [True, False])
def test_converts_mutation_tree_to_phylogenetic_tree(tmpdir, invoke):
    # given
    in_gml = tmpdir / "mutation_tree.gml"
    out_gml = tmpdir / "phylo_tree.gml"
    out_prefix = str(out_gml).rpartition(".gml")[0]
    in_h5 = tmpdir / "in.h5"
    g = nx.relabel_nodes(
        nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph), lambda n: n - 1
    )
    nx.write_gml(g, path=str(in_gml))
    with h5py.File(in_h5, "w") as fh:
        fh.create_dataset(
            "map_tree/map_sample_attachments", data=np.array([-1, 0, 5])
        )

    runner = CliRunner()

    # when
    args = [
        "convert",
        "--mutation-tree",
        str(in_gml),
        "--hdf5",
        str(in_h5),
        "--hdf5-sample-attachment-descriptor",
        "map_tree/map_sample_attachments",
        "--hdf5-mutation-attachment-descriptor",
        "map_phylogenetic_tree/mutation_attachments",
        out_prefix,
    ]
    if invoke:
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 0, result.output
    else:
        with pytest.raises(SystemExit):
            cli.cli(args)

    # then
    new_g = nx.read_gml(str(out_gml))
    assert sorted(set(new_g.edges())) == sorted(
        {("-1", "0"), ("-1", "3"), ("3", "1"), ("-1", "4"), ("4", "2")}
    )
    with h5py.File(out_prefix + ".h5", "r") as fh:
        assert list(
            fh["map_phylogenetic_tree/mutation_attachments/mutation_ids"]
        ) == [0, 1, 5,]
        assert list(
            fh["map_phylogenetic_tree/mutation_attachments/attachments"]
        ) == [3, 4, 4,]
