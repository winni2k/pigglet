import itertools as it
import math
import shutil
import subprocess
import re

import h5py
import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner

from pigglet.cli import (
    infer,
    calc_mutation_probabilities,
    calc_branch_lengths,
    store_gls,
    extract,
)
from pigglet_testing.builders.vcf import VCFBuilder

from pigglet import cli


@pytest.mark.parametrize(
    "gl_tag,mutation_tree,invoke",
    it.product(["GL", "PL"], [True, False], [True, False]),
)
def test_single_mutation_three_samples_creates_trivial_graph(
    tmpdir, gl_tag, mutation_tree, invoke
):
    # given
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_gml = str(prefix) + ".map_tree.gml"
    out_h5 = str(prefix) + ".h5"
    out_nw = str(prefix) + ".t"

    # when
    command = [
        "pigglet",
        "infer",
        str(vcf_file),
        str(prefix),
        "--reporting-interval",
        "5",
    ]
    if mutation_tree:
        command.append("--mutation-tree")
    else:
        command.append("--no-mutation-tree")
    runner = CliRunner()

    if invoke:
        result = runner.invoke(infer, command[2:], catch_exceptions=False)
        assert result.exit_code == 0
    else:
        subprocess.run(command)

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
        g = nx.read_gml(out_gml)
        assert nx.is_directed_acyclic_graph(g)
        assert len(g) == 5
        assert len([u for u, d in g.in_degree() if d == 0]) == 1
        assert len([u for u, d in g.in_degree() if d == 1]) == 4
        assert len([u for u, d in g.out_degree() if d == 0]) == 3
        assert len([u for u, d in g.out_degree() if d == 2]) == 2
        with h5py.File(out_h5, "r") as fh:
            mut_probs = fh["phylo_tree/mutation_probabilities"]
            assert mut_probs.shape == (1, 3)
            # assert mut_probs[0] == pytest.approx([sample_prob]*3)
            assert len([e for e in fh["map_phylo_tree/edge_list"]]) == 4
        with open(out_nw) as fh:
            lines = fh.readlines()
        assert len(lines) == 10
        assert set(set("".join(lines))) == set("012()\n, ")


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


def test_defers_mutation_prob_calc(tmpdir):
    # given
    gl_tag = "PL"
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_h5 = str(prefix) + ".h5"
    n_sampling = 10

    # when
    command = ["pigglet", "infer", str(vcf_file), str(prefix)]
    command += (
        f"--burnin 10 --sampling {n_sampling} --no-mutation-tree "
        "--defer-mutation-probability-calc"
    ).split()
    runner = CliRunner()
    result = runner.invoke(infer, command[2:], catch_exceptions=False)

    # then
    assert result.exit_code == 0
    with h5py.File(out_h5, "r") as fh:
        assert len(fh["phylo_tree/samples"]) == n_sampling
        for idx in range(n_sampling):
            assert len([e for e in fh[f"phylo_tree/samples/{idx}"]]) == 4
        assert "phylo_tree/mutation_probabilities" not in fh


def test_calc_phylo_mutation_probs(tmpdir):
    # given
    gl_tag = "PL"
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_h5 = str(prefix) + ".h5"
    out_h5_2 = str(prefix) + ".2.h5"
    n_sampling = 10

    # when
    command = ["pigglet", "infer", str(vcf_file), str(prefix)]
    command += (
        f"--seed 42 --burnin 10 --sampling {n_sampling} --no-mutation-tree "
        "--store-gls"
    ).split()
    runner = CliRunner()
    result = runner.invoke(infer, command[2:], catch_exceptions=False)
    shutil.copy(out_h5, out_h5_2)
    with h5py.File(out_h5_2, "a") as fh:
        del fh["phylo_tree/mutation_probabilities"]
    result2 = runner.invoke(
        calc_mutation_probabilities,
        [out_h5_2, "--jobs", "2"],
        catch_exceptions=False,
    )

    # then
    assert result.exit_code == 0
    assert result2.exit_code == 0
    with h5py.File(out_h5, "r") as fh:
        assert len(fh["phylo_tree/samples"]) == n_sampling
        for idx in range(n_sampling):
            assert len([e for e in fh[f"phylo_tree/samples/{idx}"]]) == 4
        mut_probs_agg = fh["phylo_tree/mutation_probabilities"][:]

    with h5py.File(out_h5_2, "r") as fh:
        mut_probs = fh["phylo_tree/mutation_probabilities"][:]
    assert mut_probs.shape == (2, 3)
    assert np.alltrue(np.exp(mut_probs) > 0.5)
    assert mut_probs == pytest.approx(mut_probs_agg)


def test_calc_phylo_branch_lengths(tmpdir):
    # given
    num_sites = 2
    gl_tag = "PL"
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    for _ in range(num_sites):
        b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_h5 = str(prefix) + ".h5"
    out_h5_2 = str(prefix) + ".2.h5"
    n_sampling = 10

    # when
    runner = CliRunner()
    command = ["pigglet", "infer", str(vcf_file), str(prefix)]
    command += (
        f"--seed 42 --burnin 10 --sampling {n_sampling} --no-mutation-tree "
        "--store-gls"
    ).split()
    result = runner.invoke(infer, command[2:], catch_exceptions=False)
    assert result.exit_code == 0

    shutil.copy(out_h5, out_h5_2)
    with h5py.File(out_h5_2, "a") as fh:
        del fh["phylo_tree/mutation_probabilities"]
    result2 = runner.invoke(
        calc_branch_lengths, [out_h5_2, "--jobs", "2"], catch_exceptions=False,
    )
    assert result2.exit_code == 0

    # then
    with h5py.File(out_h5_2, "r") as fh:
        assert len(fh["phylo_tree/samples"]) == n_sampling
        for idx in range(n_sampling):
            assert len([e for e in fh[f"phylo_tree/samples/{idx}"]]) == 4
            br_lens = list(fh[f"phylo_tree/samples_br_lens/{idx}"])
            assert len(br_lens) == 5
            assert sum(br_lens) == num_sites


def test_store_gls(tmpdir):
    # given
    gl_tag = "PL"
    b = VCFBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, 0, -1], [-1, 0, -1], [-1, 0, -1])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_h5 = str(prefix) + ".h5"

    # when
    command = f"pigglet store-gls {vcf_file} {out_h5}".split()
    runner = CliRunner()
    result = runner.invoke(store_gls, command[2:], catch_exceptions=False)

    # then
    assert result.exit_code == 0
    with h5py.File(out_h5, "r") as fh:
        assert "input/gls" in fh
        assert fh["input/gls"].shape == (1, 3, 3)


def test_extract_trees(tmpdir):
    # given
    b = VCFBuilder(tmpdir)
    b.with_site_gls([-100, 0, -100], [-100, 0, -100], [-100, 0, -100])
    vcf_file = b.build()
    prefix = tmpdir / "out"
    out_h5 = str(prefix) + ".h5"
    out_nw = str(prefix) + ".nws"
    out_nw_map = str(prefix) + ".map.nw"
    out_nw_no_label = str(prefix) + ".nl.nws"
    out_nw_map_no_label = str(prefix) + ".nl.map.nw"

    # when
    command = ["pigglet", "infer", str(vcf_file), str(prefix)]
    command += (
        "--burnin 1 --sampling 1 --no-mutation-tree "
        "--defer-mutation-probability-calc"
    ).split()
    runner = CliRunner()
    result = runner.invoke(infer, command[2:], catch_exceptions=False)
    assert result.exit_code == 0

    result = runner.invoke(
        calc_branch_lengths, [f"{out_h5}"], catch_exceptions=False,
    )
    assert result.exit_code == 0

    result = runner.invoke(
        extract,
        f"{out_h5} --phylo-map-newick {out_nw_map}"
        f" --phylo-newicks {out_nw}"
        f" --label-leaves"
        f" --branch-lengths".split(),
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    result = runner.invoke(
        extract,
        f"{out_h5} --phylo-map-newick {out_nw_map_no_label}"
        f" --phylo-newicks {out_nw_no_label}"
        f" --no-label-leaves"
        f" --branch-lengths".split(),
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # then
    for nw in [out_nw, out_nw_map]:
        with open(nw) as fh:
            tree = fh.readline().rstrip()
        print(nw, tree)
        for sample in b.sample_names:
            assert f"{sample}:0.0" in tree
        assert re.search(r"\):1.0*;$", tree)
    for nw in [out_nw_no_label, out_nw_map_no_label]:
        with open(nw) as fh:
            tree = fh.readline().rstrip()
        print(nw, tree)
        for sample in range(1, len(b.sample_names) + 1):
            assert f"{sample}:0.0" in tree
        assert re.search(r"\):1.0*;$", tree)


@pytest.mark.parametrize(
    "tree_str,one_base,label_leaves",
    [
        ("((1,2),3);", True, False),
        ("((0,1),2);", False, False),
        ("((1,2),3);", True, True),
        ("((0,1),2);", False, True),
    ],
)
def test_calc_tree_stats(tmp_path, tree_str, one_base, label_leaves):
    # given
    gl_tag = "PL"
    b = VCFBuilder(tmp_path)
    b.with_tag(gl_tag)
    mut = (-100, 0, -100)
    no_mut = (0, -100, -100)
    b.with_site_gls(mut, mut, no_mut)
    b.with_site_gls(mut, no_mut, no_mut)
    b.with_site_gls(no_mut, mut, no_mut)
    b.with_site_gls(no_mut, no_mut, mut)
    b.with_site_gls(no_mut, no_mut, mut)
    vcf_file = b.build()
    prefix = tmp_path / "out"
    out_h5 = str(prefix) + ".h5"
    test_nw = tmp_path / "test.nw"

    runner = CliRunner()
    command = [str(vcf_file), str(out_h5)]
    result = runner.invoke(cli.store_gls, command, catch_exceptions=False)
    assert result.exit_code == 0

    with open(str(test_nw), "w") as fh:
        fh.write(tree_str)

    # when
    expected_nw = "((1:1, 2:1):1, 3:2):0"
    command = [out_h5, str(test_nw)]
    if one_base:
        command += ["--one-based"]
    else:
        command += ["--no-one-based"]
    if label_leaves:
        command += ["--label-leaves"]
        expected_nw = re.sub(
            r"(\d):(\d)",
            lambda m: f"sample_{int(m.group(1))-1}:{m.group(2)}",
            expected_nw,
        )

    result = runner.invoke(
        cli.calc_tree_stats, command, catch_exceptions=False
    )

    # then
    print(result.output)
    assert result.exit_code == 0
    output = result.output.rstrip().split("\n")[-1]
    output = re.sub(r"\.0+", "", output)
    assert (
        output == f"tree_log_likelihood=0\tnewick_branch_lengths={expected_nw}"
    )
