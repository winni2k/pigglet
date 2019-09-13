import math

import h5py
import networkx as nx
import pytest
from click.testing import CliRunner

from pigglet import cli
from pigglet_testing.builders.vcf import VCFBuilder


@pytest.mark.parametrize('with_click', [False])
def test_single_mutation_one_sample_creates_trivial_graph(tmpdir, with_click):
    # given
    b = VCFBuilder(tmpdir)
    b.with_site_gls([0, 1, 0])
    vcf_file = b.build()
    prefix = tmpdir / 'out'
    runner = CliRunner()
    out_gml = str(prefix) + '.map_tree.gml'
    out_h5 = str(prefix) + '.h5'

    # when
    if with_click:
        result = runner.invoke(cli.cli, [str(vcf_file), str(prefix)])
        assert result.exit_code == 0
    else:
        result = cli.run(str(vcf_file), str(prefix))

    # then
    assert list(nx.read_gml(out_gml).edges) == [('-1', '0')]
    with h5py.File(out_h5, 'r') as fh:
        assert list(fh['map_tree/mutation_probabilities']) == pytest.approx(
            [math.e / (math.e + 1)])
        assert list(fh['map_tree/map_sample_attachments']) == [0]
        assert list(fh['map_tree/edge_list'][0]) == [-1, 0]
