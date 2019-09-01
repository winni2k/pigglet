import subprocess

import networkx as nx
import pytest

from pigglet_testing.builders.vcf import VCFBuilder


class CLIRunner:
    def __init__(self, vcf_file=None, prefix=None):
        self.vcf_file = vcf_file
        self.prefix = prefix

    def run(self):
        subprocess.run(['pigglet', '--vcf', self.vcf_file, '--prefix', self.prefix])


@pytest.mark.xfail(reason='Implementation incomplete')
def test_single_mutation_one_sample(tmpdir):
    # given
    b = VCFBuilder(tmpdir)
    b.with_site_gls([0, 1, 0])
    vcf_file = b.build()
    prefix = tmpdir / 'out'
    r = CLIRunner(vcf_file=vcf_file, prefix=prefix)
    out_gml = str(prefix) + '.gml'

    # when
    r.run()

    # then
    assert nx.read_gml(out_gml).edges == [(-1, 0)]
