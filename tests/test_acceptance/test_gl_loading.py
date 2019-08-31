import numpy as np
from pysam import VariantFile

from pigglet.constants import PL_DTYPE


class LikelihoodLoader:

    def __init__(self):
        self.gls = None

    def load(self, vcf_file):
        site_infos, site_pls = load_vcf_file(vcf_file)
        self.gls = np.array(site_pls, dtype=PL_DTYPE)


def load_vcf_file(vcf_file, gl_type='PL'):
    bcf_in = VariantFile(vcf_file)
    site_pls = []
    site_infos = []
    assert gl_type == 'PL'
    for site_info, pls in site_pl_iter(bcf_in.fetch()):
        site_infos.append(site_info)
        site_pls.append(pls)

    return site_infos, site_pls


def site_pl_iter(records):
    for rec in records:
        values = []
        assert len(rec.alts) == 1
        site_info = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        for sample, value in rec.samples.items():
            values.append(value.items()[0][1])
        yield site_info, values


def test_loads_gls_of_single_site_and_two_samples_from_vcf(tmpdir):
    vcf_file = tmpdir.join('input.vcf')
    with open(vcf_file, 'w') as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(
            '##FORMAT=<ID=PL,Number=G,Type=String,Description="Phred scaled likelihood">\n')
        fh.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_1\tsample_2\n")
        fh.write('20\t14370\t.\tG\tA\t29\tPASS\t.\tPL\t1,2,4\t3,7,8\n')

    loader = LikelihoodLoader()
    loader.load(vcf_file)
    assert loader.gls.shape == (1, 2, 3)
    first_row = loader.gls[0]
    assert first_row[0][0] == 1
    assert first_row[0][1] == 2
    assert first_row[0][2] == 4
    assert first_row[1][0] == 3
    assert first_row[1][1] == 7
    assert first_row[1][2] == 8
