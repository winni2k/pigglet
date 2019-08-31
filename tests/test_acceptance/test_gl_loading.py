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


class VCFBuilder:
    def __init__(self, tmpdir):
        self.vcf_file = tmpdir.join('input.vcf')
        self.gls = []

    def build(self):
        with open(self.vcf_file, 'w') as fh:
            fh.write("##fileformat=VCFv4.2\n")
            fh.write(
                '##FORMAT=<ID=PL,Number=G,Type=String,Description="Phred scaled likelihood">\n')
            fh.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_1\tsample_2\n")
            for idx, site_gls in enumerate(self.gls):
                row = f'20\t{idx + 1}\t.\tG\tA\t29\tPASS\t.\tPL'
                for tripple in site_gls:
                    tripple = [str(v) for v in tripple]
                    row += '\t' + ','.join(tripple)
                row += '\n'
                fh.write(row)
        loader = LikelihoodLoader()
        loader.load(self.vcf_file)
        return loader

    def with_site_gls(self, gls):
        self.gls.append(gls)
        return self


def test_loads_gls_of_single_site_and_two_samples_from_vcf(tmpdir):
    b = VCFBuilder(tmpdir)
    b.with_site_gls([[1, 2, 4], [3, 7, 8]])
    loader = b.build()

    assert loader.gls.shape == (1, 2, 3)
    first_row = loader.gls[0]
    assert first_row[0][0] == 1
    assert first_row[0][1] == 2
    assert first_row[0][2] == 4
    assert first_row[1][0] == 3
    assert first_row[1][1] == 7
    assert first_row[1][2] == 8


def test_loads_gls_of_two_sites_and_two_samples_from_vcf(tmpdir):
    b = VCFBuilder(tmpdir)
    b.with_site_gls([[1, 2, 3], [4, 5, 6]])
    b.with_site_gls([[7, 8, 9], [10, 11, 12]])
    loader = b.build()

    assert loader.gls.shape == (2, 2, 3)
    first_row = loader.gls[1]
    assert first_row[0][0] == 7
    assert first_row[0][1] == 8
    assert first_row[0][2] == 9
    assert first_row[1][0] == 10
    assert first_row[1][1] == 11
    assert first_row[1][2] == 12
