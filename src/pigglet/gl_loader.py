import numpy as np
from pysam.libcbcf import VariantFile

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
