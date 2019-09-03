import numpy as np
from pysam.libcbcf import VariantFile

from pigglet.constants import GL_DTYPE


class LikelihoodLoader:
    """Loads GLs from a VCF"""

    def __init__(self, vcf_file=None):
        self.vcf_file = vcf_file
        self.bcf_in = None
        self.gl_field_preference = ['GL']
        self.gl_field_idx = 1
        self.gl_field = 1
        self.gls = None

    def load(self):
        """Extract GLs from a VCF

        The VCF contains m sites, n samples, and g likelihoods per site and sample.
        Converts to log10 space.

        :return numpy array of shape (m, n, g)"""
        self.bcf_in = VariantFile(self.vcf_file)
        self._determine_field()
        self._extract_gls()
        return self.gls

    def _determine_field(self):
        formats = list(self.bcf_in.header.formats)
        for field in self.gl_field_preference:
            try:
                self.gl_field_idx = formats.index(field)
                return
            except ValueError:
                pass
        raise ValueError(
            f'Could not find a genotype likelihood format field in VCF ({self.vcf_file})'
        )

    def _extract_gls(self):
        site_gls = []
        infos = []
        for site_info, gls in site_gl_iter(self.bcf_in.fetch(), self.gl_field_idx):
            infos.append(site_info)
            site_gls.append(gls)
        self.gls = np.array(site_gls, dtype=GL_DTYPE)


def site_gl_iter(records, record_idx):
    for rec in records:
        values = []
        assert len(rec.alts) == 1
        site_info = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
        for sample, value in rec.samples.items():
            values.append(value.items()[record_idx][1])
        yield site_info, values
