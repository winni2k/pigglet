import logging

import numpy as np
from pysam import VariantFile

from pigglet.constants import GL_DTYPE


def convert_missing_entries(gls):
    for gl in gls:
        if gl == (None,) or "." in gl:
            yield (0, 0, 0)
        else:
            yield gl


class LikelihoodLoader:
    """Loads GLs from a VCF"""

    def __init__(self, vcf_file=None):
        self.vcf_file = vcf_file
        self.bcf_in = None
        self.gl_field_preference = ["GL", "PL"]
        self.gl_field_idx = None
        self.gl_field_name = None
        self.gls = None
        self.infos = None

    def load(self):
        """Extract GLs from a VCF

        The VCF contains m sites, n samples, and g likelihoods per site and
        sample. Converts to log10 space.

        :return numpy array of shape (m, n, g)"""
        self.bcf_in = VariantFile(self.vcf_file)
        self._determine_field()
        logging.info(
            'Extracting GLs from genotype field "%s"', self.gl_field_name
        )
        self._extract_gls()
        return self.gls

    def _determine_field(self):
        formats = list(self.bcf_in.header.formats)
        for field in self.gl_field_preference:
            if field in formats:
                self.gl_field_name = field
                self._determine_field_idx()
                return
        raise ValueError(
            "Could not find a genotype likelihood format "
            "field in VCF header: (%s)",
            self.vcf_file,
        )

    def _determine_field_idx(self):
        rec = next(self.bcf_in.fetch())
        assert (
            len(rec.samples) > 0
        ), "Expected at least one sample in input VCF"
        for idx, field in enumerate(rec.samples[0].keys()):
            if field == self.gl_field_name:
                self.gl_field_idx = idx
                return
        raise ValueError(
            f"Could not find the genotype likelihood field "
            f"{self.gl_field_name} in the first line of the "
            f"VCF {self.vcf_file}"
        )

    def _extract_gls(self):
        site_gls = []
        self.infos = []
        current_chrom = None
        for site_info, gls in site_gl_iter(
            self.bcf_in.fetch(), self.gl_field_idx
        ):
            if site_info[0] != current_chrom:
                current_chrom = site_info[0]
                logging.info("Loading chromosome %s", current_chrom)
            self.infos.append(site_info)
            gls = list(convert_missing_entries(gls))
            site_gls.append(gls)
        self.gls = np.array(site_gls, dtype=GL_DTYPE)
        if self.gl_field_name == "PL":
            self.gls = self.gls / -10
        if not np.all(self.gls <= 0):
            raise ValueError(
                "Not all input genotype likelihoods are in the"
                " interval [0, 1] "
                "(inclusive, log likelihoods have been exponentiated)"
            )


def site_gl_iter(records, record_idx):
    for rec in records:
        values = []
        assert len(rec.alts) == 1
        site_info = (rec.chrom, rec.pos, rec.id, rec.ref, rec.alts[0])
        for sample, value in rec.samples.items():
            values.append(value.items()[record_idx][1])
        yield site_info, values
