import shutil
from contextlib import closing

from pysam import VariantFile

from pigglet.gl_loader import LikelihoodLoader

GL_HEADER_LINES = {
    "GL": (
        "##FORMAT=<ID=GL,Number=G,Type=String,Description="
        '"Log10 scaled genotype likelihood">\n'
    ),
    "PL": (
        "##FORMAT=<ID=PL,Number=G,Type=Integer,Description="
        '"List of Phred-scaled genotype likelihoods">\n'
    ),
}


class VCFBuilder:
    def __init__(self, tmpdir):
        self.vcf_file = tmpdir.join("input.vcf")
        self.gls = []
        self.n_samples = None
        self.known_tags = {"GL", "PL"}
        self.likelihood_tag = "GL"
        self.gl_header_line = GL_HEADER_LINES["GL"]
        self.extra_header_lines = []
        self.use_bcf = False
        self.geno = "./."

    @property
    def sample_names(self):
        return [f"sample_{sample_idx}" for sample_idx in range(self.n_samples)]

    def build_header(self):
        header = [
            "##fileformat=VCFv4.2\n",
            "##contig=<ID=20,length=63025520>\n",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">\n',
        ]
        header += self.extra_header_lines
        header += self.gl_header_line
        return "".join(header)

    def build(self):
        header = self.build_header()

        with open(self.vcf_file, "w") as fh:
            fh.write(header)
            fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            fh.write("\t" + "\t".join(self.sample_names) + "\n")
            for idx, site_gls in enumerate(self.gls):
                row = (
                    f"20\t{idx + 1}\t.\tG\tA\t29\tPASS\t.\t"
                    f"GT:{self.likelihood_tag}"
                )
                for tripple in site_gls:
                    if self.likelihood_tag == "PL":
                        tripple = [round(-gl * 10) for gl in tripple]
                    tripple = [str(v) for v in tripple]
                    row += "\t" + f"{self.geno}:" + ",".join(tripple)
                row += "\n"
                fh.write(row)

        if self.use_bcf:
            tmp_vcf = self.vcf_file + ".tmp"
            shutil.move(self.vcf_file, tmp_vcf)
            with closing(VariantFile(tmp_vcf)) as bcf_in:
                with closing(
                    VariantFile(self.vcf_file, "wb", header=bcf_in.header)
                ) as bcf_out:
                    for rec in bcf_in.fetch():
                        bcf_out.write(rec)
        return self.vcf_file

    def with_extra_header_line(self, line):
        self.extra_header_lines += line
        return self

    def with_site_gls(self, *gls):
        if self.n_samples is None:
            self.n_samples = len(gls)
        assert self.n_samples == len(gls)
        self.gls.append(gls)
        return self

    def with_tag(self, tag):
        assert tag in self.known_tags
        self.likelihood_tag = tag
        self.gl_header_line = GL_HEADER_LINES[tag]
        return self

    def with_geno(self, geno):
        self.geno = geno
        return self

    def with_bcf(self):
        self.use_bcf = True
        return self


class VCFLoaderBuilder(VCFBuilder):
    def build(self):
        return LikelihoodLoader(super().build())


class VCFLoadedGLBuilder(VCFLoaderBuilder):
    def build(self):
        loader = super().build()
        return loader.load()
