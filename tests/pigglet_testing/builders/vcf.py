from pigglet.gl_loader import LikelihoodLoader


class VCFBuilder:
    def __init__(self, tmpdir):
        self.vcf_file = tmpdir.join('input.vcf')
        self.gls = []
        self.n_samples = None

    def build(self):
        with open(self.vcf_file, 'w') as fh:
            fh.write("##fileformat=VCFv4.2\n")
            fh.write(
                '##FORMAT=<ID=PL,Number=G,Type=String,Description="Phred scaled likelihood">\n')
            fh.write(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
            for samp_num in range(self.n_samples):
                fh.write(f'\tsample_{samp_num}')
            fh.write('\n')
            for idx, site_gls in enumerate(self.gls):
                row = f'20\t{idx + 1}\t.\tG\tA\t29\tPASS\t.\tPL'
                for tripple in site_gls:
                    tripple = [str(v) for v in tripple]
                    row += '\t' + ','.join(tripple)
                row += '\n'
                fh.write(row)
        return self.vcf_file

    def with_site_gls(self, *gls):
        if self.n_samples is None:
            self.n_samples = len(gls)
        assert self.n_samples == len(gls)
        self.gls.append(gls)
        return self


class VCFLoaderBuilder(VCFBuilder):
    def build(self):
        vcf_file = super().build()
        loader = LikelihoodLoader()
        loader.load(vcf_file)
        return loader