from pigglet.gl_loader import LikelihoodLoader


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
