import pytest


class Vcf:

    def __init__(self, vcf_file):
        pass

    def gls(self):
        pass


def load_vcf_file(vcf_file):
    pass


@pytest.mark.xfail
def test_loads_gls_of_single_site_and_two_samples_from_vcf(tmpdir):
    vcf_file = tmpdir.join('input.vcf')
    with open(vcf_file, 'w') as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##FORMAT=<ID=PL,Number=G,Type=String,Description="Phred scaled likelihood">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample_1\tsample_2\n")
        fh.write('20\t14370\t.\tG\tA\t29\tPASS\tPL\t1,2,4\t3,7,8\n')

    vcf = load_vcf_file(vcf_file)
    assert vcf.gls().shape == (2, 3)
    assert vcf.gls()[0][0] == 1
    assert vcf.gls()[0][1] == 2
    assert vcf.gls()[0][2] == 4
    assert vcf.gls()[1][0] == 3
    assert vcf.gls()[1][1] == 7
    assert vcf.gls()[1][2] == 8
