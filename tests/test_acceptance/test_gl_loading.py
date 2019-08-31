from pigglet_testing.builders.vcf import VCFBuilder


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
