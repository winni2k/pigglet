import pytest
from pigglet_testing.builders.vcf import VCFLoadedGLBuilder, VCFLoaderBuilder


def test_raises_if_gl_larger_than_one(tmpdir):
    b = VCFLoaderBuilder(tmpdir)
    b.with_tag("GL")
    b.with_site_gls([0.1, 0, 0])
    loader = b.build()

    with pytest.raises(ValueError):
        loader.load()


@pytest.mark.parametrize("gl_tag", ["GL", "PL"])
def test_loads_gls_of_single_site_and_one_sample_from_vcf(tmpdir, gl_tag):
    b = VCFLoadedGLBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, -2, -4], [-3, -7, -8])
    gls = b.build()

    assert gls.shape == (1, 2, 3)
    first_row = gls[0]
    assert list(first_row[0]) == [-1, -2, -4]
    assert list(first_row[1]) == [-3, -7, -8]


@pytest.mark.parametrize("gl_tag", ["GL", "PL"])
def test_loads_gls_of_two_sites_and_two_samples_from_vcf(tmpdir, gl_tag):
    b = VCFLoadedGLBuilder(tmpdir)
    b.with_tag(gl_tag)
    b.with_site_gls([-1, -2, -3], [-4, -5, -6])
    b.with_site_gls([-7, -8, -9], [-10, -11, -12])
    gls = b.build()

    assert gls.shape == (2, 2, 3)
    second_row = gls[1]
    assert second_row[0][0] == -7
    assert second_row[0][1] == -8
    assert second_row[0][2] == -9
    assert second_row[1][0] == -10
    assert second_row[1][1] == -11
    assert second_row[1][2] == -12


def test_raises_on_vcf_with_unacceptable_likelihood_encoding(tmpdir):
    # given
    b = VCFLoaderBuilder(tmpdir)
    b.with_site_gls([-1, -2, -4], [-3, -7, -8])
    b.gl_header_line = b.gl_header_line.replace("GL", "GP")
    loader = b.build()

    # when/then
    with pytest.raises(ValueError):
        loader.load()


def test_handles_extra_unused_format_header_line(tmpdir):
    # given
    b = VCFLoadedGLBuilder(tmpdir)
    b.with_site_gls([-1, -2, -4], [-3, -7, -8])
    b.with_extra_header_line(
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">\n'
    )

    # when
    gls = b.build()

    # when/then
    assert tuple(gls[0][0]) == (-1, -2, -4)
    assert tuple(gls[0][1]) == (-3, -7, -8)


def test_missing_gls_are_coded_as_all_zeros(tmpdir):
    # given
    b = VCFLoadedGLBuilder(tmpdir)
    b.with_site_gls([".", ".", "."])

    # when
    gls = b.build()

    # when/then
    assert tuple(gls[0][0]) == (0, 0, 0)


@pytest.mark.parametrize("use_bcf", (True, False))
def test_missing_gl_entry_is_coded_as_all_zeros(tmpdir, use_bcf):
    # given
    b = VCFLoadedGLBuilder(tmpdir)
    b.with_site_gls(["."])
    b.with_geno(".")
    if use_bcf:
        b.with_bcf()

    # when
    gls = b.build()

    # then
    assert tuple(gls[0][0]) == (0, 0, 0)
