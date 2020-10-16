from pathlib import Path

import pytest
from click.testing import CliRunner

from pigglet.cli import infer

FIXTURE_DIR = Path(__file__).resolve().parent / "test_fixture_files"

pytestmark = pytest.mark.skipif(
    len(list(FIXTURE_DIR.glob("*"))) == 1,
    reason="Proprietary test fixtures are not available.",
)


@pytest.mark.datafiles(FIXTURE_DIR / "9_colonies.vcf")
def test_nine_colonies(tmp_path, datafiles):
    # given
    vcf = str(datafiles.listdir()[0])
    print(vcf)
    prefix = tmp_path / "out"

    # when
    command = ["pigglet", "infer", str(vcf), str(prefix)]
    command += (
        "--num-actors 1 --burnin 1000 --reporting-interval 100 --seed 42"
    ).split()
    runner = CliRunner()
    result = runner.invoke(infer, command[2:], catch_exceptions=False)
    print(result.output)
    assert result.exit_code == 0
