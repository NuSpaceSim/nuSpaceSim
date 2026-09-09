from click.testing import CliRunner

from nuspacesim import version
from nuspacesim.apps.cli import cli


def test_cli_version_option_reports_package_version():
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"nuspacesim, version {version}\n"
