from click.testing import CliRunner

from nuspacesim import version
from nuspacesim.apps.cli import cli


def test_cli_version_option_reports_package_version():
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert result.output == f"nuspacesim, version {version}\n"


def test_cli_registers_all_commands():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    for cmd in ("run", "stream", "create-config", "show-plot"):
        assert cmd in result.output


def test_stream_help_lists_override_options():
    runner = CliRunner()

    result = runner.invoke(cli, ["stream", "--help"])

    assert result.exit_code == 0
    for opt in ("--rel-unc", "--reservoir-size", "--batch-size"):
        assert opt in result.output
