from io import StringIO
from pathlib import Path

from minicoder.cli import main


def test_check_config_prints_safe_summary(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--check-config", "--workspace", str(tmp_path)],
        environ={"DEEPSEEK_API_KEY": "never-print-this"},
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "configuration is valid" in stdout.getvalue()
    assert "deepseek-v4-pro" in stdout.getvalue()
    assert "never-print-this" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_check_config_returns_two_for_user_configuration_error(
    tmp_path: Path,
) -> None:
    stderr = StringIO()

    exit_code = main(
        ["--check-config", "--workspace", str(tmp_path)],
        environ={},
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "DEEPSEEK_API_KEY is required" in stderr.getvalue()
