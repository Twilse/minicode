from __future__ import annotations

from pathlib import Path

import pytest

from minicoder.adapters.verification_config import load_verification_commands
from minicoder.domain.errors import ConfigurationError


def test_absent_verification_config_loads_no_commands(tmp_path: Path) -> None:
    assert load_verification_commands(tmp_path) == ()


def test_verification_config_loads_exact_argv_alternatives(tmp_path: Path) -> None:
    (tmp_path / ".minicoder.toml").write_text(
        """
[project]
name = "example"

[verification]
commands = [
  ["zig", "build", "test"],
  ["./scripts/verify", "--all"],
]
""".strip(),
        encoding="utf-8",
    )

    commands = load_verification_commands(tmp_path)

    assert tuple(command.argv for command in commands) == (
        ("zig", "build", "test"),
        ("./scripts/verify", "--all"),
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("verification = []", "TOML table"),
        ("[verification]\ncommands = 'pytest'", "must be an array"),
        ("[verification]\nunknown = true", "unsupported fields"),
        ("[verification]\ncommands = [[]]", "non-empty argv"),
        ("[verification]\ncommands = [['pytest', '']]", "non-blank text"),
        (
            "[verification]\ncommands = [['pytest'], ['pytest']]",
            "duplicates an earlier command",
        ),
    ],
)
def test_verification_config_rejects_invalid_schema(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    (tmp_path / ".minicoder.toml").write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_verification_commands(tmp_path)


def test_verification_config_rejects_invalid_toml(tmp_path: Path) -> None:
    (tmp_path / ".minicoder.toml").write_text(
        "[verification\ncommands = []",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="not valid TOML"):
        load_verification_commands(tmp_path)
