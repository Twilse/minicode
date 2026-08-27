from pathlib import Path

import pytest

from minicoder.config import AppConfig
from minicoder.domain.errors import ConfigurationError


def test_from_environment_uses_safe_defaults(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {"DEEPSEEK_API_KEY": "secret-value"},
        workspace=tmp_path,
    )

    assert config.api_key == "secret-value"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-pro"
    assert config.workspace == tmp_path.resolve()
    assert config.max_steps == 20
    assert config.command_timeout_seconds == 30.0


def test_from_environment_accepts_explicit_overrides(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "DEEPSEEK_API_KEY": "secret-value",
            "DEEPSEEK_BASE_URL": "http://localhost:9000/",
            "DEEPSEEK_MODEL": "test-model",
            "MINICODER_MAX_STEPS": "7",
            "MINICODER_COMMAND_TIMEOUT_SECONDS": "2.5",
            "MINICODER_MAX_TOOL_OUTPUT_CHARS": "800",
            "MINICODER_CONTEXT_BUDGET_CHARS": "9000",
        },
        workspace=tmp_path,
    )

    assert config.base_url == "http://localhost:9000"
    assert config.model == "test-model"
    assert config.max_steps == 7
    assert config.command_timeout_seconds == 2.5
    assert config.max_tool_output_chars == 800
    assert config.context_budget_chars == 9000


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "DEEPSEEK_API_KEY is required"),
        (
            {"DEEPSEEK_API_KEY": "key", "DEEPSEEK_BASE_URL": "not-a-url"},
            "absolute http or https URL",
        ),
        (
            {"DEEPSEEK_API_KEY": "key", "MINICODER_MAX_STEPS": "zero"},
            "must be an integer",
        ),
        (
            {"DEEPSEEK_API_KEY": "key", "MINICODER_MAX_STEPS": "0"},
            "must be greater than zero",
        ),
    ],
)
def test_from_environment_rejects_invalid_values(
    tmp_path: Path,
    environ: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        AppConfig.from_environment(environ, workspace=tmp_path)


def test_from_environment_rejects_missing_workspace(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ConfigurationError, match="workspace does not exist"):
        AppConfig.from_environment(
            {"DEEPSEEK_API_KEY": "key"},
            workspace=missing,
        )


def test_config_repr_does_not_reveal_api_key(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {"DEEPSEEK_API_KEY": "never-print-this"},
        workspace=tmp_path,
    )

    assert "never-print-this" not in repr(config)
    assert "<hidden>" in repr(config)
