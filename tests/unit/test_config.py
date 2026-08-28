from pathlib import Path

import pytest

from minicoder.config import AppConfig
from minicoder.domain.errors import ConfigurationError

REQUIRED_MODEL_ENV = {
    "MINICODER_API_KEY": "secret-value",
    "MINICODER_BASE_URL": "https://api.deepseek.com",
    "MINICODER_MODEL": "deepseek-v4-pro",
}


def test_from_environment_loads_required_model_settings_and_safe_defaults(
    tmp_path: Path,
) -> None:
    config = AppConfig.from_environment(
        REQUIRED_MODEL_ENV,
        workspace=tmp_path,
    )

    assert config.api_key == "secret-value"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-pro"
    assert config.workspace == tmp_path.resolve()
    assert config.max_steps == 20
    assert config.model_timeout_seconds == 60.0
    assert config.command_timeout_seconds == 30.0


def test_from_environment_accepts_explicit_overrides(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "secret-value",
            "MINICODER_BASE_URL": "http://localhost:9000/",
            "MINICODER_MODEL": "test-model",
            "MINICODER_MAX_STEPS": "7",
            "MINICODER_MODEL_TIMEOUT_SECONDS": "15.5",
            "MINICODER_COMMAND_TIMEOUT_SECONDS": "2.5",
            "MINICODER_MAX_TOOL_OUTPUT_CHARS": "800",
            "MINICODER_CONTEXT_BUDGET_CHARS": "9000",
        },
        workspace=tmp_path,
    )

    assert config.base_url == "http://localhost:9000"
    assert config.model == "test-model"
    assert config.max_steps == 7
    assert config.model_timeout_seconds == 15.5
    assert config.command_timeout_seconds == 2.5
    assert config.max_tool_output_chars == 800
    assert config.context_budget_chars == 9000


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "MINICODER_API_KEY is required"),
        (
            {"MINICODER_API_KEY": "key"},
            "MINICODER_BASE_URL is required",
        ),
        (
            {
                "MINICODER_API_KEY": "key",
                "MINICODER_BASE_URL": "https://example.com/v1",
            },
            "MINICODER_MODEL is required",
        ),
        (
            {
                "MINICODER_API_KEY": "key",
                "MINICODER_BASE_URL": "not-a-url",
                "MINICODER_MODEL": "model",
            },
            "absolute http or https URL",
        ),
        (
            {**REQUIRED_MODEL_ENV, "MINICODER_MAX_STEPS": "zero"},
            "must be an integer",
        ),
        (
            {**REQUIRED_MODEL_ENV, "MINICODER_MAX_STEPS": "0"},
            "must be greater than zero",
        ),
        (
            {**REQUIRED_MODEL_ENV, "MINICODER_MAX_TOOL_OUTPUT_CHARS": "511"},
            "must be at least 512",
        ),
        (
            {**REQUIRED_MODEL_ENV, "MINICODER_MODEL_TIMEOUT_SECONDS": "nan"},
            "finite number greater than zero",
        ),
        (
            {**REQUIRED_MODEL_ENV, "MINICODER_MODEL_TIMEOUT_SECONDS": "inf"},
            "finite number greater than zero",
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
            REQUIRED_MODEL_ENV,
            workspace=missing,
        )


def test_model_settings_are_provider_neutral(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "another-provider-key",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "another-coding-model",
        },
        workspace=tmp_path,
    )

    assert config.base_url == "https://models.example.com/v1"
    assert config.model == "another-coding-model"


def test_config_repr_does_not_reveal_api_key(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {**REQUIRED_MODEL_ENV, "MINICODER_API_KEY": "never-print-this"},
        workspace=tmp_path,
    )

    assert "never-print-this" not in repr(config)
    assert "<hidden>" in repr(config)
