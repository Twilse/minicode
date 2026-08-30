"""Validated startup configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from minicoder.domain.errors import ConfigurationError

_MIN_TOOL_OUTPUT_CHARS = 512


@dataclass(frozen=True, slots=True)
class AppConfig:
    """All runtime settings required before application adapters are created."""

    api_key: str
    base_url: str
    model: str
    workspace: Path
    max_steps: int  # Maximum agent iterations before forced termination.
    model_timeout_seconds: float  # Timeout for one model API request in seconds.
    command_timeout_seconds: float  # Per-command execution timeout in seconds.
    max_tool_output_chars: int  # Maximum characters returned by one tool call.
    context_budget_chars: int  # Approximate character budget for conversation history.
    memory_enabled: bool  # Whether successful turns create local project memory.
    planning_enabled: bool  # Whether each user turn starts with a no-tool plan.

    def __repr__(self) -> str:
        """Return a debug representation that never includes the secret key."""

        return (
            "AppConfig("
            "api_key='<hidden>', "
            f"base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"workspace={self.workspace!r}, "
            f"max_steps={self.max_steps!r}, "
            f"model_timeout_seconds={self.model_timeout_seconds!r}, "
            f"command_timeout_seconds={self.command_timeout_seconds!r}, "
            f"max_tool_output_chars={self.max_tool_output_chars!r}, "
            f"context_budget_chars={self.context_budget_chars!r}, "
            f"memory_enabled={self.memory_enabled!r}, "
            f"planning_enabled={self.planning_enabled!r}"
            ")"
        )

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> AppConfig:
        """Build and validate configuration without mutating process state."""

        source = os.environ if environ is None else environ

        api_key = _required_text(source, "MINICODER_API_KEY")
        base_url = _validated_base_url(
            _required_text(source, "MINICODER_BASE_URL")
        )
        model = _required_text(source, "MINICODER_MODEL")

        workspace_value = (
            workspace
            if workspace is not None
            else source.get("MINICODER_WORKSPACE", ".")
        )
        workspace_path = Path(workspace_value).expanduser().resolve()
        if not workspace_path.exists():
            raise ConfigurationError(f"workspace does not exist: {workspace_path}")
        if not workspace_path.is_dir():
            raise ConfigurationError(f"workspace is not a directory: {workspace_path}")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            workspace=workspace_path,
            max_steps=_positive_int(source, "MINICODER_MAX_STEPS", 20),
            model_timeout_seconds=_positive_float(
                source,
                "MINICODER_MODEL_TIMEOUT_SECONDS",
                60.0,
            ),
            command_timeout_seconds=_positive_float(
                source,
                "MINICODER_COMMAND_TIMEOUT_SECONDS",
                30.0,
            ),
            max_tool_output_chars=_int_at_least(
                source,
                "MINICODER_MAX_TOOL_OUTPUT_CHARS",
                12_000,
                minimum=_MIN_TOOL_OUTPUT_CHARS,
            ),
            context_budget_chars=_positive_int(
                source,
                "MINICODER_CONTEXT_BUDGET_CHARS",
                60_000,
            ),
            memory_enabled=_boolean(
                source,
                "MINICODER_MEMORY_ENABLED",
                False,
            ),
            planning_enabled=_boolean(
                source,
                "MINICODER_PLANNING_ENABLED",
                True,
            ),
        )


def _validated_base_url(raw_value: str) -> str:
    value = raw_value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(
            "MINICODER_BASE_URL must be an absolute http or https URL"
        )
    return value


def _required_text(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required")
    return value


def _positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _positive_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw_value = environ.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a finite number greater than zero")
    return value


def _int_at_least(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = _positive_int(environ, name, default)
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _boolean(
    environ: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    raw_value = environ.get(name, str(default)).strip().casefold()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} must be one of true, false, 1, 0, yes, no, on, or off"
    )
