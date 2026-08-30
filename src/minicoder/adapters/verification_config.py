"""Read trusted project verification commands once during application startup."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from minicoder.application.verification import ConfiguredVerificationCommand
from minicoder.domain.errors import ConfigurationError

VERIFICATION_CONFIG_NAME = ".minicoder.toml"

_MAX_CONFIG_BYTES = 100_000
_MAX_COMMANDS = 20
_MAX_ARGUMENTS = 100
_MAX_ARGUMENT_CHARS = 20_000


def load_verification_commands(
    workspace: Path,
) -> tuple[ConfiguredVerificationCommand, ...]:
    """Load exact alternative verifier commands from the workspace root."""

    config_path = workspace / VERIFICATION_CONFIG_NAME
    if not config_path.exists() and not config_path.is_symlink():
        return ()
    try:
        resolved_workspace = workspace.resolve(strict=True)
        resolved_config = config_path.resolve(strict=True)
        resolved_config.relative_to(resolved_workspace)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"{VERIFICATION_CONFIG_NAME} must resolve inside the workspace"
        ) from exc
    if not resolved_config.is_file():
        raise ConfigurationError(f"{VERIFICATION_CONFIG_NAME} must be a file")
    try:
        if resolved_config.stat().st_size > _MAX_CONFIG_BYTES:
            raise ConfigurationError(
                f"{VERIFICATION_CONFIG_NAME} must not exceed {_MAX_CONFIG_BYTES} bytes"
            )
        with resolved_config.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(
            f"{VERIFICATION_CONFIG_NAME} is not valid TOML: {exc}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(
            f"{VERIFICATION_CONFIG_NAME} could not be read"
        ) from exc

    return _parse_verification_commands(document)


def _parse_verification_commands(
    document: dict[str, Any],
) -> tuple[ConfiguredVerificationCommand, ...]:
    section = document.get("verification")
    if section is None:
        return ()
    if not isinstance(section, dict):
        raise ConfigurationError("[verification] must be a TOML table")
    unknown_fields = sorted(set(section) - {"commands"})
    if unknown_fields:
        raise ConfigurationError(
            "[verification] contains unsupported fields: "
            + ", ".join(unknown_fields)
        )

    raw_commands = section.get("commands", [])
    if not isinstance(raw_commands, list):
        raise ConfigurationError("verification.commands must be an array")
    if len(raw_commands) > _MAX_COMMANDS:
        raise ConfigurationError(
            f"verification.commands must contain at most {_MAX_COMMANDS} commands"
        )

    commands: list[ConfiguredVerificationCommand] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw_command in enumerate(raw_commands, start=1):
        argv = _parse_argv(raw_command, index=index)
        if argv in seen:
            raise ConfigurationError(
                f"verification.commands[{index}] duplicates an earlier command"
            )
        seen.add(argv)
        commands.append(ConfiguredVerificationCommand(argv=argv))
    return tuple(commands)


def _parse_argv(raw_command: Any, *, index: int) -> tuple[str, ...]:
    if not isinstance(raw_command, list) or not raw_command:
        raise ConfigurationError(
            f"verification.commands[{index}] must be a non-empty argv array"
        )
    if len(raw_command) > _MAX_ARGUMENTS:
        raise ConfigurationError(
            f"verification.commands[{index}] must contain at most "
            f"{_MAX_ARGUMENTS} arguments"
        )
    if any(
        not isinstance(argument, str)
        or not argument.strip()
        or len(argument) > _MAX_ARGUMENT_CHARS
        for argument in raw_command
    ):
        raise ConfigurationError(
            f"verification.commands[{index}] arguments must be non-blank text "
            f"of at most {_MAX_ARGUMENT_CHARS} characters"
        )
    return tuple(raw_command)
