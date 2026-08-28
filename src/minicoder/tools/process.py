"""Model-facing command tools built on provider-neutral process and output services."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from minicoder.application.ports import ProcessPort
from minicoder.domain.errors import ConfigurationError
from minicoder.domain.models import ProcessResult, ToolDefinition, ToolResult
from minicoder.tools.base import ToolCommand
from minicoder.tools.command_safety import CommandPolicyError, CommandSafetyPolicy
from minicoder.tools.output import (
    ArtifactStoreError,
    OutputCompactionStrategy,
    ToolOutputArtifactStore,
)

COMMAND_FAILED = "COMMAND_FAILED"
COMMAND_LAUNCH_FAILED = "COMMAND_LAUNCH_FAILED"
COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
COMMAND_PERMISSION_DENIED = "COMMAND_PERMISSION_DENIED"
COMMAND_TIMED_OUT = "COMMAND_TIMED_OUT"
OUTPUT_STORAGE_FAILED = "OUTPUT_STORAGE_FAILED"

_MAX_COMMAND_ARGUMENTS = 100
_MAX_COMMAND_ARGUMENT_CHARS = 20_000
_OUTPUT_HEADER_RESERVE = 256


class RunCommandTool:
    """Validate and execute one bounded, non-interactive command vector."""

    def __init__(
        self,
        *,
        processes: ProcessPort,
        policy: CommandSafetyPolicy,
        artifacts: ToolOutputArtifactStore,
        compactor: OutputCompactionStrategy,
        workspace: Path,
        timeout_seconds: float,
        max_output_chars: int,
    ) -> None:
        if timeout_seconds <= 0:
            raise ConfigurationError("run_command timeout_seconds must be positive")
        if max_output_chars < _OUTPUT_HEADER_RESERVE * 2:
            raise ConfigurationError(
                "run_command max_output_chars must be at least "
                f"{_OUTPUT_HEADER_RESERVE * 2}"
            )
        self._processes = processes
        self._policy = policy
        self._artifacts = artifacts
        self._compactor = compactor
        self._workspace = workspace
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars
        self._definition = ToolDefinition(
            name="run_command",
            description=(
                "Run a non-interactive command in the workspace using an argv list. "
                "Shell syntax such as pipes, redirects, and && is not supported."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_COMMAND_ARGUMENTS,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_COMMAND_ARGUMENT_CHARS,
                        },
                    }
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        raw_argv = command.arguments["argv"]
        try:
            argv = self._policy.validate_and_normalize(raw_argv)
            process_result = self._processes.run(
                argv=argv,
                cwd=self._workspace,
                timeout_seconds=self._timeout_seconds,
            )
        except CommandPolicyError as exc:
            return _failure(command, exc.error_code, str(exc))
        except FileNotFoundError:
            return _failure(
                command,
                COMMAND_NOT_FOUND,
                f"Command {raw_argv[0]!r} was not found.",
            )
        except PermissionError:
            return _failure(
                command,
                COMMAND_PERMISSION_DENIED,
                f"Command {raw_argv[0]!r} could not be started due to permissions.",
            )
        except OSError as exc:
            return _failure(
                command,
                COMMAND_LAUNCH_FAILED,
                f"Command {raw_argv[0]!r} could not be started.",
                metadata={"exception_type": type(exc).__name__},
            )

        try:
            content, output_metadata = _format_process_output(
                process_result,
                artifacts=self._artifacts,
                compactor=self._compactor,
                max_chars=self._max_output_chars,
            )
        except OSError as exc:
            return _failure(
                command,
                OUTPUT_STORAGE_FAILED,
                "Command completed, but its full output could not be stored safely.",
                metadata={"exception_type": type(exc).__name__},
            )

        metadata: dict[str, Any] = {
            "argv": argv,
            "exit_code": process_result.exit_code,
            "timed_out": process_result.timed_out,
            "duration_seconds": process_result.duration_seconds,
            **output_metadata,
        }
        if process_result.timed_out:
            return _failure(
                command,
                COMMAND_TIMED_OUT,
                content,
                metadata=metadata,
            )
        if process_result.exit_code != 0:
            return _failure(
                command,
                COMMAND_FAILED,
                content,
                metadata=metadata,
            )
        return _success(command, content, metadata=metadata)


class ReadToolOutputTool:
    """Read one bounded character range from current-session full command output."""

    def __init__(
        self,
        artifacts: ToolOutputArtifactStore,
        *,
        max_output_chars: int,
    ) -> None:
        if max_output_chars < _OUTPUT_HEADER_RESERVE * 2:
            raise ConfigurationError(
                "read_tool_output max_output_chars must be at least 512"
            )
        if artifacts.max_read_chars > max_output_chars - _OUTPUT_HEADER_RESERVE:
            raise ConfigurationError(
                "artifact read limit leaves insufficient space for the result header"
            )
        self._artifacts = artifacts
        self._definition = ToolDefinition(
            name="read_tool_output",
            description=(
                "Read another character range from a truncated command output_id "
                "created during this session."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "output_id": {"type": "string", "minLength": 1},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": artifacts.max_read_chars,
                        "default": artifacts.max_read_chars,
                    },
                },
                "required": ["output_id"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        output_id = command.arguments["output_id"]
        offset = command.arguments.get("offset", 0)
        limit = command.arguments.get("limit", self._artifacts.max_read_chars)
        try:
            chunk = self._artifacts.read(output_id, offset=offset, limit=limit)
        except ArtifactStoreError as exc:
            return _failure(command, exc.error_code, str(exc))
        except OSError as exc:
            return _failure(
                command,
                OUTPUT_STORAGE_FAILED,
                "The stored command output could not be read.",
                metadata={"exception_type": type(exc).__name__},
            )

        next_offset = chunk.end if chunk.has_more else None
        header = (
            f"[output_id={chunk.output_id!r} chars={chunk.offset}:{chunk.end} "
            f"total_chars={chunk.total_chars} "
            f"has_more={'true' if chunk.has_more else 'false'}"
        )
        if next_offset is not None:
            header += f" next_offset={next_offset}"
        content = f"{header}]\n{chunk.content}"
        return _success(
            command,
            content,
            metadata={
                "output_id": chunk.output_id,
                "offset": chunk.offset,
                "end": chunk.end,
                "total_chars": chunk.total_chars,
                "has_more": chunk.has_more,
                "next_offset": next_offset,
            },
        )


def _format_process_output(
    result: ProcessResult,
    *,
    artifacts: ToolOutputArtifactStore,
    compactor: OutputCompactionStrategy,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    complete_output = result.combined_output()
    visible_output = complete_output or "(no output)"
    short_header = _status_header(
        result,
        truncated=False,
        output_id=None,
        original_chars=len(complete_output),
        preview_chars=len(visible_output),
    )
    short_content = f"{short_header}\n{visible_output}"
    if len(short_content) <= max_chars:
        ranges = () if not complete_output else ((0, len(complete_output)),)
        return short_content, {
            "output_id": None,
            "original_chars": len(complete_output),
            "returned_chars": len(visible_output),
            "truncated": False,
            "included_ranges": ranges,
        }

    output_id = artifacts.save(complete_output)
    available = max_chars - _OUTPUT_HEADER_RESERVE
    seen_budgets: set[int] = set()
    while available not in seen_budgets:
        seen_budgets.add(available)
        compacted = compactor.compact(complete_output, max_chars=available)
        header = _status_header(
            result,
            truncated=True,
            output_id=output_id,
            original_chars=len(complete_output),
            preview_chars=compacted.returned_chars,
        )
        next_available = max(1, max_chars - len(header) - 1)
        if next_available == available:
            break
        available = next_available
    else:
        available = min(available, next_available)
        compacted = compactor.compact(complete_output, max_chars=available)
        header = _status_header(
            result,
            truncated=True,
            output_id=output_id,
            original_chars=len(complete_output),
            preview_chars=compacted.returned_chars,
        )

    content = f"{header}\n{compacted.content}"
    included_ranges = tuple(
        (current.start, current.end) for current in compacted.included_ranges
    )
    return content, {
        "output_id": output_id,
        "original_chars": compacted.original_chars,
        "returned_chars": compacted.returned_chars,
        "truncated": compacted.truncated,
        "included_ranges": included_ranges,
    }


def _status_header(
    result: ProcessResult,
    *,
    truncated: bool,
    output_id: str | None,
    original_chars: int,
    preview_chars: int,
) -> str:
    exit_code = "none" if result.exit_code is None else str(result.exit_code)
    header = (
        f"[command exit_code={exit_code} "
        f"timed_out={'true' if result.timed_out else 'false'} "
        f"duration_seconds={result.duration_seconds:.3f} "
        f"truncated={'true' if truncated else 'false'} "
        f"original_chars={original_chars} preview_chars={preview_chars}"
    )
    if output_id is not None:
        header += f" output_id={output_id}"
    return f"{header}]"


def _success(
    command: ToolCommand,
    content: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=command.call_id,
        tool_name=command.tool_name,
        ok=True,
        content=content,
        metadata={} if metadata is None else metadata,
    )


def _failure(
    command: ToolCommand,
    error_code: str,
    content: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=command.call_id,
        tool_name=command.tool_name,
        ok=False,
        content=content,
        error_code=error_code,
        metadata={} if metadata is None else metadata,
    )
