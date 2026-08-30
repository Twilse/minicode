from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from minicoder.domain.models import ProcessResult, ToolCall, ToolResult
from minicoder.tools.command_safety import COMMAND_REJECTED, CommandSafetyPolicy
from minicoder.tools.output import StreamAwareOutputCompactor, ToolOutputArtifactStore
from minicoder.tools.process import (
    COMMAND_FAILED,
    COMMAND_NOT_FOUND,
    COMMAND_PERMISSION_DENIED,
    COMMAND_TIMED_OUT,
    ReadToolOutputTool,
    RunCommandTool,
)
from minicoder.tools.registry import INVALID_ARGUMENTS, ToolRegistry


class RecordingProcessAdapter:
    def __init__(
        self,
        result: ProcessResult | None = None,
        *,
        error: OSError | None = None,
    ) -> None:
        self.result = result or ProcessResult(
            stdout="ok\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            duration_seconds=0.1,
        )
        self.error = error
        self.requests: list[tuple[tuple[str, ...], Path, float]] = []

    def run(
        self,
        *,
        argv: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
    ) -> ProcessResult:
        self.requests.append((tuple(argv), cwd, timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.result


def _registry(
    workspace: Path,
    process: RecordingProcessAdapter,
    *,
    max_output_chars: int = 512,
) -> tuple[ToolRegistry, ToolOutputArtifactStore]:
    artifacts = ToolOutputArtifactStore(
        max_read_chars=max_output_chars - 256,
    )
    tools = ToolRegistry(
        (
            RunCommandTool(
                processes=process,
                policy=CommandSafetyPolicy(python_executable="/runtime/python"),
                artifacts=artifacts,
                compactor=StreamAwareOutputCompactor(),
                workspace=workspace,
                timeout_seconds=3.5,
                max_output_chars=max_output_chars,
            ),
            ReadToolOutputTool(
                artifacts,
                max_output_chars=max_output_chars,
            ),
        )
    )
    return tools, artifacts


def _execute(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
) -> ToolResult:
    return registry.execute(
        ToolCall(
            id=f"call-{name}",
            name=name,
            arguments_json=json.dumps(arguments),
        )
    )


def test_run_command_uses_normalized_argv_fixed_workspace_and_timeout(
    tmp_path: Path,
) -> None:
    process = RecordingProcessAdapter()
    registry, artifacts = _registry(tmp_path, process)

    result = _execute(
        registry,
        "run_command",
        {"argv": ["python3", "-m", "pytest", "-q"]},
    )

    assert result.ok is True
    assert process.requests == [
        (("/runtime/python", "-m", "pytest", "-q"), tmp_path, 3.5)
    ]
    assert "exit_code=0" in result.content
    assert result.content.endswith("ok\n")
    assert result.metadata["argv"] == (
        "/runtime/python",
        "-m",
        "pytest",
        "-q",
    )
    assert result.metadata["requested_argv"] == (
        "python3",
        "-m",
        "pytest",
        "-q",
    )
    assert result.metadata["purpose"] == "general"
    artifacts.close()


def test_run_command_preserves_explicit_verification_purpose(
    tmp_path: Path,
) -> None:
    process = RecordingProcessAdapter()
    registry, artifacts = _registry(tmp_path, process)

    result = _execute(
        registry,
        "run_command",
        {"argv": ["g++", "main.cpp", "-o", "main"], "purpose": "verification"},
    )

    assert result.ok is True
    assert result.metadata["purpose"] == "verification"
    assert result.metadata["requested_argv"] == (
        "g++",
        "main.cpp",
        "-o",
        "main",
    )
    artifacts.close()


def test_run_command_rejects_dangerous_command_before_process_execution(
    tmp_path: Path,
) -> None:
    process = RecordingProcessAdapter()
    registry, artifacts = _registry(tmp_path, process)

    result = _execute(
        registry,
        "run_command",
        {"argv": ["rm", "-rf", "."]},
    )

    assert result.error_code == COMMAND_REJECTED
    assert process.requests == []
    artifacts.close()


@pytest.mark.parametrize(
    ("process_result", "error_code"),
    [
        (
            ProcessResult(
                stdout="",
                stderr="tests failed\n",
                exit_code=2,
                timed_out=False,
                duration_seconds=1.25,
            ),
            COMMAND_FAILED,
        ),
        (
            ProcessResult(
                stdout="partial output\n",
                stderr="",
                exit_code=None,
                timed_out=True,
                duration_seconds=3.5,
            ),
            COMMAND_TIMED_OUT,
        ),
    ],
)
def test_run_command_maps_process_failure_semantics(
    tmp_path: Path,
    process_result: ProcessResult,
    error_code: str,
) -> None:
    registry, artifacts = _registry(
        tmp_path,
        RecordingProcessAdapter(process_result),
    )

    result = _execute(registry, "run_command", {"argv": ["pytest"]})

    assert result.ok is False
    assert result.error_code == error_code
    assert result.metadata["timed_out"] is process_result.timed_out
    assert result.metadata["exit_code"] == process_result.exit_code
    artifacts.close()


@pytest.mark.parametrize(
    ("error", "error_code"),
    [
        (FileNotFoundError(), COMMAND_NOT_FOUND),
        (PermissionError(), COMMAND_PERMISSION_DENIED),
    ],
)
def test_run_command_maps_expected_launch_errors(
    tmp_path: Path,
    error: OSError,
    error_code: str,
) -> None:
    registry, artifacts = _registry(
        tmp_path,
        RecordingProcessAdapter(error=error),
    )

    result = _execute(registry, "run_command", {"argv": ["missing-command"]})

    assert result.error_code == error_code
    artifacts.close()


def test_long_output_is_compacted_stored_and_read_back(tmp_path: Path) -> None:
    complete_output = (
        "test session starts\n"
        + "setup\n" * 200
        + "ERROR important failure\n"
        + "details\n" * 200
        + "failure summary\n"
    )
    process = RecordingProcessAdapter(
        ProcessResult(
            stdout=complete_output,
            stderr="",
            exit_code=1,
            timed_out=False,
            duration_seconds=2.0,
        )
    )
    registry, artifacts = _registry(tmp_path, process)

    result = _execute(registry, "run_command", {"argv": ["pytest", "-q"]})

    assert result.error_code == COMMAND_FAILED
    assert len(result.content) <= 512
    assert "ERROR important failure" in result.content
    assert "output_id=" in result.content
    assert result.metadata["truncated"] is True
    output_id = result.metadata["output_id"]

    first = _execute(
        registry,
        "read_tool_output",
        {"output_id": output_id, "offset": 0, "limit": 256},
    )
    stored = artifacts.read(output_id, offset=0, limit=256)

    assert first.ok is True
    assert "has_more=true" in first.content
    assert first.content.endswith(stored.content)
    assert len(first.content) <= 512
    assert artifacts.read(
        output_id,
        offset=0,
        limit=artifacts.max_read_chars,
    ).content == complete_output[:256]
    artifacts.close()


def test_long_dual_stream_output_keeps_both_channel_labels(tmp_path: Path) -> None:
    process = RecordingProcessAdapter(
        ProcessResult(
            stdout="stdout start\n" + "stdout noise\n" * 200,
            stderr="stderr start\n" + "stderr noise\n" * 200,
            exit_code=1,
            timed_out=False,
            duration_seconds=2.0,
        )
    )
    registry, artifacts = _registry(tmp_path, process)

    result = _execute(registry, "run_command", {"argv": ["pytest", "-q"]})

    assert result.error_code == COMMAND_FAILED
    assert len(result.content) <= 512
    assert result.content.count("[stdout]") == 1
    assert result.content.count("[stderr]") == 1
    assert result.metadata["truncated"] is True
    artifacts.close()


def test_read_tool_output_rejects_a_forged_id(tmp_path: Path) -> None:
    registry, artifacts = _registry(tmp_path, RecordingProcessAdapter())

    result = _execute(
        registry,
        "read_tool_output",
        {"output_id": "../../secret", "offset": 0, "limit": 10},
    )

    assert result.ok is False
    assert "unavailable in this session" in result.content
    artifacts.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": "pytest -q"},
        {"argv": []},
        {"argv": ["pytest"], "extra": True},
        {"argv": ["pytest"], "purpose": "testing"},
    ],
)
def test_run_command_schema_rejects_invalid_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    registry, artifacts = _registry(tmp_path, RecordingProcessAdapter())

    result = _execute(registry, "run_command", arguments)

    assert result.error_code == INVALID_ARGUMENTS
    artifacts.close()
