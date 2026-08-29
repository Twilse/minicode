import json
from io import StringIO
from pathlib import Path

import pytest

from minicoder.bootstrap import ApplicationFactory
from minicoder.cli import main
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.models import AssistantTurn
from tests.fakes import FakeModelAdapter


def test_check_config_prints_safe_summary(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--check-config", "--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "never-print-this",
            "MINICODER_BASE_URL": "https://api.deepseek.com",
            "MINICODER_MODEL": "deepseek-v4-pro",
        },
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
    assert "MINICODER_API_KEY is required" in stderr.getvalue()


def test_cli_runs_one_task_with_console_events_and_jsonl_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter([AssistantTurn(content="Task completed safely.")])
    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: model,
    )
    stdout = StringIO()
    stderr = StringIO()
    trace_path = tmp_path / "trace.jsonl"
    task_marker = "inspect this private task"

    exit_code = main(
        [
            "--workspace",
            str(tmp_path),
            "--trace",
            str(trace_path),
            task_marker,
        ],
        environ={
            "MINICODER_API_KEY": "secret-key-not-in-trace",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
        },
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        "[TASK] started (7 tools, max 20 model steps)",
        "[MODEL] step 1 requested (2 messages)",
        "[DONE] completed after 1 model steps",
        "Task completed safely.",
    ]
    assert stderr.getvalue() == ""
    trace_text = trace_path.read_text(encoding="utf-8")
    assert task_marker not in trace_text
    assert "secret-key-not-in-trace" not in trace_text
    assert [
        json.loads(line)["type"] for line in trace_text.splitlines()
    ] == ["task_started", "model_requested", "task_completed"]


def test_cli_rejects_trace_path_with_missing_parent(tmp_path: Path) -> None:
    stderr = StringIO()

    exit_code = main(
        [
            "--workspace",
            str(tmp_path),
            "--trace",
            str(tmp_path / "missing" / "trace.jsonl"),
            "inspect project",
        ],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
        },
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "trace error" in stderr.getvalue()


def test_cli_returns_one_for_a_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("network unavailable")

    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: FailingModel(),
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--workspace", str(tmp_path), "inspect project"],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
        },
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert "[FAILED] model_error" in stdout.getvalue()
    assert "agent failed: Model request failed" in stderr.getvalue()
