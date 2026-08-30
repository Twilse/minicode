import json
from io import StringIO
from pathlib import Path

import pytest

from minicoder.bootstrap import ApplicationFactory
from minicoder.cli import main
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.models import AssistantTurn, MessageRole
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
    assert "verification_commands=0" in stdout.getvalue()
    assert "planning_enabled=True" in stdout.getvalue()
    assert "memory_enabled=False" in stdout.getvalue()
    assert "never-print-this" not in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_check_config_validates_and_counts_project_verifiers(
    tmp_path: Path,
) -> None:
    (tmp_path / ".minicoder.toml").write_text(
        "[verification]\ncommands = [['zig', 'build', 'test']]\n",
        encoding="utf-8",
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--check-config", "--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "secret",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "model",
        },
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "verification_commands=1" in stdout.getvalue()
    assert stderr.getvalue() == ""


def test_check_config_rejects_invalid_project_verifiers(tmp_path: Path) -> None:
    (tmp_path / ".minicoder.toml").write_text(
        "[verification]\ncommands = 'pytest'\n",
        encoding="utf-8",
    )
    stderr = StringIO()

    exit_code = main(
        ["--check-config", "--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "secret",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "model",
        },
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "verification.commands must be an array" in stderr.getvalue()


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
    model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Inspect the project.\n2. Report the result."),
            AssistantTurn(content="Task completed safely."),
        ]
    )
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
        "[开始] 正在处理你的任务（本轮最多 20 个步骤）",
        "[计划] 正在制定本轮执行计划…",
        "[计划] 已生成，开始按照计划处理",
        "[分析] 正在规划下一步（步骤 2）",
        "[完成] 任务已完成（共 2 个步骤）",
        "Task completed safely.",
    ]
    assert stderr.getvalue() == ""
    trace_text = trace_path.read_text(encoding="utf-8")
    assert task_marker not in trace_text
    assert "secret-key-not-in-trace" not in trace_text
    assert [
        json.loads(line)["type"] for line in trace_text.splitlines()
    ] == [
        "task_started",
        "planning_started",
        "model_requested",
        "planning_completed",
        "model_requested",
        "task_completed",
    ]


def test_cli_without_a_task_runs_an_interactive_multi_turn_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Inspect project metadata.\n2. Answer."),
            AssistantTurn(
                content="The project uses Python.",
                reasoning_content="first turn state",
            ),
            AssistantTurn(content="1. Reuse the prior context.\n2. Answer."),
            AssistantTurn(content="It requires Python 3.11."),
        ]
    )
    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: model,
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        ["--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
        },
        stdin=StringIO(
            "Which language does this project use?\n"
            "What is the minimum version?\n"
            "/exit\n"
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert "MiniCoder 交互模式" in stdout.getvalue()
    assert "The project uses Python." in stdout.getvalue()
    assert "It requires Python 3.11." in stdout.getvalue()
    assert stdout.getvalue().count("[开始]") == 2
    assert stderr.getvalue() == ""
    second_request = model.requests[3].messages
    assert [message.role for message in second_request[-4:]] == [
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert second_request[-4].reasoning_content == "first turn state"
    assert "What is the minimum version?" in (
        second_request[-3].content or ""
    )


def test_interactive_cli_exits_cleanly_on_eof_without_calling_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter([])
    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: model,
    )

    exit_code = main(
        ["--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
        },
        stdin=StringIO(""),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert model.requests == []


def test_interactive_cli_returns_130_when_input_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingInput(StringIO):
        def readline(self, size: int = -1) -> str:
            raise KeyboardInterrupt

    model = FakeModelAdapter([])
    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: model,
    )
    stderr = StringIO()

    exit_code = main(
        ["--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
        },
        stdin=InterruptingInput(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 130
    assert "interrupted by user" in stderr.getvalue()
    assert model.requests == []


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
    assert "[失败] 模型服务请求失败" in stdout.getvalue()
    assert "agent failed: Model request failed" in stderr.getvalue()


def test_cli_renders_model_markdown_instead_of_printing_fence_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Explain the input format."),
            AssistantTurn(
                content=(
                    "输入格式：\n\n"
                    "```text\n"
                    "n m\n"
                    "u v w   # 共 m 行\n"
                    "source\n"
                    "```"
                )
            )
        ]
    )
    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: model,
    )
    stdout = StringIO()

    exit_code = main(
        ["--workspace", str(tmp_path), "Explain the input format"],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
        },
        stdout=stdout,
        stderr=StringIO(),
    )

    rendered = stdout.getvalue()
    assert exit_code == 0
    assert "```" not in rendered
    assert "n m" in rendered
    assert "u v w   # 共 m 行" in rendered
