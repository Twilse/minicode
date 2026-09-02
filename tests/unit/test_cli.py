import json
from io import StringIO
from pathlib import Path

import pytest

from minicoder.bootstrap import ApplicationFactory
from minicoder.cli import main
from minicoder.adapters.jsonl_session import JsonlSessionArchive
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.models import AssistantTurn, Message, MessageRole, ToolCall
from minicoder.domain.state import AgentPhase, AgentRunResult, AgentStopReason
from tests.fakes import FakeModelAdapter


def _finish_plan_step(step: int, *, suffix: str = "") -> AssistantTurn:
    return AssistantTurn(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"call-finish-{step}{suffix}",
                name="finish_plan_step",
                arguments_json=json.dumps(
                    {"step": step, "summary": f"Step {step} completed."}
                ),
            ),
        ),
    )


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
    assert "context_budget_chars=180000" in stdout.getvalue()
    assert "context_response_reserve_chars=8000" in stdout.getvalue()
    assert "planning_enabled=True" in stdout.getvalue()
    assert "memory_enabled=True" in stdout.getvalue()
    assert "session_archive_enabled=True" in stdout.getvalue()
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
    assert "verification.commands 必须是数组" in stderr.getvalue()


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
    assert "缺少必需的环境变量 MINICODER_API_KEY" in stderr.getvalue()


def test_cli_runs_one_task_with_console_events_and_jsonl_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content="Plan:\n1. Inspect the project.\n2. Report the result."
            ),
            _finish_plan_step(1),
            _finish_plan_step(2),
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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue().splitlines() == [
        "[开始] 正在处理你的任务（本轮最多 40 次模型调用）",
        "[计划] 正在制定本轮执行计划…",
        "[计划] 已生成，共 2 项：",
        "  1. Inspect the project.",
        "  2. Report the result.",
        "[进行中] 1/2 Inspect the project.",
        "[分析] 正在请求模型（第 2 次）",
        "[已完成] 1/2 Inspect the project.",
        "[进行中] 2/2 Report the result.",
        "[分析] 正在请求模型（第 3 次）",
        "[分析] 正在请求模型（第 4 次）",
        "[已完成] 2/2 Report the result.",
        "[计划] 全部 2 项已完成",
        "[完成] 任务已完成（共 4 次模型调用）",
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
        "plan_step_started",
        "model_requested",
        "plan_step_completed",
        "plan_step_started",
        "model_requested",
        "model_requested",
        "plan_step_completed",
        "plan_completed",
        "task_completed",
    ]


def test_cli_without_a_task_runs_an_interactive_multi_turn_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content="Plan:\n1. Inspect project metadata.\n2. Answer."
            ),
            _finish_plan_step(1, suffix="-first"),
            _finish_plan_step(2, suffix="-first"),
            AssistantTurn(
                content="The project uses Python.",
                reasoning_content="first turn state",
            ),
            AssistantTurn(
                content="Plan:\n1. Reuse the prior context.\n2. Answer."
            ),
            _finish_plan_step(1, suffix="-second"),
            _finish_plan_step(2, suffix="-second"),
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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
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
    second_request = model.requests[4].messages
    assert any(
        message.reasoning_content == "first turn state"
        for message in second_request
        if message.role is MessageRole.ASSISTANT
    )
    assert "What is the minimum version?" in (
        second_request[-1].content or ""
    )


def test_cli_replays_complete_project_dialogue_before_the_new_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "session-store"
    previous = JsonlSessionArchive(
        workspace=tmp_path,
        storage_root=archive_root,
    )
    messages = (
        Message(role=MessageRole.SYSTEM, content="hidden system prompt"),
        Message(role=MessageRole.USER, content="augmented internal user message"),
        Message(role=MessageRole.ASSISTANT, content="hidden planning response"),
    )
    previous.record_turn_started(
        task="请介绍这个项目",
        history=(),
        turn_index=1,
    )
    previous.record_turn_result(
        task="请介绍这个项目",
        result=AgentRunResult(
            phase=AgentPhase.COMPLETE,
            stop_reason=AgentStopReason.FINAL_RESPONSE,
            model_steps=2,
            messages=messages,
            final_response="这是一个 **Python** 项目。",
        ),
        turn_index=1,
    )
    previous.close()

    monkeypatch.setattr(
        ApplicationFactory,
        "create_model_adapter",
        lambda config: FakeModelAdapter([]),
    )
    monkeypatch.setattr(
        ApplicationFactory,
        "create_session_archive",
        lambda config: JsonlSessionArchive(
            workspace=tmp_path,
            storage_root=archive_root,
        ),
    )
    stdout = StringIO()

    exit_code = main(
        ["--workspace", str(tmp_path)],
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
            "MINICODER_MEMORY_ENABLED": "false",
        },
        stdin=StringIO("/exit\n"),
        stdout=stdout,
        stderr=StringIO(),
    )

    rendered = stdout.getvalue()
    assert exit_code == 0
    assert "你：\n请介绍这个项目" in rendered
    assert "MiniCoder：" in rendered
    assert "请介绍这个项目" in rendered
    assert "这是一个 Python 项目。" in rendered
    assert "hidden system prompt" not in rendered
    assert "hidden planning response" not in rendered
    assert "[历史" not in rendered
    assert "回放结束" not in rendered
    recovery_message = (
        "[上下文] 已从最近的会话档案恢复连续对话上下文（状态：complete）"
    )
    assert rendered.index("你：") < rendered.index(recovery_message)
    assert rendered.index(recovery_message) < rendered.index("MiniCoder 交互模式")


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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        stdin=InterruptingInput(),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 130
    assert "任务被用户中断" in stderr.getvalue()
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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert "跟踪文件配置错误" in stderr.getvalue()


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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert "[失败] 模型服务请求失败" in stdout.getvalue()
    assert "模型服务请求失败" in stderr.getvalue()
    assert "network unavailable" not in stderr.getvalue()


def test_cli_renders_model_markdown_instead_of_printing_fence_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(content="Plan:\n1. Explain the input format."),
            _finish_plan_step(1, suffix="-markdown"),
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
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        stdout=stdout,
        stderr=StringIO(),
    )

    rendered = stdout.getvalue()
    assert exit_code == 0
    assert "```" not in rendered
    assert "n m" in rendered
    assert "u v w   # 共 m 行" in rendered
