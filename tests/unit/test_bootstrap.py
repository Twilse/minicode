from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import minicoder.bootstrap as bootstrap_module
from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.adapters.subprocess_runner import (
    PosixSubprocessAdapter,
    WindowsSubprocessAdapter,
)
from minicoder.application.completion import EvidenceBasedCompletionPolicy
from minicoder.application.context import ContextManager
from minicoder.application.retry import ExponentialBackoffRetryStrategy
from minicoder.bootstrap import ApplicationFactory
from minicoder.config import AppConfig
from minicoder.domain.models import AssistantTurn, ToolCall
from minicoder.domain.state import AgentPhase
from minicoder.platforms import OperatingSystem
from minicoder.tools.output import (
    StreamAwareOutputCompactor,
    ToolOutputArtifactStore,
)
from tests.fakes import FakeModelAdapter


def test_factory_creates_validated_bootstrap_context(tmp_path: Path) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "key",
            "MINICODER_BASE_URL": "https://api.deepseek.com",
            "MINICODER_MODEL": "deepseek-v4-pro",
        },
        workspace=tmp_path,
        platform_name="win32",
    )

    assert context.config.workspace == tmp_path.resolve()
    assert context.operating_system is OperatingSystem.WINDOWS


def test_factory_configures_sdk_client_without_hidden_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "secret-key",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "coding-model",
            "MINICODER_MODEL_TIMEOUT_SECONDS": "12.5",
        },
        workspace=tmp_path,
    )
    captured: dict[str, Any] = {}
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=object()))

    def fake_openai(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(bootstrap_module, "OpenAI", fake_openai)

    adapter = ApplicationFactory.create_model_adapter(config)

    assert isinstance(adapter, OpenAICompatibleChatAdapter)
    assert captured == {
        "api_key": "secret-key",
        "base_url": "https://models.example.com/v1",
        "timeout": 12.5,
        "max_retries": 0,
    }


def test_factory_builds_workspace_scoped_file_tools(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "secret-key",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "coding-model",
            "MINICODER_MAX_TOOL_OUTPUT_CHARS": "512",
        },
        workspace=tmp_path,
    )

    artifacts = ToolOutputArtifactStore(max_read_chars=256)
    try:
        tools = ApplicationFactory.create_tool_registry(
            config,
            processes=PosixSubprocessAdapter(),
            artifacts=artifacts,
        )
        definitions = tools.definitions()
        result = tools.execute(
            ToolCall(
                id="call-create",
                name="create_file",
                arguments_json='{"path":"created.txt","content":"hello"}',
            )
        )
    finally:
        artifacts.close()

    assert [definition.name for definition in definitions] == [
        "list_files",
        "read_file",
        "search_text",
        "create_file",
        "replace_text",
        "run_command",
        "read_tool_output",
    ]
    read_definition = next(
        definition for definition in definitions if definition.name == "read_file"
    )
    assert read_definition.parameters_schema["properties"]["limit"]["maximum"] == 512
    assert result.ok is True
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.parametrize(
    ("operating_system", "expected_type"),
    [
        (OperatingSystem.MACOS, PosixSubprocessAdapter),
        (OperatingSystem.LINUX, PosixSubprocessAdapter),
        (OperatingSystem.WINDOWS, WindowsSubprocessAdapter),
    ],
)
def test_factory_selects_platform_process_adapter(
    operating_system: OperatingSystem,
    expected_type: type[object],
) -> None:
    adapter = ApplicationFactory.create_process_adapter(operating_system)

    assert isinstance(adapter, expected_type)


def test_factory_selects_stream_aware_output_compactor() -> None:
    compactor = ApplicationFactory.create_output_compactor()

    assert isinstance(compactor, StreamAwareOutputCompactor)


def test_factory_selects_i09_context_and_retry_strategies(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "secret-key",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "coding-model",
            "MINICODER_CONTEXT_BUDGET_CHARS": "4321",
        },
        workspace=tmp_path,
    )

    context = ApplicationFactory.create_context_manager(config)
    retries = ApplicationFactory.create_retry_strategy()

    assert isinstance(context, ContextManager)
    assert context.budget_chars == 4321
    assert isinstance(retries, ExponentialBackoffRetryStrategy)


def test_factory_selects_the_evidence_based_completion_policy() -> None:
    policy = ApplicationFactory.create_completion_policy()

    assert isinstance(policy, EvidenceBasedCompletionPolicy)


def test_factory_creates_one_shot_agent_session_with_injected_adapters(
    tmp_path: Path,
) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
        },
        workspace=tmp_path,
        platform_name="darwin",
    )
    model = FakeModelAdapter([AssistantTurn(content="done")])
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
    )

    result = session.run("Inspect the workspace")

    assert result.phase is AgentPhase.COMPLETE
    assert result.final_response == "done"
    assert session.closed is True
