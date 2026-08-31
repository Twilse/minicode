from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import AssistantTurn, ToolCall
from minicoder.domain.state import AgentPhase, AgentStopReason
from tests.fakes import FakeModelAdapter, MemoryEventSink


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
def test_agent_accepts_a_real_compiled_and_executed_cpp_change(
    tmp_path: Path,
) -> None:
    source = (
        "#include <iostream>\n"
        "\n"
        "int main() {\n"
        "    std::cout << 3 + 5 << '\\n';\n"
        "    return 0;\n"
        "}\n"
    )
    create = ToolCall(
        id="call-create-cpp",
        name="create_file",
        arguments_json=json.dumps({"path": "two_sum.cpp", "content": source}),
    )
    compile_source = ToolCall(
        id="call-compile-cpp",
        name="run_command",
        arguments_json=json.dumps(
            {
                "argv": [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "two_sum.cpp",
                    "-o",
                    "two_sum",
                ],
                "purpose": "verification",
            }
        ),
    )
    execute_binary = ToolCall(
        id="call-run-cpp",
        name="run_command",
        arguments_json=json.dumps(
            {"argv": ["./two_sum"], "purpose": "verification"}
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content="1. Create the C++ file.\n2. Compile it.\n3. Run it."
            ),
            AssistantTurn(content=None, tool_calls=(create,)),
            AssistantTurn(content=None, tool_calls=(compile_source,)),
            AssistantTurn(content=None, tool_calls=(execute_binary,)),
            AssistantTurn(content="Created, compiled, and ran the C++ example."),
        ]
    )
    events = MemoryEventSink()
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
            "MINICODER_COMMAND_TIMEOUT_SECONDS": "10",
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        workspace=tmp_path,
    )
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=ApplicationFactory.create_process_adapter(
            context.operating_system
        ),
        event_sinks=(events,),
    )

    result = session.run("Create and verify a C++ two-sum example")

    assert result.phase is AgentPhase.COMPLETE
    assert result.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert result.model_steps == 5
    assert (tmp_path / "two_sum.cpp").read_text(encoding="utf-8") == source
    assert (tmp_path / "two_sum").is_file()
    verification_events = [
        event
        for event in events.events
        if event.kind is AgentEventKind.VERIFICATION_PASSED
    ]
    assert [event.details["verification_kind"] for event in verification_events] == [
        "C/C++ compiler"
    ]


def test_agent_uses_a_startup_configured_project_verifier(tmp_path: Path) -> None:
    (tmp_path / ".minicoder.toml").write_text(
        "[verification]\ncommands = [['python', 'verify.py']]\n",
        encoding="utf-8",
    )
    (tmp_path / "verify.py").write_text(
        "print('project verification passed')\n",
        encoding="utf-8",
    )
    create = ToolCall(
        id="call-create-zig",
        name="create_file",
        arguments_json=json.dumps(
            {"path": "main.zig", "content": 'const std = @import("std");\n'}
        ),
    )
    verify = ToolCall(
        id="call-project-verify",
        name="run_command",
        arguments_json=json.dumps(
            {"argv": ["python", "verify.py"], "purpose": "verification"}
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Create the file.\n2. Run verification."),
            AssistantTurn(content=None, tool_calls=(create,)),
            AssistantTurn(content=None, tool_calls=(verify,)),
            AssistantTurn(content="Created and verified the Zig source."),
        ]
    )
    events = MemoryEventSink()
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "false",
        },
        workspace=tmp_path,
    )
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=ApplicationFactory.create_process_adapter(
            context.operating_system
        ),
        event_sinks=(events,),
    )

    result = session.run("Create a Zig file and use the project verifier")

    assert result.phase is AgentPhase.COMPLETE
    verification_events = [
        event
        for event in events.events
        if event.kind is AgentEventKind.VERIFICATION_PASSED
    ]
    assert [event.details["verification_kind"] for event in verification_events] == [
        "configured verifier"
    ]
