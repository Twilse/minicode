import json
from pathlib import Path

from minicoder.adapters.jsonl_trace import JsonlTraceSink
from minicoder.adapters.subprocess_runner import PosixSubprocessAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.models import AssistantTurn, ToolCall
from minicoder.domain.state import AgentPhase
from tests.fakes import FakeModelAdapter


def test_agent_jsonl_trace_excludes_message_and_tool_bodies(tmp_path: Path) -> None:
    api_key_marker = "api-key-must-not-appear"
    task_marker = "task-body-must-not-appear"
    argument_marker = "tool-argument-body-must-not-appear"
    reasoning_marker = "reasoning-must-not-appear"
    final_marker = "final-response-body-must-not-appear"
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": api_key_marker,
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
        },
        workspace=tmp_path,
        platform_name="darwin",
    )
    call = ToolCall(
        id="call-create",
        name="create_file",
        arguments_json=json.dumps(
            {
                "path": "private.py",
                "content": f"value = {argument_marker!r}\n",
            }
        ),
    )
    verify_call = ToolCall(
        id="call-verify",
        name="run_command",
        arguments_json=(
            '{"argv":["python","-m","py_compile","private.py"]}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=None,
                tool_calls=(call,),
                reasoning_content=reasoning_marker,
            ),
            AssistantTurn(content=None, tool_calls=(verify_call,)),
            AssistantTurn(content=final_marker),
        ]
    )
    trace_path = tmp_path / "agent-trace.jsonl"
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
        event_sinks=(JsonlTraceSink(trace_path),),
    )

    result = session.run(task_marker)

    assert result.phase is AgentPhase.COMPLETE
    trace_text = trace_path.read_text(encoding="utf-8")
    for forbidden in (
        api_key_marker,
        task_marker,
        argument_marker,
        reasoning_marker,
        final_marker,
    ):
        assert forbidden not in trace_text
    records = [json.loads(line) for line in trace_text.splitlines()]
    assert [record["sequence"] for record in records] == list(range(1, 11))
    assert [record["type"] for record in records] == [
        "task_started",
        "model_requested",
        "tool_called",
        "tool_finished",
        "model_requested",
        "tool_called",
        "tool_finished",
        "verification_passed",
        "model_requested",
        "task_completed",
    ]
    assert records[3]["details"]["content_chars"] > 0
