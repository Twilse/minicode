from __future__ import annotations

import json

from minicoder.application.progress import PlanProgress, tool_display_details
from minicoder.domain.models import ToolCall


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        name=name,
        arguments_json=json.dumps(arguments, ensure_ascii=False),
    )


def test_plan_progress_parses_items_and_infers_tool_stages() -> None:
    progress = PlanProgress.from_model_text(
        "1. 检查现有文件\n"
        "   并定位相关代码\n"
        "2. 实现所需修改\n"
        "3. 运行测试并验证结果"
    )

    first = progress.begin()
    inspection = progress.advance_for_tool(
        _call("run_command", {"argv": ["git", "status", "--short"]}),
        assistant_content=None,
    )
    mutation = progress.advance_for_tool(
        _call("replace_text", {"path": "app.py"}),
        assistant_content=None,
    )
    verification = progress.advance_for_tool(
        _call("run_command", {"argv": ["python", "-m", "pytest"]}),
        assistant_content=None,
    )

    assert progress.items == (
        "检查现有文件 并定位相关代码",
        "实现所需修改",
        "运行测试并验证结果",
    )
    assert first.index == 1
    assert inspection is None
    assert mutation is not None and mutation.index == 2
    assert verification is not None and verification.index == 3
    assert progress.finish() is None


def test_explicit_plan_step_marker_takes_priority_over_tool_inference() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect files.\n2. Implement changes.\n3. Verify behavior."
    )
    progress.begin()

    selected = progress.advance_for_tool(
        _call("read_file", {"path": "app.py"}),
        assistant_content="[plan_step=2] Continuing the implementation.",
    )

    assert selected is not None
    assert selected.index == 2
    assert selected.text == "Implement changes."


def test_tool_display_details_expose_targets_without_file_bodies() -> None:
    private_body = "private-body-must-not-be-displayed"
    create = _call(
        "create_file",
        {"path": "src/new.py", "content": private_body},
    )
    replace = _call(
        "replace_text",
        {
            "path": "src/app.py",
            "old_text": private_body,
            "new_text": "another-private-body",
        },
    )

    create_details = tool_display_details(create)
    replace_details = tool_display_details(replace)

    assert create_details == {"display_path": "src/new.py"}
    assert replace_details == {"display_path": "src/app.py"}
    assert private_body not in repr(create_details)
    assert private_body not in repr(replace_details)


def test_command_display_is_bounded_and_redacts_common_secret_arguments() -> None:
    api_key = "sk-this-value-must-never-be-displayed"
    command = _call(
        "run_command",
        {
            "argv": [
                "curl",
                "--header",
                f"Authorization: Bearer {api_key}",
                "--api-key",
                api_key,
                "https://example.com/" + "x" * 600,
            ]
        },
    )

    details = tool_display_details(command)
    displayed = str(details["display_command"])

    assert api_key not in displayed
    assert displayed.count("<redacted>") == 2
    assert len(displayed) <= 500


def test_invalid_tool_arguments_do_not_leak_raw_text_to_events() -> None:
    call = ToolCall(
        id="call-invalid",
        name="create_file",
        arguments_json="not-json private-body",
    )

    assert tool_display_details(call) == {}


def test_display_text_removes_terminal_control_sequences() -> None:
    progress = PlanProgress.from_model_text("1. \x1b[31m检查文件\x1b[0m")
    call = _call("read_file", {"path": "\x1b[2Jsrc/app.py"})

    assert progress.items == ("检查文件",)
    assert tool_display_details(call) == {
        "display_path": "src/app.py",
        "display_offset": 0,
    }
