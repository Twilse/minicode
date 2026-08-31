from __future__ import annotations

import json

from minicoder.application.progress import (
    PlanProgress,
    tool_display_details,
)
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
        explicit_step=None,
    )
    mutation = progress.advance_for_tool(
        _call("replace_text", {"path": "app.py"}),
        explicit_step=None,
    )
    verification = progress.advance_for_tool(
        _call("run_command", {"argv": ["python", "-m", "pytest"]}),
        explicit_step=None,
    )

    assert progress.items == (
        "检查现有文件 并定位相关代码",
        "实现所需修改",
        "运行测试并验证结果",
    )
    assert first.step.index == 1 and first.completed is False
    assert inspection.updates == ()
    assert inspection.untracked == ()
    assert [
        (update.step.index, update.completed) for update in mutation.updates
    ] == [
        (1, True),
        (2, False),
    ]
    assert [
        (update.step.index, update.completed)
        for update in verification.updates
    ] == [
        (2, True),
        (3, False),
    ]
    assert [
        (update.step.index, update.completed)
        for update in progress.finish().updates
    ] == [(3, True)]


def test_compatible_explicit_plan_step_takes_priority_over_inference() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect files.\n2. Implement changes.\n3. Verify behavior."
    )
    progress.begin()

    transition = progress.advance_for_tool(
        _call("replace_text", {"path": "app.py"}),
        explicit_step=2,
    )

    assert transition.updates[-1].step.index == 2
    assert transition.updates[-1].step.text == "Implement changes."
    assert transition.updates[-1].completed is False


def test_incompatible_explicit_step_cannot_override_real_tool_activity() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect files.\n2. Implement changes.\n3. Verify behavior."
    )
    progress.begin()

    transition = progress.advance_for_tool(
        _call("read_file", {"path": "app.py"}),
        explicit_step=2,
    )

    assert transition.updates == ()
    assert progress.current_index == 1


def test_plan_progress_reports_untracked_items_instead_of_fake_updates() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect.\n2. Design.\n3. Implement.\n4. Verify."
    )
    progress.begin()
    progress.advance_for_tool(
        _call("read_file", {"path": "app.py"}),
        explicit_step=1,
    )

    transition = progress.advance_for_tool(
        _call("run_command", {"argv": ["python", "-m", "pytest"]}),
        explicit_step=4,
    )

    assert [
        (update.step.index, update.completed)
        for update in transition.updates
    ] == [
        (1, True),
        (4, False),
    ]
    assert [step.index for step in transition.untracked] == [2, 3]
    assert progress.untracked_count == 2


def test_plan_finish_closes_only_the_active_item_before_whole_plan_completion() -> None:
    progress = PlanProgress.from_model_text(
        "1. Answer the question.\n2. Add supporting details.\n3. Summarize."
    )
    progress.begin()

    transition = progress.finish()

    assert [
        (update.step.index, update.completed)
        for update in transition.updates
    ] == [(1, True)]
    assert [step.index for step in transition.untracked] == [2, 3]
    assert progress.current_index == 3
    assert progress.untracked_count == 2


def test_realistic_tool_targets_follow_each_six_step_plan_item() -> None:
    progress = PlanProgress.from_model_text(
        "1. 检查 models.py、storage.py、cli.py、tests 和 README。\n"
        "2. 在 models.py 和 storage.py 增加数据与存储功能。\n"
        "3. 在 cli.py 接入新命令。\n"
        "4. 为新功能补充 tests 测试。\n"
        "5. 更新 README 文档。\n"
        "6. 运行 unittest 验证。"
    )
    progress.begin()
    calls = (
        _call("read_file", {"path": "todo_cli/models.py"}),
        _call("replace_text", {"path": "todo_cli/models.py"}),
        _call("replace_text", {"path": "todo_cli/cli.py"}),
        _call("replace_text", {"path": "tests/test_todo_cli.py"}),
        _call("create_file", {"path": "README.md"}),
        _call(
            "run_command",
            {"argv": ["python", "-m", "unittest", "discover"]},
        ),
    )

    transitions = [
        progress.advance_for_tool(call, explicit_step=None) for call in calls
    ]

    started = [
        update.step.index
        for transition in transitions
        for update in transition.updates
        if not update.completed
    ]
    assert started == [2, 3, 4, 5, 6]
    assert all(not transition.untracked for transition in transitions)
    assert progress.untracked_count == 0


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
