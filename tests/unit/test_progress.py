from __future__ import annotations

import json

import pytest

from minicoder.application.progress import (
    PlanProgress,
    tool_display_details,
)
from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import ToolCall


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=f"call-{name}",
        name=name,
        arguments_json=json.dumps(arguments, ensure_ascii=False),
    )


@pytest.mark.parametrize(
    "heading",
    [
        "计划：",
        "## 规划",
        "**方案：**",
        "执行步骤",
        "实现思路",
        "Plan:",
        "Implementation Plan",
        "Roadmap",
        "Next Steps:",
    ],
)
def test_planning_response_accepts_common_headings_and_explicit_items(
    heading: str,
) -> None:
    progress = PlanProgress.from_planning_response(
        f"{heading}\n1. 检查文件\n2. 输出结果"
    )

    assert progress.items == ("检查文件", "输出结果")


def test_planning_response_accepts_bulleted_items() -> None:
    progress = PlanProgress.from_planning_response(
        "方案\n- 检查现有实现\n- 运行测试"
    )

    assert progress.items == ("检查现有实现", "运行测试")


@pytest.mark.parametrize(
    "text",
    [
        "1. 检查文件\n2. 输出结果",
        "这个计划还没有生成。\n- 检查文件",
        "计划：\n这里没有明确分点",
        "",
    ],
)
def test_planning_response_rejects_missing_heading_or_explicit_items(
    text: str,
) -> None:
    with pytest.raises(DomainValidationError, match="planning response"):
        PlanProgress.from_planning_response(text)


def test_plan_progress_advances_only_after_exact_current_step_report() -> None:
    progress = PlanProgress.from_model_text(
        "1. 检查现有文件\n"
        "   并定位相关代码\n"
        "2. 实现所需修改\n"
        "3. 运行测试并验证结果"
    )

    first = progress.begin()
    second = progress.complete_current(1)
    third = progress.complete_current(2)
    final_report = progress.complete_current(3)

    assert progress.items == (
        "检查现有文件 并定位相关代码",
        "实现所需修改",
        "运行测试并验证结果",
    )
    assert first.step.index == 1 and first.completed is False
    assert [
        (update.step.index, update.completed) for update in second.updates
    ] == [
        (1, True),
        (2, False),
    ]
    assert [
        (update.step.index, update.completed)
        for update in third.updates
    ] == [
        (2, True),
        (3, False),
    ]
    assert final_report.updates == ()
    assert progress.all_steps_reported is True
    assert [
        (update.step.index, update.completed)
        for update in progress.finish().updates
    ] == [(3, True)]


def test_plan_progress_rejects_skipped_or_repeated_step_numbers() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect files.\n2. Implement changes.\n3. Verify behavior."
    )
    progress.begin()

    with pytest.raises(DomainValidationError, match="current plan step is 1"):
        progress.complete_current(2)

    assert progress.current_index == 1
    progress.complete_current(1)
    with pytest.raises(DomainValidationError, match="current plan step is 2"):
        progress.complete_current(1)
    assert progress.current_index == 2


def test_plan_progress_cannot_be_started_twice() -> None:
    progress = PlanProgress.from_model_text("1. Inspect files.")
    progress.begin()

    with pytest.raises(DomainValidationError, match="already started"):
        progress.begin()


def test_plan_finish_requires_every_step_reported() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect files.\n2. Implement changes.\n3. Verify behavior."
    )
    progress.begin()

    with pytest.raises(DomainValidationError, match="final step"):
        progress.finish()

    assert progress.current_index == 1


def test_final_step_stays_active_until_final_response_is_accepted() -> None:
    progress = PlanProgress.from_model_text(
        "1. Inspect.\n2. Implement.\n3. Verify."
    )
    progress.begin()
    progress.complete_current(1)
    progress.complete_current(2)

    transition = progress.complete_current(3)

    assert transition.updates == ()
    assert progress.current_index == 3
    assert progress.current_step.text == "Verify."
    assert progress.all_steps_reported is True


def test_plan_progress_supports_at_most_five_items() -> None:
    progress = PlanProgress.from_model_text(
        "1. 检查现有 todo_cli 的 models.py、storage.py、cli.py、测试与项目结构。\n"
        "2. 扩展 models.py 和 storage.py：增加标签、搜索、筛选与批量清理。\n"
        "3. 扩展 cli.py：新增筛选、批量命令和彩色输出。\n"
        "4. 增加 pyproject.toml 打包配置，并更新 README 文档。\n"
        "5. 新增或更新 tests 单元测试与 CLI 集成测试。\n"
        "6. 运行 python -m unittest discover 和冒烟测试。"
    )

    assert progress.total == 5
    assert progress.items[-1].startswith("新增或更新 tests")


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
