from __future__ import annotations

from collections.abc import Sequence

import pytest

from minicoder.application.completion import (
    CompletionReason,
    EvidenceBasedCompletionPolicy,
)
from minicoder.domain.models import ToolCall, ToolResult


def _call(
    name: str,
    arguments_json: str = "{}",
    *,
    call_id: str | None = None,
) -> ToolCall:
    return ToolCall(
        id=call_id or f"call-{name}",
        name=name,
        arguments_json=arguments_json,
    )


def _result(
    call: ToolCall,
    *,
    ok: bool,
    metadata: dict[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=ok,
        content="completed" if ok else "failed",
        error_code=None if ok else "TOOL_FAILED",
        metadata={} if metadata is None else metadata,
    )


def _command_result(
    argv: Sequence[str],
    *,
    ok: bool,
    call_id: str = "call-command",
) -> tuple[ToolCall, ToolResult]:
    call = _call("run_command", call_id=call_id)
    return call, _result(
        call,
        ok=ok,
        metadata={
            "argv": tuple(argv),
            "exit_code": 0 if ok else 1,
            "timed_out": False,
        },
    )


def test_policy_accepts_read_only_work_without_verification() -> None:
    policy = EvidenceBasedCompletionPolicy()
    read = _call("read_file", '{"path":"app.py"}')

    policy.observe_tool(read, _result(read, ok=True), model_step=1)
    decision = policy.evaluate()

    assert decision.accepted is True
    assert decision.reason is CompletionReason.NO_MUTATIONS
    assert decision.feedback is None


def test_policy_requires_verification_after_a_successful_mutation() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"src/new.py"}')

    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "src/new.py"}),
        model_step=2,
    )
    decision = policy.evaluate()

    assert decision.accepted is False
    assert decision.reason is CompletionReason.VERIFICATION_REQUIRED
    assert "src/new.py" in (decision.feedback or "")
    assert policy.modified_files == ("src/new.py",)
    assert policy.last_mutation_step == 2
    assert policy.last_successful_verification_step is None


def test_failed_mutation_does_not_create_modification_evidence() -> None:
    policy = EvidenceBasedCompletionPolicy()
    replace = _call("replace_text", '{"path":"app.py"}')

    policy.observe_tool(replace, _result(replace, ok=False), model_step=1)

    assert policy.evaluate().reason is CompletionReason.NO_MUTATIONS
    assert policy.modified_files == ()


def test_successful_pytest_after_mutation_allows_completion() -> None:
    policy = EvidenceBasedCompletionPolicy()
    replace = _call("replace_text", '{"path":"app.py"}')
    policy.observe_tool(
        replace,
        _result(replace, ok=True, metadata={"path": "app.py"}),
        model_step=1,
    )
    command, result = _command_result(
        ("/runtime/python3.14", "-m", "pytest", "-q"),
        ok=True,
    )

    observation = policy.observe_tool(command, result, model_step=2)
    decision = policy.evaluate()

    assert observation is not None
    assert observation.kind == "pytest"
    assert observation.passed is True
    assert decision.accepted is True
    assert decision.reason is CompletionReason.VERIFIED
    assert policy.last_successful_verification_step == 2


def test_a_later_failed_verification_invalidates_an_earlier_success() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"app.py"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "app.py"}),
        model_step=1,
    )
    passed_call, passed = _command_result(("pytest", "-q"), ok=True)
    failed_call, failed = _command_result(
        ("pytest", "-q"),
        ok=False,
        call_id="call-failed-test",
    )
    policy.observe_tool(passed_call, passed, model_step=2)

    observation = policy.observe_tool(failed_call, failed, model_step=3)
    decision = policy.evaluate()

    assert observation is not None and observation.passed is False
    assert decision.accepted is False
    assert decision.reason is CompletionReason.VERIFICATION_FAILED
    assert "failed" in (decision.feedback or "").casefold()
    assert "Latest verification failed" in (policy.unfinished_summary() or "")


def test_mutation_after_verification_requires_a_new_check_even_in_same_step() -> None:
    policy = EvidenceBasedCompletionPolicy()
    command, command_result = _command_result(("pytest", "-q"), ok=True)
    create = _call("create_file", '{"path":"later.py"}')

    policy.observe_tool(command, command_result, model_step=1)
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "later.py"}),
        model_step=1,
    )

    assert policy.evaluate().reason is CompletionReason.VERIFICATION_REQUIRED

    new_command, new_result = _command_result(
        ("pytest", "-q"),
        ok=True,
        call_id="call-new-test",
    )
    policy.observe_tool(new_command, new_result, model_step=1)
    assert policy.evaluate().reason is CompletionReason.VERIFIED


@pytest.mark.parametrize(
    ("argv", "expected_kind"),
    [
        (("pytest", "-q"), "pytest"),
        (("python.exe", "-m", "unittest"), "unittest"),
        (("python3", "-m", "compileall", "-q", "src"), "compileall"),
        (("ruff", "check", "."), "ruff"),
        (("ruff", "format", "--check", "."), "ruff"),
        (("npm.cmd", "run", "test"), "npm test"),
        (("go", "test", "./..."), "go test"),
        (("cargo", "check"), "cargo check"),
        (("dotnet", "build"), "dotnet build"),
        (("mvnw", "verify"), "maven"),
        (("gradlew.bat", "test"), "gradle"),
        (("make", "lint"), "make"),
    ],
)
def test_policy_recognizes_cross_platform_verification_commands(
    argv: tuple[str, ...],
    expected_kind: str,
) -> None:
    policy = EvidenceBasedCompletionPolicy()
    command, result = _command_result(argv, ok=True)

    observation = policy.observe_tool(command, result, model_step=1)

    assert observation is not None
    assert observation.kind == expected_kind


@pytest.mark.parametrize(
    "argv",
    [
        ("pwd",),
        ("git", "status"),
        ("pytest", "--collect-only"),
        ("python", "-m", "pytest", "--help"),
        ("ruff", "format", "."),
        ("npm", "install"),
    ],
)
def test_policy_rejects_commands_that_are_not_actual_verification(
    argv: tuple[str, ...],
) -> None:
    policy = EvidenceBasedCompletionPolicy()
    command, result = _command_result(argv, ok=True)

    assert policy.observe_tool(command, result, model_step=1) is None


def test_policy_reset_clears_evidence_for_a_new_task() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"old.py"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "old.py"}),
        model_step=1,
    )

    policy.reset()

    assert policy.modified_files == ()
    assert policy.last_mutation_step is None
    assert policy.evaluate().reason is CompletionReason.NO_MUTATIONS
