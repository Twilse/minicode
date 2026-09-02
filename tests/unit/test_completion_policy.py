from __future__ import annotations

from collections.abc import Sequence

import pytest

from minicoder.application.completion import (
    CompletionReason,
    EvidenceBasedCompletionPolicy,
)
from minicoder.application.verification import (
    CommandVerificationClassifier,
    ConfiguredVerificationCommand,
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
    purpose: str | None = None,
    requested_argv: Sequence[str] | None = None,
) -> tuple[ToolCall, ToolResult]:
    call = _call("run_command", call_id=call_id)
    metadata: dict[str, object] = {
        "argv": tuple(argv),
        "exit_code": 0 if ok else 1,
        "timed_out": False,
    }
    if purpose is not None:
        metadata["purpose"] = purpose
    if requested_argv is not None:
        metadata["requested_argv"] = tuple(requested_argv)
    return call, _result(
        call,
        ok=ok,
        metadata=metadata,
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


def test_successful_write_file_requires_verification() -> None:
    policy = EvidenceBasedCompletionPolicy()
    write = _call("write_file", '{"path":"empty.py"}')

    policy.observe_tool(
        write,
        _result(write, ok=True, metadata={"path": "empty.py"}),
        model_step=1,
    )

    assert policy.evaluate().reason is CompletionReason.VERIFICATION_REQUIRED
    assert policy.modified_files == ("empty.py",)


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
        ("python", "dijkstra.py"),
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


def test_unknown_declared_verifier_gets_one_correction_before_stopping() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"main.zig"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "main.zig"}),
        model_step=1,
    )
    command, result = _command_result(
        ("zig", "build", "test"),
        ok=True,
        purpose="verification",
    )

    assert policy.observe_tool(command, result, model_step=2) is None
    assert "not supported" in (policy.unfinished_summary() or "")

    first_decision = policy.evaluate()

    assert first_decision.accepted is False
    assert first_decision.reason is CompletionReason.VERIFICATION_UNSUPPORTED
    assert first_decision.terminal is False
    assert "python -m py_compile" in (first_decision.feedback or "")
    assert "Directly running python <file>.py" in (
        first_decision.feedback or ""
    )

    second_decision = policy.evaluate()

    assert second_decision.accepted is False
    assert second_decision.reason is CompletionReason.VERIFICATION_UNSUPPORTED
    assert second_decision.terminal is True
    assert ".minicoder.toml" in (second_decision.feedback or "")


def test_new_mutation_restores_the_unsupported_verifier_correction() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"main.py"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "main.py"}),
        model_step=1,
    )
    first_command, first_result = _command_result(
        ("python", "main.py"),
        ok=True,
        purpose="verification",
    )
    policy.observe_tool(first_command, first_result, model_step=2)
    assert policy.evaluate().terminal is False
    assert policy.evaluate().terminal is True

    replace = _call("replace_text", '{"path":"main.py"}')
    policy.observe_tool(
        replace,
        _result(replace, ok=True, metadata={"path": "main.py"}),
        model_step=3,
    )
    second_command, second_result = _command_result(
        ("python", "main.py"),
        ok=True,
        purpose="verification",
        call_id="call-command-after-edit",
    )
    policy.observe_tool(second_command, second_result, model_step=4)

    assert policy.evaluate().terminal is False


def test_failed_unknown_declared_verifier_can_still_be_repaired() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"main.zig"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "main.zig"}),
        model_step=1,
    )
    command, result = _command_result(
        ("zig", "build", "test"),
        ok=False,
        purpose="verification",
    )

    policy.observe_tool(command, result, model_step=2)
    decision = policy.evaluate()

    assert decision.reason is CompletionReason.VERIFICATION_REQUIRED
    assert decision.terminal is False


def test_unmarked_unknown_command_does_not_become_verification() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"notes.txt"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "notes.txt"}),
        model_step=1,
    )
    command, result = _command_result(("echo", "done"), ok=True)

    policy.observe_tool(command, result, model_step=2)

    assert policy.evaluate().reason is CompletionReason.VERIFICATION_REQUIRED


def test_configured_exact_command_supplies_verification_evidence() -> None:
    policy = EvidenceBasedCompletionPolicy(
        verification=CommandVerificationClassifier(
            (ConfiguredVerificationCommand(("zig", "build", "test")),)
        )
    )
    create = _call("create_file", '{"path":"main.zig"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "main.zig"}),
        model_step=1,
    )
    command, result = _command_result(
        ("zig", "build", "test"),
        ok=True,
    )

    observation = policy.observe_tool(command, result, model_step=2)

    assert observation is not None
    assert observation.kind == "configured verifier"
    assert policy.evaluate().reason is CompletionReason.VERIFIED


def test_unknown_runtime_command_does_not_override_a_successful_compile() -> None:
    policy = EvidenceBasedCompletionPolicy()
    create = _call("create_file", '{"path":"main.cpp"}')
    policy.observe_tool(
        create,
        _result(create, ok=True, metadata={"path": "main.cpp"}),
        model_step=1,
    )
    compiler, compiled = _command_result(
        ("g++", "main.cpp", "-o", "main"),
        ok=True,
        purpose="verification",
    )
    executable, executed = _command_result(
        ("./main",),
        ok=True,
        call_id="call-executable",
        purpose="verification",
    )

    policy.observe_tool(compiler, compiled, model_step=2)
    policy.observe_tool(executable, executed, model_step=3)

    assert policy.evaluate().reason is CompletionReason.VERIFIED


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
