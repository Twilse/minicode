"""Evidence-based policy for accepting a coding agent's final response."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from minicoder.domain.models import ToolCall, ToolResult

_MUTATION_TOOLS = frozenset({"create_file", "replace_text"})
_EXECUTABLE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_PYTHON_PROGRAM = re.compile(r"python(?:3(?:\.\d+)?)?", re.IGNORECASE)
_DIRECT_VERIFIERS = frozenset(
    {
        "flake8",
        "mypy",
        "nox",
        "pytest",
        "pyright",
        "tox",
    }
)
_PYTHON_MODULE_VERIFIERS = frozenset(
    {"compileall", "mypy", "py_compile", "pytest", "unittest"}
)
_MAKE_TARGETS = frozenset({"build", "check", "lint", "test", "verify"})
_PACKAGE_SCRIPT_TARGETS = frozenset({"build", "check", "lint", "test", "verify"})


class CompletionReason(str, Enum):
    """Stable explanation for accepting or rejecting a proposed final response."""

    NO_MUTATIONS = "no_mutations"
    VERIFIED = "verified"
    VERIFICATION_REQUIRED = "verification_required"
    VERIFICATION_FAILED = "verification_failed"


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    """Policy result for one model-proposed final response."""

    accepted: bool  # Whether AgentEngine may finish with the proposed response.
    reason: CompletionReason  # Stable decision reason for events and tests.
    feedback: str | None = None  # Host instruction appended after a rejection.


@dataclass(frozen=True, slots=True)
class VerificationObservation:
    """One recognized verification command and its actual process outcome."""

    kind: str  # Provider-neutral verifier label such as pytest or compileall.
    passed: bool  # True only for a completed command with exit code zero.
    model_step: int  # Agent model step that requested this command.


class CompletionPolicy(Protocol):
    """Track tool evidence and decide whether a final response is justified."""

    @property
    def modified_files(self) -> tuple[str, ...]:
        """Return known successfully changed workspace-relative paths."""

        ...

    def reset(self) -> None:
        """Clear all evidence before one fresh AgentEngine task."""

        ...

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        model_step: int,
    ) -> VerificationObservation | None:
        """Record one correlated tool outcome and return verification facts."""

        ...

    def evaluate(self) -> CompletionDecision:
        """Evaluate the evidence available when the model proposes completion."""

        ...

    def unfinished_summary(self) -> str | None:
        """Describe unverified work when the model-step limit is reached."""

        ...


class EvidenceBasedCompletionPolicy:
    """Require a successful recognized check after the latest file mutation."""

    def __init__(self) -> None:
        self.reset()

    @property
    def modified_files(self) -> tuple[str, ...]:
        return tuple(self._modified_files)

    @property
    def last_mutation_step(self) -> int | None:
        return self._last_mutation_step

    @property
    def last_successful_verification_step(self) -> int | None:
        return self._last_successful_verification_step

    def reset(self) -> None:
        self._modified_files: dict[str, None] = {}
        self._mutation_generation = 0
        self._last_mutation_step: int | None = None
        self._last_successful_verification_step: int | None = None
        self._last_verification_generation: int | None = None
        self._last_verification_passed: bool | None = None

    def observe_tool(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        model_step: int,
    ) -> VerificationObservation | None:
        if result.ok and result.tool_name in _MUTATION_TOOLS:
            path = _mutation_path(call, result.metadata)
            if path is not None:
                self._modified_files[path] = None
            self._mutation_generation += 1
            self._last_mutation_step = model_step

        verification_kind = _verification_kind(result)
        if verification_kind is None:
            return None

        passed = (
            result.ok
            and result.metadata.get("exit_code") == 0
            and result.metadata.get("timed_out") is False
        )
        self._last_verification_generation = self._mutation_generation
        self._last_verification_passed = passed
        if passed:
            self._last_successful_verification_step = model_step
        return VerificationObservation(
            kind=verification_kind,
            passed=passed,
            model_step=model_step,
        )

    def evaluate(self) -> CompletionDecision:
        if self._mutation_generation == 0:
            return CompletionDecision(
                accepted=True,
                reason=CompletionReason.NO_MUTATIONS,
            )

        if (
            self._last_verification_generation == self._mutation_generation
            and self._last_verification_passed is True
        ):
            return CompletionDecision(
                accepted=True,
                reason=CompletionReason.VERIFIED,
            )

        files = _display_files(self.modified_files)
        if (
            self._last_verification_generation == self._mutation_generation
            and self._last_verification_passed is False
        ):
            return CompletionDecision(
                accepted=False,
                reason=CompletionReason.VERIFICATION_FAILED,
                feedback=(
                    "[MiniCoder completion policy]\n"
                    "The latest verification after the most recent file change "
                    f"failed. Inspect its tool output, fix the affected work{files}, "
                    "and run a relevant test, compile, or static-check command again "
                    "before returning a final response."
                ),
            )

        return CompletionDecision(
            accepted=False,
            reason=CompletionReason.VERIFICATION_REQUIRED,
            feedback=(
                "[MiniCoder completion policy]\n"
                "Files were changed but have not been successfully verified after "
                f"the latest change{files}. Run a relevant test, compile, or "
                "static-check command with run_command before returning a final "
                "response."
            ),
        )

    def unfinished_summary(self) -> str | None:
        decision = self.evaluate()
        if decision.accepted:
            return None
        files = _display_files(self.modified_files)
        if decision.reason is CompletionReason.VERIFICATION_FAILED:
            return f"Latest verification failed after the most recent change{files}."
        return f"Modified work was not verified after the latest change{files}."


def _verification_kind(result: ToolResult) -> str | None:
    if result.tool_name != "run_command":
        return None
    argv = result.metadata.get("argv")
    if (
        not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes))
        or not argv
        or any(not isinstance(argument, str) for argument in argv)
    ):
        return None

    normalized = tuple(argument.casefold() for argument in argv)
    program = _program_name(normalized[0])
    arguments = normalized[1:]

    if program in _DIRECT_VERIFIERS:
        if not _is_informational(arguments):
            return program
    if _PYTHON_PROGRAM.fullmatch(program) and len(arguments) >= 2:
        if (
            arguments[0] == "-m"
            and arguments[1] in _PYTHON_MODULE_VERIFIERS
            and not _is_informational(arguments[2:])
        ):
            return arguments[1]
    if program == "ruff" and arguments:
        if arguments[0] == "check" or (
            arguments[0] == "format" and "--check" in arguments[1:]
        ):
            return "ruff"
    if program in {"npm", "pnpm", "yarn"}:
        return _package_script_kind(program, arguments)
    if program == "go" and arguments[:1] == ("test",):
        return "go test"
    if program == "cargo" and arguments[:1] in {("test",), ("check",)}:
        return f"cargo {arguments[0]}"
    if program == "dotnet" and arguments[:1] in {("test",), ("build",)}:
        return f"dotnet {arguments[0]}"
    if program in {"mvn", "mvnw"} and any(
        argument in {"test", "verify"} for argument in arguments
    ):
        return "maven"
    if program in {"gradle", "gradlew"} and any(
        argument in _MAKE_TARGETS for argument in arguments
    ):
        return "gradle"
    if program == "make" and any(
        argument in _MAKE_TARGETS for argument in arguments
    ):
        return "make"
    return None


def _package_script_kind(program: str, arguments: tuple[str, ...]) -> str | None:
    if not arguments:
        return None
    if arguments[0] in _PACKAGE_SCRIPT_TARGETS:
        return f"{program} {arguments[0]}"
    if (
        arguments[0] == "run"
        and len(arguments) >= 2
        and arguments[1] in _PACKAGE_SCRIPT_TARGETS
    ):
        return f"{program} {arguments[1]}"
    return None


def _is_informational(arguments: Sequence[str]) -> bool:
    return any(
        argument in {"-h", "--help", "--version", "--collect-only"}
        for argument in arguments
    )


def _program_name(raw_program: str) -> str:
    name = raw_program.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _mutation_path(
    call: ToolCall,
    metadata: Mapping[str, Any],
) -> str | None:
    path = metadata.get("path")
    if isinstance(path, str) and path.strip():
        return path
    try:
        arguments = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) and path.strip() else None


def _display_files(files: Sequence[str]) -> str:
    if not files:
        return ""
    selected = files[:8]
    suffix = f", and {len(files) - len(selected)} more" if len(files) > 8 else ""
    return f" ({', '.join(selected)}{suffix})"
