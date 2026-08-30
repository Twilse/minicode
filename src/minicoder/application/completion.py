"""Evidence-based policy for accepting a coding agent's final response."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from minicoder.application.verification import (
    CommandVerificationClassifier,
    VerificationClassifier,
)
from minicoder.domain.models import ToolCall, ToolResult

_MUTATION_TOOLS = frozenset({"create_file", "replace_text"})


class CompletionReason(str, Enum):
    """Stable explanation for accepting or rejecting a proposed final response."""

    NO_MUTATIONS = "no_mutations"
    VERIFIED = "verified"
    VERIFICATION_REQUIRED = "verification_required"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_UNSUPPORTED = "verification_unsupported"


@dataclass(frozen=True, slots=True)
class CompletionDecision:
    """Policy result for one model-proposed final response."""

    accepted: bool  # Whether AgentEngine may finish with the proposed response.
    reason: CompletionReason  # Stable decision reason for events and tests.
    feedback: str | None = None  # Host instruction appended after a rejection.
    terminal: bool = False  # Whether retrying inside this session cannot resolve it.


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

    def __init__(
        self,
        *,
        verification: VerificationClassifier | None = None,
    ) -> None:
        self._verification = (
            CommandVerificationClassifier()
            if verification is None
            else verification
        )
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
        self._unsupported_verification_generation: int | None = None

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
            self._unsupported_verification_generation = None

        classification = self._verification.classify(result)
        if classification is None:
            return None
        if classification.kind is None:
            if (
                classification.attempted
                and result.ok
                and result.metadata.get("exit_code") == 0
                and result.metadata.get("timed_out") is False
                and self._last_verification_generation
                != self._mutation_generation
            ):
                self._unsupported_verification_generation = (
                    self._mutation_generation
                )
            return None

        passed = (
            result.ok
            and result.metadata.get("exit_code") == 0
            and result.metadata.get("timed_out") is False
        )
        self._last_verification_generation = self._mutation_generation
        self._last_verification_passed = passed
        self._unsupported_verification_generation = None
        if passed:
            self._last_successful_verification_step = model_step
        return VerificationObservation(
            kind=classification.kind,
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

        if (
            self._unsupported_verification_generation
            == self._mutation_generation
        ):
            return CompletionDecision(
                accepted=False,
                reason=CompletionReason.VERIFICATION_UNSUPPORTED,
                feedback=(
                    "A command marked for verification completed, but MiniCoder "
                    "does not recognize it as built-in or startup-configured "
                    "verification. Add its exact argv to [verification].commands "
                    "in .minicoder.toml, review that file, and start a new session."
                ),
                terminal=True,
            )

        return CompletionDecision(
            accepted=False,
            reason=CompletionReason.VERIFICATION_REQUIRED,
            feedback=(
                "[MiniCoder completion policy]\n"
                "Files were changed but have not been successfully verified after "
                f"the latest change{files}. Run a relevant test, compile, or "
                "static-check command with run_command and set purpose to "
                "'verification' before returning a final response."
                f"{self._configured_command_guidance()}"
            ),
        )

    def unfinished_summary(self) -> str | None:
        decision = self.evaluate()
        if decision.accepted:
            return None
        files = _display_files(self.modified_files)
        if decision.reason is CompletionReason.VERIFICATION_FAILED:
            return f"Latest verification failed after the most recent change{files}."
        if decision.reason is CompletionReason.VERIFICATION_UNSUPPORTED:
            return "The attempted verification method is not supported in this session."
        return f"Modified work was not verified after the latest change{files}."

    def _configured_command_guidance(self) -> str:
        count = self._verification.configured_command_count
        if count == 0:
            return ""
        noun = "command" if count == 1 else "commands"
        return (
            f" This session loaded {count} exact alternative verification "
            f"{noun} from .minicoder.toml."
        )


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
