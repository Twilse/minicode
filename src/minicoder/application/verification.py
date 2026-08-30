"""Classify host-observed commands as supported verification evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import ToolResult

VERIFICATION_PURPOSE = "verification"

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
_C_FAMILY_COMPILERS = frozenset(
    {"cc", "c++", "gcc", "g++", "clang", "clang++", "cl"}
)
_C_FAMILY_SOURCE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".m",
    ".mm",
)
_PREPROCESS_ONLY_FLAGS = frozenset(
    {"-###", "-e", "-m", "-mm", "/e", "/ep", "/p"}
)
_MAKE_NON_EXECUTING_FLAGS = frozenset(
    {"-n", "--dry-run", "-q", "--question", "-p", "--print-data-base"}
)


@dataclass(frozen=True, slots=True)
class ConfiguredVerificationCommand:
    """One exact command trusted from the startup workspace configuration."""

    argv: tuple[str, ...]  # Exact model-requested argv accepted as one verifier.

    def __post_init__(self) -> None:
        if not isinstance(self.argv, tuple) or not self.argv or any(
            not isinstance(argument, str) or not argument.strip()
            for argument in self.argv
        ):
            raise DomainValidationError(
                "configured verification argv must contain non-blank text"
            )


@dataclass(frozen=True, slots=True)
class VerificationClassification:
    """Whether one process result is supported or declared verification."""

    kind: str | None  # Safe verifier label, or None when unsupported.
    attempted: bool  # Whether the command was explicitly intended as verification.

    def __post_init__(self) -> None:
        if not isinstance(self.attempted, bool):
            raise DomainValidationError("verification attempted must be boolean")
        if self.kind is not None and not self.kind.strip():
            raise DomainValidationError("verification kind must be non-blank text")
        if self.kind is not None and not self.attempted:
            raise DomainValidationError(
                "supported verification must also be an attempted verification"
            )


class VerificationClassifier(Protocol):
    """Interpret command metadata without deciding whether the Agent may finish."""

    @property
    def configured_command_count(self) -> int:
        """Return the number of startup-configured accepted commands."""

        ...

    def classify(
        self,
        result: ToolResult,
    ) -> VerificationClassification | None:
        """Return command classification, or None for non-process results."""

        ...


class CommandVerificationClassifier:
    """Recognize common ecosystems plus exact startup-configured commands."""

    def __init__(
        self,
        configured_commands: Sequence[ConfiguredVerificationCommand] = (),
    ) -> None:
        commands = tuple(configured_commands)
        if any(
            not isinstance(command, ConfiguredVerificationCommand)
            for command in commands
        ):
            raise DomainValidationError(
                "configured commands must be ConfiguredVerificationCommand values"
            )
        self._configured_commands = commands

    @property
    def configured_command_count(self) -> int:
        return len(self._configured_commands)

    def classify(
        self,
        result: ToolResult,
    ) -> VerificationClassification | None:
        if result.tool_name != "run_command":
            return None

        argv = _metadata_argv(result, "argv")
        if argv is None:
            return None
        requested_argv = _metadata_argv(result, "requested_argv") or argv
        kind = self._configured_kind(requested_argv) or _builtin_kind(argv)
        declared = result.metadata.get("purpose") == VERIFICATION_PURPOSE
        return VerificationClassification(
            kind=kind,
            attempted=kind is not None or declared,
        )

    def _configured_kind(self, argv: tuple[str, ...]) -> str | None:
        if any(command.argv == argv for command in self._configured_commands):
            return "configured verifier"
        return None


def _metadata_argv(result: ToolResult, name: str) -> tuple[str, ...] | None:
    value = result.metadata.get(name)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(argument, str) for argument in value)
    ):
        return None
    return tuple(value)


def _builtin_kind(argv: tuple[str, ...]) -> str | None:
    normalized = tuple(argument.casefold() for argument in argv)
    program = _program_name(normalized[0])
    arguments = normalized[1:]

    if program in _DIRECT_VERIFIERS and not _is_informational(arguments):
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
    if program in _C_FAMILY_COMPILERS and _is_c_family_compilation(arguments):
        return "C/C++ compiler"
    if program == "cmake" and "--build" in arguments:
        if not _is_informational(arguments):
            return "cmake build"
    if program == "ctest" and _is_ctest_execution(arguments):
        return "ctest"
    if program == "ninja" and _is_build_execution(arguments):
        return "ninja"
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
    if program == "make" and _is_make_execution(arguments):
        return "make"
    return None


def _is_c_family_compilation(arguments: tuple[str, ...]) -> bool:
    if _is_informational(arguments) or any(
        argument in _PREPROCESS_ONLY_FLAGS for argument in arguments
    ):
        return False
    return any(argument.endswith(_C_FAMILY_SOURCE_SUFFIXES) for argument in arguments)


def _is_ctest_execution(arguments: tuple[str, ...]) -> bool:
    if _is_informational(arguments):
        return False
    return not any(
        argument == "-n" or argument.startswith("--show-only")
        for argument in arguments
    )


def _is_build_execution(arguments: tuple[str, ...]) -> bool:
    if _is_informational(arguments):
        return False
    return not any(
        argument in {"-n", "--dry-run", "-t"} for argument in arguments
    )


def _is_make_execution(arguments: tuple[str, ...]) -> bool:
    if _is_informational(arguments) or any(
        argument in _MAKE_NON_EXECUTING_FLAGS for argument in arguments
    ):
        return False
    explicit_targets = tuple(
        argument for argument in arguments if argument and not argument.startswith("-")
    )
    return not explicit_targets or any(
        target in _MAKE_TARGETS for target in explicit_targets
    )


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
        argument in {"-h", "--help", "/?", "--version", "--collect-only"}
        for argument in arguments
    )


def _program_name(raw_program: str) -> str:
    name = raw_program.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
