"""Deterministic host-side policy for model-proposed command arguments."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence

COMMAND_REJECTED = "COMMAND_REJECTED"
INVALID_COMMAND = "INVALID_COMMAND"

_BLOCKED_PROGRAMS = frozenset(
    {
        "bash",
        "cmd",
        "diskpart",
        "doas",
        "format",
        "halt",
        "init",
        "kill",
        "killall",
        "mkfs",
        "pkill",
        "poweroff",
        "powershell",
        "pwsh",
        "reboot",
        "shutdown",
        "sh",
        "su",
        "sudo",
        "taskkill",
        "zsh",
    }
)
_EXECUTABLE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_PYTHON_COMMAND = re.compile(r"python(?:3(?:\.\d+)?)?", re.IGNORECASE)


class CommandPolicyError(ValueError):
    """A command vector that is malformed or denied by host policy."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class CommandSafetyPolicy:
    """Validate command shape, reject obvious hazards, and normalize Python."""

    def __init__(self, *, python_executable: str | None = None) -> None:
        executable = sys.executable if python_executable is None else python_executable
        if (
            not isinstance(executable, str)
            or not executable.strip()
            or "\x00" in executable
        ):
            raise ValueError("python_executable must be non-empty text")
        self._python_executable = executable

    def validate_and_normalize(self, argv: Sequence[str]) -> tuple[str, ...]:
        """Return an immutable safe command vector or reject it before execution."""

        if isinstance(argv, (str, bytes)) or not argv:
            raise CommandPolicyError(
                INVALID_COMMAND,
                "argv must be a non-empty sequence of argument strings",
            )
        if any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in argv
        ):
            raise CommandPolicyError(
                INVALID_COMMAND,
                "every command argument must be non-empty text without null bytes",
            )

        normalized = tuple(argv)
        program = _program_name(normalized[0])
        if program in _BLOCKED_PROGRAMS or program.startswith("mkfs."):
            raise _rejected(program, "program is blocked")
        if program == "rm" and _contains_recursive_remove(normalized[1:]):
            raise _rejected(program, "recursive removal is blocked")
        if program == "rmdir" and any(
            argument.casefold() == "/s" for argument in normalized[1:]
        ):
            raise _rejected(program, "recursive removal is blocked")
        if program == "find" and "-delete" in normalized[1:]:
            raise _rejected(program, "recursive deletion is blocked")
        if program == "git":
            _reject_destructive_git(normalized[1:])

        if _is_python_command(normalized[0]):
            normalized = (self._python_executable, *normalized[1:])
        return normalized


def _program_name(raw_program: str) -> str:
    name = raw_program.replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_python_command(raw_program: str) -> bool:
    return _PYTHON_COMMAND.fullmatch(_program_name(raw_program)) is not None


def _contains_recursive_remove(arguments: Sequence[str]) -> bool:
    for argument in arguments:
        if argument == "--recursive":
            return True
        if argument.startswith("-") and not argument.startswith("--"):
            if "r" in argument.casefold()[1:]:
                return True
    return False


def _reject_destructive_git(arguments: Sequence[str]) -> None:
    lowered = tuple(argument.casefold() for argument in arguments)
    if "clean" in lowered:
        raise _rejected("git", "git clean is blocked")
    if "reset" in lowered and "--hard" in lowered:
        raise _rejected("git", "git reset --hard is blocked")
    if "checkout" in lowered and "--" in lowered:
        raise _rejected("git", "git checkout -- is blocked")
    if "restore" in lowered:
        raise _rejected("git", "git restore is blocked")


def _rejected(program: str, reason: str) -> CommandPolicyError:
    return CommandPolicyError(
        COMMAND_REJECTED,
        f"Command {program!r} was rejected: {reason}.",
    )
