from __future__ import annotations

import pytest

from minicoder.tools.command_safety import (
    COMMAND_REJECTED,
    INVALID_COMMAND,
    CommandPolicyError,
    CommandSafetyPolicy,
)


def test_policy_allows_normal_commands_and_returns_an_immutable_vector() -> None:
    policy = CommandSafetyPolicy(python_executable="/runtime/python")

    result = policy.validate_and_normalize(["pytest", "-q", "tests/unit"])

    assert result == ("pytest", "-q", "tests/unit")


@pytest.mark.parametrize("program", ["python", "python3", "python3.11", "PYTHON.EXE"])
def test_policy_normalizes_bare_python_commands(program: str) -> None:
    policy = CommandSafetyPolicy(python_executable="/runtime/python")

    result = policy.validate_and_normalize([program, "-m", "pytest"])

    assert result == ("/runtime/python", "-m", "pytest")


def test_policy_normalizes_an_explicit_python_path_for_portability() -> None:
    policy = CommandSafetyPolicy(python_executable="/runtime/python")

    result = policy.validate_and_normalize(["/other/runtime/python", "script.py"])

    assert result == ("/runtime/python", "script.py")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        [""],
        ["pytest", ""],
        ["pytest", "bad\x00argument"],
        "pytest -q",
    ],
)
def test_policy_rejects_malformed_argument_vectors(argv: object) -> None:
    policy = CommandSafetyPolicy()

    with pytest.raises(CommandPolicyError) as captured:
        policy.validate_and_normalize(argv)  # type: ignore[arg-type]

    assert captured.value.error_code == INVALID_COMMAND


@pytest.mark.parametrize(
    "argv",
    [
        ["sudo", "pytest"],
        ["SH.EXE", "-c", "pytest && rm file"],
        ["mkfs.ext4", "/dev/example"],
        ["rm", "-rf", "build"],
        ["rm", "--recursive", "build"],
        ["rmdir", "/S", "build"],
        ["find", ".", "-delete"],
        ["git", "clean", "-fd"],
        ["git", "reset", "--hard"],
        ["git", "checkout", "--", "src"],
        ["git", "restore", "src"],
        ["pkill", "python"],
    ],
)
def test_policy_rejects_obviously_dangerous_commands(argv: list[str]) -> None:
    policy = CommandSafetyPolicy()

    with pytest.raises(CommandPolicyError) as captured:
        policy.validate_and_normalize(argv)

    assert captured.value.error_code == COMMAND_REJECTED


def test_policy_allows_non_recursive_single_file_removal() -> None:
    policy = CommandSafetyPolicy()

    result = policy.validate_and_normalize(["rm", "generated.txt"])

    assert result == ("rm", "generated.txt")
