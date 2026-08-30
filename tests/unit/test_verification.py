from __future__ import annotations

from collections.abc import Sequence

import pytest

from minicoder.application.verification import (
    CommandVerificationClassifier,
    ConfiguredVerificationCommand,
)
from minicoder.domain.models import ToolResult


def _command_result(
    argv: Sequence[str],
    *,
    purpose: str = "general",
    requested_argv: Sequence[str] | None = None,
) -> ToolResult:
    metadata: dict[str, object] = {
        "argv": tuple(argv),
        "purpose": purpose,
        "exit_code": 0,
        "timed_out": False,
    }
    if requested_argv is not None:
        metadata["requested_argv"] = tuple(requested_argv)
    return ToolResult(
        call_id="call-command",
        tool_name="run_command",
        ok=True,
        content="completed",
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("argv", "expected_kind"),
    [
        (("g++", "-std=c++17", "-Wall", "main.cpp", "-o", "main"), "C/C++ compiler"),
        (("/usr/bin/clang++", "-fsyntax-only", "main.cxx"), "C/C++ compiler"),
        (("cl.exe", "/std:c++17", "main.cpp"), "C/C++ compiler"),
        (("gcc", "-fsyntax-only", "main.c"), "C/C++ compiler"),
        (("cmake", "--build", "build"), "cmake build"),
        (("ctest", "--test-dir", "build", "--output-on-failure"), "ctest"),
        (("ninja", "-C", "build"), "ninja"),
        (("make",), "make"),
    ],
)
def test_classifier_recognizes_c_cpp_and_build_verifiers(
    argv: tuple[str, ...],
    expected_kind: str,
) -> None:
    classifier = CommandVerificationClassifier()

    classification = classifier.classify(_command_result(argv))

    assert classification is not None
    assert classification.kind == expected_kind
    assert classification.attempted is True


@pytest.mark.parametrize(
    "argv",
    [
        ("g++", "--version"),
        ("g++", "-E", "main.cpp"),
        ("g++", "-###", "main.cpp"),
        ("cmake", "-S", ".", "-B", "build"),
        ("ctest", "-N"),
        ("ninja", "-t", "targets"),
        ("make", "--dry-run"),
        ("echo", "done"),
    ],
)
def test_classifier_does_not_trust_non_executing_or_arbitrary_commands(
    argv: tuple[str, ...],
) -> None:
    classifier = CommandVerificationClassifier()

    classification = classifier.classify(_command_result(argv))

    assert classification is not None
    assert classification.kind is None
    assert classification.attempted is False


def test_classifier_marks_an_unknown_declared_verifier_as_attempted() -> None:
    classifier = CommandVerificationClassifier()

    classification = classifier.classify(
        _command_result(("zig", "build", "test"), purpose="verification")
    )

    assert classification is not None
    assert classification.kind is None
    assert classification.attempted is True


def test_classifier_matches_configured_command_against_model_requested_argv() -> None:
    classifier = CommandVerificationClassifier(
        (ConfiguredVerificationCommand(("python", "verify.py")),)
    )

    classification = classifier.classify(
        _command_result(
            ("/runtime/python3", "verify.py"),
            requested_argv=("python", "verify.py"),
        )
    )

    assert classifier.configured_command_count == 1
    assert classification is not None
    assert classification.kind == "configured verifier"
    assert classification.attempted is True


def test_classifier_ignores_non_process_tool_results() -> None:
    classifier = CommandVerificationClassifier()
    result = ToolResult(
        call_id="call-read",
        tool_name="read_file",
        ok=True,
        content="contents",
    )

    assert classifier.classify(result) is None
