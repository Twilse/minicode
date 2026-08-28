from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from minicoder.adapters.subprocess_runner import (
    PosixSubprocessAdapter,
    WindowsSubprocessAdapter,
)
from minicoder.application.ports import ProcessPort


def _current_adapter() -> ProcessPort:
    if os.name == "nt":
        return WindowsSubprocessAdapter()
    return PosixSubprocessAdapter()


def test_adapter_runs_in_workspace_and_captures_both_streams(
    tmp_path: Path,
) -> None:
    adapter = _current_adapter()

    result = adapter.run(
        argv=(
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "print(os.getcwd()); "
                "sys.stderr.write('warning\\n')"
            ),
        ),
        cwd=tmp_path,
        timeout_seconds=5.0,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.strip() == str(tmp_path.resolve())
    assert result.stderr == "warning\n"
    assert result.duration_seconds >= 0


def test_adapter_preserves_a_nonzero_exit_code(tmp_path: Path) -> None:
    result = _current_adapter().run(
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        cwd=tmp_path,
        timeout_seconds=5.0,
    )

    assert result.exit_code == 7
    assert result.timed_out is False


def test_adapter_times_out_and_keeps_partial_output(tmp_path: Path) -> None:
    started_at = time.monotonic()

    result = _current_adapter().run(
        argv=(
            sys.executable,
            "-c",
            "import time; print('before timeout', flush=True); time.sleep(10)",
        ),
        cwd=tmp_path,
        timeout_seconds=0.1,
    )

    assert result.timed_out is True
    assert result.exit_code is None
    assert result.stdout == "before timeout\n"
    assert time.monotonic() - started_at < 5.0


def test_adapter_replaces_invalid_utf8_and_normalizes_newlines(
    tmp_path: Path,
) -> None:
    result = _current_adapter().run(
        argv=(
            sys.executable,
            "-c",
            "import os; os.write(1, b'bad\\xff\\r\\nline\\r')",
        ),
        cwd=tmp_path,
        timeout_seconds=5.0,
    )

    assert result.stdout == "bad�\nline\n"


def test_posix_adapter_requests_a_new_session() -> None:
    adapter = PosixSubprocessAdapter()

    assert adapter._platform_options() == {"start_new_session": True}


def test_windows_adapter_requests_a_new_process_group() -> None:
    adapter = WindowsSubprocessAdapter()

    assert adapter._platform_options() == {
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    }
