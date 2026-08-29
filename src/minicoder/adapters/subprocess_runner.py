"""Synchronous subprocess adapters with platform-specific tree termination."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from minicoder.domain.models import ProcessResult


class _SubprocessAdapter:
    """Template for shared launch/capture logic and platform termination hooks."""

    def run(
        self,
        *,
        argv: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
    ) -> ProcessResult:
        command = tuple(argv)
        started_at = time.monotonic()
        process: subprocess.Popen[bytes] = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **self._platform_options(),
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = process.communicate(
                timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process_tree(process)
            stdout_bytes, stderr_bytes = process.communicate()
        except KeyboardInterrupt:
            try:
                self._terminate_process_tree(process)
                process.communicate()
            except Exception:
                pass
            raise

        duration = max(0.0, time.monotonic() - started_at)
        return ProcessResult(
            stdout=_decode_output(stdout_bytes),
            stderr=_decode_output(stderr_bytes),
            exit_code=None if timed_out else process.returncode,
            timed_out=timed_out,
            duration_seconds=duration,
        )

    def _platform_options(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def _terminate_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        raise NotImplementedError


class PosixSubprocessAdapter(_SubprocessAdapter):
    """Run commands in a new POSIX session so the whole group can be killed."""

    def _platform_options(self) -> Mapping[str, Any]:
        return {"start_new_session": True}

    def _terminate_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()


class WindowsSubprocessAdapter(_SubprocessAdapter):
    """Run commands in a Windows process group and terminate its descendants."""

    def _platform_options(self) -> Mapping[str, Any]:
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flag}

    def _terminate_process_tree(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()


def _decode_output(value: bytes | None) -> str:
    if not value:
        return ""
    decoded = value.decode("utf-8", errors="replace")
    return decoded.replace("\r\n", "\n").replace("\r", "\n")
