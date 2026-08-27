"""Cross-platform detection kept separate from process execution details."""

from __future__ import annotations

import sys
from enum import Enum

from minicoder.domain.errors import UnsupportedPlatformError


class OperatingSystem(str, Enum):
    """Operating systems with an explicitly supported adapter plan."""

    MACOS = "macos"
    LINUX = "linux"
    WINDOWS = "windows"


def detect_operating_system(platform_name: str | None = None) -> OperatingSystem:
    """Map Python's platform identifier to MiniCoder's stable internal value."""

    value = (sys.platform if platform_name is None else platform_name).lower()
    if value == "darwin":
        return OperatingSystem.MACOS
    if value.startswith("linux"):
        return OperatingSystem.LINUX
    if value.startswith(("win32", "cygwin", "msys")):
        return OperatingSystem.WINDOWS
    raise UnsupportedPlatformError(f"unsupported platform: {value}")
