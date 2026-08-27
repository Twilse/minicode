import pytest

from minicoder.domain.errors import UnsupportedPlatformError
from minicoder.platforms import OperatingSystem, detect_operating_system


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [
        ("darwin", OperatingSystem.MACOS),
        ("linux", OperatingSystem.LINUX),
        ("linux2", OperatingSystem.LINUX),
        ("win32", OperatingSystem.WINDOWS),
        ("cygwin", OperatingSystem.WINDOWS),
        ("msys", OperatingSystem.WINDOWS),
    ],
)
def test_detect_operating_system_maps_python_platforms(
    platform_name: str,
    expected: OperatingSystem,
) -> None:
    assert detect_operating_system(platform_name) is expected


def test_detect_operating_system_rejects_unknown_platform() -> None:
    with pytest.raises(UnsupportedPlatformError, match="unsupported platform"):
        detect_operating_system("plan9")
