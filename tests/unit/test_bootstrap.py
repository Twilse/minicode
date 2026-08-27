from pathlib import Path

from minicoder.bootstrap import ApplicationFactory
from minicoder.platforms import OperatingSystem


def test_factory_creates_validated_bootstrap_context(tmp_path: Path) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={"DEEPSEEK_API_KEY": "key"},
        workspace=tmp_path,
        platform_name="win32",
    )

    assert context.config.workspace == tmp_path.resolve()
    assert context.operating_system is OperatingSystem.WINDOWS
