from __future__ import annotations

import json
from pathlib import Path

from minicoder.bootstrap import ApplicationFactory
from minicoder.config import AppConfig
from minicoder.domain.models import ToolCall
from minicoder.platforms import detect_operating_system
from minicoder.tools.output import ToolOutputArtifactStore


def test_factory_registry_runs_a_real_local_python_command(tmp_path: Path) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
            "MINICODER_COMMAND_TIMEOUT_SECONDS": "5",
            "MINICODER_MAX_TOOL_OUTPUT_CHARS": "800",
        },
        workspace=tmp_path,
    )
    artifacts = ToolOutputArtifactStore(
        max_read_chars=config.max_tool_output_chars // 2,
    )
    try:
        tools = ApplicationFactory.create_tool_registry(
            config,
            processes=ApplicationFactory.create_process_adapter(
                detect_operating_system()
            ),
            artifacts=artifacts,
        )
        result = tools.execute(
            ToolCall(
                id="call-process",
                name="run_command",
                arguments_json=json.dumps(
                    {
                        "argv": [
                            "python",
                            "-c",
                            "from pathlib import Path; print(Path.cwd().name)",
                        ]
                    }
                ),
            )
        )
    finally:
        artifacts.close()

    assert result.ok is True
    assert result.metadata["exit_code"] == 0
    assert result.metadata["argv"][0] != "python"
    assert result.content.endswith(f"{tmp_path.name}\n")
