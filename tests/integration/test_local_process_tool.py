from __future__ import annotations

import json
from pathlib import Path

from minicoder.bootstrap import ApplicationFactory
from minicoder.config import AppConfig
from minicoder.domain.models import ToolCall


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
    tools = ApplicationFactory.create_tool_registry(config)

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

    assert result.ok is True
    assert result.metadata["exit_code"] == 0
    assert result.metadata["argv"][0] != "python"
    assert result.content.endswith(f"{tmp_path.name}\n")
