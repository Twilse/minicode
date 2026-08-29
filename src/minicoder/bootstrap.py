"""Composition root for constructing application objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from openai import OpenAI

from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.adapters.subprocess_runner import (
    PosixSubprocessAdapter,
    WindowsSubprocessAdapter,
)
from minicoder.application.ports import ModelPort, ProcessPort, ToolPort
from minicoder.config import AppConfig
from minicoder.platforms import OperatingSystem, detect_operating_system
from minicoder.tools.command_safety import CommandSafetyPolicy
from minicoder.tools.files import (
    CreateFileTool,
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
)
from minicoder.tools.output import (
    OutputCompactionStrategy,
    StreamAwareOutputCompactor,
    ToolOutputArtifactStore,
)
from minicoder.tools.process import ReadToolOutputTool, RunCommandTool
from minicoder.tools.registry import ToolRegistry
from minicoder.tools.safety import WorkspacePathPolicy


@dataclass(frozen=True, slots=True)
class BootstrapContext:
    """The validated values needed before concrete adapters are assembled."""

    config: AppConfig
    operating_system: OperatingSystem


class ApplicationFactory:
    """Create the application object graph in one visible composition root."""

    @staticmethod
    def create_bootstrap_context(
        *,
        environ: Mapping[str, str] | None = None,
        workspace: str | Path | None = None,
        platform_name: str | None = None,
    ) -> BootstrapContext:
        config = AppConfig.from_environment(environ, workspace=workspace)
        operating_system = detect_operating_system(platform_name)
        return BootstrapContext(
            config=config,
            operating_system=operating_system,
        )

    @staticmethod
    def create_model_adapter(config: AppConfig) -> ModelPort:
        """Create the configured synchronous model adapter without making a request."""

        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.model_timeout_seconds,
            max_retries=0,
        )
        return OpenAICompatibleChatAdapter(
            client=client,
            model=config.model,
        )

    @staticmethod
    def create_process_adapter(operating_system: OperatingSystem) -> ProcessPort:
        """Select the process-tree implementation for the detected platform."""

        if operating_system is OperatingSystem.WINDOWS:
            return WindowsSubprocessAdapter()
        return PosixSubprocessAdapter()

    @staticmethod
    def create_output_compactor() -> OutputCompactionStrategy:
        """Select the default process-output compaction strategy."""

        return StreamAwareOutputCompactor()

    @staticmethod
    def create_tool_registry(
        config: AppConfig,
        *,
        operating_system: OperatingSystem | None = None,
        process_adapter: ProcessPort | None = None,
    ) -> ToolPort:
        """Create the workspace-scoped collection of local coding tools."""

        paths = WorkspacePathPolicy(config.workspace)
        if process_adapter is None:
            selected_os = (
                detect_operating_system()
                if operating_system is None
                else operating_system
            )
            processes = ApplicationFactory.create_process_adapter(selected_os)
        else:
            processes = process_adapter
        artifacts = ToolOutputArtifactStore(
            max_read_chars=config.max_tool_output_chars // 2,
        )
        return ToolRegistry(
            (
                ListFilesTool(paths),
                ReadFileTool(paths, max_chars=config.max_tool_output_chars),
                SearchTextTool(paths),
                CreateFileTool(paths),
                ReplaceTextTool(paths),
                RunCommandTool(
                    processes=processes,
                    policy=CommandSafetyPolicy(),
                    artifacts=artifacts,
                    compactor=ApplicationFactory.create_output_compactor(),
                    workspace=config.workspace,
                    timeout_seconds=config.command_timeout_seconds,
                    max_output_chars=config.max_tool_output_chars,
                ),
                ReadToolOutputTool(
                    artifacts,
                    max_output_chars=config.max_tool_output_chars,
                ),
            )
        )
