"""Composition root for constructing application objects."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Mapping

from openai import OpenAI

from minicoder.adapters.jsonl_memory import JsonlProjectMemoryStore
from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.adapters.subprocess_runner import (
    PosixSubprocessAdapter,
    WindowsSubprocessAdapter,
)
from minicoder.adapters.verification_config import load_verification_commands
from minicoder.application.agent_engine import AgentEngine
from minicoder.application.completion import (
    CompletionPolicy,
    EvidenceBasedCompletionPolicy,
)
from minicoder.application.context import ContextManager
from minicoder.application.event_bus import EventBus, EventDeliveryFailure
from minicoder.application.memory import MemorySummarizer, ModelMemorySummarizer
from minicoder.application.ports import (
    EventSinkPort,
    ModelPort,
    ProcessPort,
    ProjectMemoryPort,
    ToolPort,
)
from minicoder.application.retry import (
    ExponentialBackoffRetryStrategy,
    RetryStrategy,
)
from minicoder.application.verification import (
    CommandVerificationClassifier,
    ConfiguredVerificationCommand,
)
from minicoder.config import AppConfig
from minicoder.domain.errors import MemoryPersistenceError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import Message
from minicoder.domain.state import AgentPhase, AgentRunResult
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
    verification_commands: tuple[
        ConfiguredVerificationCommand, ...
    ]  # Exact project verifiers frozen at startup.


class AgentSession:
    """Own one multi-turn engine, its history, artifacts, and observer failures."""

    def __init__(
        self,
        *,
        engine: AgentEngine,
        artifacts: ToolOutputArtifactStore,
        events: EventBus,
        memory_store: ProjectMemoryPort | None = None,
        memory_summarizer: MemorySummarizer | None = None,
        initial_memory: Sequence[ProjectMemoryRecord] = (),
    ) -> None:
        if (memory_store is None) != (memory_summarizer is None):
            raise ValueError(
                "memory store and summarizer must be configured together"
            )
        self._engine = engine
        self._artifacts = artifacts
        self._events = events
        self._memory_store = memory_store
        self._memory_summarizer = memory_summarizer
        self._initial_memory = tuple(initial_memory)
        self._history: tuple[Message, ...] = ()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def event_failures(self) -> tuple[EventDeliveryFailure, ...]:
        """Expose non-fatal observer errors after or during the task."""

        return self._events.failures

    @property
    def history(self) -> tuple[Message, ...]:
        """Return the immutable full conversation accumulated so far."""

        return self._history

    def submit(self, user_message: str) -> AgentRunResult:
        """Run one user turn without closing the shared session resources."""

        if self._closed:
            raise RuntimeError("agent session is already closed")
        result = self._engine.run_turn(
            user_message,
            history=self._history,
            project_memory=self._initial_memory if not self._history else (),
        )
        self._history = result.messages
        if (
            result.phase is AgentPhase.COMPLETE
            and result.final_response is not None
            and self._memory_store is not None
            and self._memory_summarizer is not None
        ):
            summary = self._memory_summarizer.summarize(
                task=user_message,
                outcome=result.final_response,
                model_step=result.model_steps,
            )
            record = ProjectMemoryRecord(
                recorded_at=datetime.now(timezone.utc),
                summary=summary,
            )
            try:
                self._memory_store.append(record)
            except MemoryPersistenceError as exc:
                self._events.publish(
                    AgentEventKind.MEMORY_OPERATION_FAILED,
                    model_step=result.model_steps,
                    details={
                        "operation": "append",
                        "error_type": type(exc).__name__,
                    },
                )
            else:
                self._events.publish(
                    AgentEventKind.MEMORY_SAVED,
                    model_step=result.model_steps,
                    details={"summary_chars": len(summary)},
                )
        return result

    def run(self, task: str) -> AgentRunResult:
        """Run one turn and always release resources for one-shot callers."""

        try:
            return self.submit(task)
        finally:
            self.close()

    def __enter__(self) -> AgentSession:
        if self._closed:
            raise RuntimeError("agent session is already closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Idempotently invalidate output IDs and remove temporary files."""

        if self._closed:
            return
        self._closed = True
        self._artifacts.close()


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
        verification_commands = load_verification_commands(config.workspace)
        return BootstrapContext(
            config=config,
            operating_system=operating_system,
            verification_commands=verification_commands,
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
    def create_context_manager(config: AppConfig) -> ContextManager:
        """Create deterministic context budgeting from provider-neutral config."""

        return ContextManager(budget_chars=config.context_budget_chars)

    @staticmethod
    def create_completion_policy(
        configured_commands: Sequence[ConfiguredVerificationCommand] = (),
    ) -> CompletionPolicy:
        """Create the host-side evidence gate for final model responses."""

        return EvidenceBasedCompletionPolicy(
            verification=CommandVerificationClassifier(configured_commands),
        )

    @staticmethod
    def create_retry_strategy() -> RetryStrategy:
        """Create the single visible policy for transient model failures."""

        return ExponentialBackoffRetryStrategy(
            max_retries=2,
            initial_delay_seconds=0.5,
            multiplier=2.0,
        )

    @staticmethod
    def create_project_memory_store(config: AppConfig) -> ProjectMemoryPort:
        """Create private JSONL persistence bound to the canonical workspace."""

        return JsonlProjectMemoryStore(
            workspace=config.workspace,
            sensitive_values=(config.api_key,),
        )

    @staticmethod
    def create_memory_summarizer(
        config: AppConfig,
        *,
        model: ModelPort,
        events: EventBus,
    ) -> MemorySummarizer:
        """Create the one-shot no-tool model memory summarizer."""

        return ModelMemorySummarizer(
            model=model,
            events=events,
            sensitive_values=(config.api_key,),
        )

    @staticmethod
    def create_tool_registry(
        config: AppConfig,
        *,
        processes: ProcessPort,
        artifacts: ToolOutputArtifactStore,
    ) -> ToolPort:
        """Create the workspace-scoped collection of local coding tools."""

        paths = WorkspacePathPolicy(config.workspace)
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

    @staticmethod
    def create_agent_session(
        context: BootstrapContext,
        *,
        model_adapter: ModelPort | None = None,
        process_adapter: ProcessPort | None = None,
        event_sinks: Sequence[EventSinkPort] = (),
        memory_store: ProjectMemoryPort | None = None,
    ) -> AgentSession:
        """Assemble one task session while retaining ownership of its resources."""

        config = context.config
        model = (
            ApplicationFactory.create_model_adapter(config)
            if model_adapter is None
            else model_adapter
        )
        processes = (
            ApplicationFactory.create_process_adapter(context.operating_system)
            if process_adapter is None
            else process_adapter
        )
        events = EventBus(event_sinks)
        active_memory_store: ProjectMemoryPort | None = None
        memory_summarizer: MemorySummarizer | None = None
        initial_memory: tuple[ProjectMemoryRecord, ...] = ()
        if config.memory_enabled:
            try:
                active_memory_store = (
                    ApplicationFactory.create_project_memory_store(config)
                    if memory_store is None
                    else memory_store
                )
                initial_memory = tuple(active_memory_store.load_recent())
            except MemoryPersistenceError as exc:
                events.publish(
                    AgentEventKind.MEMORY_OPERATION_FAILED,
                    model_step=0,
                    details={
                        "operation": "load",
                        "error_type": type(exc).__name__,
                    },
                )
                active_memory_store = None
            else:
                if initial_memory:
                    events.publish(
                        AgentEventKind.MEMORY_LOADED,
                        model_step=0,
                        details={"record_count": len(initial_memory)},
                    )
                memory_summarizer = ApplicationFactory.create_memory_summarizer(
                    config,
                    model=model,
                    events=events,
                )
        artifacts = ToolOutputArtifactStore(
            max_read_chars=config.max_tool_output_chars // 2,
        )
        try:
            tools = ApplicationFactory.create_tool_registry(
                config,
                processes=processes,
                artifacts=artifacts,
            )
            engine = AgentEngine(
                model=model,
                tools=tools,
                max_steps=config.max_steps,
                events=events,
                context=ApplicationFactory.create_context_manager(config),
                retries=ApplicationFactory.create_retry_strategy(),
                completion=ApplicationFactory.create_completion_policy(
                    context.verification_commands,
                ),
                planning_enabled=config.planning_enabled,
            )
        except Exception:
            artifacts.close()
            raise
        return AgentSession(
            engine=engine,
            artifacts=artifacts,
            events=events,
            memory_store=active_memory_store,
            memory_summarizer=memory_summarizer,
            initial_memory=initial_memory,
        )
