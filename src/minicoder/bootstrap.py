"""Composition root for constructing application objects."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Mapping

from openai import OpenAI

from minicoder.adapters.jsonl_memory import JsonlProjectMemoryStore
from minicoder.adapters.jsonl_session import JsonlSessionArchive
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
from minicoder.application.context import ContextManager, ModelContextSummary
from minicoder.application.event_bus import EventBus, EventDeliveryFailure
from minicoder.application.memory import (
    LongTermMemoryMaintainer,
    ModelLongTermMemoryMaintainer,
)
from minicoder.application.ports import (
    EventSinkPort,
    ModelPort,
    ProcessPort,
    ProjectMemoryPort,
    SessionArchivePort,
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
from minicoder.application.session_context import format_recent_session_boundary
from minicoder.domain.errors import (
    MemoryPersistenceError,
    SessionPersistenceError,
)
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import Message
from minicoder.domain.session import (
    ArchivedDialogueTurn,
    ContextCheckpoint,
    RecentSessionContext,
)
from minicoder.domain.state import AgentRunResult
from minicoder.platforms import OperatingSystem, detect_operating_system
from minicoder.tools.command_safety import CommandSafetyPolicy
from minicoder.tools.files import (
    CreateFileTool,
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    WriteFileTool,
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
        memory_maintainer: LongTermMemoryMaintainer | None = None,
        initial_memory: Sequence[ProjectMemoryRecord] = (),
        archive: SessionArchivePort | None = None,
        recent_session_context: RecentSessionContext | None = None,
        dialogue_history: Sequence[ArchivedDialogueTurn] = (),
    ) -> None:
        self._engine = engine
        self._artifacts = artifacts
        self._events = events
        self._memory_store = memory_store
        self._memory_maintainer = memory_maintainer
        self._project_memory = list(initial_memory)
        self._archive = archive
        self._recovery_context = format_recent_session_boundary(
            recent_session_context
        )
        self._history = (
            () if recent_session_context is None else recent_session_context.messages
        )
        self._dialogue_history = tuple(dialogue_history)
        self._turn_index = 0
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
        """Return exact User/Assistant/Tool history without a System message."""

        return self._history

    @property
    def dialogue_history(self) -> tuple[ArchivedDialogueTurn, ...]:
        """Return exact external turns recovered from earlier processes."""

        return self._dialogue_history

    def submit(self, user_message: str) -> AgentRunResult:
        """Run one user turn without closing the shared session resources."""

        if self._closed:
            raise RuntimeError("agent session is already closed")
        self._turn_index += 1
        turn_index = self._turn_index
        self._archive_safely(
            "record_turn_started",
            lambda: self._archive.record_turn_started(
                task=user_message,
                history=self._history,
                turn_index=turn_index,
            )
            if self._archive is not None
            else None,
            model_step=0,
        )
        previous_message_count = len(self._history)
        result = self._engine.run_turn(
            user_message,
            history=self._history,
            project_memory=tuple(self._project_memory),
            reference_context=self._recovery_context,
            turn_index=turn_index,
        )
        self._recovery_context = ""
        self._history = result.messages
        self._archive_safely(
            "record_turn_result",
            lambda: self._archive.record_turn_result(
                task=user_message,
                result=result,
                turn_index=turn_index,
            )
            if self._archive is not None
            else None,
            model_step=result.model_steps,
        )
        if self._memory_maintainer is not None:
            decision = self._memory_maintainer.maintain(
                task=user_message,
                result=result,
                turn_messages=result.messages[previous_message_count:],
                project_memory=tuple(self._project_memory),
                model_step=result.model_steps,
                turn_index=turn_index,
            )
            self._archive_safely(
                "record_maintenance",
                lambda: self._archive.record_maintenance(
                    memory_summary=decision.memory_summary,
                    used_fallback=decision.used_fallback,
                    turn_index=turn_index,
                    model_step=result.model_steps,
                )
                if self._archive is not None
                else None,
                model_step=result.model_steps,
            )
            if decision.memory_summary is not None and self._memory_store is not None:
                record = ProjectMemoryRecord(
                    recorded_at=datetime.now(timezone.utc),
                    summary=decision.memory_summary,
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
                    self._project_memory.append(record)
                    self._events.publish(
                        AgentEventKind.MEMORY_SAVED,
                        model_step=result.model_steps,
                        details={
                            "summary_chars": len(decision.memory_summary),
                        },
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
        """Flush the session archive, then remove temporary tool artifacts."""

        if self._closed:
            return
        self._closed = True
        checkpoint = self._engine.context_checkpoint
        if checkpoint is not None:
            self._archive_safely(
                "record_context_checkpoint",
                lambda: self._archive.record_context_checkpoint(
                    checkpoint=checkpoint,
                    turn_index=self._turn_index,
                    model_step=0,
                )
                if self._archive is not None
                else None,
                model_step=0,
            )
        self._archive_safely(
            "close",
            lambda: self._archive.close()
            if self._archive is not None
            else None,
            model_step=0,
        )
        self._artifacts.close()

    def _archive_safely(
        self,
        operation: str,
        action: Callable[[], object],
        *,
        model_step: int,
    ) -> None:
        if self._archive is None:
            return
        try:
            action()
        except SessionPersistenceError as exc:
            self._events.publish(
                AgentEventKind.SESSION_ARCHIVE_FAILED,
                model_step=model_step,
                details={
                    "operation": operation,
                    "error_type": type(exc).__name__,
                },
            )


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
    def create_context_manager(
        config: AppConfig,
        *,
        model: ModelPort,
        events: EventBus,
        archive: SessionArchivePort | None = None,
        initial_checkpoint: ContextCheckpoint | None = None,
    ) -> ContextManager:
        """Create total-request budgeting with model-based semantic compaction."""

        return ContextManager(
            budget_chars=config.context_budget_chars,
            response_reserve_chars=config.context_response_reserve_chars,
            initial_checkpoint=initial_checkpoint,
            archive=archive,
            events=events,
            summary_strategy=ModelContextSummary(
                model=model,
                events=events,
                archive=archive,
                source_chars=max(
                    1_000,
                    min(
                        120_000,
                        config.context_budget_chars
                        - config.context_response_reserve_chars
                        - 2_000,
                    ),
                ),
                request_input_budget_chars=(
                    config.context_budget_chars
                    - config.context_response_reserve_chars
                ),
            ),
        )

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
    def create_session_archive(config: AppConfig) -> SessionArchivePort:
        """Create the exact private archive for one local process."""

        return JsonlSessionArchive(workspace=config.workspace)

    @staticmethod
    def create_memory_maintainer(
        config: AppConfig,
        *,
        model: ModelPort,
        events: EventBus,
        archive: SessionArchivePort | None,
    ) -> LongTermMemoryMaintainer:
        """Create one selective post-turn long-term-memory decision service."""

        source_budget = max(
            1_024,
            config.context_budget_chars
            - config.context_response_reserve_chars
            - 2_500,
        )
        task_chars = min(2_000, max(128, source_budget // 12))
        outcome_chars = min(4_000, max(256, source_budget // 6))
        transcript_chars = max(
            256,
            min(
                120_000,
                source_budget
                - task_chars
                - outcome_chars,
            ),
        )
        maintenance_payload_chars = max(
            128,
            config.context_response_reserve_chars - 256,
        )
        return ModelLongTermMemoryMaintainer(
            model=model,
            events=events,
            sensitive_values=(config.api_key,),
            task_input_chars=task_chars,
            outcome_input_chars=outcome_chars,
            transcript_input_chars=transcript_chars,
            memory_summary_chars=min(
                1_200,
                max(32, maintenance_payload_chars),
            ),
            request_input_budget_chars=(
                config.context_budget_chars
                - config.context_response_reserve_chars
            ),
            archive=archive,
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
                WriteFileTool(paths),
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
        session_archive: SessionArchivePort | None = None,
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
        active_archive: SessionArchivePort | None = None
        recent_session_context: RecentSessionContext | None = None
        dialogue_history: tuple[ArchivedDialogueTurn, ...] = ()
        if config.session_archive_enabled:
            try:
                active_archive = (
                    ApplicationFactory.create_session_archive(config)
                    if session_archive is None
                    else session_archive
                )
                recent_session_context = active_archive.load_latest_context()
                dialogue_history = tuple(active_archive.load_dialogue_history())
            except SessionPersistenceError as exc:
                events.publish(
                    AgentEventKind.SESSION_ARCHIVE_FAILED,
                    model_step=0,
                    details={
                        "operation": "load_latest_context",
                        "error_type": type(exc).__name__,
                    },
                )
                active_archive = None
            else:
                if recent_session_context is not None:
                    events.publish(
                        AgentEventKind.SESSION_CONTEXT_LOADED,
                        model_step=0,
                        details={
                            "previous_status": recent_session_context.status.value,
                            "previous_stop_reason": (
                                recent_session_context.stop_reason
                            ),
                            "restored_message_count": len(
                                recent_session_context.messages
                            ),
                        },
                    )
        active_memory_store: ProjectMemoryPort | None = None
        initial_memory: tuple[ProjectMemoryRecord, ...] = ()
        if config.memory_enabled:
            try:
                active_memory_store = (
                    ApplicationFactory.create_project_memory_store(config)
                    if memory_store is None
                    else memory_store
                )
                initial_memory = tuple(active_memory_store.load_all())
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
        memory_maintainer: LongTermMemoryMaintainer | None = None
        if config.memory_enabled:
            memory_maintainer = ApplicationFactory.create_memory_maintainer(
                config,
                model=model,
                events=events,
                archive=active_archive,
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
                context=ApplicationFactory.create_context_manager(
                    config,
                    model=model,
                    events=events,
                    archive=active_archive,
                    initial_checkpoint=(
                        None
                        if recent_session_context is None
                        else recent_session_context.context_checkpoint
                    ),
                ),
                retries=ApplicationFactory.create_retry_strategy(),
                completion=ApplicationFactory.create_completion_policy(
                    context.verification_commands,
                ),
                planning_enabled=config.planning_enabled,
                archive=active_archive,
            )
        except Exception:
            artifacts.close()
            raise
        return AgentSession(
            engine=engine,
            artifacts=artifacts,
            events=events,
            memory_store=active_memory_store,
            memory_maintainer=memory_maintainer,
            initial_memory=initial_memory,
            archive=active_archive,
            recent_session_context=recent_session_context,
            dialogue_history=dialogue_history,
        )
