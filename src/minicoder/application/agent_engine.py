"""Synchronous provider-neutral model/tool loop for one coding task."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from minicoder.application.completion import (
    CompletionPolicy,
    EvidenceBasedCompletionPolicy,
    VerificationObservation,
)
from minicoder.application.context import ContextManager
from minicoder.application.event_bus import EventBus
from minicoder.application.model_protocol import decode_assistant_turn
from minicoder.application.ports import ModelPort, SessionArchivePort, ToolPort
from minicoder.application.progress import (
    PlanProgress,
    PlanTransition,
    PlanStepUpdate,
    tool_display_details,
)
from minicoder.application.retry import (
    ExponentialBackoffRetryStrategy,
    RetryAttempt,
    RetryStrategy,
)
from minicoder.domain.errors import (
    DomainValidationError,
    ModelError,
    SessionPersistenceError,
)
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import (
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from minicoder.domain.state import (
    AgentPhase,
    AgentRunResult,
    AgentStateMachine,
    AgentStopReason,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are MiniCoder. Stay in workspace; inspect first. Memory may be stale; the "
    "current request and safety win. Follow the plan unless evidence changes it. "
    "After edits, run a recognized test or compiler with purpose='verification'; "
    "direct application runs are general. Reply only when complete."
)

_PROJECT_MEMORY_CONTEXT_CHARS = 6_000
_PLANNING_REQUIREMENT = (
    "[Host planning requirement]\n"
    "Before any tool use, return only a concise numbered action plan proportional "
    "to the current request. Use exactly 1 step for a direct answer that needs no "
    "tools, 2 to 3 steps for read-only inspection, and 3 to 7 steps for code changes. "
    "Describe actions, not an outline of the final answer. Base the plan on available "
    "conversation and project memory. Include inspection before editing and relevant "
    "verification after changes. Every item in a tool-using plan must correspond to "
    "observable tool work; do not add hidden thinking-only or design-only items. Do "
    "not execute the task, call tools, include answer facts, or claim completion in "
    "this response."
)
_EXECUTION_REQUIREMENT = (
    "[Host execution requirement]\n"
    "Now execute the plan above as the default execution contract. Do not ignore it "
    "without evidence. If file contents, tool results, errors, or safety rules "
    "invalidate a step, adapt the remaining steps while preserving the current user "
    "goal. When calling tools, put [plan_step=N] in the assistant content to identify "
    "the current numbered item. This is a host annotation: use it only in a response "
    "that calls tools, never in the final answer. Complete each numbered item before "
    "calling tools for the next item; the host rejects out-of-order tool calls. Group "
    "multiple safe exact edits to one file in replace_text.replacements to reduce "
    "model round trips. Do not skip required verification."
)


class AgentEngine:
    """Drive sequential model turns and execute every requested tool call."""

    def __init__(
        self,
        *,
        model: ModelPort,
        tools: ToolPort,
        max_steps: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        events: EventBus | None = None,
        context: ContextManager | None = None,
        retries: RetryStrategy | None = None,
        completion: CompletionPolicy | None = None,
        planning_enabled: bool = False,
        archive: SessionArchivePort | None = None,
    ) -> None:
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise DomainValidationError("agent max_steps must be a positive integer")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise DomainValidationError("agent system_prompt must be non-blank text")
        if not isinstance(planning_enabled, bool):
            raise DomainValidationError("agent planning_enabled must be a boolean")
        self._model = model
        self._tools = tools
        self._max_steps = max_steps
        self._system_prompt = system_prompt
        self._events = EventBus() if events is None else events
        self._context = (
            ContextManager(budget_chars=60_000) if context is None else context
        )
        self._retries = (
            ExponentialBackoffRetryStrategy(max_retries=0)
            if retries is None
            else retries
        )
        self._completion = (
            EvidenceBasedCompletionPolicy() if completion is None else completion
        )
        self._planning_enabled = planning_enabled
        self._archive = archive

    def run(self, task: str) -> AgentRunResult:
        """Run one fresh task until final text, model failure, or the step limit."""

        self._completion.reset()
        return self._run_turn(task, history=())

    def run_turn(
        self,
        user_message: str,
        *,
        history: Sequence[Message] = (),
        project_memory: Sequence[ProjectMemoryRecord] = (),
        reference_context: str = "",
        turn_index: int = 1,
    ) -> AgentRunResult:
        """Run one user turn while preserving an existing conversation history."""

        history_snapshot = tuple(history)
        if not history_snapshot:
            self._completion.reset()
        memory_snapshot = tuple(project_memory)
        if any(
            not isinstance(record, ProjectMemoryRecord)
            for record in memory_snapshot
        ):
            raise DomainValidationError(
                "project memory must contain ProjectMemoryRecord values"
            )
        if not isinstance(reference_context, str):
            raise DomainValidationError("agent reference_context must be text")
        if (
            not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or turn_index <= 0
        ):
            raise DomainValidationError("agent turn_index must be a positive integer")
        return self._run_turn(
            user_message,
            history=history_snapshot,
            project_memory=memory_snapshot,
            reference_context=reference_context,
            turn_index=turn_index,
        )

    def _run_turn(
        self,
        user_message: str,
        *,
        history: tuple[Message, ...],
        project_memory: tuple[ProjectMemoryRecord, ...] = (),
        reference_context: str = "",
        turn_index: int = 1,
    ) -> AgentRunResult:
        if not isinstance(user_message, str) or not user_message.strip():
            raise DomainValidationError("agent task must be non-blank text")

        messages = list(self._validated_history(history))
        current_user_index = len(messages)
        definitions = tuple(self._tools.definitions())
        current_user_content = user_message
        if self._planning_enabled:
            current_user_content = (
                f"{current_user_content}\n\n"
                f"{_planning_requirement(definitions)}"
            )
        messages.append(
            Message(
                role=MessageRole.USER,
                content=current_user_content,
            )
        )
        model_reference_context = _model_reference_context(
            reference_context,
            project_memory,
        )
        state = AgentStateMachine(max_steps=self._max_steps)
        self._events.publish(
            AgentEventKind.TASK_STARTED,
            model_step=0,
            details={
                "task_chars": len(user_message),
                "history_message_count": len(history),
                "max_steps": self._max_steps,
                "tool_count": len(definitions),
                "planning_enabled": self._planning_enabled,
            },
        )
        planning_pending = self._planning_enabled
        plan_progress: PlanProgress | None = None
        if planning_pending:
            self._events.publish(
                AgentEventKind.PLANNING_STARTED,
                model_step=0,
                details={"history_message_count": len(history)},
            )

        while True:
            if not state.can_call_model:
                state.fail()
                failure_message = (
                    f"Agent reached the maximum of {self._max_steps} model steps "
                    "before returning a final response."
                )
                unfinished = self._completion.unfinished_summary()
                if unfinished is not None:
                    failure_message = f"{failure_message} {unfinished}"
                self._events.publish(
                    AgentEventKind.TASK_FAILED,
                    model_step=state.model_steps,
                    details={
                        "reason": AgentStopReason.MAX_STEPS.value,
                        "message": failure_message,
                    },
                )
                return AgentRunResult(
                    phase=state.phase,
                    stop_reason=AgentStopReason.MAX_STEPS,
                    model_steps=state.model_steps,
                    messages=tuple(messages),
                    failure_message=failure_message,
                )

            next_model_step = state.model_steps + 1
            advertised_definitions = () if planning_pending else definitions
            window = self._context.prepare(
                messages,
                current_user_index=current_user_index,
                reference_context=model_reference_context,
                tools=advertised_definitions,
                turn_index=turn_index,
                model_step=next_model_step,
            )
            if window.compacted:
                self._events.publish(
                    AgentEventKind.CONTEXT_COMPACTED,
                    model_step=next_model_step,
                    details={
                        "budget_chars": window.budget_chars,
                        "original_chars": window.original_chars,
                        "prepared_chars": window.prepared_chars,
                        "omitted_message_count": window.omitted_message_count,
                        "shortened_message_count": window.shortened_message_count,
                        "budget_exceeded": window.budget_exceeded,
                        "tool_definition_chars": window.tool_definition_chars,
                        "response_reserve_chars": window.response_reserve_chars,
                    },
                )
            if window.budget_exceeded:
                state.fail()
                failure_message = (
                    "Model request was not sent because permanent instructions, "
                    "pinned recent/project memory, the current user input, current "
                    "tool definitions, and response reserve require approximately "
                    f"{window.request_chars} characters, above the configured "
                    f"{window.budget_chars}."
                )
                self._events.publish(
                    AgentEventKind.TASK_FAILED,
                    model_step=state.model_steps,
                    details={
                        "reason": (
                            AgentStopReason.CONTEXT_BUDGET_EXCEEDED.value
                        ),
                        "message": failure_message,
                    },
                )
                return AgentRunResult(
                    phase=state.phase,
                    stop_reason=AgentStopReason.CONTEXT_BUDGET_EXCEEDED,
                    model_steps=state.model_steps,
                    messages=tuple(messages),
                    failure_message=failure_message,
                )
            state.begin_model_call()
            request_kind = "planning" if planning_pending else "execution"
            self._archive_safely(
                "record_model_request",
                lambda: self._archive.record_model_request(
                    messages=window.messages,
                    tools=advertised_definitions,
                    request_kind=request_kind,
                    turn_index=turn_index,
                    model_step=state.model_steps,
                )
                if self._archive is not None
                else None,
                model_step=state.model_steps,
            )
            self._events.publish(
                AgentEventKind.MODEL_REQUESTED,
                model_step=state.model_steps,
                details={
                    "message_count": len(window.messages),
                    "tool_count": len(advertised_definitions),
                    "request_kind": request_kind,
                },
            )
            try:
                raw_turn = self._retries.run(
                    lambda: self._model.complete(
                        messages=window.messages,
                        tools=advertised_definitions,
                    ),
                    on_retry=lambda attempt: self._record_model_retry(
                        state,
                        attempt,
                    ),
                )
                self._archive_safely(
                    "record_model_response",
                    lambda: self._archive.record_model_response(
                        turn=raw_turn,
                        request_kind=request_kind,
                        turn_index=turn_index,
                        model_step=state.model_steps,
                    )
                    if self._archive is not None
                    else None,
                    model_step=state.model_steps,
                )
                decoded_turn = decode_assistant_turn(raw_turn)
                turn = decoded_turn.turn
                annotated_plan_step = decoded_turn.plan_step
            except KeyboardInterrupt:
                self._record_user_interruption(state)
                raise
            except ModelError as exc:
                state.fail()
                failure_message = f"Model request failed: {exc}"
                self._events.publish(
                    AgentEventKind.TASK_FAILED,
                    model_step=state.model_steps,
                    details={
                        "reason": AgentStopReason.MODEL_ERROR.value,
                        "message": failure_message,
                    },
                )
                return AgentRunResult(
                    phase=state.phase,
                    stop_reason=AgentStopReason.MODEL_ERROR,
                    model_steps=state.model_steps,
                    messages=tuple(messages),
                    failure_message=failure_message,
                )

            if planning_pending:
                if turn.tool_calls:
                    state.fail()
                    failure_message = (
                        "Planning response requested tools even though no tools were "
                        "available; no tool was executed."
                    )
                    self._events.publish(
                        AgentEventKind.TASK_FAILED,
                        model_step=state.model_steps,
                        details={
                            "reason": AgentStopReason.PLANNING_ERROR.value,
                            "message": failure_message,
                        },
                    )
                    return AgentRunResult(
                        phase=state.phase,
                        stop_reason=AgentStopReason.PLANNING_ERROR,
                        model_steps=state.model_steps,
                        messages=tuple(messages),
                        failure_message=failure_message,
                    )
                messages.append(turn.as_message())
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=_EXECUTION_REQUIREMENT,
                    )
                )
                state.plan_ready()
                planning_pending = False
                plan_progress = PlanProgress.from_model_text(turn.content or "")
                self._events.publish(
                    AgentEventKind.PLANNING_COMPLETED,
                    model_step=state.model_steps,
                    details={
                        "plan_chars": len(turn.content or ""),
                        "plan_item_count": plan_progress.total,
                        "display_plan": plan_progress.display_text,
                    },
                )
                self._record_plan_update(
                    plan_progress.begin(),
                    model_step=state.model_steps,
                )
                continue

            messages.append(turn.as_message())
            if turn.tool_calls:
                state.begin_tool_execution()
                for call in turn.tool_calls:
                    if plan_progress is not None:
                        plan_transition = plan_progress.advance_for_tool(
                            call,
                            explicit_step=annotated_plan_step,
                        )
                        if plan_transition.blocked:
                            result = self._reject_out_of_order_tool(
                                call,
                                plan_transition,
                                model_step=state.model_steps,
                            )
                            messages.append(result.as_message())
                            self._archive_tool_result(
                                call,
                                result,
                                turn_index=turn_index,
                                model_step=state.model_steps,
                            )
                            continue
                        self._record_plan_transition(
                            plan_transition,
                            model_step=state.model_steps,
                        )
                    self._events.publish(
                        AgentEventKind.TOOL_CALLED,
                        model_step=state.model_steps,
                        details={
                            "call_id": call.id,
                            "tool_name": call.name,
                            **tool_display_details(call),
                        },
                    )
                    try:
                        result = self._tools.execute(call)
                    except KeyboardInterrupt:
                        self._record_user_interruption(state)
                        raise
                    messages.append(result.as_message())
                    self._archive_tool_result(
                        call,
                        result,
                        turn_index=turn_index,
                        model_step=state.model_steps,
                    )
                    finished_details = {
                        "call_id": result.call_id,
                        "tool_name": result.tool_name,
                        "ok": result.ok,
                        "error_code": result.error_code,
                        "content_chars": len(result.content),
                    }
                    exit_code = result.metadata.get("exit_code")
                    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                        finished_details["exit_code"] = exit_code
                    timed_out = result.metadata.get("timed_out")
                    if isinstance(timed_out, bool):
                        finished_details["timed_out"] = timed_out
                    self._events.publish(
                        AgentEventKind.TOOL_FINISHED,
                        model_step=state.model_steps,
                        details=finished_details,
                    )
                    observation = self._completion.observe_tool(
                        call,
                        result,
                        model_step=state.model_steps,
                    )
                    if observation is not None and observation.passed:
                        self._record_verification_passed(observation)
                continue

            decision = self._completion.evaluate()
            if not decision.accepted:
                feedback = decision.feedback
                if feedback is None:
                    raise DomainValidationError(
                        "rejected completion decisions require feedback"
                    )
                if decision.terminal:
                    state.fail()
                    self._events.publish(
                        AgentEventKind.TASK_FAILED,
                        model_step=state.model_steps,
                        details={
                            "reason": AgentStopReason.VERIFICATION_UNSUPPORTED.value,
                            "message": feedback,
                        },
                    )
                    return AgentRunResult(
                        phase=AgentPhase.FAILED,
                        stop_reason=AgentStopReason.VERIFICATION_UNSUPPORTED,
                        model_steps=state.model_steps,
                        messages=tuple(messages),
                        failure_message=feedback,
                    )
                messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=feedback,
                    )
                )
                state.require_revision()
                self._events.publish(
                    AgentEventKind.COMPLETION_REJECTED,
                    model_step=state.model_steps,
                    details={
                        "reason": decision.reason.value,
                        "modified_file_count": len(self._completion.modified_files),
                        "response_chars": len(turn.content or ""),
                    },
                )
                continue

            state.complete()
            if plan_progress is not None:
                self._record_plan_transition(
                    plan_progress.finish(),
                    model_step=state.model_steps,
                )
                self._events.publish(
                    AgentEventKind.PLAN_COMPLETED,
                    model_step=state.model_steps,
                    details={
                        "plan_item_count": plan_progress.total,
                        "untracked_plan_item_count": (
                            plan_progress.untracked_count
                        ),
                    },
                )
            self._events.publish(
                AgentEventKind.TASK_COMPLETED,
                model_step=state.model_steps,
                details={"response_chars": len(turn.content or "")},
            )
            return AgentRunResult(
                phase=AgentPhase.COMPLETE,
                stop_reason=AgentStopReason.FINAL_RESPONSE,
                model_steps=state.model_steps,
                messages=tuple(messages),
                final_response=turn.content,
            )

    def _archive_tool_result(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._archive_safely(
            "record_tool_result",
            lambda: self._archive.record_tool_result(
                call=call,
                result=result,
                turn_index=turn_index,
                model_step=model_step,
            )
            if self._archive is not None
            else None,
            model_step=model_step,
        )

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

    def _validated_history(
        self,
        history: tuple[Message, ...],
    ) -> tuple[Message, ...]:
        if not history:
            return (
                Message(role=MessageRole.SYSTEM, content=self._system_prompt),
            )
        if any(not isinstance(message, Message) for message in history):
            raise DomainValidationError("agent history must contain Message values")
        first = history[0]
        if (
            first.role is not MessageRole.SYSTEM
            or first.content != self._system_prompt
        ):
            raise DomainValidationError(
                "agent history must start with this engine's system prompt"
            )
        if any(message.role is MessageRole.SYSTEM for message in history[1:]):
            raise DomainValidationError(
                "agent history may contain only one system message"
            )
        return history

    def _record_user_interruption(self, state: AgentStateMachine) -> None:
        state.fail()
        self._events.publish(
            AgentEventKind.TASK_FAILED,
            model_step=state.model_steps,
            details={
                "reason": AgentStopReason.USER_INTERRUPTED.value,
                "message": "Agent task was interrupted by the user.",
            },
        )

    def _record_model_retry(
        self,
        state: AgentStateMachine,
        attempt: RetryAttempt,
    ) -> None:
        self._events.publish(
            AgentEventKind.MODEL_RETRY_SCHEDULED,
            model_step=state.model_steps,
            details={
                "retry_number": attempt.retry_number,
                "delay_seconds": attempt.delay_seconds,
                "error_type": attempt.error_type,
            },
        )

    def _record_verification_passed(
        self,
        observation: VerificationObservation,
    ) -> None:
        self._events.publish(
            AgentEventKind.VERIFICATION_PASSED,
            model_step=observation.model_step,
            details={"verification_kind": observation.kind},
        )

    def _record_plan_update(
        self,
        update: PlanStepUpdate,
        *,
        model_step: int,
    ) -> None:
        kind = (
            AgentEventKind.PLAN_STEP_COMPLETED
            if update.completed
            else AgentEventKind.PLAN_STEP_STARTED
        )
        step = update.step
        self._events.publish(
            kind,
            model_step=model_step,
            details={
                "plan_step": step.index,
                "plan_item_count": step.total,
                "display_plan_step": step.text,
            },
        )

    def _record_plan_transition(
        self,
        transition: PlanTransition,
        *,
        model_step: int,
    ) -> None:
        completed = tuple(
            update for update in transition.updates if update.completed
        )
        started = tuple(
            update for update in transition.updates if not update.completed
        )
        for update in completed:
            self._record_plan_update(update, model_step=model_step)
        if transition.untracked:
            self._events.publish(
                AgentEventKind.PLAN_STEPS_UNTRACKED,
                model_step=model_step,
                details={
                    "first_plan_step": transition.untracked[0].index,
                    "last_plan_step": transition.untracked[-1].index,
                    "untracked_plan_item_count": len(transition.untracked),
                    "plan_item_count": transition.untracked[0].total,
                },
            )
        for update in started:
            self._record_plan_update(update, model_step=model_step)

    def _reject_out_of_order_tool(
        self,
        call: ToolCall,
        transition: PlanTransition,
        *,
        model_step: int,
    ) -> ToolResult:
        expected = transition.expected
        attempted = transition.attempted
        if expected is None or attempted is None:
            raise DomainValidationError(
                "blocked plan transitions require expected and attempted items"
            )
        details = {
            "call_id": call.id,
            "tool_name": call.name,
            "expected_plan_step": expected.index,
            "attempted_plan_step": attempted.index,
            "plan_item_count": expected.total,
            **tool_display_details(call),
        }
        self._events.publish(
            AgentEventKind.PLAN_TOOL_REJECTED,
            model_step=model_step,
            details=details,
        )
        return ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_code="PLAN_STEP_OUT_OF_ORDER",
            content=(
                "PLAN_STEP_OUT_OF_ORDER: this tool call appears to belong to "
                f"plan step {attempted.index}, but plan step {expected.index} "
                "must receive its required tool-visible work first. Follow the "
                "numbered plan in order, then retry this operation."
            ),
            metadata={
                "expected_plan_step": expected.index,
                "attempted_plan_step": attempted.index,
            },
        )


def _model_reference_context(
    session_context: str,
    records: tuple[ProjectMemoryRecord, ...],
) -> str:
    sections: list[str] = []
    if session_context.strip():
        sections.append(
            "[Recent cross-process and current-session context]\n"
            f"{session_context.strip()}"
        )
    memory_text = _project_memory_text(records)
    if memory_text:
        sections.append(f"[Durable project memory]\n{memory_text}")
    return "\n\n".join(sections)


def _project_memory_text(records: tuple[ProjectMemoryRecord, ...]) -> str:
    if not records:
        return ""
    available = _PROJECT_MEMORY_CONTEXT_CHARS
    selected: list[str] = []
    used_chars = 0
    for record in reversed(records):
        date = record.recorded_at.date().isoformat()
        entry = f"- [{date}] {record.summary.strip()}"
        separator_chars = 1 if selected else 0
        if used_chars + separator_chars + len(entry) > available:
            if not selected and available > 0:
                selected.append(entry[:available])
            break
        selected.append(entry)
        used_chars += separator_chars + len(entry)
    selected.reverse()
    return "\n".join(selected)


def _planning_requirement(definitions: tuple[ToolDefinition, ...]) -> str:
    catalog_lines = [
        "[Current capability catalog — planning only; tools are not callable yet]"
    ]
    remaining = 3_000 - len(catalog_lines[0])
    for definition in definitions:
        description = " ".join(definition.description.split())
        line = f"- {definition.name}: {description[:180]}"
        if len(line) + 1 > remaining:
            break
        catalog_lines.append(line)
        remaining -= len(line) + 1
    return f"{_PLANNING_REQUIREMENT}\n\n" + "\n".join(catalog_lines)
