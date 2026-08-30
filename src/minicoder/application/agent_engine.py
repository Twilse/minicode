"""Synchronous provider-neutral model/tool loop for one coding task."""

from __future__ import annotations

from minicoder.application.completion import (
    CompletionPolicy,
    EvidenceBasedCompletionPolicy,
    VerificationObservation,
)
from minicoder.application.context import ContextManager
from minicoder.application.event_bus import EventBus
from minicoder.application.ports import ModelPort, ToolPort
from minicoder.application.retry import (
    ExponentialBackoffRetryStrategy,
    RetryAttempt,
    RetryStrategy,
)
from minicoder.domain.errors import DomainValidationError, ModelError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import Message, MessageRole
from minicoder.domain.state import (
    AgentPhase,
    AgentRunResult,
    AgentStateMachine,
    AgentStopReason,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are MiniCoder, a local coding agent. Use the available tools to inspect "
    "and modify only the configured workspace, run relevant checks, and return a "
    "concise final response when the task is complete. After changing files, run "
    "relevant tests, compilation, or static checks before claiming completion."
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
    ) -> None:
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise DomainValidationError("agent max_steps must be a positive integer")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise DomainValidationError("agent system_prompt must be non-blank text")
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

    def run(self, task: str) -> AgentRunResult:
        """Run one fresh task until final text, model failure, or the step limit."""

        if not isinstance(task, str) or not task.strip():
            raise DomainValidationError("agent task must be non-blank text")

        self._completion.reset()
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._system_prompt),
            Message(role=MessageRole.USER, content=task),
        ]
        definitions = tuple(self._tools.definitions())
        state = AgentStateMachine(max_steps=self._max_steps)
        self._events.publish(
            AgentEventKind.TASK_STARTED,
            model_step=0,
            details={
                "task_chars": len(task),
                "max_steps": self._max_steps,
                "tool_count": len(definitions),
            },
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

            state.begin_model_call()
            window = self._context.prepare(messages)
            if window.compacted:
                self._events.publish(
                    AgentEventKind.CONTEXT_COMPACTED,
                    model_step=state.model_steps,
                    details={
                        "budget_chars": window.budget_chars,
                        "original_chars": window.original_chars,
                        "prepared_chars": window.prepared_chars,
                        "omitted_message_count": window.omitted_message_count,
                        "shortened_message_count": window.shortened_message_count,
                        "budget_exceeded": window.budget_exceeded,
                    },
                )
            self._events.publish(
                AgentEventKind.MODEL_REQUESTED,
                model_step=state.model_steps,
                details={
                    "message_count": len(window.messages),
                    "tool_count": len(definitions),
                },
            )
            try:
                turn = self._retries.run(
                    lambda: self._model.complete(
                        messages=window.messages,
                        tools=definitions,
                    ),
                    on_retry=lambda attempt: self._record_model_retry(
                        state,
                        attempt,
                    ),
                )
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

            messages.append(turn.as_message())
            if turn.tool_calls:
                state.begin_tool_execution()
                for call in turn.tool_calls:
                    self._events.publish(
                        AgentEventKind.TOOL_CALLED,
                        model_step=state.model_steps,
                        details={
                            "call_id": call.id,
                            "tool_name": call.name,
                        },
                    )
                    try:
                        result = self._tools.execute(call)
                    except KeyboardInterrupt:
                        self._record_user_interruption(state)
                        raise
                    messages.append(result.as_message())
                    self._events.publish(
                        AgentEventKind.TOOL_FINISHED,
                        model_step=state.model_steps,
                        details={
                            "call_id": result.call_id,
                            "tool_name": result.tool_name,
                            "ok": result.ok,
                            "error_code": result.error_code,
                            "content_chars": len(result.content),
                        },
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
