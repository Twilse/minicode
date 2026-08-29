"""Synchronous provider-neutral model/tool loop for one coding task."""

from __future__ import annotations

from minicoder.application.event_bus import EventBus
from minicoder.application.ports import ModelPort, ToolPort
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
    "concise final response when the task is complete."
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

    def run(self, task: str) -> AgentRunResult:
        """Run one fresh task until final text, model failure, or the step limit."""

        if not isinstance(task, str) or not task.strip():
            raise DomainValidationError("agent task must be non-blank text")

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
            self._events.publish(
                AgentEventKind.MODEL_REQUESTED,
                model_step=state.model_steps,
                details={
                    "message_count": len(messages),
                    "tool_count": len(definitions),
                },
            )
            try:
                turn = self._model.complete(
                    messages=tuple(messages),
                    tools=definitions,
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
