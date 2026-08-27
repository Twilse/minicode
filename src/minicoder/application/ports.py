"""Ports owned by the application core and implemented by outer adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from minicoder.domain.models import AssistantTurn, Message, ToolDefinition


class ModelPort(Protocol):
    """Obtain one normalized assistant turn without exposing an SDK type."""

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        """Send one conversation snapshot and return one assistant turn."""

        ...
