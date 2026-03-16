from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    """Single conversational turn stored in short-term memory."""

    role: Role
    content: str


class ConversationMemory:
    """Simple in-memory chat history for the current agent session."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages: list[Message] = list(messages or [])

    def add(self, role: Role, content: str) -> None:
        self._messages.append(Message(role=role, content=content))

    def recent(self, limit: int | None = None) -> list[Message]:
        if limit is None:
            return list(self._messages)
        return list(self._messages[-limit:])

    def as_prompt(self, limit: int | None = None) -> str:
        messages = self.recent(limit=limit)
        if not messages:
            return "No prior conversation."

        return "\n".join(f"{message.role.title()}: {message.content}" for message in messages)


class LongTermMemoryStore:
    """
    Placeholder interface for a future vector database integration.

    The methods exist so the agent can depend on a stable API today and swap
    the implementation later without changing the core loop.
    """

    def __init__(self) -> None:
        self._documents: list[tuple[str, dict[str, str]]] = []

    def add(self, text: str, metadata: dict[str, str] | None = None) -> None:
        self._documents.append((text, metadata or {}))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        _ = (query, top_k)
        return []
