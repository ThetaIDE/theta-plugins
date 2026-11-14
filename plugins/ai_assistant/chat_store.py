"""Saved conversations: one JSON file per chat in the workspace's .thetaide folder.

A chat holds the transcript as shown and the provider-neutral message history, so it can be
reopened and continued with any provider. API keys are never part of a chat. Files are written
atomically so a crash mid-save cannot corrupt an existing chat.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .providers.chat_types import Message, ToolCall

FORMAT = 1
MAX_CHATS = 100


def message_to_dict(message: Message) -> dict[str, Any]:
    data = asdict(message)
    return {k: v for k, v in data.items() if v not in (None, "", [], False)} | {"role": message.role}


def message_from_dict(data: dict[str, Any]) -> Message:
    calls = [
        ToolCall(c["id"], c["name"], c.get("arguments") or {}, c.get("extra") or {}) for c in data.get("tool_calls", [])
    ]
    return Message(
        role=data["role"],
        content=data.get("content", ""),
        tool_calls=calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        is_error=bool(data.get("is_error", False)),
    )


@dataclass
class Chat:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    provider: str = ""
    model: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)  # the transcript as rendered
    history: list[Message] = field(default_factory=list)
    cost: float = 0.0  # estimated US dollars so far; models without a known price add nothing

    @staticmethod
    def title_for(text: str) -> str:
        line = " ".join(text.split())
        return line if len(line) <= 60 else line[:57] + "…"


@dataclass(frozen=True)
class ChatSummary:
    id: str
    title: str
    updated: float


class ChatStore:
    def __init__(self, folder: Path, max_chats: int = MAX_CHATS):
        self.folder = Path(folder)
        self.max_chats = max_chats

    def _path(self, chat_id: str) -> Path:
        if not chat_id or not all(c.isalnum() or c in "-_" for c in chat_id):
            raise ValueError(f"Bad chat id {chat_id!r}")
        return self.folder / f"{chat_id}.json"

    def save(self, chat: Chat) -> None:
        if not chat.history and not chat.entries:
            return  # nothing worth keeping
        chat.updated = time.time()
        data = {
            "format": FORMAT,
            **{
                k: getattr(chat, k)
                for k in (
                    "id",
                    "title",
                    "created",
                    "updated",
                    "provider",
                    "model",
                    "entries",
                    "cost",
                )
            },
            "history": [message_to_dict(m) for m in chat.history],
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self._path(chat.id)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
        self._prune()

    def load(self, chat_id: str) -> Chat:
        data = json.loads(self._path(chat_id).read_text(encoding="utf-8"))
        return Chat(
            id=data["id"],
            title=data.get("title", ""),
            created=data.get("created", 0.0),
            updated=data.get("updated", 0.0),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            entries=list(data.get("entries", [])),
            history=[message_from_dict(m) for m in data.get("history", [])],
            cost=float(data.get("cost", 0.0)),
        )

    def list(self) -> list[ChatSummary]:
        """Saved chats, most recently updated first. Unreadable files are skipped, not fatal."""
        summaries = []
        for path in self.folder.glob("*.json") if self.folder.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(
                    ChatSummary(
                        data["id"],
                        data.get("title") or "Untitled",
                        data.get("updated", 0.0),
                    )
                )
            except (OSError, ValueError, KeyError):
                continue
        return sorted(summaries, key=lambda s: s.updated, reverse=True)

    def delete(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)

    def _prune(self) -> None:
        for old in self.list()[self.max_chats :]:
            self.delete(old.id)
