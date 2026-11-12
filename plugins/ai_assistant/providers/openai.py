"""OpenAI Chat Completions adapter. Also serves any OpenAI-compatible server via a custom base URL."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence

from .base import ProviderAdapter, register_provider
from .chat_types import (
    Completion,
    Message,
    ProviderUnavailableError,
    StreamEvent,
    System,
    TextDelta,
    ToolCall,
    ToolSpec,
    Usage,
    system_text,
)
from .transport import iter_sse, parse_arguments


@register_provider
class OpenAIAdapter(ProviderAdapter):
    id = "openai"
    display_name = "OpenAI"
    default_model = "gpt-4.1-mini"
    default_base_url = "https://api.openai.com/v1"

    def stream_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.config.api_key}"}

    @staticmethod
    def to_wire(message: Message) -> dict:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        wire: dict = {"role": message.role, "content": message.content or None}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in message.tool_calls
            ]
        elif wire["content"] is None:
            wire["content"] = ""
        return wire

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict:
        # OpenAI caches prompt prefixes automatically: stable text first keeps the prefix identical.
        wire = [{"role": "system", "content": system_text(system)}] if system else []
        wire += [self.to_wire(m) for m in messages]
        payload: dict = {
            "model": self.config.model,
            "messages": wire,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if not allow_tools:
                payload["tool_choice"] = "none"
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]:
        text: list[str] = []
        partial: dict[int, dict] = {}  # tool call index -> {"id", "name", "arguments"}
        stop_reason, usage = "", Usage()
        for _event, data in iter_sse(lines):
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise ProviderUnavailableError(
                    f"OpenAI: {chunk['error'].get('message', chunk['error'])}",
                    provider="OpenAI",
                )
            if chunk.get("usage"):
                u = chunk["usage"]
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), cached)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    yield TextDelta(delta["content"])
                for fragment in delta.get("tool_calls") or []:
                    slot = partial.setdefault(
                        fragment.get("index", 0),
                        {"id": "", "name": "", "arguments": ""},
                    )
                    slot["id"] = fragment.get("id") or slot["id"]
                    function = fragment.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                stop_reason = choice.get("finish_reason") or stop_reason
        calls = [
            ToolCall(
                id=s["id"] or f"call_{i}",
                name=s["name"],
                arguments=parse_arguments(s["arguments"]),
            )
            for i, s in sorted(partial.items())
        ]
        yield Completion(Message("assistant", "".join(text), tool_calls=calls), stop_reason, usage)

    def _list_models(self) -> list[str]:
        data = self.transport.get_json(f"{self.base_url}/models", self.headers(), 30)
        return [m["id"] for m in data.get("data", [])]
