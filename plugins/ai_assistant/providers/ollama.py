"""Ollama adapter for local models. No key; data never leaves the machine."""

from __future__ import annotations

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
from .transport import iter_ndjson, parse_arguments


@register_provider
class OllamaAdapter(ProviderAdapter):
    id = "ollama"
    display_name = "Ollama (local)"
    default_model = "qwen3"
    default_base_url = "http://localhost:11434"
    requires_key = False

    def stream_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def headers(self) -> dict[str, str]:
        # A reverse proxy in front of Ollama may want a token; plain Ollama ignores it.
        return {"authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    @staticmethod
    def to_wire(message: Message) -> dict:
        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
                "tool_name": message.name or "",
            }
        wire: dict = {"role": message.role, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls]
        return wire

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict:
        wire = [{"role": "system", "content": system_text(system)}] if system else []
        wire += [self.to_wire(m) for m in messages]
        options: dict = {"num_predict": self.config.max_tokens}
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        payload: dict = {
            "model": self.config.model,
            "messages": wire,
            "stream": True,
            "options": options,
        }
        if tools and allow_tools:  # Ollama has no tool_choice; omitting the tools disables them
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
        return payload

    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]:
        text: list[str] = []
        calls: list[ToolCall] = []
        stop_reason, usage = "", Usage()
        for chunk in iter_ndjson(lines):
            if chunk.get("error"):
                raise ProviderUnavailableError(f"Ollama: {chunk['error']}", provider=self.display_name)
            message = chunk.get("message") or {}
            if message.get("content"):
                text.append(message["content"])
                yield TextDelta(message["content"])
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append(
                    ToolCall(
                        call.get("id") or f"call_{len(calls)}",
                        function.get("name", ""),
                        parse_arguments(function.get("arguments")),
                    )
                )
            if chunk.get("done"):
                stop_reason = chunk.get("done_reason") or "stop"
                usage = Usage(chunk.get("prompt_eval_count", 0), chunk.get("eval_count", 0))
                break
        yield Completion(Message("assistant", "".join(text), tool_calls=calls), stop_reason, usage)

    def _list_models(self) -> list[str]:
        data = self.transport.get_json(f"{self.base_url}/api/tags", self.headers(), 10)
        return [m["name"] for m in data.get("models", [])]
