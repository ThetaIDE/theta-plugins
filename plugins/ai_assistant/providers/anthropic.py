"""Anthropic Messages API adapter."""

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
)
from .transport import error_for_status, iter_sse, parse_arguments

API_VERSION = "2023-06-01"


@register_provider
class AnthropicAdapter(ProviderAdapter):
    id = "anthropic"
    display_name = "Anthropic"
    default_model = "claude-sonnet-5-5"
    default_base_url = "https://api.anthropic.com/v1"

    def stream_url(self) -> str:
        return f"{self.base_url}/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.api_key or "",
            "anthropic-version": API_VERSION,
        }

    @staticmethod
    def to_wire(messages: Sequence[Message]) -> list[dict]:
        """Anthropic returns tool results as `user` turns, and wants consecutive ones merged."""
        wire: list[dict] = []
        for m in messages:
            if m.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }
                if m.is_error:
                    block["is_error"] = True
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [
                    {
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    }
                    for c in m.tool_calls
                ]
                wire.append(
                    {
                        "role": "assistant",
                        "content": blocks or [{"type": "text", "text": ""}],
                    }
                )
            elif m.role == "user":
                wire.append({"role": "user", "content": m.content})
        return wire

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict:
        payload: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": self.to_wire(messages),
            "stream": True,
        }
        segments = [system] if isinstance(system, str) else [part for part in system if part]
        if segments:
            payload["system"] = [{"type": "text", "text": text} for text in segments]
            # Cache breakpoint 1: tools + the stable system segment (everything but the last segment
            # when there are several). Breakpoint 2: the conversation so far, so each tool-loop step
            # re-reads the previous steps from cache instead of paying for them again.
            stable = len(segments) - 2 if len(segments) > 1 else 0
            payload["system"][stable]["cache_control"] = {"type": "ephemeral"}
        if payload["messages"]:
            last = payload["messages"][-1]
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            if last["content"]:
                last["content"][-1]["cache_control"] = {"type": "ephemeral"}
        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]
            if not allow_tools:
                payload["tool_choice"] = {"type": "none"}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        return payload

    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]:
        text: list[str] = []
        blocks: dict[int, dict] = {}  # content block index -> tool_use being assembled
        stop_reason, input_tokens, output_tokens, cached = "", 0, 0, 0
        for event, data in iter_sse(lines):
            payload = json.loads(data)
            kind = payload.get("type", event)
            if kind == "message_start":
                usage = payload["message"].get("usage", {})
                cached = usage.get("cache_read_input_tokens") or 0
                # input_tokens excludes cache reads and writes; report the whole prompt like the others.
                input_tokens = (
                    (usage.get("input_tokens") or 0) + cached + (usage.get("cache_creation_input_tokens") or 0)
                )
            elif kind == "content_block_start":
                block = payload["content_block"]
                if block["type"] == "tool_use":
                    blocks[payload["index"]] = {
                        "id": block["id"],
                        "name": block["name"],
                        "json": "",
                    }
            elif kind == "content_block_delta":
                delta = payload["delta"]
                if delta["type"] == "text_delta":
                    text.append(delta["text"])
                    yield TextDelta(delta["text"])
                elif delta["type"] == "input_json_delta":
                    blocks[payload["index"]]["json"] += delta.get("partial_json", "")
            elif kind == "message_delta":
                stop_reason = payload.get("delta", {}).get("stop_reason") or stop_reason
                output_tokens = payload.get("usage", {}).get("output_tokens", output_tokens)
            elif kind == "error":
                error = payload.get("error", {})
                status = 529 if error.get("type") == "overloaded_error" else 500
                raise error_for_status(status, json.dumps(payload), provider="Anthropic")
            elif kind == "message_stop":
                break
        else:
            if not stop_reason:
                raise ProviderUnavailableError("Anthropic: stream ended before message_stop", provider="Anthropic")
        calls = [ToolCall(b["id"], b["name"], parse_arguments(b["json"] or "{}")) for _i, b in sorted(blocks.items())]
        yield Completion(
            Message("assistant", "".join(text), tool_calls=calls),
            stop_reason,
            Usage(input_tokens, output_tokens, cached),
        )

    def _list_models(self) -> list[str]:
        data = self.transport.get_json(f"{self.base_url}/models?limit=100", self.headers(), 30)
        return [m["id"] for m in data.get("data", [])]
