"""Google Gemini (Generative Language API) adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

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
from .transport import iter_sse

# Gemini accepts an OpenAPI subset of JSON Schema and rejects requests that use the rest.
_UNSUPPORTED_SCHEMA_KEYS = {
    "additionalProperties",
    "$schema",
    "default",
    "examples",
    "title",
}


def gemini_schema(schema: Any) -> Any:
    """`schema` with the keywords Gemini rejects removed, recursively."""
    if isinstance(schema, dict):
        return {k: gemini_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(schema, list):
        return [gemini_schema(v) for v in schema]
    return schema


@register_provider
class GeminiAdapter(ProviderAdapter):
    id = "gemini"
    display_name = "Google Gemini"
    default_model = "gemini-2.5-flash"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def stream_url(self) -> str:
        return f"{self.base_url}/models/{self.config.model}:streamGenerateContent?alt=sse"

    def headers(self) -> dict[str, str]:
        # A header rather than ?key=, so the key never appears in proxy or error logs.
        return {"x-goog-api-key": self.config.api_key or ""}

    @staticmethod
    def to_wire(messages: Sequence[Message]) -> list[dict]:
        contents: list[dict] = []
        for m in messages:
            if m.role == "tool":
                response = {"error": m.content} if m.is_error else {"content": m.content}
                part = {"functionResponse": {"name": m.name or "", "response": response}}
                if contents and contents[-1]["role"] == "user" and "functionResponse" in contents[-1]["parts"][0]:
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            elif m.role == "assistant":
                parts: list[dict] = [{"text": m.content}] if m.content else []
                for c in m.tool_calls:
                    part = {"functionCall": {"name": c.name, "args": c.arguments}}
                    if c.extra.get("thoughtSignature"):
                        part["thoughtSignature"] = c.extra["thoughtSignature"]
                    parts.append(part)
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
        return contents

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict:
        generation: dict = {"maxOutputTokens": self.config.max_tokens}
        if self.config.temperature is not None:
            generation["temperature"] = self.config.temperature
        payload: dict = {
            "contents": self.to_wire(messages),
            "generationConfig": generation,
        }
        if system:
            # Gemini 2.5+ caches repeated prefixes implicitly; stable text first maximises hits.
            payload["systemInstruction"] = {"parts": [{"text": system_text(system)}]}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": gemini_schema(t.parameters),
                        }
                        for t in tools
                    ]
                }
            ]
            if not allow_tools:
                payload["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
        return payload

    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]:
        text: list[str] = []
        calls: list[ToolCall] = []
        stop_reason, usage = "", Usage()
        for _event, data in iter_sse(lines):
            chunk = json.loads(data)
            if "error" in chunk:
                raise ProviderUnavailableError(f"Gemini: {chunk['error'].get('message')}", provider="Google Gemini")
            if chunk.get("usageMetadata"):
                u = chunk["usageMetadata"]
                usage = Usage(
                    u.get("promptTokenCount", 0),
                    u.get("candidatesTokenCount", 0),
                    u.get("cachedContentTokenCount", 0),
                )
            for candidate in chunk.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text") and not part.get("thought"):
                        text.append(part["text"])
                        yield TextDelta(part["text"])
                    elif "functionCall" in part:
                        call = part["functionCall"]
                        extra = {"thoughtSignature": part["thoughtSignature"]} if part.get("thoughtSignature") else {}
                        calls.append(
                            ToolCall(
                                call.get("id") or f"call_{len(calls)}",
                                call["name"],
                                call.get("args") or {},
                                extra,
                            )
                        )
                stop_reason = candidate.get("finishReason") or stop_reason
        yield Completion(Message("assistant", "".join(text), tool_calls=calls), stop_reason, usage)

    def _list_models(self) -> list[str]:
        data = self.transport.get_json(f"{self.base_url}/models?pageSize=200", self.headers(), 30)
        return [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
