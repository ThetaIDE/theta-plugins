"""OpenRouter adapter: one key for hundreds of models from many vendors.

OpenRouter speaks the OpenAI Chat Completions protocol, so this reuses the OpenAI adapter and changes
only what differs: the endpoint, attribution headers, error labels, and the cost OpenRouter reports for
each request (exact, so no price table is needed). Model ids are "vendor/model", e.g.
"anthropic/claude-sonnet-4.5" or "openai/gpt-4o-mini".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence

from .base import register_provider
from .chat_types import (
    Completion,
    Message,
    ProviderUnavailableError,
    StreamEvent,
    System,
    ToolSpec,
    Usage,
)
from .openai import OpenAIAdapter

# Generic attribution so OpenRouter can show the app in its rankings; nothing about the user.
APP_URL = "https://github.com/ThetaIDE/ThetaIDE"
APP_TITLE = "ThetaIDE"


@register_provider
class OpenRouterAdapter(OpenAIAdapter):
    id = "openrouter"
    display_name = "OpenRouter"
    default_model = "openai/gpt-4o-mini"
    default_base_url = "https://openrouter.ai/api/v1"

    def headers(self) -> dict[str, str]:
        return {**super().headers(), "http-referer": APP_URL, "x-title": APP_TITLE}

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict:
        payload = super().build_payload(messages, tools, system, allow_tools)
        # OpenRouter's name for the output limit; it ignores max_completion_tokens on some routes.
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload["usage"] = {"include": True}  # adds the request's cost in US dollars to the final chunk
        return payload

    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]:
        cost: list[float] = []

        def watch(source: Iterable[str]) -> Iterator[str]:
            # OpenRouter reports errors that happen after the stream opened as a chunk with "error",
            # and the cost in the usage chunk; peek at both before the OpenAI parser sees them.
            for line in source:
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data.startswith("{"):
                        chunk = json.loads(data)
                        if "error" in chunk:
                            error = chunk["error"]
                            message = error.get("message", error) if isinstance(error, dict) else error
                            raise ProviderUnavailableError(f"OpenRouter: {message}", provider="OpenRouter")
                        reported = (chunk.get("usage") or {}).get("cost")
                        if isinstance(reported, (int, float)):
                            cost.append(float(reported))
                yield line

        for event in super().parse_stream(watch(lines)):
            if isinstance(event, Completion) and cost:
                u = event.usage
                event = Completion(
                    event.message,
                    event.stop_reason,
                    Usage(u.input_tokens, u.output_tokens, u.cached_input_tokens, cost=cost[-1]),
                )
            yield event
