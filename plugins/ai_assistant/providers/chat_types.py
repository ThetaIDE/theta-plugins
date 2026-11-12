"""Provider-neutral chat types shared by every adapter, the agent loop and the tools.

Each adapter translates these to and from its own wire format, so nothing above the
adapter layer knows which vendor is on the other end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    """A function the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model, with its arguments already parsed."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque provider state that must be echoed back with the call (Gemini's thought signatures).
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class Message:
    """One turn of the conversation.

    A `tool` message answers the assistant's ToolCall with the same `tool_call_id`; `name`
    repeats the tool's name because Gemini and Ollama match results by name, not by id.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    """Token counts. input_tokens is the whole prompt; cached_input_tokens is the part read from the
    provider's prompt cache (billed at a fraction of the normal rate)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text, emitted as soon as it arrives."""

    text: str


@dataclass(frozen=True)
class Completion:
    """The last event of a stream: the whole assistant message and why generation stopped."""

    message: Message
    stop_reason: str = ""
    usage: Usage = Usage()


StreamEvent = TextDelta | Completion

# A system prompt is a string, or segments ordered from most to least stable. Providers with explicit
# prompt caching cache the stable segments; the rest join them into one string.
System = str | tuple[str, ...] | list[str]


def system_text(system: System) -> str:
    return system if isinstance(system, str) else "\n\n".join(part for part in system if part)


# ── Errors ───────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base for every provider failure; `retryable` tells the retry loop whether to try again."""

    retryable = False

    def __init__(self, message: str, *, status: int | None = None, provider: str = ""):
        super().__init__(message)
        self.status = status
        self.provider = provider


class AuthenticationError(LLMError):
    """The API key is missing, wrong or lacks permission (HTTP 401/403)."""


class BadRequestError(LLMError):
    """The provider rejected the request itself (HTTP 400/404/422): unknown model, bad schema."""


class RateLimitError(LLMError):
    """HTTP 429. `retry_after` is the server's requested wait in seconds, when it sent one."""

    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ProviderUnavailableError(LLMError):
    """A 5xx, an overloaded provider, a timeout, or no connection at all."""

    retryable = True


class StreamCancelled(LLMError):
    """The user pressed Stop while a response was streaming."""
