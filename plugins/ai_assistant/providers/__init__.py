"""LLM provider adapters behind one interface. Importing this package registers all four."""

from . import anthropic, gemini, ollama, openai  # noqa: F401  (registration side effect)
from .base import PROVIDERS, ProviderAdapter, ProviderConfig, create_provider
from .chat_types import (
    AuthenticationError,
    BadRequestError,
    Completion,
    LLMError,
    Message,
    ProviderUnavailableError,
    RateLimitError,
    StreamCancelled,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "PROVIDERS",
    "AuthenticationError",
    "BadRequestError",
    "Completion",
    "LLMError",
    "Message",
    "ProviderAdapter",
    "ProviderConfig",
    "ProviderUnavailableError",
    "RateLimitError",
    "StreamCancelled",
    "StreamEvent",
    "TextDelta",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "create_provider",
]
