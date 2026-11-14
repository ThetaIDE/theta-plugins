"""The common adapter interface every LLM provider implements, and the provider registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from .chat_types import (
    AuthenticationError,
    LLMError,
    Message,
    StreamEvent,
    System,
    ToolSpec,
)
from .transport import RetryPolicy, Transport, UrllibTransport, call_with_retries


@dataclass
class ProviderConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    timeout: float = 120.0


class ProviderAdapter(ABC):
    """Translate provider-neutral messages and tools to one vendor's API and back.

    Subclasses supply the endpoint, headers, request body and stream parser; the base
    class owns key checks and retries so every provider behaves the same under failure.
    Only opening the stream is retried: once tokens have reached the user, replaying
    the request would duplicate them, so a mid-stream failure is reported instead.
    """

    id: ClassVar[str]
    display_name: ClassVar[str]
    default_model: ClassVar[str]
    default_base_url: ClassVar[str]
    requires_key: ClassVar[bool] = True

    def __init__(
        self,
        config: ProviderConfig,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        on_retry: Callable[[int, float, LLMError], None] | None = None,
        sleep: Callable[[float], None] | None = None,
    ):
        self.config = config
        self.transport = transport or UrllibTransport(self.display_name)
        self.retry = retry or RetryPolicy()
        self.on_retry = on_retry
        self._sleep = sleep

    @property
    def base_url(self) -> str:
        return (self.config.base_url or self.default_base_url).rstrip("/")

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System = "",
        allow_tools: bool = True,
    ) -> Iterator[StreamEvent]:
        """Send the conversation; yield TextDelta events, then exactly one Completion.

        allow_tools=False keeps the tools declared (providers reject tool history without them)
        but tells the model it may not call one, so it must answer in text.
        """
        if self.requires_key and not self.config.api_key:
            raise AuthenticationError(
                f"{self.display_name}: no API key. Add one in Settings → Plugins → AI Assistant.",
                provider=self.display_name,
            )
        url, headers, payload = (
            self.stream_url(),
            self.headers(),
            self.build_payload(messages, tools, system, allow_tools),
        )
        kwargs = {"on_retry": self.on_retry}
        if self._sleep is not None:
            kwargs["sleep"] = self._sleep
        lines = call_with_retries(
            lambda: self.transport.post_stream(url, headers, payload, self.config.timeout),
            self.retry,
            **kwargs,
        )
        yield from self.parse_stream(lines)

    def list_models(self) -> list[str]:
        """Model ids this key (or local server) can use, alphabetically."""
        return sorted(self._list_models())

    # ── Provider specifics ───────────────────────────────────────────────────

    @abstractmethod
    def stream_url(self) -> str: ...

    @abstractmethod
    def headers(self) -> dict[str, str]: ...

    @abstractmethod
    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: System,
        allow_tools: bool = True,
    ) -> dict: ...

    @abstractmethod
    def parse_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]: ...

    @abstractmethod
    def _list_models(self) -> list[str]: ...


PROVIDERS: dict[str, type[ProviderAdapter]] = {}


def register_provider(cls: type[ProviderAdapter]) -> type[ProviderAdapter]:
    PROVIDERS[cls.id] = cls
    return cls


def create_provider(provider_id: str, config: ProviderConfig, **kwargs) -> ProviderAdapter:
    try:
        cls = PROVIDERS[provider_id]
    except KeyError:
        raise ValueError(f"Unknown provider {provider_id!r}; choose one of {', '.join(PROVIDERS)}") from None
    return cls(config, **kwargs)
