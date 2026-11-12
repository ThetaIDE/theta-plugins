"""HTTP transport, error mapping, retries and stream framing shared by the adapters.

The adapters use the standard library rather than four vendor SDKs: the IDE gains no
dependencies, every provider fails the same way, and tests swap in a fake `Transport`
so CI never needs a real key or network access.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .chat_types import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)

T = TypeVar("T")


class Transport(Protocol):
    """What an adapter needs from HTTP. Both methods raise LLMError for any failure."""

    def post_stream(self, url: str, headers: dict[str, str], payload: dict, timeout: float) -> Iterator[str]:
        """POST JSON and return the response body as decoded lines.

        The request is sent and the status checked *before* this returns, so a retry
        wrapper around the call sees HTTP errors; only the body is read lazily.
        """
        ...

    def get_json(self, url: str, headers: dict[str, str], timeout: float) -> Any: ...


def _message_from_body(body: str) -> str:
    """The human-readable message inside an error body, in any of the providers' shapes."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body.strip()[:500]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or error)[:500]
        if isinstance(error, str):
            return error[:500]
    return body.strip()[:500]


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None  # an HTTP date; fall back to backoff


def error_for_status(status: int, body: str, headers: Any = None, provider: str = "") -> LLMError:
    """Map an HTTP failure onto the shared error hierarchy."""
    detail = _message_from_body(body) or f"HTTP {status}"
    if status in (401, 403):
        return AuthenticationError(
            f"{provider}: authentication failed ({detail})",
            status=status,
            provider=provider,
        )
    if status == 429:
        return RateLimitError(
            f"{provider}: rate limited ({detail})",
            retry_after=_retry_after(headers),
            status=status,
            provider=provider,
        )
    if status >= 500 or status == 408 or status == 529:
        return ProviderUnavailableError(
            f"{provider}: service error {status} ({detail})",
            status=status,
            provider=provider,
        )
    return BadRequestError(f"{provider}: request rejected ({detail})", status=status, provider=provider)


class UrllibTransport:
    """Default transport over urllib. Honours HTTPS_PROXY and the OS certificate store via Python."""

    def __init__(self, provider: str = ""):
        self.provider = provider

    def _open(self, request: urllib.request.Request, timeout: float):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise error_for_status(exc.code, body, exc.headers, self.provider) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderUnavailableError(
                f"{self.provider}: could not reach {request.full_url.split('?')[0]} ({reason})",
                provider=self.provider,
            ) from None

    def post_stream(self, url: str, headers: dict[str, str], payload: dict, timeout: float) -> Iterator[str]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"content-type": "application/json", **headers},
        )
        response = self._open(request, timeout)
        return self._lines(response)

    def _lines(self, response) -> Iterator[str]:
        try:
            for raw in response:
                yield raw.decode("utf-8", errors="replace").rstrip("\r\n")
        except (TimeoutError, ConnectionError) as exc:
            raise ProviderUnavailableError(
                f"{self.provider}: stream interrupted ({exc})", provider=self.provider
            ) from None
        finally:
            response.close()

    def get_json(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        request = urllib.request.Request(url, method="GET", headers=headers)
        with self._open(request, timeout) as response:
            return json.loads(response.read().decode("utf-8"))


# ── Retries ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter; a server's Retry-After wins when it is longer."""

    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0

    def delay(self, attempt: int, error: LLMError) -> float:
        backoff = min(self.max_delay, self.base_delay * (2**attempt)) * (0.5 + random.random() / 2)
        hinted = getattr(error, "retry_after", None)
        return min(self.max_delay, max(backoff, hinted)) if hinted is not None else backoff


def call_with_retries(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, float, LLMError], None] | None = None,
) -> T:
    """Call `fn`, retrying rate limits and outages. Other errors propagate at once."""
    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except LLMError as error:
            if not error.retryable or attempt == policy.max_attempts - 1:
                raise
            wait = policy.delay(attempt, error)
            if on_retry is not None:
                on_retry(attempt + 1, wait, error)
            sleep(wait)
    raise AssertionError("unreachable")


# ── Stream framing ───────────────────────────────────────────────────────────


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Server-sent events as (event name, data) pairs. Comments and ids are ignored."""
    event, data = "", []
    for line in lines:
        if not line:
            if data:
                yield event or "message", "\n".join(data)
            event, data = "", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield event or "message", "\n".join(data)


def iter_ndjson(lines: Iterable[str]) -> Iterator[dict]:
    """Newline-delimited JSON objects (Ollama's stream format)."""
    for line in lines:
        if line.strip():
            yield json.loads(line)


def parse_arguments(raw: str | dict | None) -> dict:
    """Tool arguments as a dict, whether the provider sent JSON text, an object, or nothing."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"_unparsed_arguments": raw}
    return value if isinstance(value, dict) else {"value": value}
