"""Sign in with OpenRouter: OAuth with PKCE that ends with an API key, no copy and paste.

The flow (https://openrouter.ai/docs/use-cases/oauth-pkce):
1. Listen on a loopback port and open https://openrouter.ai/auth in the browser with a PKCE challenge,
   a random `state` and `callback_url=http://localhost:<port>/callback`.
2. The user approves; OpenRouter redirects the browser to the callback with `code` and `state`.
3. POST the code and the PKCE verifier to https://openrouter.ai/api/v1/auth/keys, which returns a key.

The key then goes to the keychain like a pasted one. Nothing here imports Qt: the panel opens the browser
on the GUI thread (start) and waits for the redirect and exchange on a background thread (finish).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

AUTH_URL = "https://openrouter.ai/auth"
EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
KEY_LABEL = "ThetaIDE"  # the name the new key gets in the user's OpenRouter account
CALLBACK_PATH = "/callback"
TIMEOUT_S = 300.0  # OpenRouter's codes expire after 10 minutes; give up waiting before that

PostJson = Callable[[str, dict[str, Any]], dict[str, Any]]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ThetaIDE</title>
<style>body{{font-family:system-ui,sans-serif;margin:15vh auto;max-width:28em;text-align:center;color:#333}}
h1{{font-size:1.4em}}</style></head><body><h1>{title}</h1><p>{body}</p></body></html>"""


class OAuthError(Exception):
    """Sign-in did not produce a key; the message says why, in words for the user."""


def pkce_pair() -> tuple[str, str]:
    """(verifier, S256 challenge) as in RFC 7636: 64 URL-safe characters, and their SHA-256 in base64url."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": "ThetaIDE"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        if exc.code == 403:
            raise OAuthError("OpenRouter rejected the sign-in code (it may have expired). Try again.") from exc
        raise OAuthError(f"OpenRouter could not create a key (HTTP {exc.code}). {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OAuthError(f"Could not reach OpenRouter: {exc}") from exc


class _Receiver:
    """A one-shot HTTP server on the loopback interface that waits for the browser's redirect.

    It listens on 127.0.0.1 and, when possible, on ::1 at the same port, because browsers resolve
    `localhost` to either."""

    def __init__(self) -> None:
        self.params: dict[str, str] | None = None
        self._got = threading.Event()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802  (http.server API)
                url = urlparse(self.path)
                if url.path != CALLBACK_PATH or receiver._got.is_set():
                    self.send_error(404)
                    return
                params = {k: v[0] for k, v in parse_qs(url.query).items()}
                ok = "code" in params
                page = PAGE.format(
                    title="Signed in to OpenRouter" if ok else "Sign-in did not finish",
                    body="You can close this tab and return to ThetaIDE."
                    if ok
                    else "ThetaIDE did not receive an authorization. Return to ThetaIDE to try again.",
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                receiver.params = params
                receiver._got.set()

            def log_message(self, *_args: Any) -> None:  # keep the IDE's console quiet
                pass

        self.servers: list[ThreadingHTTPServer] = [ThreadingHTTPServer(("127.0.0.1", 0), Handler)]
        self.port = self.servers[0].server_address[1]
        if socket.has_ipv6:
            try:

                class V6(ThreadingHTTPServer):
                    address_family = socket.AF_INET6

                self.servers.append(V6(("::1", self.port), Handler))
            except OSError:
                pass  # no IPv6 loopback, or the port is taken there: IPv4 is enough
        for server in self.servers:
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.port}{CALLBACK_PATH}"

    def wait(self, timeout: float, cancelled: Callable[[], bool]) -> dict[str, str]:
        deadline = time.monotonic() + timeout
        while not self._got.wait(0.1):
            if cancelled():
                raise OAuthError("Sign-in was cancelled.")
            if time.monotonic() > deadline:
                raise OAuthError("Sign-in timed out. Try again, and approve the request in your browser.")
        return dict(self.params or {})

    def close(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()


class OpenRouterSignIn:
    """One sign-in attempt. start() on the GUI thread returns the URL to open; finish() blocks until the
    browser comes back, then exchanges the code for a key. close() always, even on failure."""

    def __init__(self, post: PostJson | None = None, timeout: float = TIMEOUT_S):
        self._post = post or post_json
        self._timeout = timeout
        self._verifier, self._challenge = pkce_pair()
        self._state = secrets.token_urlsafe(24)
        self._receiver: _Receiver | None = None

    def start(self) -> str:
        self._receiver = _Receiver()
        query = {
            "callback_url": self._receiver.redirect_uri,
            "code_challenge": self._challenge,
            "code_challenge_method": "S256",
            "state": self._state,
            "key_label": KEY_LABEL,
        }
        return f"{AUTH_URL}?{urlencode(query)}"

    def finish(self, cancelled: Callable[[], bool] = lambda: False) -> str:
        if self._receiver is None:
            raise OAuthError("Sign-in was not started.")
        try:
            params = self._receiver.wait(self._timeout, cancelled)
        finally:
            self.close()
        if "code" not in params:
            raise OAuthError("Sign-in was declined or cancelled in the browser.")
        if not secrets.compare_digest(params.get("state", ""), self._state):
            raise OAuthError("The sign-in response did not match this request, so it was ignored. Try again.")
        response = self._post(
            EXCHANGE_URL,
            {"code": params["code"], "code_verifier": self._verifier, "code_challenge_method": "S256"},
        )
        key = response.get("key") if isinstance(response, dict) else None
        if not isinstance(key, str) or not key.strip():
            raise OAuthError("OpenRouter did not return an API key.")
        return key.strip()

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
