"""A Model Context Protocol server for ThetaIDE's tools, on the standard library only.

It speaks MCP's Streamable HTTP transport (https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
in its simplest form: the client POSTs one JSON-RPC message to /mcp and gets one JSON reply. Nothing is streamed
back, so GET (the optional server-to-client stream) is refused with 405, as the specification allows.

Only this machine can reach it, and only with the session's token:
- it listens on 127.0.0.1 alone;
- every request needs `Authorization: Bearer <token>`, a random token made when the server starts;
- the Host header must name the loopback address and an Origin header, if any, must too. A web page in a
  browser therefore cannot call it, even through DNS rebinding.

Nothing here imports Qt. The plugin passes in two callables: one that lists the tools and one that runs a tool,
which may block for as long as the user takes to approve a change in the IDE.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Newest first. A client asking for one of these gets it; any other gets the newest.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "thetaide", "title": "ThetaIDE", "version": "1.0.0"}
ENDPOINT = "/mcp"
MAX_BODY = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

INSTRUCTIONS = """\
These tools act on the ThetaIDE workspace the user has open: its training runs, configs, logs, metrics and \
figures. Reading is free. Writing or editing files, launching or stopping runs and installing Hub components \
open an approval dialog in ThetaIDE; if a call comes back declined, do not retry it, ask the user instead. \
Run ids come from list_runs; 'selected' is the run open in the IDE's Training Monitor and 'latest' the newest. \
To compare variants of an experiment, launch_experiment it once per variant with a label, overrides and the same \
seeds, then wait_for_runs and compare_runs (it reports the mean and spread across seeds)."""

# How known clients are named in the IDE's approval dialogs ("Claude Code wants to edit ...").
CLIENT_NAMES = {
    "claude-code": "Claude Code",
    "claude-ai": "Claude",
    "codex-mcp-client": "Codex",
    "cursor-vscode": "Cursor",
    "gemini-cli-mcp-client": "Gemini CLI",
    "Visual Studio Code": "VS Code",
}
UNKNOWN_CLIENT = "An external agent"

ListTools = Callable[[], list[dict[str, Any]]]
# (name, arguments, who) -> {"text": str, "is_error": bool, "images": [[path, caption], ...]}
CallTool = Callable[[str, dict[str, Any], str], dict[str, Any]]

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


def client_label(info: Any) -> str:
    """The name to show the user for a client's `clientInfo`."""
    if not isinstance(info, dict):
        return UNKNOWN_CLIENT
    name = str(info.get("name") or "")
    title = str(info.get("title") or "")
    if name.startswith("pi-mcp-"):  # pi's MCP adapter names itself after the server: pi-mcp-<server>
        return "pi"
    return CLIENT_NAMES.get(name) or title or name or UNKNOWN_CLIENT


def image_content(path: str) -> dict[str, Any] | None:
    """An MCP image block for a raster image file, or None (other formats, too large, unreadable)."""
    file = Path(path)
    mime = IMAGE_TYPES.get(file.suffix.lower())
    try:
        if mime is None or file.stat().st_size > MAX_IMAGE_BYTES:
            return None
        data = file.read_bytes()
    except OSError:
        return None
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": mime}


class MCPHandler:
    """JSON-RPC over MCP, without the transport: one message in, one reply (or None for a notification) out."""

    def __init__(self, list_tools: ListTools, call_tool: CallTool, instructions: str = INSTRUCTIONS):
        self._list_tools = list_tools
        self._call_tool = call_tool
        self._instructions = instructions
        self._sessions: dict[str, str] = {}  # Mcp-Session-Id -> the client's name
        self._lock = threading.Lock()

    def new_session(self, client: str) -> str:
        session = uuid.uuid4().hex
        with self._lock:
            self._sessions[session] = client
        return session

    def client(self, session: str | None) -> str | None:
        """The client's name for a session id; None for an id this server never issued."""
        if not session:
            return UNKNOWN_CLIENT
        with self._lock:
            return self._sessions.get(session)

    def end_session(self, session: str) -> bool:
        with self._lock:
            return self._sessions.pop(session, None) is not None

    def handle(self, message: Any, client: str = UNKNOWN_CLIENT) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "Expected a JSON-RPC 2.0 message.")
        if "method" not in message:
            return None  # a response to something we never asked: nothing to say
        request_id = message.get("id")
        is_notification = "id" not in message
        method = message.get("method")
        params = message.get("params") or {}
        try:
            if method == "initialize":
                result: Any = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [self._tool(t) for t in self._list_tools()]}
            elif method == "tools/call":
                result = self._call(params, client)
            elif is_notification:
                return None  # notifications/initialized, cancelled, ...: nothing to do
            else:
                return _error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except _InvalidParams as exc:
            return _error(request_id, INVALID_PARAMS, str(exc))
        except Exception as exc:  # a bug here must not take the server down
            return _error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        return {
            "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": self._instructions,
        }

    @staticmethod
    def _tool(definition: dict[str, Any]) -> dict[str, Any]:
        mutating = bool(definition.get("mutating"))
        return {
            "name": definition["name"],
            "description": definition.get("description", ""),
            "inputSchema": definition.get("input_schema") or {"type": "object", "properties": {}},
            "annotations": {
                "readOnlyHint": not mutating,
                "destructiveHint": mutating,
                "openWorldHint": False,
            },
        }

    def _call(self, params: dict[str, Any], client: str) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments")
        arguments = {} if arguments is None else arguments  # absent is fine; [] or "x" is not
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise _InvalidParams("tools/call needs a tool name and an arguments object.")
        if name not in {t["name"] for t in self._list_tools()}:
            raise _InvalidParams(f"Unknown tool: {name}")
        outcome = self._call_tool(name, arguments, client)
        content: list[dict[str, Any]] = [{"type": "text", "text": str(outcome.get("text", ""))}]
        for path, caption in outcome.get("images") or ():
            block = image_content(path)
            if block is not None:
                content.append(block)
            else:
                content.append({"type": "text", "text": f"Image not attached ({caption or 'figure'}): {path}"})
        return {"content": content, "isError": bool(outcome.get("is_error"))}


class _InvalidParams(Exception):
    pass


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class MCPServer:
    """The HTTP side: listens on 127.0.0.1 and hands each authorised message to an MCPHandler."""

    def __init__(
        self,
        list_tools: ListTools,
        call_tool: CallTool,
        *,
        token: str | None = None,
        port: int = 0,
        instructions: str = INSTRUCTIONS,
    ):
        self.token = token or secrets.token_urlsafe(32)
        self.handler = MCPHandler(list_tools, call_tool, instructions)
        self._requested_port = port
        self._http: ThreadingHTTPServer | None = None

    @property
    def running(self) -> bool:
        return self._http is not None

    @property
    def port(self) -> int:
        if self._http is None:
            raise RuntimeError("The MCP server is not running.")
        return self._http.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}{ENDPOINT}"

    def start(self) -> None:
        if self._http is not None:
            return
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802  (http.server API)
                server._post(self)

            def do_GET(self) -> None:  # noqa: N802
                if server._checked(self):
                    server._reply(self, 405, None, {"Allow": "POST, DELETE"})

            def do_DELETE(self) -> None:  # noqa: N802
                if server._checked(self):
                    ended = server.handler.end_session(self.headers.get("Mcp-Session-Id", ""))
                    server._reply(self, 200 if ended else 404, None)

            def log_message(self, *_args: Any) -> None:  # keep the IDE's console quiet
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", self._requested_port), Handler)
        self._http.daemon_threads = True
        threading.Thread(target=self._http.serve_forever, name="theta-mcp", daemon=True).start()

    def stop(self) -> None:
        if self._http is not None:
            self._http.shutdown()
            self._http.server_close()
            self._http = None

    # ── Requests ──────────────────────────────────────────────────────────

    def _checked(self, request: BaseHTTPRequestHandler) -> bool:
        """Path, Host, Origin and token; replies with the error and returns False when one is wrong."""
        if urlparse(request.path).path != ENDPOINT:
            self._reply(request, 404, None)
            return False
        loopback = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if request.headers.get("Host", "") not in loopback:
            self._reply(request, 403, None)  # DNS rebinding: a page on another name reached this port
            return False
        origin = request.headers.get("Origin")
        if origin is not None and urlparse(origin).netloc not in loopback:
            self._reply(request, 403, None)  # a web page in a browser
            return False
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token.strip().encode("utf-8"), self.token.encode("utf-8")
        ):
            self._reply(request, 401, None, {"WWW-Authenticate": 'Bearer realm="ThetaIDE"'})
            return False
        return True

    @staticmethod
    def _body(request: BaseHTTPRequestHandler) -> bytes | int:
        """The request body, from Content-Length or chunked transfer encoding (which Claude Code uses).
        An HTTP status instead when it is missing, malformed or too large."""
        if "chunked" in request.headers.get("Transfer-Encoding", "").lower():
            data = bytearray()
            while True:
                size_line = request.rfile.readline(1024)
                try:
                    size = int(size_line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    return 400
                if size == 0:
                    while request.rfile.readline(1024) not in (b"\r\n", b"\n", b""):
                        pass  # trailers
                    return bytes(data) if data else 400
                if len(data) + size > MAX_BODY:
                    return 413
                data += request.rfile.read(size)
                request.rfile.readline(1024)  # the CRLF after each chunk
        try:
            length = int(request.headers.get("Content-Length") or 0)
        except ValueError:
            return 400
        if length > MAX_BODY:
            return 413
        return request.rfile.read(length) if length > 0 else 400

    def _post(self, request: BaseHTTPRequestHandler) -> None:
        if not self._checked(request):
            return
        body = self._body(request)
        if isinstance(body, int):
            self._reply(request, body, None)
            return
        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._reply(request, 400, _error(None, PARSE_ERROR, "The body is not valid JSON."))
            return
        if isinstance(message, list):  # batches were dropped from MCP in 2025-06-18
            self._reply(request, 400, _error(None, INVALID_REQUEST, "Send one JSON-RPC message per request."))
            return

        headers: dict[str, str] = {}
        if isinstance(message, dict) and message.get("method") == "initialize":
            client = client_label((message.get("params") or {}).get("clientInfo"))
            headers["Mcp-Session-Id"] = self.handler.new_session(client)
        else:
            client = self.handler.client(request.headers.get("Mcp-Session-Id"))
            if client is None:  # an id from before a restart: the client must initialize again
                self._reply(request, 404, _error(None, INVALID_REQUEST, "Unknown session; initialize again."))
                return
        reply = self.handler.handle(message, client)
        if reply is None:
            self._reply(request, 202, None, headers)
        else:
            self._reply(request, 200, reply, headers)

    @staticmethod
    def _reply(
        request: BaseHTTPRequestHandler, status: int, body: dict[str, Any] | None, headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        request.send_response(status)
        for key, value in (headers or {}).items():
            request.send_header(key, value)
        if body is not None:
            request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(data)))
        # One request per connection: a refused request's unread body can never be taken for the next one.
        request.send_header("Connection", "close")
        request.close_connection = True
        request.end_headers()
        if data:
            request.wfile.write(data)
