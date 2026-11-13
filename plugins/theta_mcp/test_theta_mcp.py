"""Tests for the Theta MCP Server: the protocol, the HTTP guard, the workspace config and the plugin."""

from __future__ import annotations

import base64
import http.client
import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.theta_mcp import project_config
from plugins.theta_mcp.server import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSIONS,
    UNKNOWN_CLIENT,
    MCPHandler,
    MCPServer,
    client_label,
)

TOOLS = [
    {
        "name": "list_runs",
        "description": "List runs.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        "mutating": False,
    },
    {
        "name": "write_file",
        "description": "Write a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "mutating": True,
    },
]


def tiny_png(path: Path) -> Path:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"\x00\xff\x00\x00"  # one red pixel
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


class Calls:
    def __init__(self, outcome=None):
        self.seen: list[tuple] = []
        self.outcome = outcome or {"text": '{"total": 0}', "is_error": False, "images": []}

    def __call__(self, name, arguments, who):
        self.seen.append((name, arguments, who))
        return self.outcome


def request(method, params=None, rid=1):
    message = {"jsonrpc": "2.0", "method": method, "id": rid}
    if params is not None:
        message["params"] = params
    return message


# ── The protocol ─────────────────────────────────────────────────────────────


def test_initialize_negotiates_the_protocol_version_and_offers_tools():
    handler = MCPHandler(lambda: TOOLS, Calls())
    reply = handler.handle(request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}))
    result = reply["result"]
    assert result["protocolVersion"] == "2025-03-26"  # one we speak: kept
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "thetaide" and "approval" in result["instructions"]
    newer = handler.handle(request("initialize", {"protocolVersion": "2099-01-01"}))
    assert newer["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]  # unknown: our newest
    assert handler.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handler.handle(request("ping"))["result"] == {}


def test_tools_are_listed_with_hints_about_what_they_change():
    tools = MCPHandler(lambda: TOOLS, Calls()).handle(request("tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == ["list_runs", "write_file"]
    assert tools[0]["inputSchema"] == TOOLS[0]["input_schema"]
    assert tools[0]["annotations"]["readOnlyHint"] is True and tools[0]["annotations"]["destructiveHint"] is False
    assert tools[1]["annotations"]["readOnlyHint"] is False and tools[1]["annotations"]["destructiveHint"] is True


def test_a_tool_call_runs_for_the_named_client_and_returns_text_and_images(tmp_path):
    png = tiny_png(tmp_path / "curve.png")
    calls = Calls({"text": "Plotted.", "is_error": False, "images": [[str(png), "reward"], ["/no/such.svg", "x"]]})
    handler = MCPHandler(lambda: TOOLS, calls)
    reply = handler.handle(request("tools/call", {"name": "list_runs", "arguments": {"limit": 3}}), "Claude Code")
    assert calls.seen == [("list_runs", {"limit": 3}, "Claude Code")]
    content = reply["result"]["content"]
    assert content[0] == {"type": "text", "text": "Plotted."} and reply["result"]["isError"] is False
    assert content[1]["type"] == "image" and content[1]["mimeType"] == "image/png"
    assert base64.b64decode(content[1]["data"]) == png.read_bytes()
    assert content[2]["type"] == "text" and "/no/such.svg" in content[2]["text"]  # not attachable: named


def test_tool_errors_are_results_and_bad_calls_are_protocol_errors():
    handler = MCPHandler(lambda: TOOLS, Calls({"text": "The user declined this action.", "is_error": True}))
    declined = handler.handle(request("tools/call", {"name": "write_file", "arguments": {"path": "a.yaml"}}))
    assert declined["result"]["isError"] is True  # the agent reads it and adapts
    unknown = handler.handle(request("tools/call", {"name": "rm_rf", "arguments": {}}))
    assert unknown["error"]["code"] == INVALID_PARAMS and "rm_rf" in unknown["error"]["message"]
    assert (
        handler.handle(request("tools/call", {"name": "list_runs", "arguments": []}))["error"]["code"] == INVALID_PARAMS
    )
    assert handler.handle(request("resources/list"))["error"]["code"] == METHOD_NOT_FOUND


def test_clients_are_named_for_the_approval_dialog():
    assert client_label({"name": "claude-code", "version": "2.1"}) == "Claude Code"
    assert client_label({"name": "my-agent", "title": "My Agent"}) == "My Agent"
    assert client_label({"name": "my-agent"}) == "my-agent"
    assert client_label({"name": "pi-mcp-thetaide", "version": "7.0.1"}) == "pi"
    assert client_label(None) == UNKNOWN_CLIENT


# ── HTTP ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def server():
    calls = Calls()
    mcp = MCPServer(lambda: TOOLS, calls)
    mcp.start()
    mcp.calls = calls
    yield mcp
    mcp.stop()


def post(server, body, *, token=None, session=None, host=None, origin=None, method="POST"):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Host": host or f"127.0.0.1:{server.port}",
    }
    if token is not False:
        headers["Authorization"] = f"Bearer {token or server.token}"
    if session:
        headers["Mcp-Session-Id"] = session
    if origin:
        headers["Origin"] = origin
    data = json.dumps(body).encode() if body is not None else None
    connection.request(method, "/mcp", body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    return response.status, dict(response.getheaders()), (json.loads(raw) if raw else None)


def test_a_session_over_http(server):
    status, headers, body = post(
        server, request("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "claude-code"}})
    )
    assert status == 200 and body["result"]["serverInfo"]["name"] == "thetaide"
    session = headers["Mcp-Session-Id"]
    status, _, body = post(server, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session=session)
    assert status == 202 and body is None
    status, _, body = post(server, request("tools/list", rid=2), session=session)
    assert [t["name"] for t in body["result"]["tools"]] == ["list_runs", "write_file"]
    post(server, request("tools/call", {"name": "list_runs", "arguments": {}}, rid=3), session=session)
    assert server.calls.seen == [("list_runs", {}, "Claude Code")]  # named from the session's clientInfo

    assert post(server, request("ping"), session="not-a-session")[0] == 404  # restart: initialize again
    assert post(server, None, session=session, method="GET")[0] == 405  # no server-to-client stream
    assert post(server, None, session=session, method="DELETE")[0] == 200
    assert post(server, request("ping"), session=session)[0] == 404  # ended


def test_only_this_machine_with_the_token_gets_in(server):
    ping = request("ping")
    assert post(server, ping)[0] == 200
    status, headers, _ = post(server, ping, token=False)
    assert status == 401 and headers["WWW-Authenticate"].startswith("Bearer")
    assert post(server, ping, token="guess")[0] == 401
    assert post(server, ping, token="guéss")[0] == 401  # not ASCII: refused, not a crashed request
    assert post(server, ping, host="evil.example:80")[0] == 403  # DNS rebinding
    assert post(server, ping, origin="https://evil.example")[0] == 403  # a web page
    assert post(server, ping, origin=f"http://localhost:{server.port}")[0] == 200
    assert server.calls.seen == []


def test_a_chunked_body_is_read_like_claude_code_sends_it(server):
    # Claude Code streams its JSON-RPC body (Transfer-Encoding: chunked, no Content-Length).
    body = json.dumps(request("initialize", {"clientInfo": {"name": "claude-code"}})).encode()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.request(
        "POST",
        "/mcp",
        body=iter([body[:20], body[20:]]),
        headers={"Authorization": f"Bearer {server.token}", "Content-Type": "application/json"},
        encode_chunked=True,
    )
    response = connection.getresponse()
    assert response.status == 200 and json.loads(response.read())["result"]["serverInfo"]["name"] == "thetaide"
    assert response.getheader("Connection") == "close"  # nothing left over for a next request to trip on
    connection.close()


def test_a_newer_clients_discovery_probe_gets_method_not_found_so_it_falls_back_to_initialize():
    handler = MCPHandler(lambda: TOOLS, Calls())
    probe = {"jsonrpc": "2.0", "id": "server-discover-probe-1", "method": "server/discover", "params": {}}
    assert handler.handle(probe)["error"]["code"] == METHOD_NOT_FOUND


def test_bad_bodies_are_refused(server):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.request(
        "POST",
        "/mcp",
        body=b"{not json",
        headers={"Authorization": f"Bearer {server.token}", "Host": f"127.0.0.1:{server.port}"},
    )
    response = connection.getresponse()
    assert response.status == 400 and json.loads(response.read())["error"]["code"] == -32700
    connection.close()
    assert post(server, [request("ping")])[0] == 400  # batches are not part of MCP 2025-06-18


def test_each_start_has_a_new_token_and_stopping_closes_the_port():
    first, second = MCPServer(lambda: [], Calls()), MCPServer(lambda: [], Calls())
    assert first.token != second.token and len(first.token) >= 40
    first.start()
    port = first.port
    first.stop()
    with pytest.raises(OSError):
        http.client.HTTPConnection("127.0.0.1", port, timeout=2).request("GET", "/mcp")
    assert not first.running


# ── The workspace's .mcp.json and the terminal's environment ─────────────────


def test_mcp_json_names_the_server_without_its_secret(tmp_path):
    assert project_config.write_project_config(tmp_path).startswith("Added")
    config = json.loads((tmp_path / ".mcp.json").read_text())
    entry = config["mcpServers"]["thetaide"]
    assert entry["type"] == "http" and entry["url"].startswith("${THETA_MCP_URL:-")
    assert entry["headers"]["Authorization"] == "Bearer ${THETA_MCP_TOKEN:-none}"
    assert project_config.write_project_config(tmp_path) == ".mcp.json already lists ThetaIDE."


def test_mcp_json_keeps_other_servers_and_leaves_a_broken_file_alone(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"github": {"type": "http", "url": "x"}}}))
    project_config.write_project_config(tmp_path)
    servers = json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"github", "thetaide"} and servers["github"] == {"type": "http", "url": "x"}

    (tmp_path / ".mcp.json").write_text("{ oops")
    assert "not changed" in project_config.write_project_config(tmp_path)
    assert (tmp_path / ".mcp.json").read_text() == "{ oops"


def test_the_address_and_token_go_to_the_environment_not_to_disk(tmp_path):
    env: dict[str, str] = {}
    project_config.export("http://127.0.0.1:5000/mcp", "tok", env)
    assert env == {"THETA_MCP_URL": "http://127.0.0.1:5000/mcp", "THETA_MCP_TOKEN": "tok"}
    project_config.unexport(env)
    assert env == {}


# ── The plugin ───────────────────────────────────────────────────────────────


class FakeHost:
    def __init__(self):
        self.calls = Calls({"text": "ok", "is_error": False, "images": []})

    def tool_definitions(self):
        return TOOLS

    def call_tool(self, name, arguments, who):
        return self.calls(name, arguments, who)


def make_plugin(tmp_path, host=None, settings=None):
    from frontend.plugins.base import PluginManifest

    from plugins.theta_mcp import ThetaMCPPlugin

    class Context:
        window = SimpleNamespace()

        def __init__(self):
            self.settings = settings if settings is not None else {}
            self.messages: list[str] = []
            self.commands: list[str] = []

        def register_command(self, command_id, title, callback, **_):
            self.commands.append(command_id)

        def get_workspace_dir(self):
            return tmp_path

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def set_setting(self, key, value):
            self.settings[key] = value

        def log(self, message):
            self.messages.append(message)

        def show_status_message(self, message, timeout_ms=4000):
            pass

    manifest = PluginManifest(
        id="theta_mcp",
        name="Theta MCP Server",
        version="1.0.0",
        description="",
        author="",
        entry_point="ThetaMCPPlugin",
    )
    plugin = ThetaMCPPlugin(manifest)
    plugin.host = host if host is not None else FakeHost()  # the real ToolHost needs ThetaIDE's window
    plugin.home = tmp_path / "home"  # never the real home folder or token
    plugin.token_folder = tmp_path / "home" / ".config" / "thetaide"
    context = Context()
    return plugin, context


def test_starting_serves_thetaides_tools_to_new_terminals(tmp_path, monkeypatch):
    env: dict[str, str] = {}
    monkeypatch.setattr(project_config.os, "environ", env)
    host = FakeHost()
    plugin, context = make_plugin(tmp_path, host)
    plugin.activate(context)
    assert context.commands == ["theta_mcp.start", "theta_mcp.stop"] and plugin.server is None  # off by default
    try:
        assert plugin.start() is True
        assert env["THETA_MCP_URL"] == plugin.server.url and env["THETA_MCP_TOKEN"] == plugin.server.token
        assert (tmp_path / ".mcp.json").exists() and plugin.server.token not in (tmp_path / ".mcp.json").read_text()
        assert context.settings["enabled"] is True and "agent" in context.messages[-1]

        status, headers, _ = post(plugin.server, request("initialize", {"clientInfo": {"name": "claude-code"}}))
        session = headers["Mcp-Session-Id"]
        _, _, body = post(plugin.server, request("tools/call", {"name": "list_runs", "arguments": {}}), session=session)
        assert body["result"]["content"][0]["text"] == "ok"
        assert host.calls.seen == [("list_runs", {}, "Claude Code")]
    finally:
        plugin.stop()
    assert env == {} and context.settings["enabled"] is False


def test_mcp_json_goes_to_the_project_root_not_the_data_folder(tmp_path, monkeypatch):
    # In a real session get_workspace_dir() is the .thetaide data folder; WorkspacePaths.root is the project.
    monkeypatch.setattr(project_config.os, "environ", {})
    plugin, context = make_plugin(tmp_path, FakeHost())
    data = tmp_path / ".thetaide"
    data.mkdir()
    context.window.workspace_paths = SimpleNamespace(root=tmp_path, data_dir=data)
    context.get_workspace_dir = lambda: data
    plugin.activate(context)
    try:
        assert plugin.start()
        assert (tmp_path / ".mcp.json").exists() and not (data / ".mcp.json").exists()
    finally:
        plugin.stop()


def test_it_restarts_with_the_ide_and_stops_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(project_config.os, "environ", {})
    plugin, context = make_plugin(tmp_path, FakeHost(), settings={"enabled": True})
    plugin.activate(context)
    try:
        assert plugin.server is not None and plugin.server.running
    finally:
        plugin.deactivate()
    assert plugin.server is None and context.settings["enabled"] is True  # comes back when re-enabled


def test_the_settings_page_turns_the_server_on_and_off(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from plugins.theta_mcp.settings_widget import MCPSettings

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(project_config.os, "environ", {})
    plugin, context = make_plugin(tmp_path, FakeHost())
    plugin.activate(context)
    page = MCPSettings(plugin)
    try:
        assert not page.toggle.isChecked() and page.status.text().startswith("Stopped.")
        page.toggle.setChecked(True)
        assert plugin.server is not None and plugin.server.url in page.status.text()
        plugin.stop()  # from the command palette: the page follows
        assert not page.toggle.isChecked()
    finally:
        plugin.stop()
        page.deleteLater()
        app.processEvents()


def test_the_manifest_ships_every_module_and_depends_on_no_other_plugin():
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / ".theta_component.json").read_text())
    shipped = set(manifest["payload"]["implementations"])
    modules = [p for p in folder.rglob("*.py") if not p.name.startswith("test_") and "__pycache__" not in p.parts]
    assert shipped == {p.relative_to(folder).as_posix() for p in modules}
    assert manifest["dependencies"]["components"] == []
    for module in modules:  # nothing reaches into another plugin
        text = module.read_text()
        assert "ai_assistant" not in text.replace("plugins/ai_assistant", "") and "plugin_manager" not in text
    assert all((folder / r).is_file() for r in manifest["payload"]["resources"])
    plugin = json.loads((folder / "plugin.json").read_text())
    assert plugin["id"] == manifest["id"] == "theta_mcp" and plugin["version"] == manifest["version"]


def test_waits_are_kept_under_the_request_timeout_most_clients_use(tmp_path, monkeypatch):
    from plugins.theta_mcp import MAX_WAIT_SECONDS, capped_wait

    monkeypatch.setattr(project_config.os, "environ", {})
    host = FakeHost()
    plugin, context = make_plugin(tmp_path, host)
    plugin.activate(context)
    plugin.call_tool("wait_for_runs", {"run_ids": ["r1"], "timeout_minutes": 30}, "pi")
    plugin.call_tool("list_runs", {"limit": 3}, "pi")
    assert host.calls.seen[0][1] == {"run_ids": ["r1"], "timeout_minutes": MAX_WAIT_SECONDS / 60}
    assert host.calls.seen[1][1] == {"limit": 3}  # other tools are passed through untouched
    assert capped_wait(None) == capped_wait("soon") == MAX_WAIT_SECONDS / 60
    assert capped_wait(0.25) == 0.25 and capped_wait(-1) == 0.0
    assert MAX_WAIT_SECONDS < 60


# ── Each agent's own config file ────────────────────────────────────────────


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stable():
    from plugins.theta_mcp import harnesses

    return harnesses.Target("http://127.0.0.1:47321/mcp", "secret-token-" + "x" * 30)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


WINDOWS = sys.platform == "win32"


def _private(path: Path) -> bool:
    """Readable by the user only: mode 0600. Windows has no modes; the user folder's permissions apply there."""
    return WINDOWS or _mode(path) == 0o600


def test_project_files_name_the_variables_and_keep_other_servers(tmp_path):
    from plugins.theta_mcp import harnesses

    project, home = tmp_path / "p", tmp_path / "home"
    project.mkdir()
    (project / ".cursor").mkdir()
    (project / ".cursor" / "mcp.json").write_text('{"mcpServers": {"other": {"url": "http://x"}}}')
    none = harnesses.Target(None, None)
    notes = harnesses.configure(["claude", "opencode", "cursor"], project, home, none)
    assert [n.split(":")[0] for n in notes] == ["Claude Code / pi", "opencode", "Cursor"] and all(
        "added" in n for n in notes
    )

    opencode = json.loads((project / "opencode.json").read_text())
    assert opencode["$schema"] == "https://opencode.ai/config.json"
    assert opencode["mcp"]["thetaide"] == {
        "type": "remote",
        "url": "{env:THETA_MCP_URL}",
        "headers": {"Authorization": "Bearer {env:THETA_MCP_TOKEN}"},
    }
    cursor = json.loads((project / ".cursor" / "mcp.json").read_text())["mcpServers"]
    assert cursor["other"] == {"url": "http://x"}
    assert cursor["thetaide"] == {
        "url": "${env:THETA_MCP_URL}",
        "headers": {"Authorization": "Bearer ${env:THETA_MCP_TOKEN}"},
    }
    assert "type" not in cursor["thetaide"]  # Cursor's CLI drops the whole file for "streamable-http"
    assert not home.exists()  # nothing outside the project without the fixed address

    again = harnesses.configure(["claude", "opencode", "cursor"], project, home, none)
    assert all("already lists" in n for n in again)
    removed = harnesses.unconfigure(["opencode", "cursor"], project, home)
    assert all("removed" in n for n in removed)
    assert json.loads((project / ".cursor" / "mcp.json").read_text()) == {"mcpServers": {"other": {"url": "http://x"}}}


def test_opencode_follows_the_files_layout_and_leaves_commented_files_alone(tmp_path):
    from plugins.theta_mcp import harnesses

    none = harnesses.Target(None, None)
    v2 = tmp_path / "v2"
    (v2 / ".opencode").mkdir(parents=True)
    (v2 / ".opencode" / "opencode.json").write_text('{"mcp": {"servers": {"ctx": {"type": "remote", "url": "u"}}}}')
    harnesses.configure(["opencode"], v2, tmp_path / "home", none)
    servers = json.loads((v2 / ".opencode" / "opencode.json").read_text())["mcp"]["servers"]
    assert set(servers) == {"ctx", "thetaide"}
    assert not (v2 / "opencode.json").exists()

    commented = tmp_path / "c"
    commented.mkdir()
    text = '{\n  // my settings\n  "model": "x"\n}\n'
    (commented / "opencode.jsonc").write_text(text)
    [note] = harnesses.configure(["opencode"], commented, tmp_path / "home", none)
    assert "not plain JSON" in note and "by hand" in note
    assert (commented / "opencode.jsonc").read_text() == text


def test_agents_that_need_the_fixed_address_wait_for_it(tmp_path):
    from plugins.theta_mcp import harnesses

    notes = harnesses.configure(["codex", "antigravity"], tmp_path, tmp_path / "home", harnesses.Target(None, None))
    assert all("needs the fixed address" in n for n in notes)
    assert not (tmp_path / "home").exists()


def test_cursor_with_the_fixed_address_uses_one_private_global_entry(tmp_path):
    from plugins.theta_mcp import harnesses

    project, home = tmp_path / "p", tmp_path / "home"
    project.mkdir()
    harnesses.configure(["cursor"], project, home, harnesses.Target(None, None))
    [note] = harnesses.configure(["cursor"], project, home, _stable())
    assert note.startswith("Cursor: added ThetaIDE in ~/.cursor/mcp.json") or "added ThetaIDE in" in note
    global_file = home / ".cursor" / "mcp.json"
    entry = json.loads(global_file.read_text())["mcpServers"]["thetaide"]
    assert entry == {"url": _stable().url, "headers": {"Authorization": f"Bearer {_stable().token}"}}
    assert _private(global_file)
    # the project's variable-based entry would win over the global one in the editor, where they are unset
    assert "thetaide" not in json.loads((project / ".cursor" / "mcp.json").read_text())["mcpServers"]


def test_codex_gets_a_marked_block_that_parses_and_keeps_the_rest(tmp_path):
    import tomllib

    from plugins.theta_mcp import harnesses

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    config = home / ".codex" / "config.toml"
    original = 'model = "gpt-5"\n\n[mcp_servers.figma]\nurl = "https://mcp.figma.com/mcp"\n'
    config.write_text(original)
    config.chmod(0o644)
    [note] = harnesses.configure(["codex"], tmp_path, home, _stable())
    assert "added" in note
    parsed = tomllib.loads(config.read_text())
    assert parsed["model"] == "gpt-5" and parsed["mcp_servers"]["figma"]["url"] == "https://mcp.figma.com/mcp"
    entry = parsed["mcp_servers"]["thetaide"]
    assert entry["url"] == _stable().url
    assert entry["http_headers"] == {"Authorization": f"Bearer {_stable().token}"}
    assert entry["tool_timeout_sec"] == 120 and entry["default_tools_approval_mode"] == "writes"
    assert _private(config)

    assert "already lists" in harnesses.configure(["codex"], tmp_path, home, _stable())[0]
    renewed = harnesses.Target(_stable().url, "another-token-" + "y" * 30)
    assert "updated" in harnesses.configure(["codex"], tmp_path, home, renewed)[0]
    text = config.read_text()
    assert text.count(harnesses.CODEX_BEGIN) == 1 and _stable().token not in text
    assert "removed" in harnesses.unconfigure(["codex"], tmp_path, home)[0]
    assert config.read_text() == original


def test_codex_leaves_a_hand_written_thetaide_table_alone(tmp_path):
    from plugins.theta_mcp import harnesses

    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    mine = '[mcp_servers.thetaide]\nurl = "http://127.0.0.1:1/mcp"\n'
    (home / ".codex" / "config.toml").write_text(mine)
    [note] = harnesses.configure(["codex"], tmp_path, home, _stable())
    assert "was not changed" in note
    assert (home / ".codex" / "config.toml").read_text() == mine


def test_antigravity_uses_server_url_and_updates_the_older_location_too(tmp_path):
    from plugins.theta_mcp import harnesses

    home = tmp_path / "home"
    (home / ".gemini" / "antigravity").mkdir(parents=True)
    (home / ".gemini" / "antigravity" / "mcp_config.json").write_text('{"mcpServers": {}}')
    notes = harnesses.configure(["antigravity"], tmp_path, home, _stable())[0]
    assert notes.count("added") == 2
    for folder in ("config", "antigravity"):
        path = home / ".gemini" / folder / "mcp_config.json"
        entry = json.loads(path.read_text())["mcpServers"]["thetaide"]
        assert entry == {
            "serverUrl": _stable().url,
            "headers": {"Authorization": f"Bearer {_stable().token}"},
            "timeoutSeconds": 120,
        }
        assert _private(path)


def test_the_fixed_token_lives_in_a_private_user_file(tmp_path):
    folder = tmp_path / "cfg"
    token = project_config.fixed_token(folder)
    assert project_config.fixed_token(folder) == token and len(token) >= 32
    assert _private(folder / "mcp_token")
    assert project_config.fixed_token(folder, renew=True) != token
    assert project_config.user_config_dir({"XDG_CONFIG_HOME": "/x"}) in (Path("/x/thetaide"), Path("/x") / "thetaide")


def test_with_the_fixed_address_restarts_keep_the_address_and_token(tmp_path, monkeypatch):
    import tomllib

    monkeypatch.setattr(project_config.os, "environ", {})
    port = _free_port()
    plugin, context = make_plugin(tmp_path, FakeHost(), settings={"agents": ["claude", "codex"], "port": port})
    plugin.activate(context)
    try:
        assert plugin.start()
        assert "Codex: needs the fixed address" in " ".join(plugin.notes)  # still a new address per start
        plugin.set_fixed_address(True)  # restarts the running server
        url, token = plugin.server.url, plugin.server.token
        assert url == f"http://127.0.0.1:{port}/mcp"
        codex = tomllib.loads((plugin.home / ".codex" / "config.toml").read_text())["mcp_servers"]["thetaide"]
        assert codex["url"] == url and codex["http_headers"]["Authorization"] == f"Bearer {token}"
        plugin.stop()
        plugin.start()
        assert (plugin.server.url, plugin.server.token) == (url, token)
        assert token not in (tmp_path / ".mcp.json").read_text()  # the project file still names variables only
        assert token not in json.dumps(context.settings)  # nor the workspace's plugin settings

        plugin.renew_token()
        assert plugin.server.token != token
        assert plugin.server.token in (plugin.home / ".codex" / "config.toml").read_text()

        plugin.set_agents(["claude"])  # dropping an agent removes ThetaIDE from its file
        assert "thetaide" not in (plugin.home / ".codex" / "config.toml").read_text()
        plugin.set_agents(["claude", "codex"])
        plugin.set_fixed_address(False)  # the fixed token is taken out of agents' files
        assert "thetaide" not in (plugin.home / ".codex" / "config.toml").read_text()
        assert plugin.server.token != project_config.fixed_token(plugin.token_folder)
    finally:
        plugin.stop()


def test_the_settings_page_greys_out_agents_that_need_the_fixed_address(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from plugins.theta_mcp.settings_widget import MCPSettings

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(project_config.os, "environ", {})
    plugin, context = make_plugin(tmp_path, FakeHost(), settings={"port": _free_port()})
    plugin.activate(context)
    page = MCPSettings(plugin)
    try:
        boxes = page.agent_boxes
        assert boxes["claude"].isChecked() and not boxes["codex"].isEnabled() and not page.renew.isEnabled()
        boxes["opencode"].setChecked(True)
        assert context.settings["agents"] == ["claude", "opencode"]
        page.fixed.setChecked(True)
        assert boxes["codex"].isEnabled() and page.renew.isEnabled()
        boxes["codex"].setChecked(True)
        page.toggle.setChecked(True)
        assert (plugin.home / ".codex" / "config.toml").exists()
        assert any(n.startswith("Codex: added") for n in plugin.notes) and "Codex: added" in page.notes.text()
    finally:
        plugin.stop()
        page.deleteLater()
        app.processEvents()


def test_with_the_fixed_address_claude_pi_and_opencode_work_from_any_terminal(tmp_path):
    from plugins.theta_mcp import harnesses

    project, home = tmp_path / "p", tmp_path / "home"
    project.mkdir()
    home.mkdir()
    claude_state = {"numStartups": 12, "projects": {"/x": {"allowedTools": []}}, "mcpServers": {"gh": {"url": "u"}}}
    (home / ".claude.json").write_text(json.dumps(claude_state))
    harnesses.configure(["claude", "opencode"], project, home, harnesses.Target(None, None))
    assert "thetaide" in json.loads((project / ".mcp.json").read_text())["mcpServers"]

    stable = _stable()
    notes = harnesses.configure(["claude", "opencode"], project, home, stable)
    assert all("(and removed ThetaIDE from" in n for n in notes)  # the project entries would win over these
    bearer = {"Authorization": f"Bearer {stable.token}"}
    claude = json.loads((home / ".claude.json").read_text())
    assert claude["numStartups"] == 12 and claude["projects"] == claude_state["projects"]  # the rest is kept
    assert claude["mcpServers"]["gh"] == {"url": "u"}
    assert claude["mcpServers"]["thetaide"] == {"type": "http", "url": stable.url, "headers": bearer}
    pi = json.loads((home / ".config" / "mcp" / "mcp.json").read_text())["mcpServers"]["thetaide"]
    assert pi == claude["mcpServers"]["thetaide"]
    opencode = json.loads((home / ".config" / "opencode" / "opencode.json").read_text())
    assert opencode["mcp"]["thetaide"] == {"type": "remote", "url": stable.url, "headers": bearer}
    for path in (
        home / ".claude.json",
        home / ".config" / "mcp" / "mcp.json",
        home / ".config" / "opencode" / "opencode.json",
    ):
        assert _private(path)
    assert "thetaide" not in json.loads((project / ".mcp.json").read_text())["mcpServers"]
    assert "thetaide" not in json.loads((project / "opencode.json").read_text())["mcp"]

    harnesses.unconfigure(["claude", "opencode"], project, home)  # turning the fixed address off
    for path in (
        home / ".claude.json",
        home / ".config" / "mcp" / "mcp.json",
        home / ".config" / "opencode" / "opencode.json",
    ):
        assert stable.token not in path.read_text()
    assert json.loads((home / ".claude.json").read_text())["mcpServers"] == {"gh": {"url": "u"}}


def test_turning_the_fixed_address_off_brings_back_the_project_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(project_config.os, "environ", {})
    plugin, context = make_plugin(tmp_path, FakeHost(), settings={"agents": ["claude"], "port": _free_port()})
    plugin.activate(context)
    try:
        plugin.set_fixed_address(True)  # while stopped: nothing to restart
        assert plugin.server is None
        plugin.start()
        token = plugin.server.token
        assert token in (plugin.home / ".claude.json").read_text()
        assert not (tmp_path / ".mcp.json").exists()  # the project gets no entry, and no file is made for one
        plugin.set_fixed_address(False)
        assert token not in (plugin.home / ".claude.json").read_text()
        assert json.loads((tmp_path / ".mcp.json").read_text())["mcpServers"]["thetaide"] == project_config.ENTRY
    finally:
        plugin.stop()


# ── The tools: a generated copy, run by ToolHost ────────────────────────────


def test_the_copied_tools_match_the_ai_assistants():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "sync_mcp_tools.py"
    if not script.is_file():
        pytest.skip("runs in the theta-plugins repository")
    spec = importlib.util.spec_from_file_location("sync_mcp_tools", script)
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert sync.stale() == [], "Run: python scripts/sync_mcp_tools.py"


def _host(tmp_path, ran):
    from plugins.theta_mcp.host import ToolHost
    from plugins.theta_mcp.ide_tools.tools import Tool, ToolError, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        Tool(
            "list_runs", "List runs.", {"type": "object", "properties": {}}, lambda: ran.append("list") or {"total": 0}
        )
    )
    registry.register(
        Tool(
            "launch_experiment",
            "Launch.",
            {"type": "object", "properties": {}},
            lambda: ran.append("launch") or "queued",
            mutating=True,
        )
    )

    def impossible(path):
        raise ToolError(f"{path} is outside the workspace.")

    registry.register(
        Tool(
            "write_file",
            "Write.",
            {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            lambda path: ran.append("write") or "written",
            mutating=True,
            preview=impossible,
        )
    )
    statuses: list[str] = []
    host = ToolHost(SimpleNamespace(), tmp_path, tmp_path / "figures", status=statuses.append, tools=registry)
    return host, statuses


def test_reads_run_straight_away_and_changes_need_approval(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from plugins.theta_mcp import host as host_module

    app = QApplication.instance() or QApplication([])
    ran: list[str] = []
    host, statuses = _host(tmp_path, ran)
    asked: list[tuple] = []
    answer = {"approve": False}

    def confirm(call, preview, who):
        asked.append((call.name, who))
        return answer["approve"]

    host.confirm = confirm
    assert host.call_tool("list_runs", {}, "Claude Code") == {
        "text": '{\n "total": 0\n}',
        "is_error": False,
        "images": [],
    }
    assert asked == [] and statuses[-1] == "Claude Code used list_runs"  # reading needs no approval

    declined = host.call_tool("launch_experiment", {}, "Claude Code")
    assert declined["is_error"] and declined["text"] == host_module.DECLINED
    assert asked == [("launch_experiment", "Claude Code")] and "launch" not in ran
    answer["approve"] = True
    assert host.call_tool("launch_experiment", {}, "Claude Code")["text"] == "queued"

    failed = host.call_tool("write_file", {"path": "/etc/passwd"}, "Codex")
    assert failed["is_error"] and "outside the workspace" in failed["text"]
    assert ("write_file", "Codex") not in asked  # it could not work: the user is not asked
    assert [(d["name"], d["mutating"]) for d in host.tool_definitions()] == [
        ("list_runs", False),
        ("launch_experiment", True),
        ("write_file", True),
    ]
    host.close()
    assert host.call_tool("list_runs", {}, "pi")["text"] == '{\n "total": 0\n}'  # same thread: still answers
    app.processEvents()


def test_requests_from_the_servers_threads_run_on_the_gui_thread(tmp_path):
    import os
    import threading

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    ran: list[str] = []
    host, _ = _host(tmp_path, ran)
    threads: list[str] = []
    original = host.tools.execute

    def execute(call):
        threads.append(threading.current_thread().name)
        return original(call)

    host.tools.execute = execute
    out: dict = {}
    worker = threading.Thread(target=lambda: out.update(r=host.call_tool("list_runs", {}, "pi")), name="mcp-request")
    worker.start()
    while worker.is_alive():
        app.processEvents()
    assert out["r"]["text"] == '{\n "total": 0\n}' and threads == [threading.main_thread().name]

    host.close()  # shutting down: a request from another thread is refused, not hung
    worker = threading.Thread(target=lambda: out.update(r=host.call_tool("list_runs", {}, "pi")))
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and out["r"]["is_error"] and "shutting down" in out["r"]["text"]


def test_the_approval_dialog_names_the_agent(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from plugins.theta_mcp import host as host_module
    from plugins.theta_mcp.ide_tools.chat_types import ToolCall

    app = QApplication.instance() or QApplication([])
    host, statuses = _host(tmp_path, [])
    shown: list[tuple] = []

    class Dialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, title, summary, preview, is_diff=False, parent=None, actor="", self_named=False):
            shown.append((summary, actor))
            assert self_named  # the dialog says the name is the agent's own claim

        def exec(self):
            return 0

    monkeypatch.setattr(host_module, "ApprovalDialog", Dialog)
    assert (
        host.confirm(ToolCall("1", "write_file", {"path": "configs/a.yaml"}), "--- a/configs/a.yaml\n", "Codex")
        is False
    )
    assert shown == [("Codex wants to change <b>configs/a.yaml</b>:", "Codex")]
    assert statuses[-1] == "Rejected configs/a.yaml for Codex"
    app.processEvents()


def test_config_files_are_replaced_whole_keeping_links_and_permissions(tmp_path):
    import os

    from plugins.theta_mcp.project_config import atomic_write

    real = tmp_path / "dotfiles" / "claude.json"
    real.parent.mkdir()
    real.write_text("{}")
    os.chmod(real, 0o640)
    link = tmp_path / ".claude.json"
    try:
        link.symlink_to(real)
    except OSError:  # Windows without symlink rights
        pytest.skip("symlinks are not available")
    atomic_write(link, '{"a": 1}\n')
    assert link.is_symlink() and real.read_text() == '{"a": 1}\n'  # the link still points at the dotfile
    assert WINDOWS or _mode(real) == 0o640  # its permissions are kept
    atomic_write(link, "{}\n", private=True)
    assert _private(real)  # holding the token: the user only
    fresh = tmp_path / "new" / "token"
    atomic_write(fresh, "t\n", private=True)
    assert fresh.read_text() == "t\n" and _private(fresh)
    assert [p.name for p in tmp_path.rglob("*.tmp")] == []  # no temporary files left behind


def test_moved_config_folders_are_followed_for_the_real_home_only(tmp_path, monkeypatch):
    from plugins.theta_mcp import harnesses

    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir(), project.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    target = _stable()

    # Another home (as in every other test): the variables describe the real user's agents, so they are ignored.
    harnesses.configure(["codex", "opencode"], project, home, target)
    assert (home / ".codex" / "config.toml").is_file()
    assert (home / ".config" / "opencode" / "opencode.json").is_file()
    assert not (tmp_path / "codex-home").exists() and not (tmp_path / "xdg").exists()

    # The real home: Codex's and opencode's moved folders are where their entries go, and come out of.
    monkeypatch.setattr(Path, "home", lambda: home)
    harnesses.configure(["codex", "opencode"], project, home, target)
    codex = tmp_path / "codex-home" / "config.toml"
    opencode = tmp_path / "xdg" / "opencode" / "opencode.json"
    assert "[mcp_servers.thetaide]" in codex.read_text() and _private(codex)
    assert "thetaide" in json.loads(opencode.read_text())["mcp"] and _private(opencode)
    harnesses.unconfigure(["codex", "opencode"], project, home)
    assert "thetaide" not in codex.read_text() and "thetaide" not in json.loads(opencode.read_text())["mcp"]
