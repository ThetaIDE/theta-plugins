"""Tell each coding agent where the server is: one writer per agent's own config file.

Agents read different files with different schemas, and differ in whether those files may refer to
environment variables:

  agent         without the fixed address (ThetaIDE's terminal)    with it (anywhere), written out
  Claude Code   <project>/.mcp.json  ${THETA_MCP_URL}                  ~/.claude.json
  pi            <project>/.mcp.json  ${THETA_MCP_URL}                  ~/.config/mcp/mcp.json
  opencode      <project>/opencode.json  {env:THETA_MCP_URL}           ~/.config/opencode/opencode.json ($XDG_CONFIG_HOME)
  Cursor        <project>/.cursor/mcp.json  ${env:THETA_MCP_URL}       ~/.cursor/mcp.json
  Codex         -                                                    ~/.codex/config.toml ($CODEX_HOME)
  Antigravity   -                                                    ~/.gemini/config/mcp_config.json

Project files only name the variables, which only terminals opened in ThetaIDE have, so they hold no secret
and are safe to commit. With the fixed address (an address and token that stay the same across restarts), each
agent's entry goes to its user-wide file instead, written out (never into the project), and that file is made
readable by the user only. The project entry is removed then, since agents prefer it and outside ThetaIDE its
variables are unset. Codex (no variables in its URL) and Antigravity (no documented variables) only work this way.

Every writer only adds, updates or removes the "thetaide" entry and keeps everything else in the file. A file
it cannot parse safely (invalid JSON, JSON with comments, an existing hand-written Codex table) is left alone,
and the message says what to add by hand.

The same paths hold on macOS and Windows (~ is the user's home folder, %USERPROFILE% on Windows): these agents
keep their config under the home folder on every system. Codex and opencode let the user move theirs with
$CODEX_HOME and $XDG_CONFIG_HOME, which are followed. "Readable by the user only" is file mode 0600 on macOS and
Linux; Windows ignores modes, and there the home folder's own permissions (the user, administrators and the system)
keep the files private.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project_config import ENTRY, SERVER_NAME, TOKEN_VAR, URL_VAR, atomic_write

# Codex allows 60 s per tool call and Antigravity 180 s by default. Each wait_for_runs call stays under 50 s
# (see MAX_WAIT_SECONDS), but launches validate first and Hub installs download: give those room.
TOOL_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class Target:
    """Where the server should be, as an agent's config states it."""

    url: str | None  # the fixed address, or None: the address changes every start
    token: str | None  # the fixed token, or None

    @property
    def stable(self) -> bool:
        return self.url is not None and self.token is not None


@dataclass(frozen=True)
class Harness:
    id: str
    name: str
    needs_stable: bool  # works only with the fixed address and token
    where: str  # the file, for the settings page
    write: Callable[[Path, Path, Target], str]  # (project, home, target) -> what happened
    remove: Callable[[Path, Path], str]


# ── JSON files ─────────────────────────────────────────────────────────────


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    """The file's object ({} if it does not exist), or None and why it must be left alone."""
    if not path.exists():
        return {}, ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"could not be read ({exc})"
    try:
        data = json.loads(text or "{}")
    except json.JSONDecodeError:
        return None, "is not plain JSON (comments or a syntax error), so it was not changed"
    if not isinstance(data, dict):
        return None, "is not a JSON object, so it was not changed"
    return data, ""


def _write_file(path: Path, text: str, *, private: bool) -> None:
    atomic_write(path, text, private=private)  # private: it holds the token, so the user only


def _set_json_entry(
    path: Path,
    label: str,
    section: list[str],
    entry: dict[str, Any],
    *,
    private: bool = False,
    base: dict | None = None,
) -> str:
    data, problem = _read_json(path)
    if data is None:
        return f"{label}: {_shown(path)} {problem}. Add the ThetaIDE server to it by hand (see the plugin's README)."
    if not data and base:
        data.update(base)
    table: Any = data
    for key in section:
        table = table.setdefault(key, {})
        if not isinstance(table, dict):
            return f'{label}: {_shown(path)} was not changed: its "{key}" is not an object.'
    if table.get(SERVER_NAME) == entry:
        if private and path.exists():
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return f"{label}: {_shown(path)} already lists ThetaIDE."
    existed = SERVER_NAME in table
    table[SERVER_NAME] = entry
    try:
        _write_file(path, json.dumps(data, indent=2) + "\n", private=private)
    except OSError as exc:
        return f"{label}: {_shown(path)} could not be written ({exc})."
    return f"{label}: {'updated' if existed else 'added'} ThetaIDE in {_shown(path)}."


def _remove_json_entry(path: Path, label: str, section: list[str]) -> str:
    data, problem = _read_json(path)
    if data is None:
        return f"{label}: {_shown(path)} {problem}."
    table: Any = data
    for key in section:
        table = table.get(key) if isinstance(table, dict) else None
    if not isinstance(table, dict) or SERVER_NAME not in table:
        return f"{label}: ThetaIDE was not listed."
    del table[SERVER_NAME]
    try:
        atomic_write(path, json.dumps(data, indent=2) + "\n")
    except OSError as exc:
        return f"{label}: {_shown(path)} could not be written ({exc})."
    return f"{label}: removed ThetaIDE from {_shown(path)}."


def _shown(path: Path) -> str:
    """~/.codex/config.toml for a user file; .cursor/mcp.json or opencode.json for a project file."""
    try:
        relative = path.relative_to(Path.home())
        if relative.parts and relative.parts[0].startswith("."):
            return "~/" + relative.as_posix()
    except ValueError:
        pass
    return f"{path.parent.name}/{path.name}" if path.parent.name.startswith(".") else path.name


def _moved(variable: str, home: Path) -> Path | None:
    """A config folder the user moved with an environment variable ($CODEX_HOME, $XDG_CONFIG_HOME). Followed only
    for the real home folder: the variables describe this user's agents, not a home given for testing."""
    value = os.environ.get(variable, "").strip() if home == Path.home() else ""
    return Path(value).expanduser() if value else None


def _codex_file(home: Path) -> Path:
    return (_moved("CODEX_HOME", home) or home / ".codex") / "config.toml"


def _bearer(target: Target) -> dict[str, str]:
    return {"Authorization": f"Bearer {target.token}"}


# ── Project entries that move to the user's file with the fixed address ──────
# Project files name the variables, which only ThetaIDE's terminals have. With the fixed address, the entry goes
# to the agent's user-wide file instead, written out, so the agent works from any terminal or editor. The project
# entry is removed then: agents prefer it over the user-wide one, and outside ThetaIDE its variables are unset.


@dataclass(frozen=True)
class Placement:
    path: Path
    section: list[str]
    entry: dict[str, Any]


def _place(label: str, project: Placement, users: list[Placement], target: Target, base: dict | None = None) -> str:
    if not target.stable:
        return _set_json_entry(project.path, label, project.section, project.entry, base=base)
    removed = _remove_json_entry(project.path, label, project.section)
    notes = [_set_json_entry(u.path, label, u.section, u.entry, private=True, base=base) for u in users]
    if "removed" in removed:
        notes.append(f"(and {removed.split(': ', 1)[1]})")
    return " ".join(notes)


def _unplace(label: str, paths: list[tuple[Path, list[str]]]) -> str:
    notes = [_remove_json_entry(path, label, section) for path, section in paths]
    removed = [n for n in notes if "removed" in n]
    return " ".join(removed) if removed else notes[0]


# ── Claude Code and pi: .mcp.json; ~/.claude.json and ~/.config/mcp/mcp.json ──


def _claude_users(home: Path, target: Target) -> list[Placement]:
    entry = {"type": "http", "url": target.url, "headers": _bearer(target)}
    return [
        Placement(home / ".claude.json", ["mcpServers"], entry),  # Claude Code's user scope
        Placement(home / ".config" / "mcp" / "mcp.json", ["mcpServers"], entry),  # pi-mcp-adapter's shared file
    ]


def _claude_write(project: Path, home: Path, target: Target) -> str:
    here = Placement(project / ".mcp.json", ["mcpServers"], ENTRY)
    return _place("Claude Code / pi", here, _claude_users(home, target) if target.stable else [], target)


def _claude_remove(project: Path, home: Path) -> str:
    paths = [project / ".mcp.json", home / ".claude.json", home / ".config" / "mcp" / "mcp.json"]
    return _unplace("Claude Code / pi", [(path, ["mcpServers"]) for path in paths])


# ── opencode: opencode.json; ~/.config/opencode/opencode.json ───────────────

OPENCODE_ENTRY: dict[str, Any] = {
    "type": "remote",
    "url": f"{{env:{URL_VAR}}}",
    "headers": {"Authorization": f"Bearer {{env:{TOKEN_VAR}}}"},
}
OPENCODE_BASE = {"$schema": "https://opencode.ai/config.json"}


def _first_existing(candidates: list[Path]) -> Path:
    return next((c for c in candidates if c.exists()), candidates[0])


def _opencode_file(project: Path) -> Path:
    return _first_existing(
        [
            project / "opencode.json",
            project / "opencode.jsonc",
            project / ".opencode" / "opencode.json",
            project / ".opencode" / "opencode.jsonc",
        ]
    )


def _opencode_user_file(home: Path) -> Path:
    folder = (_moved("XDG_CONFIG_HOME", home) or home / ".config") / "opencode"
    return _first_existing([folder / "opencode.json", folder / "opencode.jsonc"])


def _opencode_section(path: Path) -> list[str]:
    """opencode 1.x lists servers under "mcp"; the 2.x docs nest them under "mcp.servers". Follow the file."""
    data, _ = _read_json(path)
    mcp = (data or {}).get("mcp")
    return ["mcp", "servers"] if isinstance(mcp, dict) and isinstance(mcp.get("servers"), dict) else ["mcp"]


def _opencode_write(project: Path, home: Path, target: Target) -> str:
    path = _opencode_file(project)
    here = Placement(path, _opencode_section(path), OPENCODE_ENTRY)
    users = []
    if target.stable:
        user = _opencode_user_file(home)
        entry = {"type": "remote", "url": target.url, "headers": _bearer(target)}
        users = [Placement(user, _opencode_section(user), entry)]
    return _place("opencode", here, users, target, base=OPENCODE_BASE)


def _opencode_remove(project: Path, home: Path) -> str:
    paths = [_opencode_file(project), _opencode_user_file(home)]
    return _unplace("opencode", [(path, _opencode_section(path)) for path in paths])


# ── Cursor: .cursor/mcp.json; ~/.cursor/mcp.json (the editor too) ───────────

CURSOR_ENTRY: dict[str, Any] = {  # no "type": the CLI drops the whole file for "streamable-http"
    "url": f"${{env:{URL_VAR}}}",
    "headers": {"Authorization": f"Bearer ${{env:{TOKEN_VAR}}}"},
}


def _cursor_write(project: Path, home: Path, target: Target) -> str:
    here = Placement(project / ".cursor" / "mcp.json", ["mcpServers"], CURSOR_ENTRY)
    users = [Placement(home / ".cursor" / "mcp.json", ["mcpServers"], {"url": target.url, "headers": _bearer(target)})]
    return _place("Cursor", here, users if target.stable else [], target)


def _cursor_remove(project: Path, home: Path) -> str:
    return _unplace(
        "Cursor", [(project / ".cursor" / "mcp.json", ["mcpServers"]), (home / ".cursor" / "mcp.json", ["mcpServers"])]
    )


# ── Antigravity: ~/.gemini/config/mcp_config.json ───────────────────────────


def _antigravity_files(home: Path) -> list[Path]:
    """The documented file, and the older location some installations still read, if it exists."""
    files = [home / ".gemini" / "config" / "mcp_config.json"]
    legacy = home / ".gemini" / "antigravity" / "mcp_config.json"
    return files + ([legacy] if legacy.exists() else [])


def _antigravity_write(project: Path, home: Path, target: Target) -> str:
    if not target.stable:
        return "Antigravity: needs the fixed address (turn it on above)."
    entry = {"serverUrl": target.url, "headers": _bearer(target), "timeoutSeconds": TOOL_TIMEOUT_SECONDS}
    return " ".join(
        _set_json_entry(path, "Antigravity", ["mcpServers"], entry, private=True) for path in _antigravity_files(home)
    )


def _antigravity_remove(project: Path, home: Path) -> str:
    return " ".join(_remove_json_entry(path, "Antigravity", ["mcpServers"]) for path in _antigravity_files(home))


# ── Codex: ~/.codex/config.toml ─────────────────────────────────────────────
# The standard library reads TOML but cannot write it, so the entry is a marked block this plugin owns.

CODEX_BEGIN = "# >>> ThetaIDE MCP server (written by ThetaIDE; changes inside this block are replaced)"
CODEX_END = "# <<< ThetaIDE MCP server"
CODEX_BLOCK = re.compile(re.escape(CODEX_BEGIN) + r".*?" + re.escape(CODEX_END) + r"\n?", re.S)


def _toml_string(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def codex_block(target: Target) -> str:
    return "\n".join(
        [
            CODEX_BEGIN,
            f"[mcp_servers.{SERVER_NAME}]",
            f"url = {_toml_string(str(target.url))}",
            f"http_headers = {{ Authorization = {_toml_string(f'Bearer {target.token}')} }}",
            f"tool_timeout_sec = {TOOL_TIMEOUT_SECONDS}",
            "# Reads run without asking; ThetaIDE itself asks before anything changes.",
            'default_tools_approval_mode = "writes"',
            CODEX_END,
            "",
        ]
    )


def _codex_parses(text: str) -> bool:
    try:
        import tomllib
    except ImportError:  # Python < 3.11: cannot check
        return True
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    return True


def _codex_write(project: Path, home: Path, target: Target) -> str:
    if not target.stable:
        return "Codex: needs the fixed address (turn it on above)."
    path = _codex_file(home)
    try:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        return f"Codex: {_shown(path)} could not be read ({exc})."
    block = codex_block(target)
    rest = CODEX_BLOCK.sub("", text)
    if re.search(rf"^\s*\[mcp_servers\.{SERVER_NAME}\]", rest, re.M) or not _codex_parses(rest):
        return (
            f"Codex: {_shown(path)} was not changed: it already has its own [mcp_servers.{SERVER_NAME}] table or "
            "does not parse. Remove that table to let ThetaIDE manage it."
        )
    if CODEX_BLOCK.search(text):
        new = CODEX_BLOCK.sub(lambda _m: block, text)
    else:
        new = text + ("" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n") + block
    if new == text:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return f"Codex: {_shown(path)} already lists ThetaIDE."
    if not _codex_parses(new):
        return f"Codex: {_shown(path)} was not changed: the result would not parse."
    try:
        _write_file(path, new, private=True)
    except OSError as exc:
        return f"Codex: {_shown(path)} could not be written ({exc})."
    return f"Codex: {'updated' if CODEX_BLOCK.search(text) else 'added'} ThetaIDE in {_shown(path)}."


def _codex_remove(project: Path, home: Path) -> str:
    path = _codex_file(home)
    try:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError as exc:
        return f"Codex: {_shown(path)} could not be read ({exc})."
    if not CODEX_BLOCK.search(text):
        return "Codex: ThetaIDE was not listed."
    new = CODEX_BLOCK.sub("", text)
    try:
        atomic_write(path, new.rstrip("\n") + "\n" if new.strip() else "")
    except OSError as exc:
        return f"Codex: {_shown(path)} could not be written ({exc})."
    return f"Codex: removed ThetaIDE from {_shown(path)}."


HARNESSES: list[Harness] = [
    Harness(
        "claude",
        "Claude Code and pi",
        False,
        ".mcp.json in the project (ThetaIDE's terminal only); with the fixed address, ~/.claude.json and "
        "~/.config/mcp/mcp.json (any terminal)",
        _claude_write,
        _claude_remove,
    ),
    Harness(
        "opencode",
        "opencode",
        False,
        "opencode.json in the project (ThetaIDE's terminal only); with the fixed address, "
        "~/.config/opencode/opencode.json, or under $XDG_CONFIG_HOME (any terminal)",
        _opencode_write,
        _opencode_remove,
    ),
    Harness(
        "cursor",
        "Cursor",
        False,
        ".cursor/mcp.json in the project (the CLI in ThetaIDE's terminal); with the fixed address, ~/.cursor/mcp.json "
        "(the editor and any terminal)",
        _cursor_write,
        _cursor_remove,
    ),
    Harness("codex", "Codex", True, "~/.codex/config.toml, or $CODEX_HOME/config.toml", _codex_write, _codex_remove),
    Harness(
        "antigravity", "Antigravity", True, "~/.gemini/config/mcp_config.json", _antigravity_write, _antigravity_remove
    ),
]
BY_ID = {h.id: h for h in HARNESSES}
DEFAULT_IDS = ["claude"]


def configure(ids: list[str], project: Path, home: Path, target: Target) -> list[str]:
    """Write each chosen agent's entry. Returns one message per agent."""
    return [BY_ID[i].write(project, home, target) for i in ids if i in BY_ID]


def unconfigure(ids: list[str], project: Path, home: Path) -> list[str]:
    return [BY_ID[i].remove(project, home) for i in ids if i in BY_ID]
