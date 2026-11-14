"""Point agents in ThetaIDE's terminal at the server: environment variables and the workspace's .mcp.json.

The server's address and token change every time it starts, so neither is written to disk. They go into
ThetaIDE's environment, which every terminal opened afterwards inherits, and .mcp.json refers to them by name:

    {"mcpServers": {"thetaide": {"type": "http", "url": "${THETA_MCP_URL:-...}",
                                 "headers": {"Authorization": "Bearer ${THETA_MCP_TOKEN:-...}"}}}}

Claude Code reads a project's .mcp.json and expands ${VAR:-default}. The defaults keep the file valid outside
ThetaIDE, where the server is simply unreachable. The file holds no secret, so it is safe to commit.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

SERVER_NAME = "thetaide"
URL_VAR = "THETA_MCP_URL"
TOKEN_VAR = "THETA_MCP_TOKEN"
CONFIG_FILE = ".mcp.json"
# Where ${VAR:-default} points when ThetaIDE is not running: port 9 (discard) refuses at once.
UNREACHABLE_URL = "http://127.0.0.1:9/mcp"

ENTRY: dict[str, Any] = {
    "type": "http",
    "url": f"${{{URL_VAR}:-{UNREACHABLE_URL}}}",
    "headers": {"Authorization": f"Bearer ${{{TOKEN_VAR}:-none}}"},
}


# The fixed address (Settings → "Keep the same address"): agents whose config cannot refer to environment
# variables, and editors not started from ThetaIDE's terminal, need an address and token that never change.
FIXED_PORT = 47321
TOKEN_FILE = "mcp_token"


def user_config_dir(environ: MutableMapping[str, str] | None = None) -> Path:
    """ThetaIDE's folder in the user's (not the project's) config directory."""
    env = os.environ if environ is None else environ
    if sys.platform == "win32":
        base = Path(env.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "thetaide"


def atomic_write(path: Path, text: str, *, private: bool = False) -> None:
    """Replace a file's contents all at once: a crash or another program writing at the same moment never
    leaves it half-written. A symlink (e.g. a dotfiles-managed ~/.claude.json) is followed, not replaced.
    `private` files (they hold the token) are readable by the user only; others keep their mode."""
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        mode = None
    if private:
        mode = stat.S_IRUSR | stat.S_IWUSR
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        else:  # a new file: the usual permissions for the user's umask
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temporary, 0o666 & ~umask)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fixed_token(folder: Path, *, renew: bool = False) -> str:
    """The token that stays the same across restarts, kept in a file only the user can read. It is never
    stored in the workspace (whose .thetaide folder may be committed). `renew` replaces it."""
    path = folder / TOKEN_FILE
    if not renew:
        try:
            token = path.read_text(encoding="utf-8").strip()
            if len(token) >= 32:
                return token
        except OSError:
            pass
    token = secrets.token_urlsafe(32)
    atomic_write(path, token + "\n", private=True)
    return token


def export(url: str, token: str, environ: MutableMapping[str, str] | None = None) -> None:
    """Make the address and token visible to terminals (and their agents) started from now on."""
    env = os.environ if environ is None else environ
    env[URL_VAR] = url
    env[TOKEN_VAR] = token


def unexport(environ: MutableMapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    env.pop(URL_VAR, None)
    env.pop(TOKEN_VAR, None)


def write_project_config(workspace: Path) -> str:
    """Add the server to <workspace>/.mcp.json, keeping every other server there. Returns what happened,
    in words for the user. A file that is not valid JSON is left alone."""
    path = workspace / CONFIG_FILE
    config: dict[str, Any] = {}
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            return f"{CONFIG_FILE} was not changed: it could not be read as JSON ({exc})."
        if not isinstance(config, dict):
            return f"{CONFIG_FILE} was not changed: it is not a JSON object."
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        return f'{CONFIG_FILE} was not changed: its "mcpServers" is not an object.'
    if servers.get(SERVER_NAME) == ENTRY:
        return f"{CONFIG_FILE} already lists ThetaIDE."
    existed = SERVER_NAME in servers
    servers[SERVER_NAME] = ENTRY
    try:
        atomic_write(path, json.dumps(config, indent=2) + "\n")
    except OSError as exc:
        return f"{CONFIG_FILE} could not be written: {exc}"
    return f"{'Updated' if existed else 'Added'} ThetaIDE in {path.name}."
