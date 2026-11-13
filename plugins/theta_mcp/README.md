# Theta MCP Server

Let Claude Code, Codex and other agents that speak the [Model Context Protocol](https://modelcontextprotocol.io) use ThetaIDE. Turn on the server, open a terminal in ThetaIDE and run your agent there. It gets the same tools as the AI Assistant plugin:

- list and diagnose runs from their configs, logs and metrics;
- compare runs and plot metric curves (charts come back as images);
- read experiment and component configs, and validate an experiment without running it;
- search the Component Hub;
- with **your approval in ThetaIDE**: create or edit configs, launch or stop runs, and install Hub components.

It needs no other plugin, and no LLM API key: the agent in the terminal brings its own model.

## Use it with Claude Code

1. Turn it on in **Settings → Community Plugins → Theta MCP Server → ⚙ (the gear on its row)**, or run **MCP: Start the ThetaIDE MCP server** from the command launcher. It stays on across restarts until you turn it off.
2. Open a **new** terminal in ThetaIDE. Terminals opened before the server started don't have its address.
3. Run `claude`. The first time, Claude Code asks whether to trust the workspace's `thetaide` server; approve it.
4. Ask things like *why did my last run diverge?*, or *add an option to toggle my change to the PPO loss, then compare it on and off over three seeds*. Claude Code edits the code with its own tools, then launches each variant with `launch_experiment` (a label, overrides and seeds), waits with `wait_for_runs` and compares with `compare_runs`.

Starting the server adds a `thetaide` entry to the workspace's `.mcp.json` and keeps any other servers listed there. The entry refers to the address and token through environment variables, so the file holds no secret and is safe to commit. Outside ThetaIDE it points at a closed port, so Claude Code just reports the server as unreachable.

```json
{
  "mcpServers": {
    "thetaide": {
      "type": "http",
      "url": "${THETA_MCP_URL:-http://127.0.0.1:9/mcp}",
      "headers": { "Authorization": "Bearer ${THETA_MCP_TOKEN:-none}" }
    }
  }
}
```

## Use it with pi

pi needs the [pi-mcp-adapter](https://github.com/nicobailon/pi-mcp-adapter) extension (`pi install npm:pi-mcp-adapter`), which reads the same `.mcp.json`. Start the server, open a new terminal and run `pi`. The first time, pi asks whether to allow the project's `thetaide` server. By default the adapter offers the tools through one `mcp` proxy tool; its `directTools` setting exposes them as ordinary tools instead.

## Choosing agents

Each agent reads its own config file, so the settings page has a checkbox per agent (Claude Code and pi are on by default). When the server starts, ThetaIDE adds a `thetaide` entry to the chosen agents' files. Unchecking an agent removes the entry; everything else in the file is kept. A file that can't be edited safely (JSON with comments, or a Codex table you wrote yourself) is left alone, and the settings page says what to add by hand.

| Agent | Without the fixed address: ThetaIDE's terminal only | With the fixed address: any terminal or editor |
| --- | --- | --- |
| Claude Code | `.mcp.json` in the project | `~/.claude.json` (Claude Code's user scope) |
| pi | `.mcp.json` in the project | `~/.config/mcp/mcp.json` (read by pi-mcp-adapter) |
| opencode | `opencode.json` in the project (or the `opencode.json(c)` / `.opencode/` file it already uses) | `~/.config/opencode/opencode.json` |
| Cursor | `.cursor/mcp.json` in the project, which serves the CLI | `~/.cursor/mcp.json`, which serves the editor and the CLI |
| Codex | Not available | `~/.codex/config.toml`, in a block marked as ThetaIDE's |
| Antigravity | Not available | `~/.gemini/config/mcp_config.json` (and `~/.gemini/antigravity/mcp_config.json` if that exists) |

Project files only name the environment variables (`THETA_MCP_URL`, `THETA_MCP_TOKEN`), which only terminals opened in ThetaIDE have. They hold no secret and are safe to commit.

With the fixed address, each agent's entry moves to its user-wide file, with the address and token written out, and the project entry is removed. Agents prefer a project entry, and outside ThetaIDE its variables are unset. Either way, ThetaIDE must be running with the server on. Agents act on the workspace open in ThetaIDE, whichever folder they were started in, and approvals appear in ThetaIDE's window.

### The fixed address

To use agents outside ThetaIDE's terminal, turn on **Keep the same address**. Codex and Antigravity need it in any case: Codex can't read its server's address from the environment, and Antigravity's files don't document variables. It makes the server always use port 47321 (or the `port` setting) and one token. It writes that address and token into those agents' files in your home folder.

- The token is kept in `~/.config/thetaide/mcp_token` (`%APPDATA%\thetaide` on Windows), readable by you only. It's never kept in the workspace, whose `.thetaide` folder may be committed.
- Agents' files that contain it are also made readable by you only.
- Any program running as you can still read it, so leave this off if you only use agents in ThetaIDE's terminal.
- **New token** replaces the token and rewrites the files.
- Turning the option off removes the token from the files.
- Only one ThetaIDE window can use the fixed port at a time.

Codex's entry sets `default_tools_approval_mode = "writes"`. Reading tools then run without Codex asking, because the server marks them read-only. ThetaIDE still asks before anything changes. The entry also raises `tool_timeout_sec` to 120.

### Any other agent

Any MCP client that supports the Streamable HTTP transport can connect. Inside a ThetaIDE terminal:

- **URL:** `$THETA_MCP_URL` (for example `http://127.0.0.1:43117/mcp`)
- **Header:** `Authorization: Bearer $THETA_MCP_TOKEN`

## Long waits

Most MCP clients give up on a tool call after 60 seconds (pi's adapter and Codex by default). So one `wait_for_runs` call waits at most 50 seconds. If runs are still going, its reply says so, and the agent calls it again.

## Approvals

Reading needs no approval. Anything that changes the workspace or starts compute opens an approval dialog in ThetaIDE. The dialog names the agent ("Claude Code wants to change `in/config/experiment/cartpole.yaml`") and shows the diff for file changes. If you reject a change, the agent is told not to retry it. Outside agents are always asked. The AI Assistant's **Launch and stop runs without asking** setting doesn't apply to them.

## Security

- The server listens on `127.0.0.1` only, on a random port unless you set `port` or turn on the fixed address.
- Each start makes a new random token, and every request must carry it. The token lives only in ThetaIDE's memory and the environment of terminals it opens. With the fixed address, the token instead stays the same and is stored as described above.
- Requests whose `Host` or `Origin` isn't the loopback address are refused, so a web page can't reach the server, even through DNS rebinding.
- Each request gets its own connection, and request bodies over 4 MB are refused.
- Training runs don't inherit the address or token, so experiment and Hub component code can't reach the tools that way.
- An agent's name in the approval dialog is what its MCP client says it is. The dialog says so: approve only what you expect.
- Agents' config files are replaced whole: written to a temporary file, then renamed. A crash, or the agent writing the same file at the same moment, can't leave one half-written. Symlinked files (dotfiles) stay links.

## Design

- `server.py`: the MCP server on Python's standard library: JSON-RPC over HTTP POST with one JSON reply per request. It offers no server-to-client stream (GET returns 405), which the protocol allows. Protocol versions 2025-06-18, 2025-03-26 and 2024-11-05 are supported. Newer clients that first probe with `server/discover` get "method not found" and fall back to `initialize`. No Qt.
- `project_config.py`: the environment variables, the `.mcp.json` entry, and the fixed token file.
- `harnesses.py`: one writer per agent's config file. Each one adds, updates or removes only the `thetaide` entry. Codex's TOML is edited as a marked block, and the result is parsed before it's written.
- `host.py`: runs each tool for the server. Window reads run on the GUI thread. Tools that wait (Hub installs, launches, `wait_for_runs`) stay on the request's thread, so the IDE stays responsive. Changes are previewed, then approved or rejected.
- `ide_tools/`: the tools themselves, a copy of the AI Assistant's tool code, so both plugins offer the same tools under the same rules and neither needs the other. These files are generated, so don't edit them. After changing `plugins/ai_assistant`, run `python scripts/sync_mcp_tools.py` from theta-plugins. A test fails while the copy is out of date.
- `__init__.py`: the plugin: commands, settings, starting and stopping the server, and writing agents' config files.

Tests: `QT_QPA_PLATFORM=offscreen python -m pytest plugins/theta_mcp` from theta-plugins. They include the real HTTP server, chunked request bodies (which Claude Code sends), token, host and origin checks, and the `.mcp.json` merge.
