"""Run ThetaIDE's tools for outside agents: on the right thread, and with the user's approval for changes.

The tools themselves are ide_tools (a copy of the AI Assistant's, so both offer the same tools under the same
rules). Requests arrive on the MCP server's threads. Reads that touch the window run on the GUI thread; tools
that wait on the network or on runs (Hub installs, launches, wait_for_runs) stay on the request's thread so the
IDE stays responsive. Anything that changes the workspace or starts compute is previewed first; if the preview
shows it cannot work the agent is told why, otherwise the user approves or rejects it in a dialog naming the agent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

from .ide_tools.approval import ApprovalDialog, describe_request
from .ide_tools.bridge import WindowBridge
from .ide_tools.chat_types import ToolCall
from .ide_tools.tools import ToolError, ToolRegistry, ToolResult, build_ide_tools

DECLINED = "The user declined this action in ThetaIDE. Do not retry it; ask them what they would like instead."


class MainThreadInvoker(QObject):
    """Run a callable on the thread this object lives in (the GUI thread) and return its result."""

    _call = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.closing = False
        self._call.connect(self._run, Qt.ConnectionType.BlockingQueuedConnection)

    @pyqtSlot(object)
    def _run(self, box: dict) -> None:
        if self.closing:
            box["error"] = RuntimeError("ThetaIDE's MCP server is shutting down.")
            return
        try:
            box["result"] = box["fn"]()
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    def invoke(self, fn: Callable[[], Any]) -> Any:
        if QThread.currentThread() is self.thread():
            return fn()
        if self.closing:
            raise RuntimeError("ThetaIDE's MCP server is shutting down.")
        box: dict[str, Any] = {"fn": fn}
        self._call.emit(box)
        if "error" in box:
            raise box["error"]
        return box.get("result")


class ToolHost:
    """The tools, as plain dicts for the server: tool_definitions() and call_tool(name, arguments, who)."""

    def __init__(
        self,
        window: Any,
        workspace: Path,
        figures_dir: Path,
        *,
        status: Callable[[str], None] | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.window = window
        self.invoker = MainThreadInvoker()  # created on the GUI thread, so it runs things there
        self.bridge = WindowBridge(window, workspace)
        self.bridge.figures_dir = figures_dir
        self.bridge.on_gui = self.invoker.invoke
        self.bridge.hub.on_gui = self.invoker.invoke
        self.bridge.should_stop = lambda: self.invoker.closing
        self.tools = tools if tools is not None else build_ide_tools(self.bridge)
        self.status = status or (lambda _text: None)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters, "mutating": t.mutating}
            for t in self.tools.tools()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any], who: str) -> dict[str, Any]:
        """Run a tool for `who`, from any thread. Returns {text, is_error, images: [[path, caption], ...]}."""
        result = self._call(ToolCall(id="mcp", name=name, arguments=dict(arguments or {})), who)
        return {
            "text": result.content,
            "is_error": result.is_error,
            "images": [[path, caption] for path, caption in result.figures],
        }

    def close(self) -> None:
        """Refuse further GUI work: the window may be going away."""
        self.invoker.closing = True

    # ── Internals ──────────────────────────────────────────────────────────

    def _call(self, call: ToolCall, who: str) -> ToolResult:
        try:
            if self.tools.is_mutating(call.name):
                try:
                    preview = self._on_tool_thread(call, self.tools.preview)
                except ToolError as exc:  # it would fail anyway: no need to ask the user
                    return ToolResult(str(exc), is_error=True)
                if not self.invoker.invoke(lambda: self.confirm(call, preview, who)):
                    return ToolResult(DECLINED, is_error=True)
            result = self._on_tool_thread(call, self.tools.execute)
        except RuntimeError as exc:  # shutting down
            return ToolResult(str(exc), is_error=True)
        self.invoker.invoke(lambda: self.status(f"{who} used {call.name}"))
        return result

    def _on_tool_thread(self, call: ToolCall, fn: Callable[[ToolCall], Any]) -> Any:
        """Tools that wait (network, installs, runs) stay on the request's thread; the rest run on the GUI's."""
        if self.tools.runs_in_background(call.name):
            return fn(call)
        return self.invoker.invoke(lambda: fn(call))

    def confirm(self, call: ToolCall, preview: str, who: str) -> bool:
        """Ask the user, on the GUI thread. Outside agents are always asked."""
        current = None
        if call.name == "launch_experiment":
            current = getattr(self.window, "current_experiment", None)
        title, summary, shown, is_diff = describe_request(call.name, call.arguments, preview, who, current)
        parent = self.window if hasattr(self.window, "windowHandle") else None
        dialog = ApprovalDialog(title, summary, shown, is_diff=is_diff, parent=parent, actor=who, self_named=True)
        approved = dialog.exec() == ApprovalDialog.DialogCode.Accepted
        what = call.arguments.get("path") or call.name
        self.status(f"{'Approved' if approved else 'Rejected'} {what} for {who}")
        return approved
