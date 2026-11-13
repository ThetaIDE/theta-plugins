"""The dialog that asks the user to approve or reject an action the assistant (or an outside agent) wants to take."""

from __future__ import annotations

import html

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Translucent tints read on both light and dark themes.
_ADDED = "rgba(46, 160, 67, 0.22)"
_REMOVED = "rgba(248, 81, 73, 0.22)"
_HUNK = "rgba(56, 139, 253, 0.18)"


def diff_html(diff: str) -> str:
    rows = []
    for line in diff.splitlines():
        text = html.escape(line) or "&nbsp;"
        if line.startswith(("+++", "---")):
            rows.append(f"<div><b>{text}</b></div>")
        elif line.startswith("@@"):
            rows.append(f'<div style="background:{_HUNK}">{text}</div>')
        elif line.startswith("+"):
            rows.append(f'<div style="background:{_ADDED}">{text}</div>')
        elif line.startswith("-"):
            rows.append(f'<div style="background:{_REMOVED}">{text}</div>')
        else:
            rows.append(f"<div>{text}</div>")
    return '<div style="white-space:pre">' + "".join(rows) + "</div>"


def describe_request(name: str, arguments: dict, preview: str, actor: str, current_experiment: str | None = None):
    """(title, summary, preview, is_diff) for the dialog asking about tool `name`. `actor` is who asked:
    "The assistant", or an outside agent such as "Claude Code". The summary is HTML."""
    is_diff = name in ("write_file", "edit_file")
    path = html.escape(str(arguments.get("path", "")))
    if is_diff:
        new = preview.startswith("--- /dev/null")
        return (
            ("Create file?" if new else "Edit file?"),
            f"{actor} wants to {'create' if new else 'change'} <b>{path}</b>:",
            preview,
            True,
        )
    if name == "launch_experiment":
        target = html.escape(str(arguments.get("path") or current_experiment or "the selected experiment"))
        label = arguments.get("label")
        summary = f"{actor} wants to queue and run <b>{target}</b>" + (
            f" as <b>{html.escape(str(label))}</b>." if label else "."
        )
        return "Launch run?", summary, preview, False
    if name == "install_from_hub":
        component = html.escape(str(arguments.get("component_id", "")))
        return "Install from the Component Hub?", f"{actor} wants to install <b>{component}</b>:", preview, False
    if name == "stop_run":
        run = html.escape(str(arguments.get("run_id", "selected")))
        return "Stop run?", f"{actor} wants to stop run <b>{run}</b>.", "", False
    return "Allow action?", f"{actor} wants to run <b>{html.escape(name)}</b>:", preview, False


class ApprovalDialog(QDialog):
    """Approve or Reject, with the exact change shown. Escape and closing the window reject."""

    def __init__(
        self,
        title: str,
        summary: str,
        preview: str = "",
        is_diff: bool = False,
        parent: QWidget | None = None,
        actor: str = "The assistant",
        self_named: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        label = QLabel(summary)
        label.setWordWrap(True)
        layout.addWidget(label)
        if preview:
            view = QTextEdit()
            view.setReadOnly(True)
            view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
            font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            font.setStyleHint(QFont.StyleHint.Monospace)
            view.setFont(font)
            if is_diff:
                view.setHtml(diff_html(preview))
            else:
                view.setPlainText(preview)
            layout.addWidget(view, 1)
            self.resize(760, 520)
        if self_named:  # an outside agent's name is whatever its MCP client says it is
            note = QLabel(
                f"Asked through ThetaIDE's MCP server by a program that calls itself {actor}. "
                "Approve only what you expect. Nothing changes unless you approve."
            )
            note.setWordWrap(True)
        else:
            note = QLabel(f"{actor} asked for this. Nothing changes unless you approve.")
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QDialogButtonBox()
        self.approve_button = buttons.addButton("Approve", QDialogButtonBox.ButtonRole.AcceptRole)
        self.reject_button = buttons.addButton("Reject", QDialogButtonBox.ButtonRole.RejectRole)
        self.reject_button.setDefault(True)  # Enter never approves by accident
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
