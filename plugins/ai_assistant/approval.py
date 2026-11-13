"""The dialog that asks the user to approve or reject an action the assistant wants to take."""

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


class ApprovalDialog(QDialog):
    """Approve or Reject, with the exact change shown. Escape and closing the window reject."""

    def __init__(
        self,
        title: str,
        summary: str,
        preview: str = "",
        is_diff: bool = False,
        parent: QWidget | None = None,
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
        note = QLabel("The assistant asked for this. Nothing changes unless you approve.")
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QDialogButtonBox()
        self.approve_button = buttons.addButton("Approve", QDialogButtonBox.ButtonRole.AcceptRole)
        self.reject_button = buttons.addButton("Reject", QDialogButtonBox.ButtonRole.RejectRole)
        self.reject_button.setDefault(True)  # Enter never approves by accident
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
