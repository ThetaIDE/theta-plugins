"""Settings → Community Plugins → Theta MCP Server → gear: turn the server on or off, and choose which agents to set up."""

from __future__ import annotations

import html
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import harnesses, project_config

HOW_TO = (
    "Open a new terminal in ThetaIDE (one opened after the server started) and start your agent there. "
    "Reading runs, configs, logs and metrics needs no approval. Writing files, launching or stopping runs and "
    "installing components always ask you in ThetaIDE first."
)
FIXED_HELP = (
    f"Off, agents only find the server in terminals opened in ThetaIDE. On, the server always uses port "
    f"{project_config.FIXED_PORT} and one token, written into each chosen agent's own config file in your home "
    "folder (readable by you only, never in the project), so they work from any terminal or editor while ThetaIDE "
    "runs. Codex and Antigravity need this, and so does an editor such as Cursor opened from the Dock, Finder or "
    "Start menu instead of ThetaIDE's terminal. Any program running as you could read the token."
)


AGENTS_HELP = (
    'Each checked agent gets a "thetaide" entry in its own config file when the server starts: in the project '
    "without the same address, in your home folder with it. Unchecking removes the entry. Hover over an agent to "
    "see its files."
)


class MCPSettings(QWidget):
    def __init__(self, plugin: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.plugin = plugin
        layout = QVBoxLayout(self)
        self.toggle = QCheckBox("Run the ThetaIDE MCP server")
        self.toggle.toggled.connect(self._toggled)
        layout.addWidget(self.toggle)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # copy the URL
        layout.addWidget(self.status)

        # Plain widgets, not a group box: wrapped labels inside a QGroupBox get cut off in the settings dialog.
        self.fixed = QCheckBox("Keep the same address, so agents work outside ThetaIDE's terminal too")
        self.fixed.toggled.connect(self._fixed_toggled)
        layout.addWidget(self.fixed)
        layout.addWidget(self._small(FIXED_HELP))
        row = QHBoxLayout()
        self.renew = QPushButton("New token")
        self.renew.setToolTip("Replace the fixed token, e.g. if it was shared by mistake; agents' files are updated.")
        self.renew.clicked.connect(self.plugin.renew_token)
        row.addWidget(self.renew)
        row.addStretch(1)
        layout.addLayout(row)

        agents = QGroupBox("Set up these agents")
        agents_layout = QVBoxLayout(agents)
        self.agent_boxes: dict[str, QCheckBox] = {}
        for harness in harnesses.HARNESSES:
            box = QCheckBox(harness.name + (" (needs the same address)" if harness.needs_stable else ""))
            box.setToolTip(harness.where)  # its files: one line each would crowd the page
            box.toggled.connect(lambda _on: self._agents_changed())
            agents_layout.addWidget(box)
            self.agent_boxes[harness.id] = box
        layout.addWidget(agents)
        layout.addWidget(self._small(AGENTS_HELP))  # outside the group box, where wrapped text lays out fully

        self.notes = QLabel()
        self.notes.setWordWrap(True)
        layout.addWidget(self.notes)
        how = QLabel(HOW_TO)
        how.setWordWrap(True)
        layout.addWidget(how)
        layout.addStretch(1)
        plugin.listeners.append(self.refresh)
        self.destroyed.connect(lambda *_: self._forget())
        self.refresh()

    @staticmethod
    def _small(text: str) -> QLabel:
        label = QLabel(f"<small>{html.escape(text)}</small>")
        label.setWordWrap(True)
        return label

    def _forget(self) -> None:
        if self.refresh in self.plugin.listeners:
            self.plugin.listeners.remove(self.refresh)

    def _toggled(self, on: bool) -> None:
        if on and self.plugin.server is None:
            self.plugin.start()
        elif not on and self.plugin.server is not None:
            self.plugin.stop()
        self.refresh()

    def _agents_changed(self) -> None:
        chosen = [hid for hid, box in self.agent_boxes.items() if box.isChecked() and box.isEnabled()]
        # An agent greyed out for lack of the fixed address stays chosen, so it comes back when that is turned on.
        kept = [hid for hid in self.plugin.agent_ids() if not self.agent_boxes[hid].isEnabled()]
        self.plugin.set_agents(chosen + [hid for hid in kept if hid not in chosen])
        self.refresh()

    def _fixed_toggled(self, on: bool) -> None:
        self.plugin.set_fixed_address(on)
        self.refresh()

    def refresh(self) -> None:
        server = self.plugin.server
        fixed = self.plugin.fixed_address()
        chosen = set(self.plugin.agent_ids())
        for widget, value in ((self.toggle, server is not None), (self.fixed, fixed)):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        for hid, box in self.agent_boxes.items():
            needs = harnesses.BY_ID[hid].needs_stable
            box.blockSignals(True)
            box.setEnabled(fixed or not needs)
            box.setChecked(hid in chosen)
            box.blockSignals(False)
        self.renew.setEnabled(fixed)
        if server is not None:
            self.status.setText(
                f"Running at <code>{server.url}</code>. Terminals opened since then have "
                f"<code>${project_config.URL_VAR}</code> and <code>${project_config.TOKEN_VAR}</code>."
            )
        else:
            self.status.setText("Stopped. Agents' config files are written when it starts.")
        notes = self.plugin.notes
        self.notes.setText("<br>".join(html.escape(n) for n in notes) if notes else "")
