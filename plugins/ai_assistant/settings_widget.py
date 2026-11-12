"""Settings → Plugins → AI Assistant: keys for every provider, server URLs and agent limits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .keys import ENV_VARS, KeyStore, KeyStoreError, mask
from .providers import PROVIDERS


class AssistantSettings(QWidget):
    def __init__(
        self,
        context: Any,
        keys: KeyStore,
        on_change: Callable[[], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.context = context
        self.keys = keys
        self.on_change = on_change or (lambda: None)
        layout = QVBoxLayout(self)

        if not keys.is_secure():
            warning = QLabel(
                "No secure OS keychain was found (install `keyring` and a Secret Service such as GNOME Keyring "
                "or KWallet on Linux). Keys can still come from environment variables: "
                + ", ".join(f"${v}" for v in ENV_VARS.values())
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        form = QFormLayout()
        self.status_labels: dict[str, QLabel] = {}
        self.url_fields: dict[str, QLineEdit] = {}
        urls = context.get_setting("base_urls", {}) or {}
        for pid, cls in PROVIDERS.items():
            row = QHBoxLayout()
            status = QLabel()
            status.setObjectName("muted")
            self.status_labels[pid] = status
            row.addWidget(status, 1)
            if cls.requires_key:
                set_button = QPushButton("Set…")
                set_button.clicked.connect(lambda _=False, p=pid: self._set_key(p))
                clear_button = QPushButton("Remove")
                clear_button.clicked.connect(lambda _=False, p=pid: self._clear_key(p))
                row.addWidget(set_button)
                row.addWidget(clear_button)
            form.addRow(cls.display_name, row)
            url = QLineEdit(urls.get(pid, ""))
            url.setPlaceholderText(cls.default_base_url)
            url.setToolTip(
                "Leave empty for the default. OpenAI accepts any OpenAI-compatible server (vLLM, LM Studio…)."
            )
            url.editingFinished.connect(self._save_urls)
            self.url_fields[pid] = url
            form.addRow("    Server URL", url)
        layout.addLayout(form)

        limits = QFormLayout()
        self.max_steps = QSpinBox()
        self.max_steps.setRange(1, 30)
        self.max_steps.setValue(int(context.get_setting("max_steps", 8)))
        self.max_steps.setToolTip(
            "Steps that change something or hit an error before the assistant stops and reports back. "
            "Steps that only read data successfully are free (up to 3x this number)."
        )
        self.max_steps.valueChanged.connect(lambda v: self._set("max_steps", v))
        limits.addRow("Action steps per message", self.max_steps)
        self.auto = QCheckBox("Launch and stop runs without asking (file changes and Hub installs always ask)")
        self.auto.setChecked(bool(context.get_setting("auto_approve_actions", False)))
        self.auto.toggled.connect(lambda v: self._set("auto_approve_actions", bool(v)))
        limits.addRow("", self.auto)
        self.watch = QCheckBox("Watch training runs for NaN, divergence, stalled learning and crashes")
        self.watch.setToolTip(
            "Checks each training run's metrics every 30 seconds on this computer. No model is called until you "
            "press Ask AI on an alert."
        )
        self.watch.setChecked(bool(context.get_setting("watch_runs", True)))
        self.watch.toggled.connect(lambda v: self._set("watch_runs", bool(v)))
        limits.addRow("", self.watch)
        layout.addLayout(limits)
        layout.addStretch(1)
        self._refresh()

    def _set(self, key: str, value: Any) -> None:
        self.context.set_setting(key, value)
        self.on_change()

    def _refresh(self) -> None:
        for pid, label in self.status_labels.items():
            if not PROVIDERS[pid].requires_key:
                label.setText("No key needed")
                continue
            source = self.keys.source(pid)
            label.setText(
                f"{mask(self.keys.get(pid))} (OS keychain)"
                if source == "keychain"
                else f"from ${ENV_VARS[pid]}"
                if source == "environment"
                else "Not set"
            )

    def _set_key(self, pid: str) -> None:
        from PyQt6.QtWidgets import QInputDialog

        key, ok = QInputDialog.getText(
            self,
            f"{PROVIDERS[pid].display_name} API key",
            "API key (stored in the OS keychain):",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not key.strip():
            return
        try:
            self.keys.set(pid, key)
        except KeyStoreError as exc:
            QMessageBox.warning(self, "API key not saved", str(exc))
        self._refresh()
        self.on_change()

    def _clear_key(self, pid: str) -> None:
        self.keys.delete(pid)
        self._refresh()
        self.on_change()

    def _save_urls(self) -> None:
        urls = {pid: field.text().strip() for pid, field in self.url_fields.items() if field.text().strip()}
        self._set("base_urls", urls)
