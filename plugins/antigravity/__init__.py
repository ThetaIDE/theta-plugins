"""Antigravity Agent Plugin for ThetaIDE with configurable Full Panel & Sidepane modes."""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.app.api.desktop import Plugin, PluginManifest
from src.app.api.desktop import PluginContext

import time

from .panel import AntigravityPanel


class SidepaneEventFilter(QObject):
    """Event filter intercepting Control+Shift+Space application-wide for Antigravity."""

    def __init__(self, callback: Callable[[], None], parent: Optional[QObject] = None):
        super().__init__(parent)
        self.callback = callback

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            has_ctrl = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
            has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            is_space = (key == Qt.Key.Key_Space) or (getattr(event, "nativeVirtualKey", lambda: None)() == 49)

            if has_ctrl and has_shift and is_space:
                self.callback()
                return True
        return super().eventFilter(watched, event)


class AntigravitySettingsWidget(QWidget):
    """Configuration widget for switching between Full Panel and Sidepane modes."""

    def __init__(self, plugin: AntigravityPlugin, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.plugin = plugin
        self.context = context
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header Title
        title_lbl = QLabel("Antigravity Interface Mode")
        title_lbl.setStyleSheet("font-size: 13px; font-weight: 700; color: #83a598;")
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(
            "Configure how the Antigravity Agent chat interface is integrated into ThetaIDE:"
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #bdae93; font-size: 12px;")
        layout.addWidget(desc_lbl)

        current_mode = self.context.get_setting("display_mode", AntigravityPlugin.MODE_FULL_PANEL)

        self.btn_group = QButtonGroup(self)

        # Option 1: Full Panel
        panel_card = QFrame()
        panel_card.setFrameShape(QFrame.Shape.StyledPanel)
        panel_card.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; padding: 10px; }"
        )
        p_layout = QVBoxLayout(panel_card)
        p_layout.setSpacing(6)

        self.radio_full = QRadioButton("Full Panel (Sidebar Tab)")
        self.radio_full.setStyleSheet("font-weight: 600; font-size: 12px; color: #ebdbb2;")
        self.btn_group.addButton(self.radio_full)
        p_layout.addWidget(self.radio_full)

        p_desc = QLabel(
            "Shows Antigravity as a full-size workspace panel in the primary sidebar (current default). "
            "Selecting the sidebar tab presents the agent as the main center view."
        )
        p_desc.setWordWrap(True)
        p_desc.setStyleSheet("color: #928374; font-size: 11px; margin-left: 22px;")
        p_layout.addWidget(p_desc)
        layout.addWidget(panel_card)

        # Option 2: Sidepane with Hotkey
        side_card = QFrame()
        side_card.setFrameShape(QFrame.Shape.StyledPanel)
        side_card.setStyleSheet(
            "QFrame { background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 6px; padding: 10px; }"
        )
        s_layout = QVBoxLayout(side_card)
        s_layout.setSpacing(6)

        self.radio_side = QRadioButton("Sidepane (Control + Shift + Space hotkey)")
        self.radio_side.setStyleSheet("font-weight: 600; font-size: 12px; color: #ebdbb2;")
        self.btn_group.addButton(self.radio_side)
        s_layout.addWidget(self.radio_side)

        s_desc = QLabel(
            "Docks Antigravity as a collapsible sidepane on the right side of the IDE. "
            "Press Control + Shift + Space (⌃⇧Space) anywhere in the application to instantly open the sidechat "
            "without conflicting with the Launcher (⌃Space)."
        )
        s_desc.setWordWrap(True)
        s_desc.setStyleSheet("color: #928374; font-size: 11px; margin-left: 22px;")
        s_layout.addWidget(s_desc)

        # Hotkey badge
        hk_layout = QHBoxLayout()
        hk_layout.setContentsMargins(22, 2, 0, 0)
        hk_label = QLabel("Hotkey:")
        hk_label.setStyleSheet("color: #bdae93; font-size: 11px; font-weight: 600;")

        current_sc = self.context.command_hotkey(AntigravityPlugin.TOGGLE_COMMAND) or "Ctrl+Shift+Space"

        hk_display = current_sc.replace("Ctrl+", "⌃ ").replace("Shift+", "⇧ ").replace("Alt+", "⌥ ")
        if current_sc == "Ctrl+Shift+Space":
            hk_display = "⌃ Control + ⇧ Shift + Space"

        hk_badge = QLabel(hk_display)
        hk_badge.setStyleSheet(
            "background-color: #504945; color: #ebdbb2; font-family: monospace; font-size: 11px; font-weight: bold; padding: 2px 7px; border-radius: 4px;"
        )
        hk_layout.addWidget(hk_label)
        hk_layout.addWidget(hk_badge)

        hk_hint = QLabel("(Configurable in Settings → Hotkeys)")
        hk_hint.setStyleSheet("color: #7c6f64; font-size: 11px; font-style: italic;")
        hk_layout.addWidget(hk_hint)
        hk_layout.addStretch()
        s_layout.addLayout(hk_layout)

        layout.addWidget(side_card)

        # Initialize checked radio
        if current_mode == AntigravityPlugin.MODE_SIDEPANE:
            self.radio_side.setChecked(True)
        else:
            self.radio_full.setChecked(True)

        self.radio_full.toggled.connect(self._on_mode_toggled)
        self.radio_side.toggled.connect(self._on_mode_toggled)

        layout.addStretch()

    def _on_mode_toggled(self):
        new_mode = (
            AntigravityPlugin.MODE_SIDEPANE
            if self.radio_side.isChecked()
            else AntigravityPlugin.MODE_FULL_PANEL
        )
        if new_mode == self.plugin.current_mode:
            return
        self.context.set_setting("display_mode", new_mode)
        self.plugin.set_display_mode(new_mode)
        if new_mode == AntigravityPlugin.MODE_SIDEPANE and self.plugin.dock:
            self.plugin.dock.show()
            self.plugin.dock.raise_()
            self.plugin.dock.activateWindow()
            if self.plugin.panel and hasattr(self.plugin.panel, "focus_input"):
                self.plugin.panel.focus_input()
        mode_str = "Sidepane (Control + Shift + Space)" if new_mode == AntigravityPlugin.MODE_SIDEPANE else "Full Panel"
        self.context.show_status_message(f"Antigravity mode updated to: {mode_str}")


class AntigravityPlugin(Plugin):
    """Integrates Google Antigravity Agent into ThetaIDE with Full Panel and Sidepane modes."""

    MODE_FULL_PANEL = "full_panel"
    MODE_SIDEPANE = "sidepane"
    DOCK_ID = "antigravity_sidepane"
    # Command and [shortcuts] key; the host binds its hotkey and applies the user's setting.
    TOGGLE_COMMAND = "toggle_antigravity"

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: AntigravityPanel | None = None
        self.dock: QDockWidget | None = None
        self.shortcut: QShortcut | None = None
        self.event_filter: SidepaneEventFilter | None = None
        self.current_mode: str = self.MODE_FULL_PANEL
        self._last_toggle_time: float = 0.0

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = AntigravityPanel(context=context)
        context.register_command(
            self.TOGGLE_COMMAND,
            "Antigravity: toggle sidepane",
            self.toggle_sidepane,
            default_hotkey=context.get_setting("hotkey", "Ctrl+Shift+Space"),
            keywords=("antigravity", "agent", "chat"),
        )

        # Retrieve saved setting (defaults to "full_panel")
        saved_mode = context.get_setting("display_mode", self.MODE_FULL_PANEL)
        self.set_display_mode(saved_mode)
        context.log(f"Antigravity Agent plugin activated (mode: {saved_mode}).")

    def set_display_mode(self, mode: str) -> None:
        """Switch between 'full_panel' and 'sidepane' display modes."""
        self.current_mode = mode
        if not self.context or not self.panel:
            return

        if mode == self.MODE_SIDEPANE:
            # 1. Remove from primary sidebar if present
            self.context.remove_sidebar_tab("antigravity")

            # 2. Attach to sidepane (QDockWidget)
            if self.dock is None:
                self.dock = self.context.add_dock_widget(
                    dock_id=self.DOCK_ID,
                    widget=self.panel,
                    title="Antigravity Agent",
                    area=Qt.DockWidgetArea.RightDockWidgetArea,
                )
                if self.dock is not None:
                    self.dock.setFeatures(
                        QDockWidget.DockWidgetFeature.DockWidgetClosable |
                        QDockWidget.DockWidgetFeature.DockWidgetMovable
                    )
                    self.dock.setMinimumWidth(320)
                    self.dock.hide()

            # 3. Setup hotkey & event filter
            self._setup_hotkey()

        else:
            # Full Panel mode
            # 1. Teardown hotkey and filter
            self._teardown_hotkey()

            # 2. Teardown dock widget if present (the host detaches the panel for reuse)
            if self.dock is not None:
                self.dock.hide()
                self.context.remove_dock_widget(self.DOCK_ID)
                self.dock = None

            # 3. Add tab into the sidebar
            self.context.add_sidebar_tab(
                tab_id="antigravity",
                widget=self.panel,
                title="Antigravity Agent",
                icon_name="antigravity",
                short_label="Agent",
            )

    def _setup_hotkey(self) -> None:
        """Extra sidepane key variants; the host binds the configured hotkey for TOGGLE_COMMAND."""
        self._teardown_hotkey()

        app = QApplication.instance()
        if app is not None:
            self.event_filter = SidepaneEventFilter(self.toggle_sidepane)
            app.installEventFilter(self.event_filter)

        parent = self.context.parent_widget() if self.context else None
        if parent is not None:
            configured = QKeySequence(self.context.command_hotkey(self.TOGGLE_COMMAND).replace("Option", "Alt"))
            variants = [
                QKeySequence("Meta+Shift+Space"),  # Physical Control+Shift+Space on macOS (⌃⇧Space)
                QKeySequence("Ctrl+Shift+Space"),  # Ctrl+Shift+Space
                QKeySequence("Alt+Shift+Space"),   # Option+Shift+Space on macOS
            ]
            self.shortcuts = []
            for seq in variants:
                if seq == configured:
                    continue  # bound by the host; a duplicate would make the shortcut ambiguous
                sc = QShortcut(seq, parent)
                sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
                sc.activated.connect(self.toggle_sidepane)
                self.shortcuts.append(sc)
            self.shortcut = self.shortcuts[0] if self.shortcuts else None

    def _teardown_hotkey(self) -> None:
        if self.event_filter is not None:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self.event_filter)
            self.event_filter = None

        if hasattr(self, "shortcuts") and self.shortcuts:
            for sc in self.shortcuts:
                sc.setEnabled(False)
                sc.deleteLater()
            self.shortcuts.clear()

        if self.shortcut is not None:
            self.shortcut.setEnabled(False)
            self.shortcut.deleteLater()
            self.shortcut = None

    def toggle_sidepane(self) -> None:
        """Toggle sidepane visibility and focus prompt input when opening."""
        now = time.monotonic()
        if now - self._last_toggle_time < 0.25:
            return
        self._last_toggle_time = now

        if not self.dock:
            return

        if not self.dock.isHidden():
            self.dock.hide()
        else:
            self.dock.show()
            self.dock.raise_()
            self.dock.activateWindow()
            if self.panel and hasattr(self.panel, "focus_input"):
                self.panel.focus_input()

    def deactivate(self) -> None:
        self._teardown_hotkey()

        if self.dock is not None:
            self.dock.hide()
            if self.context:
                self.context.remove_dock_widget(self.DOCK_ID)
            self.dock = None

        if self.context:
            self.context.remove_sidebar_tab("antigravity")
            self.context.log("Antigravity Agent plugin deactivated.")
            self.context = None

        if self.panel:
            if hasattr(self.panel, "cleanup"):
                self.panel.cleanup()
            self.panel = None

    def get_settings_widget(self, context: PluginContext) -> QWidget:
        return AntigravitySettingsWidget(self, context)
