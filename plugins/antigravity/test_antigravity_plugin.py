"""Unit tests for Antigravity plugin display mode settings and Option+Space sidepane hotkey."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt, QEvent
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication, QLabel
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from src.app.api.desktop import PluginManifest
    from src.app.api.desktop import PluginContext
    from plugins.antigravity import AntigravityPlugin, AntigravitySettingsWidget
    from plugins.antigravity.panel import AntigravityPanel
    HAS_PYQT6 = True
except ImportError:
    try:
        # Fallback for running from within theta-plugins repo
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from frontend.app import Window
        from src.app.api.desktop import PluginManifest
        from src.app.api.desktop import PluginContext
        from plugins.antigravity import AntigravityPlugin, AntigravitySettingsWidget
        from plugins.antigravity.panel import AntigravityPanel
        HAS_PYQT6 = True
    except ImportError:
        HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestAntigravityPlugin(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_antigravity_")
        self.window = Window(data_dir=self.temp_dir)
        self.window.show()
        app = QApplication.instance()
        if app is not None:
            app.setActiveWindow(self.window)
        self.storage_file = Path(self.temp_dir) / "plugins" / "antigravity.json"
        self.context = PluginContext("antigravity", self.window, self.storage_file)

        self.manifest = PluginManifest(
            id="antigravity",
            name="Antigravity Agent",
            version="0.1.5",
            description="Integrated Google Antigravity Agent",
            plugin_dir=Path(__file__).resolve().parent,
            extra={"has_settings": True},
        )
        self.plugin = AntigravityPlugin(self.manifest)

    def tearDown(self):
        if self.plugin:
            self.plugin.deactivate()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_activation_full_panel_mode(self):
        """Plugin activates in full panel mode by default, registering the sidebar tab."""
        self.plugin.activate(self.context)
        self.assertEqual(self.plugin.current_mode, AntigravityPlugin.MODE_FULL_PANEL)
        self.assertIn("antigravity", self.window.tabs.tabs)
        self.assertIsNone(self.plugin.dock)
        self.assertIsNone(self.plugin.shortcut)
        self.assertIsNone(self.plugin.event_filter)

    def test_settings_widget_options_and_mode_switching(self):
        """Settings widget reflects current mode and allows switching between Full Panel and Sidepane."""
        self.plugin.activate(self.context)
        widget = self.plugin.get_settings_widget(self.context)
        self.assertIsInstance(widget, AntigravitySettingsWidget)
        self.assertTrue(widget.radio_full.isChecked())
        self.assertFalse(widget.radio_side.isChecked())

        # Switch to Sidepane mode
        widget.radio_side.setChecked(True)
        self.assertEqual(self.context.get_setting("display_mode"), AntigravityPlugin.MODE_SIDEPANE)
        self.assertEqual(self.plugin.current_mode, AntigravityPlugin.MODE_SIDEPANE)
        # Sidebar tab is removed in sidepane mode
        self.assertNotIn("antigravity", self.window.tabs.tabs)
        # Sidepane dock and hotkey are created
        self.assertIsNotNone(self.plugin.dock)
        self.assertIsNotNone(self.plugin.shortcut)
        self.assertIsNotNone(self.plugin.event_filter)

    def test_option_space_toggle_sidepane(self):
        """Option+Space toggles sidepane visibility and focuses the chat input."""
        self.context.set_setting("display_mode", AntigravityPlugin.MODE_SIDEPANE)
        self.plugin.activate(self.context)

        self.assertIsNotNone(self.plugin.dock)
        self.assertTrue(self.plugin.dock.isHidden())

        # Toggle open
        self.plugin.toggle_sidepane()
        self.assertFalse(self.plugin.dock.isHidden())
        self.assertTrue(
            self.plugin.panel.message_input.hasFocus()
            or QApplication.focusWidget() == self.plugin.panel.message_input
        )

        # Toggle closed
        self.plugin._last_toggle_time = 0
        self.plugin.toggle_sidepane()
        self.assertTrue(self.plugin.dock.isHidden())

    def test_event_filter_intercepts_control_shift_space(self):
        """Application event filter catches Control+Shift+Space and toggles sidepane."""
        self.context.set_setting("display_mode", AntigravityPlugin.MODE_SIDEPANE)
        self.plugin.activate(self.context)
        self.assertTrue(self.plugin.dock.isHidden())

        # Control + Shift + Space
        key_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
            " ",
        )
        consumed = self.plugin.event_filter.eventFilter(self.window, key_event)
        self.assertTrue(consumed)
        self.assertFalse(self.plugin.dock.isHidden())

        # Reset debounce time and press again to toggle off
        self.plugin._last_toggle_time = 0
        consumed = self.plugin.event_filter.eventFilter(self.window, key_event)
        self.assertTrue(consumed)
        self.assertTrue(self.plugin.dock.isHidden())

    def test_event_filter_does_not_steal_control_space(self):
        """Plain Control+Space is NOT consumed by Antigravity, preserving the Launcher finder."""
        self.context.set_setting("display_mode", AntigravityPlugin.MODE_SIDEPANE)
        self.plugin.activate(self.context)
        self.assertTrue(self.plugin.dock.isHidden())

        # Plain Control + Space (used by Launcher)
        key_event = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Space,
            Qt.KeyboardModifier.ControlModifier,
            " ",
        )
        consumed = self.plugin.event_filter.eventFilter(self.window, key_event)
        self.assertFalse(consumed)
        self.assertTrue(self.plugin.dock.isHidden())

    def test_settings_widget_reflects_system_settings_hotkey(self):
        """AntigravitySettingsWidget displays the hotkey configured in system settings."""
        self.window.settings_manager.set("shortcuts", "toggle_antigravity", "Ctrl+Alt+Space")
        self.plugin.activate(self.context)
        widget = self.plugin.get_settings_widget(self.context)
        texts = [lbl.text() for lbl in widget.findChildren(QLabel)]
        self.assertTrue(any("Control" in t or "Ctrl" in t for t in texts))
        # Reset back to default
        self.window.settings_manager.set("shortcuts", "toggle_antigravity", "Ctrl+Space")

    def test_switch_back_to_full_panel_mode(self):
        """Switching from sidepane back to full panel restores sidebar tab and tears down dock."""
        self.context.set_setting("display_mode", AntigravityPlugin.MODE_SIDEPANE)
        self.plugin.activate(self.context)
        self.assertIsNotNone(self.plugin.dock)

        # Switch back to full panel
        self.plugin.set_display_mode(AntigravityPlugin.MODE_FULL_PANEL)
        self.assertEqual(self.plugin.current_mode, AntigravityPlugin.MODE_FULL_PANEL)
        self.assertIsNone(self.plugin.dock)
        self.assertIsNone(self.plugin.shortcut)
        self.assertIsNone(self.plugin.event_filter)
        self.assertIn("antigravity", self.window.tabs.tabs)

    def test_deactivate_cleans_up_all_resources(self):
        """Deactivating in sidepane mode removes dock and shortcuts cleanly."""
        self.context.set_setting("display_mode", AntigravityPlugin.MODE_SIDEPANE)
        self.plugin.activate(self.context)
        self.assertIsNotNone(self.plugin.dock)

        self.plugin.deactivate()
        self.assertIsNone(self.plugin.dock)
        self.assertIsNone(self.plugin.shortcut)
        self.assertIsNone(self.plugin.event_filter)
        self.assertIsNone(self.plugin.panel)
        self.assertIsNone(self.plugin.context)


if __name__ == "__main__":
    unittest.main()
