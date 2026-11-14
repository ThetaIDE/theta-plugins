"""Unit tests for Obsidian panel widget, note management, and experiment linking."""
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from src.app.api.desktop import PluginContext
    from .panel import ObsidianPanel
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestObsidianPanel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_obsidian_")
        self.window = Window(data_dir=self.temp_dir)
        self.storage_file = Path(self.temp_dir) / "plugins" / "obsidian.json"
        self.context = PluginContext("obsidian", self.window, self.storage_file)

        # Set an isolated vault directory
        self.vault_dir = Path(self.temp_dir) / "test_vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.context.set_setting("vault_path", str(self.vault_dir))

        self.panel = ObsidianPanel(self.context)

    def tearDown(self):
        self.panel.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vault_directory_and_tree_population(self):
        """Panel loads vault and lists markdown notes in the tree."""
        note1 = self.vault_dir / "research_notes.md"
        note1.write_text("# NeSyRL Research Notes\nExploring neurosymbolic RL.", encoding="utf-8")

        subfolder = self.vault_dir / "experiments"
        subfolder.mkdir()
        note2 = subfolder / "cartpole_ppo.md"
        note2.write_text("# CartPole PPO\nResults and observations.", encoding="utf-8")

        self.panel._refresh_vault_tree()
        self.assertEqual(self.panel.tree.topLevelItemCount(), 1)
        root_item = self.panel.tree.topLevelItem(0)
        self.assertIn("test_vault", root_item.text(0))
        self.assertEqual(root_item.childCount(), 2)

    def test_open_and_save_note(self):
        """Opening a file displays its content; editing and saving writes back to disk."""
        test_file = self.vault_dir / "todo.md"
        test_file.write_text("# Todo\n- Run PPO benchmark\n", encoding="utf-8")

        self.panel.open_file(test_file)
        self.assertEqual(self.panel.current_file, test_file)
        self.assertEqual(self.panel.editor.toPlainText(), "# Todo\n- Run PPO benchmark\n")
        self.assertFalse(self.panel._is_dirty)

        # Edit text
        self.panel.editor.setPlainText("# Todo\n- Run PPO benchmark\n- Evaluate hybrid agent\n")
        self.assertTrue(self.panel._is_dirty)

        # Save note
        self.panel.save_current_note()
        self.assertFalse(self.panel._is_dirty)
        self.assertEqual(test_file.read_text(encoding="utf-8"), "# Todo\n- Run PPO benchmark\n- Evaluate hybrid agent\n")

    def test_daily_note_creation(self):
        """Clicking Daily Note creates today's note with the research template."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        expected_note = self.vault_dir / "Daily Notes" / f"{today_str}.md"

        self.assertFalse(expected_note.exists())
        self.panel.open_daily_note()

        self.assertTrue(expected_note.exists())
        content = expected_note.read_text(encoding="utf-8")
        self.assertIn(f"# Daily Log — {today_str}", content)
        self.assertIn("🎯 Objectives", content)
        self.assertIn("🧪 Experiment Runs & Hypotheses", content)
        self.assertEqual(self.panel.current_file, expected_note)

    def test_insert_active_experiment_link(self):
        """Inserts reference to the currently selected experiment in the IDE."""
        test_note = self.vault_dir / "active_log.md"
        test_note.write_text("# Research Log\n\n", encoding="utf-8")
        self.panel.open_file(test_note)

        # Set an active experiment on the window
        self.window.current_experiment = "cartpole/quick_test"
        self.panel.insert_active_experiment_link()

        editor_text = self.panel.editor.toPlainText()
        self.assertIn("Experiment Reference: `cartpole/quick_test`", editor_text)


if __name__ == "__main__":
    unittest.main()
