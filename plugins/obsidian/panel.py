"""Obsidian Panel widget for browsing, editing, and linking research notes."""
from __future__ import annotations
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from src.app.api.desktop import PluginContext


class ObsidianPanel(QWidget):
    """Integrated Obsidian vault browser, research markdown editor, and experiment logger."""

    def __init__(self, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context = context
        self.current_file: Optional[Path] = None
        self._is_dirty = False

        # Load or initialize vault directory
        saved_vault = self.context.get_setting("vault_path")
        if saved_vault and Path(saved_vault).is_dir():
            self.vault_dir = Path(saved_vault)
        else:
            # Check default candidate directories in workspace
            workspace = self.context.get_workspace_dir()
            if (workspace / "notes").is_dir():
                self.vault_dir = workspace / "notes"
            elif (workspace / ".obsidian").is_dir():
                self.vault_dir = workspace
            else:
                self.vault_dir = workspace / "notes"
                self.vault_dir.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._refresh_vault_tree()

        # Listen for active experiment changes
        self._on_exp_changed = lambda exp, path: self._handle_experiment_changed(exp, path)
        self.context.add_experiment_change_listener(self._on_exp_changed)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(10)

        # ── Eyebrow & Actions Header ─────────────────────────────────────────
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        # Eyebrow / title
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        eyebrow = QLabel("OBSIDIAN NOTES")
        eyebrow.setObjectName("eyebrow")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #83a598;")
        
        self.vault_label = QLabel(f"Vault: {self.vault_dir.name}")
        self.vault_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_box.addWidget(eyebrow)
        title_box.addWidget(self.vault_label)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Vault selector
        self.btn_select_vault = QPushButton("📁 Choose Vault…")
        self.btn_select_vault.setToolTip("Select an existing Obsidian vault directory")
        self.btn_select_vault.clicked.connect(self._choose_vault)
        header_layout.addWidget(self.btn_select_vault)

        # Daily Note button
        self.btn_daily = QPushButton("📅 Daily Note")
        self.btn_daily.setToolTip("Open or create today's daily research note (YYYY-MM-DD.md)")
        self.btn_daily.clicked.connect(self.open_daily_note)
        header_layout.addWidget(self.btn_daily)

        # New Note button
        self.btn_new = QPushButton("＋ New Note")
        self.btn_new.setToolTip("Create a new markdown note in the vault")
        self.btn_new.clicked.connect(self._create_new_note)
        header_layout.addWidget(self.btn_new)

        # Link Active Experiment button
        self.btn_link_exp = QPushButton("🔗 Link Experiment")
        self.btn_link_exp.setToolTip("Insert a Markdown link and YAML reference for the active experiment")
        self.btn_link_exp.clicked.connect(self.insert_active_experiment_link)
        header_layout.addWidget(self.btn_link_exp)

        # Save button
        self.btn_save = QPushButton("💾 Save Note")
        self.btn_save.setObjectName("primary")
        self.btn_save.setToolTip("Save changes to current note (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_current_note)
        header_layout.addWidget(self.btn_save)

        main_layout.addLayout(header_layout)

        # ── Main Splitter: File Tree + Markdown Editor ────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Pane: Vault Navigation
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter notes…")
        self.search_input.textChanged.connect(self._filter_tree)
        search_row.addWidget(self.search_input)

        btn_refresh = QToolButton()
        btn_refresh.setText("⟳")
        btn_refresh.setToolTip("Refresh vault files")
        btn_refresh.clicked.connect(self._refresh_vault_tree)
        search_row.addWidget(btn_refresh)
        left_layout.addLayout(search_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.tree, 1)

        splitter.addWidget(left_pane)

        # Right Pane: Note Editor
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Note Info Bar
        info_bar = QHBoxLayout()
        self.note_title_label = QLabel("No note selected")
        self.note_title_label.setStyleSheet("font-weight: 600; font-size: 13px;")
        info_bar.addWidget(self.note_title_label)
        info_bar.addStretch()

        self.char_count_label = QLabel("")
        self.char_count_label.setStyleSheet("color: #928374; font-size: 11px;")
        info_bar.addWidget(self.char_count_label)
        right_layout.addLayout(info_bar)

        # Markdown Text Editor
        self.editor = QPlainTextEdit()
        font = QFont("Consolas", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setPlaceholderText("Select a note from the vault or click '📅 Daily Note' to begin writing...")
        self.editor.textChanged.connect(self._on_text_changed)
        right_layout.addWidget(self.editor, 1)

        splitter.addWidget(right_pane)
        splitter.setSizes([260, 740])
        main_layout.addWidget(splitter, 1)

    # ── Vault Browsing & Tree ────────────────────────────────────────────────

    def _choose_vault(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Obsidian Vault Directory", str(self.vault_dir)
        )
        if chosen:
            self.vault_dir = Path(chosen)
            self.vault_label.setText(f"Vault: {self.vault_dir.name}")
            self.context.set_setting("vault_path", str(self.vault_dir))
            self._refresh_vault_tree()

    def _refresh_vault_tree(self) -> None:
        self.tree.clear()
        if not self.vault_dir.exists():
            return

        def populate(parent_item, folder: Path):
            try:
                entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return

            for entry in entries:
                if entry.name.startswith("."):
                    continue  # Ignore .obsidian or .git
                item = QTreeWidgetItem([entry.name])
                item.setData(0, Qt.ItemDataRole.UserRole, str(entry))
                if entry.is_dir():
                    item.setText(0, f"📁 {entry.name}")
                    populate(item, entry)
                    # Expand root directories by default
                    if folder == self.vault_dir:
                        item.setExpanded(True)
                elif entry.suffix.lower() in (".md", ".markdown", ".txt"):
                    item.setText(0, f"📄 {entry.name}")
                else:
                    continue
                parent_item.addChild(item)

        root_item = QTreeWidgetItem([f"📚 {self.vault_dir.name}"])
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(self.vault_dir))
        root_item.setExpanded(True)
        populate(root_item, self.vault_dir)
        self.tree.addTopLevelItem(root_item)

    def _filter_tree(self, text: str) -> None:
        filter_text = text.strip().lower()

        def apply_filter(item: QTreeWidgetItem) -> bool:
            child_matches = False
            for i in range(item.childCount()):
                if apply_filter(item.child(i)):
                    child_matches = True

            text_matches = filter_text in item.text(0).lower()
            is_visible = bool(not filter_text or text_matches or child_matches)
            item.setHidden(not is_visible)
            if child_matches:
                item.setExpanded(True)
            return is_visible

        for i in range(self.tree.topLevelItemCount()):
            apply_filter(self.tree.topLevelItem(i))

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in (".md", ".markdown", ".txt"):
            self.open_file(path)

    # ── Note Loading & Saving ────────────────────────────────────────────────

    def open_file(self, path: Path) -> None:
        if self._is_dirty and self.current_file and self.current_file != path:
            self.save_current_note()

        try:
            content = path.read_text(encoding="utf-8")
            self.current_file = path
            self.editor.blockSignals(True)
            self.editor.setPlainText(content)
            self.editor.blockSignals(False)
            self._is_dirty = False
            self.note_title_label.setText(path.name)
            self._update_stats()
        except Exception as exc:
            QMessageBox.warning(self, "Read Error", f"Could not open file: {exc}")

    def save_current_note(self) -> None:
        if not self.current_file:
            return
        try:
            self.current_file.parent.mkdir(parents=True, exist_ok=True)
            self.current_file.write_text(self.editor.toPlainText(), encoding="utf-8")
            self._is_dirty = False
            self.note_title_label.setText(self.current_file.name)
            self.context.show_status_message(f"Saved note: {self.current_file.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save note: {exc}")

    def _on_text_changed(self) -> None:
        if not self._is_dirty and self.current_file:
            self._is_dirty = True
            self.note_title_label.setText(f"{self.current_file.name} ●")
        self._update_stats()

    def _update_stats(self) -> None:
        text = self.editor.toPlainText()
        words = len(text.split())
        lines = len(text.splitlines()) or (1 if text else 0)
        self.char_count_label.setText(f"{words} words  ·  {lines} lines")

    # ── Research & Obsidian Workflows ────────────────────────────────────────

    def open_daily_note(self) -> None:
        """Create or open today's daily research note."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_folder = self.vault_dir / "Daily Notes"
        daily_folder.mkdir(parents=True, exist_ok=True)
        daily_file = daily_folder / f"{today_str}.md"

        if not daily_file.exists():
            template = (
                f"# Daily Log — {today_str}\n\n"
                f"## 🎯 Objectives\n- \n\n"
                f"## 🧪 Experiment Runs & Hypotheses\n\n"
                f"## 📝 Findings & Analysis\n\n"
            )
            daily_file.write_text(template, encoding="utf-8")
            self._refresh_vault_tree()

        self.open_file(daily_file)

    def _create_new_note(self) -> None:
        name, ok = QInputDialog.getText(self, "New Note", "Note title (without .md):")
        if ok and name.strip():
            clean_name = name.strip()
            if not clean_name.endswith(".md"):
                clean_name += ".md"
            target = self.vault_dir / clean_name
            if target.exists():
                QMessageBox.information(self, "Note Exists", "Opening existing note.")
            else:
                target.write_text(f"# {name.strip()}\n\n", encoding="utf-8")
                self._refresh_vault_tree()
            self.open_file(target)

    def insert_active_experiment_link(self) -> None:
        """Insert reference to the currently selected experiment in the IDE."""
        exp = self.context.get_current_experiment()
        config_path = self.context.get_active_config_path()

        if not exp and not config_path:
            exp_text = "*(No experiment currently active)*\n"
        else:
            exp_text = f"\n### 🧪 Experiment Reference: `{exp or 'Custom'}`\n"
            if config_path:
                exp_text += f"- **Config:** `[[{config_path.name}]]` (`{config_path}`)\n"
            exp_text += f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        cursor = self.editor.textCursor()
        cursor.insertText(exp_text)
        self.editor.setTextCursor(cursor)
        self.context.show_status_message("Inserted active experiment reference into note.")

    def _handle_experiment_changed(self, exp_name: Optional[str], file_path: Optional[str]) -> None:
        # Can be used to update status indicator if desired
        pass

    def cleanup(self) -> None:
        if self._is_dirty and self.current_file:
            self.save_current_note()
        self.context.remove_experiment_change_listener(self._on_exp_changed)
