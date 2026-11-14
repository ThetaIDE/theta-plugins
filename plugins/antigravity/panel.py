"""Antigravity Panel widget for chat interface with Google Antigravity Agent CLI backend."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QTextBrowser, QVBoxLayout, QWidget,
)

if TYPE_CHECKING:
    from src.app.api.desktop import PluginContext


def find_agy_binary() -> Optional[str]:
    """Find the path to the 'agy' binary across system PATH and standard locations."""
    # 1. Check system PATH
    p = shutil.which("agy")
    if p:
        return p

    # 2. Check candidate standard directories
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / "agy",
        home / ".gemini" / "antigravity-cli" / "bin" / "agy",
        Path("/opt/homebrew/bin/agy"),
        Path("/usr/local/bin/agy"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    return None


class AgentWorker(QThread):
    response_chunk = pyqtSignal(str)
    response_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    conversation_id_update = pyqtSignal(str)

    def __init__(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        super().__init__()
        self.prompt = prompt
        self.conversation_id = conversation_id
        self.workspace_dir = workspace_dir
        self.proc: Optional[subprocess.Popen] = None
        self._stopped = False

    def stop(self):
        """Terminate the running CLI subprocess gracefully."""
        self._stopped = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def run(self):
        accumulated_text = ""
        try:
            agy_bin = find_agy_binary()
            if not agy_bin:
                self.error_occurred.emit(
                    "Antigravity CLI ('agy') not found in PATH or standard locations.\n"
                    "Please ensure 'agy' is installed and accessible."
                )
                return

            cmd = [agy_bin, "--output-format", "stream-json"]
            if self.conversation_id:
                cmd.extend(["--conversation", self.conversation_id])
            cmd.extend(["--print", self.prompt])

            # Prepare environment with complete PATH
            env = dict(os.environ)
            extra_paths = [
                str(Path.home() / ".local" / "bin"),
                str(Path.home() / ".gemini" / "antigravity-cli" / "bin"),
                "/opt/homebrew/bin",
                "/usr/local/bin",
            ]
            current_path = env.get("PATH", "")
            for ep in extra_paths:
                if ep not in current_path:
                    current_path = f"{ep}:{current_path}"
            env["PATH"] = current_path

            cwd = None
            if self.workspace_dir and os.path.isdir(self.workspace_dir):
                cwd = str(self.workspace_dir)

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=cwd,
            )

            # Read stream-json events line by line
            for line in self.proc.stdout:
                if self._stopped:
                    break
                line = line.strip()
                if not line or line.startswith("warning:"):
                    continue
                try:
                    data = json.loads(line)
                    event = data.get("event")
                    if event == "init":
                        cid = data.get("conversation_id")
                        if cid:
                            self.conversation_id_update.emit(cid)
                    elif event == "step_update":
                        su = data.get("step_update", {})
                        if su.get("step_type") == "agent_response" and "text_delta" in su:
                            delta = su["text_delta"]
                            accumulated_text += delta
                            self.response_chunk.emit(delta)
                    elif event == "result":
                        res = data.get("result", {})
                        full_resp = res.get("response", "")
                        if full_resp and not accumulated_text:
                            accumulated_text = full_resp
                            self.response_chunk.emit(full_resp)
                        break
                except json.JSONDecodeError:
                    pass

            self.proc.wait()
            if self._stopped:
                return

            if self.proc.returncode != 0:
                err = self.proc.stderr.read().strip() if self.proc.stderr else ""
                if err:
                    self.error_occurred.emit(f"CLI Error ({self.proc.returncode}): {err}")
                elif not accumulated_text:
                    self.error_occurred.emit(f"Agent exited with status code {self.proc.returncode}")
            elif not accumulated_text:
                err = self.proc.stderr.read().strip() if self.proc.stderr else ""
                if err:
                    self.error_occurred.emit(f"No response returned: {err}")

        except Exception as e:
            if not self._stopped:
                self.error_occurred.emit(str(e))
        finally:
            self.response_complete.emit()


class AntigravityPanel(QWidget):
    """Integrated Antigravity Agent chat interface."""

    def __init__(self, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context = context
        self._build_ui()
        self.worker: Optional[AgentWorker] = None
        self.conversation_id: Optional[str] = None

    def _build_ui(self):
        """Construct the Antigravity chat panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        eyebrow = QLabel("ANTIGRAVITY AGENT")
        eyebrow.setObjectName("eyebrow")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #83a598;")

        title_label = QLabel("Chat with AI")
        title_label.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_box.addWidget(eyebrow)
        title_box.addWidget(title_label)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        clear_btn = QPushButton("Clear Chat")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_chat)
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

        # Chat History
        self.chat_history = QTextBrowser()
        self.chat_history.setOpenExternalLinks(True)
        self.chat_history.setStyleSheet(
            "background-color: transparent; border: 1px solid rgba(255, 255, 255, 0.1); "
            "padding: 8px; border-radius: 6px; font-size: 13px; line-height: 1.4;"
        )
        self._append_system_message("Antigravity Agent ready. Ask me to write code, explain logic, or run tasks.")
        layout.addWidget(self.chat_history)

        # Input Area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask Antigravity to write code, explain errors, or run tasks...")
        self.message_input.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.message_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

    def _append_system_message(self, text: str):
        self.chat_history.append(f"<span style='color: #83a598;'><b>[System]</b> {text}</span>")

    def _append_user_message(self, text: str):
        # Escape basic HTML entities in user input to prevent injection
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.chat_history.append(f"<br/><b style='color: #4daafc;'>[User]</b> {escaped}")

    def _append_agent_message_start(self):
        self.chat_history.append(f"<br/><b style='color: #b8bb26;'>[Antigravity]</b> ")
        cursor = self.chat_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_history.setTextCursor(cursor)

    def _append_agent_chunk(self, text: str):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_history.setTextCursor(cursor)
        self.chat_history.insertPlainText(text)
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_history.setTextCursor(cursor)

    def _send_message(self):
        if self.worker and self.worker.isRunning():
            # If already generating, clicking Stop halts the task
            self.worker.stop()
            self._append_system_message("Request stopped.")
            self._on_agent_complete()
            return

        text = self.message_input.text().strip()
        if not text:
            return

        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_btn.setText("Stop")

        self._append_user_message(text)
        self._append_agent_message_start()

        workspace = None
        if hasattr(self.context, "get_workspace_dir"):
            try:
                workspace = str(self.context.get_workspace_dir())
            except Exception:
                pass

        self.worker = AgentWorker(text, self.conversation_id, workspace_dir=workspace)
        self.worker.response_chunk.connect(self._append_agent_chunk)
        self.worker.response_complete.connect(self._on_agent_complete)
        self.worker.error_occurred.connect(self._on_agent_error)
        self.worker.conversation_id_update.connect(self._on_conversation_id_update)
        self.worker.start()

    def _on_conversation_id_update(self, cid: str):
        self.conversation_id = cid

    def _on_agent_complete(self):
        self.send_btn.setText("Send")
        self.message_input.setEnabled(True)
        self.message_input.setFocus()

    def _on_agent_error(self, err: str):
        self._append_system_message(f"Error: {err}")
        self._on_agent_complete()

    def _clear_chat(self):
        self.chat_history.clear()
        self.conversation_id = None
        self._append_system_message("Chat history cleared. Started a new conversation.")

    def focus_input(self):
        """Set focus to the chat message input field."""
        self.message_input.setFocus()
        self.message_input.selectAll()

    def cleanup(self):
        """Clean up any ongoing worker tasks."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
