"""The AI Assistant sidebar panel: provider picker, streamed chat, tool activity and confirmations.

Threading: the agent loop and its HTTP streams run on a QThread so the IDE never freezes.
Tool calls and confirmation dialogs hop back to the GUI thread through MainThreadInvoker,
because they read and change widgets that only the GUI thread may touch.
"""

from __future__ import annotations

import html
import json
import re
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .agent import Agent, ToolFinished, ToolStarted, TurnFinished
from .approval import ApprovalDialog
from .bridge import WindowBridge
from .chat_store import Chat, ChatStore
from .keys import ENV_VARS, KeyStore, KeyStoreError, mask
from .pricing import FREE_PROVIDERS, estimate, format_cost
from .providers import (
    PROVIDERS,
    LLMError,
    ProviderConfig,
    StreamCancelled,
    TextDelta,
    ToolCall,
    create_provider,
)
from .tools import ToolRegistry, ToolResult, build_ide_tools
from .watch import INTERVAL_MS, Alert, RunWatcher

QUICK_PROMPTS = (
    (
        "Why did it fail?",
        "Why did the selected run fail? Find the error and tell me how to fix it.",
    ),
    (
        "Compare last two",
        "Compare the two most recent runs: what differs in config and how did it affect results?",
    ),
    (
        "Tune it",
        "Look at the selected run's metrics and config and suggest hyperparameter changes for the next run.",
    ),
)


def workspace_root(context: Any):
    """The project root. The window's WorkspacePaths is authoritative: PluginContext.get_workspace_dir()
    returns the run store's parent, which is the .thetaide data folder in a normal session."""
    paths = getattr(context.window, "workspace_paths", None)
    return paths.root if paths is not None else context.get_workspace_dir()


def chats_folder(context: Any):
    """<workspace>/.thetaide/assistant/chats: next to the IDE's other per-workspace state."""
    from pathlib import Path

    paths = getattr(context.window, "workspace_paths", None)
    base = Path(paths.data_dir) if paths is not None else Path(context.get_data_dir())
    return base / "assistant" / "chats"


def _code_block(text: str, language: str = "") -> str:
    fence = "~~~~"
    return f"{fence}{language}\n{text.replace(fence, '~ ~ ~ ~')}\n{fence}"


def tool_details(entry: dict) -> str:
    """The expanded view of a tool line: exactly what the model sent and what it got back."""
    arguments = json.dumps(entry.get("arguments") or {}, indent=2)
    result = entry.get("result")
    if result is None:
        body = "_Still running…_"
    else:
        shown = result if len(result) <= 6000 else result[:6000] + f"\n… ({len(result) - 6000} more characters)"
        label = "Result (error)" if entry.get("is_error") else "Result"
        body = f"**{label}**\n\n" + _code_block(shown)
    return "**Arguments**\n\n" + _code_block(arguments, "json") + "\n\n" + body


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

DATA_SHARING = """\
<p>To answer, the assistant sends {provider} (<b>{host}</b>):</p>
<ul>
<li>your messages and the chat so far</li>
<li>what it reads with its tools: run configs, excerpts of training logs, metric summaries, and the \
contents of workspace files it opens</li>
<li>the names of the config files in <code>in/config</code></li>
</ul>
<p>Your API key is sent only to authenticate. What {provider} keeps, and for how long, is set by its API \
terms and your account's data settings.</p>
<p>A local model (Ollama, or an OpenAI-compatible server on this computer) keeps everything on this machine.</p>"""

_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})[^\n]*\n(.*?)^\1\2[ \t]*$", re.DOTALL | re.MULTILINE)


def with_copy_links(text: str, snippets: list[str]) -> str:
    """Add a Copy link under each complete fenced code block; `snippets` collects what each one copies.

    A block still streaming in has no closing fence yet, so it gets its link once it is complete.
    """

    def add_link(match: re.Match) -> str:
        snippets.append(match.group(3).rstrip("\n"))
        return f"{match.group(0)}\n\n[⧉ Copy](copy:{len(snippets) - 1})"

    return _FENCE.sub(add_link, text)


def _short(value) -> str:
    """An argument as shown in the transcript: file contents and long strings are abbreviated."""
    text = json.dumps(value)
    return text if len(text) <= 60 else text[:40] + f'…" ({len(text)} chars)'


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
            box["error"] = StreamCancelled("The assistant is shutting down.")
            return
        try:
            box["result"] = box["fn"]()
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    def invoke(self, fn: Callable[[], Any]) -> Any:
        if QThread.currentThread() is self.thread():
            return fn()
        box: dict[str, Any] = {"fn": fn}
        self._call.emit(box)
        if "error" in box:
            raise box["error"]
        return box.get("result")


class AgentWorker(QObject):
    """Lives on the worker thread; runs one agent turn or one model listing at a time."""

    delta = pyqtSignal(str)
    tool_started = pyqtSignal(object, bool)
    tool_finished = pyqtSignal(object, object)
    turn_finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    notice = pyqtSignal(str)
    models_listed = pyqtSignal(str, list, str)  # provider id, models, error

    def __init__(self):
        super().__init__()
        self.cancel = threading.Event()

    @pyqtSlot(object, str)
    def run_turn(self, agent: Agent, text: str) -> None:
        self.cancel.clear()
        try:
            for event in agent.run(text, self.cancel):
                if isinstance(event, TextDelta):
                    self.delta.emit(event.text)
                elif isinstance(event, ToolStarted):
                    self.tool_started.emit(event.call, event.needs_confirmation)
                elif isinstance(event, ToolFinished):
                    self.tool_finished.emit(event.call, event.result)
                elif isinstance(event, TurnFinished):
                    self.turn_finished.emit(event)
        except StreamCancelled:
            self.failed.emit("Stopped.")
        except LLMError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {type(exc).__name__}: {exc}")

    @pyqtSlot(str, object)
    def list_models(self, provider_id: str, config: ProviderConfig) -> None:
        try:
            self.models_listed.emit(provider_id, create_provider(provider_id, config).list_models(), "")
        except Exception as exc:
            self.models_listed.emit(provider_id, [], str(exc))


MAX_ALERTS = 5
TAB_ID = "ai_assistant"


class AlertRow(QFrame):
    """One watcher finding: what is wrong, Ask AI, dismiss."""

    def __init__(
        self,
        alert: Alert,
        ask: Callable[[Alert], None],
        dismiss: Callable[[AlertRow], None],
    ):
        super().__init__()
        self.alert = alert
        self.setObjectName("aiAlert")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 4, 4)
        mark = "✗" if alert.severity == "error" else "⚠"
        label = QLabel(f"{mark} <b>{html.escape(alert.label)}</b> · {html.escape(alert.message)}")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        self.ask_button = QPushButton("Ask AI")
        self.ask_button.setToolTip(alert.prompt())
        self.ask_button.clicked.connect(lambda: ask(alert))
        close = QToolButton()
        close.setText("×")
        close.setToolTip("Dismiss")
        close.clicked.connect(lambda: dismiss(self))
        row.addWidget(label, 1)
        row.addWidget(self.ask_button)
        row.addWidget(close)


class AssistantPanel(QWidget):
    _start_turn = pyqtSignal(object, str)
    _fetch_models = pyqtSignal(str, object)
    alerts_found = pyqtSignal(list)

    def __init__(self, context: Any, keys: KeyStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.context = context
        self.keys = keys
        self.bridge = WindowBridge(context.window, workspace_root(context))
        self.tools: ToolRegistry = build_ide_tools(self.bridge)
        self.invoker = MainThreadInvoker(self)
        self.agent: Agent | None = None
        self.store = ChatStore(chats_folder(context))
        self.chat = Chat()
        self.entries: list[dict] = self.chat.entries  # the transcript as rendered
        self.history: list = self.chat.history  # provider-neutral Messages, shared by every Agent built here
        self.busy = False
        self._snippets: list[str] = []  # what each Copy link in the transcript copies

        self.thread_ = QThread(self)
        self.thread_.setObjectName("ai-assistant")
        self.worker = AgentWorker()
        self.bridge.on_gui = self.invoker.invoke
        self.bridge.hub.on_gui = self.invoker.invoke
        self.bridge.hub.cancelled = self.worker.cancel.is_set
        self.worker.moveToThread(self.thread_)
        self._start_turn.connect(self.worker.run_turn)
        self._fetch_models.connect(self.worker.list_models)
        self.worker.delta.connect(self._on_delta)
        self.worker.tool_started.connect(self._on_tool_started)
        self.worker.tool_finished.connect(self._on_tool_finished)
        self.worker.turn_finished.connect(self._on_turn_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.notice.connect(lambda text: self._append("notice", text))
        self.worker.models_listed.connect(self._on_models_listed)
        self.thread_.start()

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(50)
        self._render_timer.timeout.connect(self._render)

        self._build_ui()
        self._load_selection()
        self._restore_last_chat()

        # Run watching: a local check of training runs' metrics; it never calls a model.
        self.watcher = RunWatcher()
        self._watching = False
        self.alerts_found.connect(self._on_alerts)
        self.watch_timer = QTimer(self)
        self.watch_timer.setInterval(INTERVAL_MS)
        self.watch_timer.timeout.connect(self.poll_runs)
        self.watch_timer.start()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.provider_box = QComboBox()
        for pid, cls in PROVIDERS.items():
            self.provider_box.addItem(cls.display_name, pid)
        self.provider_box.currentIndexChanged.connect(self._provider_changed)
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setMinimumWidth(160)
        self.model_box.setToolTip("Model id. Type any id the provider accepts, or press ↻ to list them.")
        self.model_box.currentTextChanged.connect(self._model_changed)
        refresh = QPushButton("↻")
        refresh.setToolTip("List the models available to this key")
        refresh.setFixedWidth(32)
        refresh.clicked.connect(self._refresh_models)
        self.history_button = QToolButton()
        self.history_button.setText("History")
        self.history_button.setToolTip("Reopen a saved chat")
        self.history_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.history_menu = QMenu(self.history_button)
        self.history_menu.aboutToShow.connect(self._fill_history_menu)
        self.history_button.setMenu(self.history_menu)
        row.addWidget(self.provider_box)
        row.addWidget(self.model_box, 1)
        row.addWidget(refresh)
        row.addWidget(self.history_button)
        layout.addLayout(row)

        key_row = QHBoxLayout()
        self.key_label = QLabel()
        self.key_label.setObjectName("muted")
        self.key_button = QPushButton("Set API key…")
        self.key_button.clicked.connect(self._set_key)
        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("Server URL")
        self.base_url.editingFinished.connect(self._base_url_changed)
        key_row.addWidget(self.key_label, 1)
        key_row.addWidget(self.base_url, 1)
        key_row.addWidget(self.key_button)
        layout.addLayout(key_row)

        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)  # tool lines are links that expand in place
        self.transcript.anchorClicked.connect(self._on_link)
        self.transcript.setPlaceholderText(
            "Ask about your runs: why one failed, how two compare, what to change next.\n\n"
            "The assistant reads run configs, logs and metrics from this workspace, and can open files, "
            "select and launch experiments, and stop runs (it asks first before starting or stopping compute)."
        )
        self.alerts_box = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_box)
        self.alerts_layout.setContentsMargins(0, 0, 0, 0)
        self.alerts_layout.setSpacing(4)
        self.alerts_box.hide()
        self._alert_rows: list[AlertRow] = []
        layout.addWidget(self.alerts_box)
        layout.addWidget(self.transcript, 1)

        quick = QHBoxLayout()
        for title, prompt in QUICK_PROMPTS:
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, p=prompt: self.ask(p))
            quick.addWidget(button)
        layout.addLayout(quick)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Message the assistant (Ctrl+Enter to send)")
        self.input.setFixedHeight(76)
        layout.addWidget(self.input)
        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self.input, activated=self._send_clicked)

        bottom = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("muted")
        new_chat = QPushButton("New chat")
        new_chat.clicked.connect(self.new_chat)
        self.send_button = QPushButton("Send")
        self.send_button.setDefault(True)
        self.send_button.clicked.connect(self._send_clicked)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(new_chat)
        bottom.addWidget(self.send_button)
        layout.addLayout(bottom)

    # ── Provider selection & settings ───────────────────────────────────────

    @property
    def provider_id(self) -> str:
        return self.provider_box.currentData() or "openai"

    def _setting(self, key: str, default: Any = None) -> Any:
        return self.context.get_setting(key, default)

    def _load_selection(self) -> None:
        index = self.provider_box.findData(self._setting("provider", "openai"))
        self.provider_box.setCurrentIndex(max(0, index))
        self._provider_changed()

    def _provider_changed(self, *_: Any) -> None:
        pid = self.provider_id
        cls = PROVIDERS[pid]
        self.context.set_setting("provider", pid)
        models = self._setting("models", {}) or {}
        known = (self._setting("model_lists", {}) or {}).get(pid, [])
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItems(known)
        self.model_box.setCurrentText(models.get(pid) or cls.default_model)
        self.model_box.blockSignals(False)
        urls = self._setting("base_urls", {}) or {}
        self.base_url.setText(urls.get(pid, ""))
        self.base_url.setPlaceholderText(cls.default_base_url)
        self.base_url.setVisible(pid == "ollama" or bool(urls.get(pid)))
        self._update_key_status()
        self.agent = None  # rebuilt with the new provider on the next message; history is carried over

    def _model_changed(self, text: str) -> None:
        models = dict(self._setting("models", {}) or {})
        models[self.provider_id] = text.strip()
        self.context.set_setting("models", models)
        self.agent = None

    def _base_url_changed(self) -> None:
        urls = dict(self._setting("base_urls", {}) or {})
        value = self.base_url.text().strip()
        if value:
            urls[self.provider_id] = value
        else:
            urls.pop(self.provider_id, None)
        self.context.set_setting("base_urls", urls)
        self.agent = None

    def _update_key_status(self) -> None:
        pid = self.provider_id
        if not PROVIDERS[pid].requires_key:
            self.key_label.setText("Local model: no key needed, nothing leaves this machine.")
            self.key_button.setText("Set token…")
            return
        source = self.keys.source(pid)
        if source == "keychain":
            self.key_label.setText(f"Key {mask(self.keys.get(pid))} in the OS keychain")
        elif source == "environment":
            self.key_label.setText(f"Key from ${ENV_VARS[pid]}")
        else:
            self.key_label.setText("No API key set")
        self.key_button.setText("Change key…" if source == "keychain" else "Set API key…")

    def _set_key(self) -> None:
        pid = self.provider_id
        name = PROVIDERS[pid].display_name
        dialog = QInputDialog(self)
        dialog.setWindowTitle(f"{name} API key")
        dialog.setLabelText(
            f"Paste your {name} API key. It is stored in the OS keychain, never in a file.\n"
            "Leave empty and press OK to remove the saved key."
        )
        dialog.setTextEchoMode(QLineEdit.EchoMode.Password)
        if not dialog.exec():
            return
        key = dialog.textValue().strip()
        try:
            if key:
                self.keys.set(pid, key)
            else:
                self.keys.delete(pid)
        except KeyStoreError as exc:
            QMessageBox.warning(self, "API key not saved", str(exc))
        self.agent = None
        self._update_key_status()

    def _provider_config(self) -> ProviderConfig:
        pid = self.provider_id
        urls = self._setting("base_urls", {}) or {}
        return ProviderConfig(
            model=self.model_box.currentText().strip() or PROVIDERS[pid].default_model,
            api_key=self.keys.get(pid),
            base_url=urls.get(pid) or None,
        )

    # ── Data leaving this computer ──────────────────────────────────────────

    def destination(self) -> str | None:
        """The host the current provider sends data to, or None when it runs on this computer."""
        cls = PROVIDERS[self.provider_id]
        host = (urlparse(self._provider_config().base_url or cls.default_base_url).hostname or "").lower()
        return None if host in LOCAL_HOSTS or host.startswith("127.") else host

    def _data_sharing_allowed(self) -> bool:
        """Once per provider and server: say what leaves the computer before the first message does."""
        host = self.destination()
        if host is None:
            return True
        key = f"{self.provider_id}@{host}"
        accepted = list(self._setting("data_sharing_ok", []) or [])
        if key in accepted:
            return True
        if not self._confirm_data_sharing(PROVIDERS[self.provider_id].display_name, host):
            self.status.setText("Not sent. Choose a local model to keep run data on this computer.")
            return False
        self.context.set_setting("data_sharing_ok", [*accepted, key])
        return True

    def _confirm_data_sharing(self, provider: str, host: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Send workspace data to " + provider + "?")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(DATA_SHARING.format(provider=html.escape(provider), host=html.escape(host)))
        send = box.addButton("Send", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is send

    def _refresh_models(self) -> None:
        self.status.setText("Listing models…")
        self._fetch_models.emit(self.provider_id, self._provider_config())

    def _on_models_listed(self, pid: str, models: list, error: str) -> None:
        if error:
            self.status.setText(f"Could not list models: {error}")
            return
        lists = dict(self._setting("model_lists", {}) or {})
        lists[pid] = models
        self.context.set_setting("model_lists", lists)
        self.status.setText(f"{len(models)} models available.")
        if pid == self.provider_id:
            current = self.model_box.currentText()
            self.model_box.blockSignals(True)
            self.model_box.clear()
            self.model_box.addItems(models)
            self.model_box.setCurrentText(current)
            self.model_box.blockSignals(False)

    # ── Agent wiring ────────────────────────────────────────────────────────

    def _ensure_agent(self) -> Agent:
        """The agent for the current provider settings; the conversation survives a provider switch."""
        if self.agent is None:
            provider = create_provider(
                self.provider_id,
                self._provider_config(),
                on_retry=lambda attempt, wait, error: self.worker.notice.emit(
                    f"{error} — retrying in {wait:.0f}s (attempt {attempt + 1})"
                ),
            )
            self.agent = Agent(
                provider,
                self.tools,
                snapshot=lambda: self.invoker.invoke(self.bridge.snapshot),
                overview=self.bridge.overview,  # file reads only: fine on the worker thread
                confirm=lambda call, preview: self.invoker.invoke(lambda: self._confirm(call, preview)),
                execute=lambda call: self._on_tool_thread(call, self.tools.execute),
                preview=lambda call: self._on_tool_thread(call, self.tools.preview),
                max_steps=int(self._setting("max_steps", 8)),
            )
            self.agent.history = self.history
        return self.agent

    def _on_tool_thread(self, call: ToolCall, fn: Callable[[ToolCall], Any]) -> Any:
        """Background tools (network, installs) stay on the worker thread; the rest run on the GUI thread."""
        if self.tools.runs_in_background(call.name):
            return fn(call)
        return self.invoker.invoke(lambda: fn(call))

    def _confirm(self, call: ToolCall, preview: str) -> bool:
        """Ask the user. File writes always ask; launch/stop may be pre-approved in Settings."""
        is_write = call.name in ("write_file", "edit_file")
        auto = call.name in ("launch_experiment", "stop_run") and self._setting("auto_approve_actions", False)
        if auto:
            return True
        path = call.arguments.get("path", "")
        if is_write:
            new = preview.startswith("--- /dev/null")
            title = "Create file?" if new else "Edit file?"
            summary = f"The assistant wants to {'create' if new else 'change'} <b>{path}</b>:"
        elif call.name == "launch_experiment":
            current = self.bridge.snapshot().get("current_experiment") or "the selected experiment"
            title, summary = (
                "Launch run?",
                f"The assistant wants to queue and run <b>{current}</b>.",
            )
            preview = ""
        elif call.name == "install_from_hub":
            title = "Install from the Component Hub?"
            summary = f"The assistant wants to install <b>{call.arguments.get('component_id', '')}</b>:"
        elif call.name == "stop_run":
            title, summary = (
                "Stop run?",
                f"The assistant wants to stop run <b>{call.arguments.get('run_id', 'selected')}</b>.",
            )
            preview = ""
        else:
            title, summary = (
                "Allow action?",
                f"The assistant wants to run <b>{call.name}</b>:",
            )
        dialog = ApprovalDialog(title, summary, preview, is_diff=is_write, parent=self)
        approved = dialog.exec() == ApprovalDialog.DialogCode.Accepted
        self.status.setText(("Approved " if approved else "Rejected ") + (path or call.name))
        return approved

    def ask(self, text: str) -> None:
        """Send `text` as the user's message (used by quick prompts and IDE commands too)."""
        text = text.strip()
        if not text or self.busy:
            return
        if not self._data_sharing_allowed():
            if not self.input.toPlainText().strip():
                self.input.setPlainText(text)  # not lost: send it later, or switch to a local model
            return
        try:
            agent = self._ensure_agent()
        except Exception as exc:
            self._append("error", str(exc))
            return
        self.busy = True
        self.send_button.setText("Stop")
        self.status.setText(f"{PROVIDERS[self.provider_id].display_name} · {agent.provider.config.model} · thinking…")
        if not self.chat.title:
            self.chat.title = Chat.title_for(text)
        self._append("user", text)
        self._append("assistant", "")
        self._start_turn.emit(agent, text)

    def _send_clicked(self) -> None:
        if self.busy:
            self.worker.cancel.set()
            self.status.setText("Stopping…")
            return
        text = self.input.toPlainText()
        self.input.clear()
        self.ask(text)

    def new_chat(self) -> None:
        if not self.busy:
            self._open_chat(Chat())

    # ── Saved chats ─────────────────────────────────────────────────────────

    def _open_chat(self, chat: Chat) -> None:
        self.chat = chat
        self.entries = chat.entries
        self.history = chat.history
        if self.agent is not None:
            self.agent.history = self.history
        self.context.set_setting("last_chat", chat.id if chat.history else None)
        self.status.setText(f"Reopened “{chat.title}”" if chat.history else "")
        self._render()
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())

    def _restore_last_chat(self) -> None:
        chat_id = self._setting("last_chat")
        if chat_id:
            try:
                self._open_chat(self.store.load(chat_id))
            except (OSError, ValueError, KeyError):
                self.context.set_setting("last_chat", None)

    def _save_chat(self) -> None:
        self.chat.provider, self.chat.model = (
            self.provider_id,
            self.model_box.currentText().strip(),
        )
        try:
            self.store.save(self.chat)
            self.context.set_setting("last_chat", self.chat.id)
        except (OSError, TypeError, ValueError) as exc:
            self.status.setText(f"Could not save this chat: {exc}")

    def _fill_history_menu(self) -> None:
        import datetime

        self.history_menu.clear()
        chats = self.store.list()
        if not chats:
            self.history_menu.addAction("No saved chats yet").setEnabled(False)
        for summary in chats[:30]:
            when = datetime.datetime.fromtimestamp(summary.updated).strftime("%b %d %H:%M")
            action = self.history_menu.addAction(f"{summary.title}    ·  {when}")
            action.setCheckable(summary.id == self.chat.id)
            action.setChecked(summary.id == self.chat.id)
            action.setEnabled(not self.busy)
            action.triggered.connect(lambda _=False, cid=summary.id: self._open_saved(cid))
        if self.chat.history:
            self.history_menu.addSeparator()
            delete = self.history_menu.addAction("Delete this chat")
            delete.setEnabled(not self.busy)
            delete.triggered.connect(self._delete_current_chat)

    def _open_saved(self, chat_id: str) -> None:
        try:
            self._open_chat(self.store.load(chat_id))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Could not open chat", str(exc))

    def _delete_current_chat(self) -> None:
        self.store.delete(self.chat.id)
        self._open_chat(Chat())

    # ── Worker events (GUI thread) ──────────────────────────────────────────

    def _append(self, kind: str, text: str) -> None:
        self.entries.append({"kind": kind, "text": text})
        self._render_timer.start()

    def _on_delta(self, text: str) -> None:
        if not self.entries or self.entries[-1]["kind"] != "assistant":
            self.entries.append({"kind": "assistant", "text": ""})
        self.entries[-1]["text"] += text
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _on_tool_started(self, call: ToolCall, needs_confirmation: bool) -> None:
        args = ", ".join(f"{k}={_short(v)}" for k, v in call.arguments.items())
        self.status.setText(f"Running {call.name}…" + (" (waiting for your approval)" if needs_confirmation else ""))
        if self.entries and self.entries[-1]["kind"] == "assistant" and not self.entries[-1]["text"].strip():
            self.entries.pop()
        self.entries.append(
            {
                "kind": "tool",
                "text": f"{call.name}({args})",
                "name": call.name,
                "arguments": call.arguments,
                "result": None,
                "is_error": False,
                "open": False,
            }
        )
        self._render_timer.start()

    def _on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        for entry in reversed(self.entries):
            if entry["kind"] == "tool" and entry.get("result") is None:
                first = result.content.strip().splitlines()[0][:140] if result.content.strip() else ""
                entry["text"] += f" — failed: {first}" if result.is_error else " ✓"
                entry["result"], entry["is_error"] = result.content, result.is_error
                break
        self._append("assistant", "")

    def _on_link(self, url: QUrl) -> None:
        if url.scheme() == "copy":
            index = int(url.path() or -1)
            if 0 <= index < len(self._snippets):
                QApplication.clipboard().setText(self._snippets[index])
                self.status.setText("Copied to the clipboard.")
        elif url.scheme() == "tool":
            index = int(url.path() or -1)
            if 0 <= index < len(self.entries) and self.entries[index]["kind"] == "tool":
                self.entries[index]["open"] = not self.entries[index].get("open")
                self._render(follow=False)
        elif url.scheme() in ("http", "https", "file", "mailto"):
            QDesktopServices.openUrl(url)

    def _on_turn_finished(self, event: TurnFinished) -> None:
        usage = event.usage
        cached = f" ({usage.cached_input_tokens:,} cached)" if usage.cached_input_tokens else ""
        status = f"{usage.input_tokens:,} in{cached} / {usage.output_tokens:,} out tokens · {event.steps} step" + (
            "s" if event.steps != 1 else ""
        )
        cost = self._turn_cost(usage)
        if cost is not None:
            self.chat.cost += cost
            status += f" · {format_cost(cost)} (chat {format_cost(self.chat.cost)})"
        self._done(status)

    def _turn_cost(self, usage) -> float | None:
        if self.provider_id in FREE_PROVIDERS:
            return None
        model = self.agent.provider.config.model if self.agent is not None else self._provider_config().model
        return estimate(usage, model, self._setting("prices", {}) or {})

    def _on_failed(self, message: str) -> None:
        self._append("error", message)
        self._done("")

    def _done(self, status: str) -> None:
        self.busy = False
        self.send_button.setText("Send")
        self.status.setText(status)
        while self.entries and self.entries[-1]["kind"] == "assistant" and not self.entries[-1]["text"].strip():
            self.entries.pop()
        self._render()
        self._save_chat()

    def _render(self, follow: bool = True) -> None:
        """Redraw the transcript. follow=False keeps the view where it is (expanding a tool line)."""
        bar = self.transcript.verticalScrollBar()
        position = bar.value()
        at_bottom = follow and position >= bar.maximum() - 4
        parts = []
        self._snippets: list[str] = []
        for index, entry in enumerate(self.entries):
            kind, text = entry["kind"], entry["text"]
            if kind == "user":
                parts.append(("---\n\n" if parts else "") + "**You**\n\n" + text)
            elif kind == "assistant":
                parts.append(with_copy_links(text, self._snippets) if text else "…")
            elif kind == "tool":
                label = text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
                marker = "▾" if entry.get("open") else "▸"
                parts.append(f"> [{marker} 🔧 {label}](tool:{index})")
                if entry.get("open"):
                    parts.append(tool_details(entry))
            elif kind == "error":
                parts.append("> ⚠ **Error:** " + text)
            else:
                parts.append("> " + text)
        self.transcript.setMarkdown("\n\n".join(p for p in parts if p))
        bar.setValue(bar.maximum() if at_bottom else position)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    # ── Run watching ─────────────────────────────────────────────────────────

    def poll_runs(self) -> None:
        """Snapshot the runs here (GUI thread), read and analyse their metrics on a background thread."""
        if self._watching or not self._setting("watch_runs", True):
            return
        try:
            targets = self.bridge.watch_targets()
        except Exception:
            return
        if not targets:
            return
        self._watching = True
        threading.Thread(target=self._check_runs, args=(targets,), name="ai-run-watch", daemon=True).start()

    def _check_runs(self, targets: list) -> None:
        alerts: list[Alert] = []
        try:
            alerts = self.watcher.check(targets)
        finally:
            try:
                self.alerts_found.emit(alerts)  # queued onto the GUI thread
            except RuntimeError:  # the panel was deleted meanwhile
                pass

    def _on_alerts(self, alerts: list) -> None:
        self._watching = False
        for alert in alerts:
            self._add_alert(alert)
            if alert.kind != "failed":  # the IDE already announces failed runs
                self._toast(alert)

    def _add_alert(self, alert: Alert) -> None:
        rows = self.alert_rows()
        if len(rows) >= MAX_ALERTS:
            self._dismiss_alert(rows[0])
        row = AlertRow(alert, self.ask_about, self._dismiss_alert)
        self._alert_rows.append(row)
        self.alerts_layout.addWidget(row)
        self.alerts_box.show()

    def alert_rows(self) -> list[AlertRow]:
        return list(self._alert_rows)

    def _dismiss_alert(self, row: AlertRow) -> None:
        if row in self._alert_rows:
            self._alert_rows.remove(row)
        self.alerts_layout.removeWidget(row)
        row.hide()
        row.deleteLater()
        self.alerts_box.setVisible(bool(self._alert_rows))

    def _toast(self, alert: Alert) -> None:
        """The IDE's own toast (a system notification when ThetaIDE is in the background), else the status bar."""
        window = self.context.window
        notifier = getattr(window, "run_notifier", None)
        try:
            from frontend.toast import Toast

            if notifier is None or not notifier.enabled("toast"):
                raise LookupError
            tray = getattr(window, "tray", None)
            title = f"{alert.label}: possible problem"
            if not notifier.in_front() and tray is not None and tray.available():
                tray.notify(title, alert.message, warning=True)
                return
            toast = Toast(
                "✗" if alert.severity == "error" else "⚠",
                title,
                alert.message,
                "Ask AI",
                "number" if alert.severity == "error" else "accent",
            )
            toast.action.connect(lambda a=alert: self.ask_about(a))
            notifier.toasts.show(toast)
        except Exception:
            self.context.show_status_message(f"AI Assistant: {alert.label}: {alert.message}", 10000)

    def ask_about(self, alert: Alert) -> None:
        """Ask AI on an alert: open the panel and send the alert's question (or leave it ready if busy)."""
        try:
            self.context.select_sidebar_tab(TAB_ID)
        except Exception:
            pass
        for row in self.alert_rows():
            if row.alert.key == alert.key:
                self._dismiss_alert(row)
        if self.busy:
            self.input.setPlainText(alert.prompt())
            self.status.setText("Send the question about the run when this reply finishes.")
        else:
            self.ask(alert.prompt())

    def reload_settings(self) -> None:
        self.agent = None
        self._provider_changed()

    def shutdown(self) -> None:
        self.watch_timer.stop()
        self.worker.cancel.set()
        self.invoker.closing = True
        self.thread_.quit()
        self.thread_.wait(3000)
