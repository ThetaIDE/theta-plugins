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
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QImage, QKeySequence, QShortcut, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
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
from .oauth import OAuthError, OpenRouterSignIn
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
from .ui_style import CrossSpinner, header_button, recolour_icons, stylesheet, svg_icon, theme_colors
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


def _escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_\[\]<>#|])", r"\\\1", text)


def figure_markdown(index: int, entry: dict) -> str:
    """A figure entry: the image (click to open it full size) and its caption."""
    path = str(entry.get("path") or "")
    caption = _escape_markdown(entry.get("text") or Path(path).name)
    if not Path(path).is_file():
        return f"> 🖼 *{caption}* (the image file no longer exists)"
    url = bytes(QUrl.fromLocalFile(path).toEncoded()).decode()
    return f"[![{caption}](figure:{index})]({url})\n\n*{caption}* · [open]({url})"


class FocusReleaser(QObject):
    """Clicking anywhere outside the message box takes the text cursor out of it.

    Qt moves focus only to widgets that accept it, so clicking a label, a chip, the empty panel or most of
    the IDE left the cursor blinking in the box. This application-wide filter clears it on any press
    outside the box, except on the widgets in `keep` (the Send button)."""

    def __init__(self, editor: QWidget, keep: tuple[QWidget, ...] = ()):
        super().__init__(editor)
        self.editor = editor
        self.keep = keep

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802  (Qt override)
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget) and self.editor.hasFocus():
            inside = watched is self.editor or self.editor.isAncestorOf(watched)
            kept = any(watched is w or w.isAncestorOf(watched) for w in self.keep)
            if not inside and not kept:
                self.editor.clearFocus()
        return False


class Transcript(QTextBrowser):
    """The chat view. It loads only the chat's own figures (figure:<entry index>), scaled to fit the width;
    other image references in model text load nothing, so a reply cannot pull arbitrary files into view."""

    def __init__(self, entries: Callable[[], list[dict]], rerender: Callable[[], None], parent=None):
        super().__init__(parent)
        self._entries = entries
        self._rerender = rerender
        self._cache: dict[tuple, QImage] = {}
        self._laid_out_width = 0

    def figure_width(self) -> int:
        return max(160, self.viewport().width() - 28)

    def loadResource(self, kind: int, url: QUrl) -> Any:  # noqa: N802  (Qt override)
        if kind != QTextDocument.ResourceType.ImageResource.value:
            return super().loadResource(kind, url)
        if url.scheme() != "figure":
            return QImage()
        try:
            entry = self._entries()[int(url.path())]
        except (ValueError, IndexError):
            return QImage()
        return self.figure(str(entry.get("path") or ""))

    def figure(self, path: str) -> QImage:
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            return QImage()
        width = self.figure_width()
        key = (path, mtime, width)
        if key not in self._cache:
            image = QImage(path)
            if not image.isNull():
                # Show at its own size, or the panel's width if narrower; keep full detail on HiDPI screens.
                target = min(image.width(), width)
                pixels = int(target * (self.devicePixelRatioF() or 1.0))
                if image.width() > pixels:
                    image = image.scaledToWidth(pixels, Qt.TransformationMode.SmoothTransformation)
                image.setDevicePixelRatio(image.width() / target)
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = image
        self._laid_out_width = width
        return self._cache[key]

    def resizeEvent(self, event) -> None:  # noqa: N802  (Qt override)
        super().resizeEvent(event)
        has_figures = any(e.get("kind") == "figure" for e in self._entries())
        if has_figures and abs(self.figure_width() - self._laid_out_width) > 24:
            self._laid_out_width = self.figure_width()
            self._rerender()


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
{relay}<p>Your API key is sent only to authenticate. What {provider} keeps, and for how long, is set by its API \
terms and your account's data settings.</p>
<p>A local model (Ollama, or an OpenAI-compatible server on this computer) keeps everything on this machine.</p>"""

# Providers that route each request to another company's model; the notice says so.
RELAY_NOTE = """\
<p>{provider} is a router: it forwards each request to the company serving the model you chose \
(for example OpenAI, Anthropic or Google), whose terms also apply.</p>
"""
RELAY_PROVIDERS = frozenset({"openrouter"})

# Providers whose key can come from signing in in the browser (OAuth with PKCE) instead of pasting it.
SIGN_IN_FLOWS: dict[str, Callable[[], Any]] = {"openrouter": OpenRouterSignIn}

KEY_URLS = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/apikey",
    "openrouter": "https://openrouter.ai/keys",
}

WELCOME = """<div style="margin-top: 28px">
<p align="center" style="font-size: 15px"><b>Ask about your runs</b></p>
<p align="center">Why did a run fail? How do two runs compare? What should I change next?</p>
<p align="center" style="color: #888">The assistant reads configs, logs and metrics in this workspace and can plot \
curves. It asks before it launches, stops or changes anything.</p>
</div>"""

DISCLAIMER = "AI can make mistakes. Check its answers and any change before you rely on them."

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
    _signed_in = pyqtSignal(str, object)  # provider id, key or OAuthError
    alerts_found = pyqtSignal(list)

    def __init__(self, context: Any, keys: KeyStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.context = context
        self.keys = keys
        self.bridge = WindowBridge(context.window, workspace_root(context))
        self.bridge.figures_dir = chats_folder(context).parent / "figures"
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
        self._render_timer.timeout.connect(lambda: self._render(tail_only=True))
        self._shown_prefix: str | None = None  # the transcript's markdown before the last entry
        self._tail_start = 0  # where the last entry starts in the transcript's document

        self._build_ui()
        self._load_selection()
        self._restore_last_chat()
        if not self.entries:
            self._render()  # the welcome text

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
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        outer.addWidget(self.splitter)

        # Saved chats, newest first: a list beside the conversation, shown with the ☰ button.
        self.history_pane = QWidget()
        history_layout = QVBoxLayout(self.history_pane)
        history_layout.setContentsMargins(8, 10, 0, 10)
        history_layout.setSpacing(6)
        heading = QLabel("Chats")
        heading.setObjectName("paneTitle")
        history_layout.addWidget(heading)
        self.history_list = QListWidget()
        self.history_list.setObjectName("chatList")
        self.history_list.setWordWrap(True)
        self.history_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.history_list.itemClicked.connect(self._history_clicked)
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._history_menu)
        QShortcut(QKeySequence.StandardKey.Delete, self.history_list, activated=self._delete_selected_chat)
        history_layout.addWidget(self.history_list, 1)
        self.splitter.addWidget(self.history_pane)

        main = QWidget()
        layout = QVBoxLayout(main)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(6)
        self.splitter.addWidget(main)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.history_toggle = header_button("chats", "Show or hide your saved chats")
        self.history_toggle.setCheckable(True)
        self.history_toggle.toggled.connect(self._set_history_visible)
        self.provider_box = QComboBox()
        for pid, cls in PROVIDERS.items():
            self.provider_box.addItem(cls.display_name, pid)
        self.provider_box.currentIndexChanged.connect(self._provider_changed)
        self.model_box = QComboBox()
        self.model_box.setEditable(True)
        self.model_box.setMinimumWidth(120)
        self.model_box.setToolTip("Model id. Type any id the provider accepts, or press ↻ to list them.")
        self.model_box.currentTextChanged.connect(self._model_changed)
        refresh = header_button("refresh", "List the models available to this key")
        refresh.clicked.connect(self._refresh_models)
        self.new_chat_button = header_button("new_chat", "New chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        self.more_button = header_button("more", "API key, server and chat options")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        self.more_menu.aboutToShow.connect(self._fill_more_menu)
        self.more_button.setMenu(self.more_menu)
        row.addWidget(self.history_toggle)
        row.addWidget(self.provider_box)
        row.addWidget(self.model_box, 1)
        row.addWidget(refresh)
        row.addWidget(self.new_chat_button)
        row.addWidget(self.more_button)
        layout.addLayout(row)

        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("Server URL")
        self.base_url.setToolTip("Where this provider's API is served; leave empty for the default")
        self.base_url.editingFinished.connect(self._base_url_changed)
        layout.addWidget(self.base_url)

        # Shown when the provider needs a key and has none, or when changing it: entered right here.
        self.key_bar = QFrame()
        self.key_bar.setObjectName("keyBar")
        key_layout = QVBoxLayout(self.key_bar)
        key_layout.setContentsMargins(10, 8, 10, 8)
        key_layout.setSpacing(6)
        self.key_prompt = QLabel()
        self.key_prompt.setWordWrap(True)
        self.key_prompt.setOpenExternalLinks(True)
        key_layout.addWidget(self.key_prompt)
        self.sign_in_button = QPushButton()
        self.sign_in_button.setObjectName("signInButton")
        self.sign_in_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sign_in_button.clicked.connect(self.sign_in)
        key_layout.addWidget(self.sign_in_button)
        self.paste_label = QLabel("or paste a key:")
        self.paste_label.setObjectName("muted")
        key_layout.addWidget(self.paste_label)
        self._sign_in: Any = None  # the attempt in progress
        self._sign_in_cancel = threading.Event()
        self._signed_in.connect(self._on_signed_in)
        entry = QHBoxLayout()
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setClearButtonEnabled(True)
        self.key_input.returnPressed.connect(self._save_key)
        self.key_save = QPushButton("Save")
        self.key_save.clicked.connect(self._save_key)
        self.key_cancel = QPushButton("Cancel")
        self.key_cancel.clicked.connect(self._cancel_key_edit)
        entry.addWidget(self.key_input, 1)
        entry.addWidget(self.key_save)
        entry.addWidget(self.key_cancel)
        key_layout.addLayout(entry)
        self.key_note = QLabel()
        self.key_note.setObjectName("muted")
        self.key_note.setWordWrap(True)
        key_layout.addWidget(self.key_note)
        layout.addWidget(self.key_bar)
        self._editing_key = False

        self.transcript = Transcript(lambda: self.entries, lambda: self._render(follow=False))
        self.transcript.setOpenLinks(False)  # tool lines are links that expand in place
        self.transcript.anchorClicked.connect(self._on_link)
        self.alerts_box = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_box)
        self.alerts_layout.setContentsMargins(0, 0, 0, 0)
        self.alerts_layout.setSpacing(4)
        self.alerts_box.hide()
        self._alert_rows: list[AlertRow] = []
        layout.addWidget(self.alerts_box)
        layout.addWidget(self.transcript, 1)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.prompt_chips: list[QToolButton] = []
        for title, prompt in QUICK_PROMPTS:
            chip = QToolButton()
            chip.setObjectName("chip")
            chip.setText(title)
            chip.setToolTip(prompt)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, p=prompt: self.ask(p))
            chips.addWidget(chip)
            self.prompt_chips.append(chip)
        chips.addStretch(1)
        layout.addLayout(chips)

        composer = QHBoxLayout()
        composer.setSpacing(8)
        self.input = QPlainTextEdit()
        self.input.setObjectName("composerInput")
        self.input.setPlaceholderText("Message the assistant (Ctrl+Enter to send)")
        self.input.setFixedHeight(68)
        self.send_button = QToolButton()  # beside the message box, not inside it
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setIconSize(QSize(24, 24))
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.clicked.connect(self._send_clicked)
        composer.addWidget(self.input, 1)
        composer.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(composer)
        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self.input, activated=self._send_clicked)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.spinner = CrossSpinner(16)  # steps round while the assistant is thinking or running tools
        self.status = QLabel("")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        status_row.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignTop)
        status_row.addWidget(self.status, 1)
        layout.addLayout(status_row)

        self.disclaimer = QLabel(DISCLAIMER)
        self.disclaimer.setObjectName("disclaimer")
        self.disclaimer.setWordWrap(True)
        self.disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.disclaimer)

        self._set_send_mode(busy=False)
        self.apply_theme()
        theme_manager = getattr(getattr(self.context, "window", None), "theme_manager", None)
        changed = getattr(theme_manager, "changed", None)
        if changed is not None:
            changed.connect(self.apply_theme)
        self._focus_releaser = FocusReleaser(self.input, keep=(self.send_button,))
        QApplication.instance().installEventFilter(self._focus_releaser)

        visible = bool(self._setting("history_open", False))
        self.history_toggle.setChecked(visible)
        self._set_history_visible(visible)
        self._refresh_history_list()

    def apply_theme(self) -> None:
        """Restyle and recolour the icons in the IDE's current theme (called again when it changes)."""
        self._colors = theme_colors()
        self.setStyleSheet(stylesheet(self._colors))
        recolour_icons(self, self._colors)
        self._set_send_mode(self.busy)
        self._render(follow=False)

    def _set_send_mode(self, busy: bool) -> None:
        colors = getattr(self, "_colors", None) or theme_colors()
        if busy:
            self.send_button.setIcon(svg_icon("stop", colors["text"], size=24))
            self.send_button.setToolTip("Stop the reply")
        else:
            self.send_button.setIcon(svg_icon("send", colors["base"], disabled=colors["disabled"], size=24))
            self.send_button.setToolTip("Send (Ctrl+Enter)")
        self.send_button.setAccessibleName(self.send_button.toolTip())
        self.spinner.set_color(colors["accent"])
        if busy and not self.spinner.is_spinning():
            self.spinner.start()
        elif not busy:
            self.spinner.stop()
        self.send_button.setProperty("busy", "true" if busy else "false")
        self.send_button.style().unpolish(self.send_button)
        self.send_button.style().polish(self.send_button)

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
        self.model_box.lineEdit().setCursorPosition(0)  # show the start of a long "vendor/model" id
        self.model_box.blockSignals(False)
        urls = self._setting("base_urls", {}) or {}
        self.base_url.setText(urls.get(pid, ""))
        self.base_url.setPlaceholderText(cls.default_base_url)
        self.base_url.setVisible(pid == "ollama" or bool(urls.get(pid)))
        self._editing_key = False
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
        cls = PROVIDERS[pid]
        missing = cls.requires_key and self.keys.source(pid) is None
        shown = missing or self._editing_key
        self.key_bar.setVisible(shown)
        self.key_cancel.setVisible(self._editing_key and not missing)
        self.more_button.setToolTip(f"{self._key_description()}\nAPI key, server and chat options")
        if not shown:
            return
        name = cls.display_name
        link = KEY_URLS.get(pid)
        can_sign_in = pid in SIGN_IN_FLOWS
        self.sign_in_button.setVisible(can_sign_in)
        self.paste_label.setVisible(can_sign_in)
        self._update_sign_in_button()
        if cls.requires_key:
            get = f' <a href="{link}">Get a key ↗</a>' if link else ""
            if can_sign_in:
                self.key_prompt.setText(f"<b>Connect your {html.escape(name)} account</b> to start chatting.")
                get_one = f' (<a href="{link}">get one ↗</a>)' if link else ""
                self.paste_label.setText(f"or paste an API key{get_one}:")
                self.paste_label.setOpenExternalLinks(True)
            else:
                self.key_prompt.setText(f"<b>Add your {html.escape(name)} API key</b> to start chatting.{get}")
            self.key_input.setPlaceholderText(f"Paste your {name} API key")
        else:
            self.key_prompt.setText(f"<b>Token for your {html.escape(name)} server</b> (only if it requires one).")
            self.key_input.setPlaceholderText("Paste a token, or leave empty to remove it")
        env = ENV_VARS.get(pid, "")
        self.key_note.setText(
            "Saved in your system keychain, never in a file. "
            + (f"Or set ${env} before starting ThetaIDE." if env else "")
        )

    def _key_description(self) -> str:
        pid = self.provider_id
        if not PROVIDERS[pid].requires_key and not self.keys.source(pid):
            return "Local model: no key needed, nothing leaves this machine."
        return {
            "session": f"Key {mask(self.keys.peek(pid))} for this session only (not saved)",
            "keychain": " ".join(filter(None, ["Key", mask(self.keys.peek(pid)), "saved in the system keychain"])),
            "environment": f"Key from ${ENV_VARS.get(pid, '')}",
        }.get(self.keys.source(pid) or "", "No API key set")

    def edit_key(self) -> None:
        self._editing_key = True
        self._update_key_status()
        self.key_input.setFocus()

    def _cancel_key_edit(self) -> None:
        self._editing_key = False
        self.key_input.clear()
        self._update_key_status()

    def _save_key(self) -> None:
        pid = self.provider_id
        key = self.key_input.text().strip()
        if not key and PROVIDERS[pid].requires_key:
            self.key_input.setFocus()
            return
        if not key:
            self.keys.delete(pid)
            self._key_stored()
            return
        if self._store_key(pid, key):
            self._key_stored()

    def _store_key(self, pid: str, key: str, how: str = "") -> bool:
        """Keychain first; with no keychain, offer this session only. False if the user declined."""
        name = PROVIDERS[pid].display_name
        try:
            self.keys.set(pid, key)
            self.status.setText(f"{how}{name} key saved in the system keychain.")
        except KeyStoreError as exc:
            if not self._use_key_for_session(str(exc)):
                return False
            self.keys.set_session(pid, key)
            self.status.setText(f"{how}Using this key until ThetaIDE closes. It was not saved.")
        return True

    def _key_stored(self) -> None:
        self.key_input.clear()
        self._editing_key = False
        self.agent = None
        self._update_key_status()

    # ── Sign in with the browser (OAuth) ────────────────────────────────────

    def _update_sign_in_button(self) -> None:
        name = PROVIDERS[self.provider_id].display_name
        waiting = self._sign_in is not None
        self.sign_in_button.setText("Cancel sign-in" if waiting else f"Sign in with {name}")
        self.sign_in_button.setToolTip(
            "Stop waiting for the browser"
            if waiting
            else f"Opens {name} in your browser. Approve ThetaIDE there and the key is saved here."
        )

    def sign_in(self) -> None:
        """Open the provider's sign-in page; when the user approves, the new key is saved like a pasted one."""
        if self._sign_in is not None:  # the button reads "Cancel sign-in"
            self._sign_in_cancel.set()
            return
        pid = self.provider_id
        flow_class = SIGN_IN_FLOWS.get(pid)
        if flow_class is None:
            return
        flow = flow_class()
        try:
            url = flow.start()
        except OSError as exc:
            self.status.setText(f"Could not start sign-in: {exc}")
            return
        self._sign_in, self._sign_in_cancel = flow, threading.Event()
        if not QDesktopServices.openUrl(QUrl(url)):
            QApplication.clipboard().setText(url)
            self.status.setText("Could not open a browser. The sign-in address is on the clipboard: paste it into one.")
        else:
            self.status.setText(f"Approve ThetaIDE in your browser to sign in to {PROVIDERS[pid].display_name}…")
        self._update_sign_in_button()
        cancel = self._sign_in_cancel
        threading.Thread(target=self._wait_for_sign_in, args=(pid, flow, cancel), daemon=True).start()

    def _wait_for_sign_in(self, pid: str, flow: Any, cancel: threading.Event) -> None:
        """Background thread: wait for the browser's redirect, exchange the code, report back."""
        try:
            result: Any = flow.finish(cancel.is_set)
        except OAuthError as exc:
            result = exc
        except Exception as exc:  # never leave the button stuck on "Cancel sign-in"
            result = OAuthError(f"Sign-in failed: {exc}")
        if not cancel.is_set() or not isinstance(result, str):
            self._signed_in.emit(pid, result)
        else:
            self._signed_in.emit(pid, OAuthError("Sign-in was cancelled."))

    def _on_signed_in(self, pid: str, result: Any) -> None:
        self._sign_in = None
        self._update_sign_in_button()
        if isinstance(result, Exception):
            self.status.setText(str(result))
            return
        if self._store_key(pid, result, how="Signed in. "):
            self._key_stored()

    def _use_key_for_session(self, reason: str) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("No keychain available")
        box.setText(reason)
        box.setInformativeText(
            "You can use the key until ThetaIDE closes without saving it. To save keys, install or start a "
            "keychain (GNOME Keyring or KWallet), or set the provider's environment variable."
        )
        use = box.addButton("Use for this session", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(use)
        box.exec()
        return box.clickedButton() is use

    def _remove_key(self) -> None:
        self.keys.delete(self.provider_id)
        self.agent = None
        self._update_key_status()

    def _fill_more_menu(self) -> None:
        menu = self.more_menu
        menu.clear()
        pid = self.provider_id
        menu.addAction(self._key_description()).setEnabled(False)
        source = self.keys.source(pid)
        label = "Set token…" if not PROVIDERS[pid].requires_key else ("Change API key…" if source else "Add API key…")
        menu.addAction(label, self.edit_key)
        if pid in SIGN_IN_FLOWS:
            menu.addAction(f"Sign in with {PROVIDERS[pid].display_name}…", self.sign_in)
        if source in ("keychain", "session"):
            menu.addAction("Remove saved key" if source == "keychain" else "Forget this key", self._remove_key)
        server = menu.addAction("Custom server URL")
        server.setCheckable(True)
        server.setChecked(self.base_url.isVisible())
        server.toggled.connect(self._toggle_base_url)
        menu.addSeparator()
        delete = menu.addAction("Delete this chat", self._delete_current_chat)
        delete.setEnabled(bool(self.chat.history) and not self.busy)

    def _toggle_base_url(self, shown: bool) -> None:
        self.base_url.setVisible(shown)
        if not shown and self.base_url.text().strip():
            self.base_url.clear()
            self._base_url_changed()
        elif shown:
            self.base_url.setFocus()

    def _provider_config(self, with_key: bool = True) -> ProviderConfig:
        """with_key=False leaves the keychain closed (it may prompt to unlock); only requests need the key."""
        pid = self.provider_id
        urls = self._setting("base_urls", {}) or {}
        return ProviderConfig(
            model=self.model_box.currentText().strip() or PROVIDERS[pid].default_model,
            api_key=self.keys.get(pid) if with_key else None,
            base_url=urls.get(pid) or None,
        )

    # ── Data leaving this computer ──────────────────────────────────────────

    def destination(self) -> str | None:
        """The host the current provider sends data to, or None when it runs on this computer."""
        cls = PROVIDERS[self.provider_id]
        host = (urlparse(self._provider_config(with_key=False).base_url or cls.default_base_url).hostname or "").lower()
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
        name = html.escape(provider)
        relay = RELAY_NOTE.format(provider=name) if self.provider_id in RELAY_PROVIDERS else ""
        box.setText(DATA_SHARING.format(provider=name, host=html.escape(host), relay=relay))
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
        self._set_send_mode(busy=True)
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
        self._refresh_history_list()
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
            self._refresh_history_list()
        except (OSError, TypeError, ValueError) as exc:
            self.status.setText(f"Could not save this chat: {exc}")

    def _set_history_visible(self, visible: bool) -> None:
        self.history_pane.setVisible(visible)
        if visible:
            total = max(self.splitter.width(), 400)
            pane = min(220, max(150, total // 3))
            self.splitter.setSizes([pane, total - pane])
            self._refresh_history_list()
        self.context.set_setting("history_open", visible)

    def _refresh_history_list(self) -> None:
        import datetime

        self.history_list.clear()
        chats = self.store.list()
        if not chats:
            placeholder = QListWidgetItem("No saved chats yet")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.history_list.addItem(placeholder)
            return
        for summary in chats[:100]:
            when = datetime.datetime.fromtimestamp(summary.updated).strftime("%b %d, %H:%M")
            item = QListWidgetItem(f"{summary.title}\n{when}")
            item.setToolTip(summary.title)
            item.setData(Qt.ItemDataRole.UserRole, summary.id)
            self.history_list.addItem(item)
            if summary.id == self.chat.id:
                item.setSelected(True)
                self.history_list.setCurrentItem(item)

    def _history_clicked(self, item: QListWidgetItem) -> None:
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        if not chat_id or chat_id == self.chat.id:
            return
        if self.busy:
            self.status.setText("Wait for the reply to finish, or press Stop, before switching chats.")
            self._refresh_history_list()
            return
        self._open_saved(chat_id)

    def _history_menu(self, position) -> None:
        item = self.history_list.itemAt(position)
        chat_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not chat_id:
            return
        menu = QMenu(self.history_list)
        menu.addAction("Open", lambda: self._history_clicked(item)).setEnabled(not self.busy)
        menu.addAction("Delete", lambda: self._delete_chat(chat_id)).setEnabled(not self.busy)
        menu.exec(self.history_list.viewport().mapToGlobal(position))

    def _delete_selected_chat(self) -> None:
        item = self.history_list.currentItem()
        chat_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if chat_id and not self.busy:
            self._delete_chat(chat_id)

    def _open_saved(self, chat_id: str) -> None:
        try:
            self._open_chat(self.store.load(chat_id))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Could not open chat", str(exc))

    def _delete_current_chat(self) -> None:
        self._delete_chat(self.chat.id)

    def _delete_chat(self, chat_id: str) -> None:
        """Delete a saved chat and the charts the assistant drew for it; images the runs produced stay."""
        if chat_id == self.chat.id:
            entries = self.entries
        else:
            try:
                entries = self.store.load(chat_id).entries
            except (OSError, ValueError, KeyError):
                entries = []
        self.store.delete(chat_id)
        own = Path(self.bridge.figures_dir).resolve()
        for entry in entries:
            path = Path(str(entry.get("path") or "")) if entry.get("kind") == "figure" else None
            if path is not None and path.resolve().is_relative_to(own):
                path.unlink(missing_ok=True)
        if chat_id == self.chat.id:
            self._open_chat(Chat())
        else:
            self._refresh_history_list()

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
        for path, caption in getattr(result, "figures", ()):
            self.entries.append({"kind": "figure", "text": caption, "path": path})
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
        if usage.cost is not None:  # the provider's own figure beats our estimate
            return usage.cost
        if self.provider_id in FREE_PROVIDERS:
            return None
        model = self.agent.provider.config.model if self.agent is not None else self._provider_config(False).model
        return estimate(usage, model, self._setting("prices", {}) or {})

    def _on_failed(self, message: str) -> None:
        self._append("error", message)
        self._done("")

    def _done(self, status: str) -> None:
        self.busy = False
        self._set_send_mode(busy=False)
        self.status.setText(status)
        while self.entries and self.entries[-1]["kind"] == "assistant" and not self.entries[-1]["text"].strip():
            self.entries.pop()
        self._render()
        self._save_chat()

    def _render(self, follow: bool = True, tail_only: bool = False) -> None:
        """Redraw the transcript. follow=False keeps the view where it is (expanding a tool line).

        tail_only (while a reply streams in) redraws just the last entry when nothing before it changed, so
        the view neither re-lays out the whole chat nor jumps; anything else rebuilds the document."""
        bar = self.transcript.verticalScrollBar()
        position = bar.value()
        at_bottom = follow and position >= bar.maximum() - 4
        self._snippets: list[str] = []
        parts = [self._entry_markdown(index, entry) for index, entry in enumerate(self.entries)]
        if not parts:
            self._shown_prefix = None
            self.transcript.setHtml(WELCOME)  # wraps to the panel's width, unlike a placeholder
            return
        prefix = "\n\n".join(p for p in parts[:-1] if p)
        if not prefix:  # a single entry: nothing to keep
            self._shown_prefix = None
            self.transcript.setMarkdown(parts[-1])
        elif not (tail_only and prefix == self._shown_prefix):
            self.transcript.setMarkdown(prefix)
            self._shown_prefix = prefix
            end = QTextCursor(self.transcript.document())
            end.movePosition(QTextCursor.MoveOperation.End)
            self._tail_start = end.position()
        if prefix:
            self._replace_tail(parts[-1])
        bar.setValue(bar.maximum() if at_bottom else position)

    def _entry_markdown(self, index: int, entry: dict) -> str:
        kind, text = entry["kind"], entry["text"]
        if kind == "user":
            return ("---\n\n" if index else "") + "**You**\n\n" + text
        if kind == "assistant":
            return with_copy_links(text, self._snippets) if text else "…"
        if kind == "tool":
            label = text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            marker = "▾" if entry.get("open") else "▸"
            line = f"> [{marker} 🔧 {label}](tool:{index})"
            return line + "\n\n" + tool_details(entry) if entry.get("open") else line
        if kind == "figure":
            return figure_markdown(index, entry)
        if kind == "error":
            return "> ⚠ **Error:** " + text
        return "> " + text

    def _replace_tail(self, markdown: str) -> None:
        """Replace everything after _tail_start with the last entry, keeping its first block's format.

        Inserting markdown into a new block would give its first block (a heading, a code block) that
        block's plain format, so a throwaway paragraph goes first and is then removed with the block break
        before it, leaving the entry's own blocks exactly as setMarkdown() would lay them out."""
        cursor = QTextCursor(self.transcript.document())
        cursor.setPosition(self._tail_start)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        if not markdown:
            return
        cursor.beginEditBlock()
        cursor.insertBlock()
        cursor.insertMarkdown("\u200b\n\n" + markdown)
        cursor.setPosition(self._tail_start)
        cursor.setPosition(self._tail_start + 2, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()

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
        self._sign_in_cancel.set()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self._focus_releaser)
        self.watch_timer.stop()
        self.worker.cancel.set()
        self.invoker.closing = True
        self.thread_.quit()
        self.thread_.wait(3000)
