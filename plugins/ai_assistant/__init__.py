"""AI Assistant plugin (Theta Hub): a tool-calling LLM agent that reads and acts on ThetaIDE's runs.

Layers, bottom up:
  providers/   one adapter interface over OpenAI, Anthropic, Gemini and Ollama (plain HTTP)
  keys.py      API keys in the OS keychain
  analysis.py  logs and metrics reduced to diagnostic evidence
  tools.py     the tools the model may call, over a narrow IDEBridge protocol
  agent.py     the tool-calling loop
  bridge.py    IDEBridge over the live window;  panel.py / settings_widget.py  the Qt UI

Everything up to agent.py is Qt-free and tested with mocked HTTP.
"""

from __future__ import annotations

from typing import Any

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

from .keys import KeyStore

TAB_ID = "ai_assistant"


class AIAssistantPlugin(Plugin):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: Any = None
        self.keys = KeyStore()

    def activate(self, context: PluginContext) -> None:
        from PyQt6.QtWidgets import QPushButton

        from .panel import AssistantPanel

        self.context = context
        # Which providers have a keychain key (names only), so startup never opens the keychain.
        self.keys.attach_index(
            lambda: context.get_setting("keychain_keys", None),
            lambda names: context.set_setting("keychain_keys", names),
        )
        self.panel = AssistantPanel(context, self.keys)
        context.add_sidebar_tab(
            tab_id=TAB_ID,
            widget=self.panel,
            title="AI Assistant",
            icon_name="ai_assistant",
            short_label="AI",
        )
        context.register_command(
            command_id="ai_assistant.open",
            title="View: AI Assistant",
            callback=self.show,
            keywords=(
                "ai",
                "assistant",
                "llm",
                "chat",
                "agent",
                "gpt",
                "claude",
                "gemini",
                "ollama",
            ),
            icon="ai_assistant",
            detail="Ask an LLM about your runs, configs and metrics",
        )
        context.register_command(
            command_id="ai_assistant.explain_run",
            title="AI: Explain the selected run",
            callback=lambda: self.ask(
                "Explain the selected run: how did it go, did anything go wrong, and what should I try next?"
            ),
            keywords=("ai", "explain", "why", "failed", "diagnose"),
            icon="ai_assistant",
            detail="Diagnose the run open in the Training Monitor",
        )

        def ask_button() -> QPushButton:
            button = QPushButton("Ask AI about this run")
            button.clicked.connect(
                lambda: self.ask(
                    "Explain the selected run: how did it go, did anything go wrong, and what should I try next?"
                )
            )
            return button

        context.contribute_to_slot("monitor.footer", "ai_assistant.ask", ask_button, priority=110)

    def show(self) -> None:
        if self.context is not None:
            self.context.select_sidebar_tab(TAB_ID)

    def ask(self, text: str) -> None:
        self.show()
        if self.panel is not None:
            self.panel.ask(text)

    def can_deactivate(self) -> bool:
        return True

    def deactivate(self) -> None:
        if self.panel is not None:
            self.panel.shutdown()
            self.panel.deleteLater()
            self.panel = None
        if self.context is not None:
            self.context.remove_sidebar_tab(TAB_ID)
            self.context = None

    def get_settings_widget(self, context: PluginContext):
        from .settings_widget import AssistantSettings

        return AssistantSettings(
            context,
            self.keys,
            on_change=self.panel.reload_settings if self.panel else None,
        )
