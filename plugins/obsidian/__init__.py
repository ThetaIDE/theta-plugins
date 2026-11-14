"""Obsidian Vault and Research Notes Plugin for ThetaIDE."""
from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext
from .panel import ObsidianPanel


class ObsidianPlugin(Plugin):
    """Integrates Obsidian research vaults and markdown note-taking into ThetaIDE."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: ObsidianPanel | None = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = ObsidianPanel(context=context)

        # Register tab into the sidebar
        context.add_sidebar_tab(
            tab_id="obsidian",
            widget=self.panel,
            title="Obsidian Notes",
            icon_name="obsidian",
            short_label="Notes",
        )
        context.log("Obsidian research notes plugin activated.")

    def deactivate(self) -> None:
        if self.context and self.panel:
            self.context.remove_sidebar_tab("obsidian")
            self.panel.cleanup()
            self.panel = None
            self.context.log("Obsidian research notes plugin deactivated.")
            self.context = None
