"""DOOM Plugin for ThetaIDE: Real 1993 id Software DOOM."""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Optional

from src.app.api.desktop import Plugin, PluginManifest
from src.app.api.desktop import PluginContext
from .panel import DoomPanel


class DoomPlugin(Plugin):
    """Launches the authentic 1993 id Software DOOM engine inside Theta IDE and provides the DOOM RL environment."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: Optional[PluginContext] = None
        self.panel: Optional[DoomPanel] = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = DoomPanel(context=context)


        # Register tab into the primary IDE sidebar
        context.add_sidebar_tab(
            tab_id="doom",
            widget=self.panel,
            title="DOOM",
            icon_name="doom",
            short_label="DOOM",
        )
        context.log("DOOM plugin activated: Can it run DOOM? YES!")

    def deactivate(self) -> None:
        if self.context and self.panel:
            self.context.remove_sidebar_tab("doom")
            self.panel.cleanup()
            self.panel = None
            self.context.log("DOOM plugin deactivated.")
            self.context = None
