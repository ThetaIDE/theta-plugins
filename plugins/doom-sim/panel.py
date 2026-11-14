"""DOOM Panel: Launches authentic 1993 id Software DOOM in Theta IDE."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

if TYPE_CHECKING:
    from src.app.api.desktop import PluginContext


HTML_PATH = Path(__file__).resolve().parent / "doom.html"


class DoomPanel(QWidget):
    """Embedded viewport for running the real 1993 id Software DOOM engine."""

    def __init__(self, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context = context
        self.web_view: Optional[QWebEngineView] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Viewport ─────────────────────────────────────────────────────────
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView(self)
            self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.web_view.setStyleSheet("background-color: #000000;")

            page = self.web_view.page()
            page.featurePermissionRequested.connect(self._on_feature_permission_requested)

            settings = self.web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)

            if HTML_PATH.exists():
                html = HTML_PATH.read_text(encoding="utf-8")
                self.web_view.setHtml(html, QUrl("https://js-dos.com/"))
            else:
                self.web_view.setUrl(QUrl("https://cdn.dos.zone/custom/dos/doom.jsdos"))

            layout.addWidget(self.web_view)
        else:
            fallback = QLabel(
                "🎮 <b>DOOM (1993) Engine</b><br><br>"
                "To run the embedded DOOM WebEngine player, install <code>PyQt6-WebEngine</code>.<br>"
                "Run: <code>pip install PyQt6-WebEngine</code>"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: #cdd6f4; font-size: 13px; padding: 40px;")
            layout.addWidget(fallback)

    def _on_feature_permission_requested(self, security_origin: QUrl, feature: QWebEnginePage.Feature) -> None:
        """Automatically grant mouse lock and necessary permissions to the DOOM game."""
        if self.web_view and self.web_view.page():
            self.web_view.page().setFeaturePermission(
                security_origin,
                feature,
                QWebEnginePage.PermissionPolicy.PermissionGrantedByUser,
            )

    def reload_game(self) -> None:
        """Reload the DOOM engine."""
        if self.web_view:
            if HTML_PATH.exists():
                html = HTML_PATH.read_text(encoding="utf-8")
                self.web_view.setHtml(html, QUrl("https://js-dos.com/"))
            else:
                self.web_view.reload()

    def cleanup(self) -> None:
        """Free resources on plugin deactivation."""
        if self.web_view:
            self.web_view.stop()
            self.web_view.load(QUrl("about:blank"))
