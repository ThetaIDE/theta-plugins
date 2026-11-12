"""The panel's look: SVG icons and a stylesheet in the IDE's theme colours.

Colours come from the IDE's active theme (frontend.theme), so the panel matches the rest of ThetaIDE and
follows theme switches. IDE builds without it fall back to the Qt palette. Icons are the plugin's own SVGs,
drawn like the IDE's (24 px grid, 1.8 px strokes, currentColor), recoloured per state when rendered.
"""

from __future__ import annotations

import functools
from pathlib import Path

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import QApplication, QToolButton, QWidget

ICON_DIR = Path(__file__).resolve().parent / "icons"
ICON_PROPERTY = "ai_assistant_icon"


def theme_colors() -> dict[str, str]:
    """The IDE theme's colour roles (base, panel, surface, raised, border, text, muted, accent, primary, ...)."""
    try:
        from frontend.theme import current_theme

        colors = dict(current_theme()["colors"])
        if colors.get("text") and colors.get("base"):
            return colors
    except Exception:  # an IDE build without themes, or tests
        pass
    palette = QApplication.palette() if QApplication.instance() else QPalette()

    def role(r: QPalette.ColorRole) -> str:
        return palette.color(r).name()

    highlight = palette.color(QPalette.ColorRole.Highlight)
    return {
        "base": role(QPalette.ColorRole.Base),
        "panel": role(QPalette.ColorRole.Window),
        "surface": role(QPalette.ColorRole.AlternateBase),
        "raised": role(QPalette.ColorRole.Button),
        "border": role(QPalette.ColorRole.Mid),
        "disabled": palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name(),
        "text": role(QPalette.ColorRole.Text),
        "muted": role(QPalette.ColorRole.PlaceholderText),
        "accent": highlight.name(),
        "primary": highlight.name(),
        "primary_hover": highlight.lighter(115).name(),
        "focus": highlight.name(),
    }


@functools.lru_cache(maxsize=256)
def _pixmap(name: str, color: str, size: int, ratio: float) -> QPixmap:
    from PyQt6.QtSvg import QSvgRenderer

    source = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(source.encode("utf-8")))
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def svg_icon(name: str, normal: str, active: str | None = None, disabled: str | None = None, size: int = 20) -> QIcon:
    """icons/<name>.svg in `normal` colour, `active` on hover and `disabled` when disabled."""
    app = QApplication.instance()
    ratio = app.devicePixelRatio() if app is not None else 1.0
    icon = QIcon()
    for mode, color in (
        (QIcon.Mode.Normal, normal),
        (QIcon.Mode.Active, active or normal),
        (QIcon.Mode.Selected, active or normal),
        (QIcon.Mode.Disabled, disabled or normal),
    ):
        pixmap = _pixmap(name, color, size, ratio)
        icon.addPixmap(pixmap, mode, QIcon.State.Off)
        icon.addPixmap(
            _pixmap(name, active or normal, size, ratio) if mode != QIcon.Mode.Disabled else pixmap,
            mode,
            QIcon.State.On,
        )
    return icon


def header_button(name: str, tooltip: str) -> QToolButton:
    """A 30 x 30 icon button like the IDE's header buttons; recolour_icons() keeps it in theme."""
    button = QToolButton()
    button.setObjectName("headerButton")
    button.setProperty(ICON_PROPERTY, name)
    button.setIconSize(QSize(18, 18))
    button.setFixedSize(30, 30)
    button.setAutoRaise(True)
    button.setToolTip(tooltip)
    button.setAccessibleName(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def recolour_icons(root: QWidget, colors: dict[str, str]) -> None:
    """Render every header button's icon under `root` in the theme's colours."""
    for button in root.findChildren(QToolButton):
        name = button.property(ICON_PROPERTY)
        if name:
            button.setIcon(svg_icon(name, colors["text"], colors["accent"], colors["disabled"]))


@functools.lru_cache(maxsize=16)
def _renderer(name: str, color: str):
    from PyQt6.QtSvg import QSvgRenderer

    source = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").replace("currentColor", color)
    return QSvgRenderer(QByteArray(source.encode("utf-8")))


class CrossSpinner(QWidget):
    """A cross that steps clockwise by 45 degrees while the assistant works: +, then x, then + again.

    It jumps from one position to the next and holds there, with no turning in between. The timer runs only
    while the spinner is shown."""

    STEP_DEGREES = 45.0
    STEP_MS = 260  # how long each + or x stays before the next step

    def __init__(self, size: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self._angle = 0.0
        self._color = "#888888"
        self.setFixedSize(size, size)
        self.setAccessibleName("Working")
        self._timer = QTimer(self)
        self._timer.setInterval(self.STEP_MS)
        self._timer.timeout.connect(self._advance)
        self.hide()

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def is_spinning(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        self._angle = 0.0
        self.show()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _advance(self) -> None:
        self._angle = (self._angle + self.STEP_DEGREES) % 360.0
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802  (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)  # positive is clockwise on screen
        half = self._size / 2
        _renderer("cross", self._color).render(painter, QRectF(-half, -half, self._size, self._size))
        painter.end()


def stylesheet(c: dict[str, str]) -> str:
    return f"""
QToolButton#headerButton {{ border: none; border-radius: 6px; padding: 0; }}
QToolButton#headerButton:hover {{ background: {c["raised"]}; }}
QToolButton#headerButton:checked {{ background: {c["surface"]}; }}
QToolButton#headerButton::menu-indicator {{ image: none; width: 0; }}
QLabel#paneTitle {{ font-weight: 600; padding-left: 4px; color: {c["text"]}; }}
QListWidget#chatList {{ border: none; background: transparent; outline: none; }}
QListWidget#chatList::item {{ padding: 6px 8px; border-radius: 6px; margin: 1px 0; color: {c["text"]}; }}
QListWidget#chatList::item:selected {{ background: {c["raised"]}; color: {c["text"]}; }}
QListWidget#chatList::item:hover:!selected {{ background: {c["surface"]}; }}
QToolButton#chip {{
    border: 1px solid {c["border"]}; border-radius: 12px; padding: 3px 11px;
    background: transparent; color: {c["text"]};
}}
QToolButton#chip:hover {{ background: {c["raised"]}; border-color: {c["muted"]}; }}
QToolButton#chip:pressed {{ background: {c["surface"]}; }}
QPlainTextEdit#composerInput {{
    border: 1px solid {c["border"]}; border-radius: 10px; padding: 6px 8px;
    background: {c["base"]}; color: {c["text"]};
}}
QPlainTextEdit#composerInput:focus {{ border-color: {c["focus"]}; }}
QToolButton#sendButton {{ border: none; border-radius: 18px; padding: 0; background: {c["primary"]}; }}
QToolButton#sendButton:hover {{ background: {c["primary_hover"]}; }}
QToolButton#sendButton:disabled {{ background: {c["raised"]}; }}
QToolButton#sendButton[busy="true"] {{ background: {c["raised"]}; border: 1px solid {c["border"]}; }}
QFrame#keyBar {{ border: 1px solid {c["border"]}; border-left: 3px solid {c["accent"]}; border-radius: 8px;
    background: {c["surface"]}; }}
QFrame#keyBar QLabel {{ background: transparent; }}
QPushButton#signInButton {{ background: {c["primary"]}; color: {c["base"]}; border: none; border-radius: 6px;
    padding: 7px 14px; font-weight: 600; }}
QPushButton#signInButton:hover {{ background: {c["primary_hover"]}; }}
QLabel#disclaimer {{ color: {c["muted"]}; font-size: 11px; }}
QLabel#muted {{ color: {c["muted"]}; }}
"""
