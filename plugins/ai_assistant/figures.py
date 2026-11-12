"""Figures in the chat: find a run's images, and draw metric curves to PNG.

Drawing uses matplotlib's object API (Figure + Agg canvas), never pyplot, so it keeps no global state and
does not touch the IDE's Qt event loop. Nothing here imports Qt.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"})
MAX_PANELS = 6  # metrics per figure
MAX_RUNS = 6  # lines per panel
MAX_FIGURES_LISTED = 60

Series = tuple[Sequence[Any], Sequence[Any]]


class PlotError(Exception):
    """The figure cannot be drawn (no matching metrics, matplotlib missing)."""


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def find_images(folders: Sequence[Path], relative_to: Path) -> list[dict[str, Any]]:
    """Image files under these folders, newest first, as workspace-relative paths."""
    seen: dict[Path, dict[str, Any]] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and is_image(path) and path not in seen:
                stat = path.stat()
                try:
                    shown = path.relative_to(relative_to).as_posix()
                except ValueError:
                    shown = str(path)
                seen[path] = {
                    "path": shown,
                    "bytes": stat.st_size,
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    "_mtime": stat.st_mtime,
                }
    found = sorted(seen.values(), key=lambda f: f.pop("_mtime"), reverse=True)
    return found[:MAX_FIGURES_LISTED]


def _points(steps: Sequence[Any], values: Sequence[Any], smooth: int) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i, (step, value) in enumerate(zip(steps, values, strict=False)):
        try:
            x = float(step)
        except (TypeError, ValueError):
            x = float(i)
        try:
            y = float(value)
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y if math.isfinite(y) else math.nan)  # NaN leaves a visible gap
    if smooth > 1 and ys:
        window: list[float] = []
        smoothed = []
        for y in ys:
            if math.isfinite(y):
                window.append(y)
                if len(window) > smooth:
                    window.pop(0)
            smoothed.append(sum(window) / len(window) if window and math.isfinite(y) else math.nan)
        ys = smoothed
    return xs, ys


def matching_metrics(runs: Mapping[str, Mapping[str, Series]], wanted: Sequence[str]) -> list[str]:
    """Metric names to plot, in first-seen order: exact names first, then substring matches."""
    names: list[str] = []
    for series in runs.values():
        for name in series:
            if name not in names:
                names.append(name)
    if not wanted:
        return names[:MAX_PANELS]
    chosen = [n for n in names if n in wanted]
    chosen += [n for n in names if n not in chosen and any(w and w in n for w in wanted)]
    return chosen[:MAX_PANELS]


def plot_metrics(
    runs: Mapping[str, Mapping[str, Series]],
    metrics: Sequence[str],
    out_dir: Path,
    *,
    title: str = "",
    smooth: int = 0,
    log_y: bool = False,
) -> tuple[Path, list[str]]:
    """Draw one panel per metric with a line per run; save a PNG under out_dir. Returns (path, metrics)."""
    names = matching_metrics(runs, metrics)
    if not names:
        available = sorted({n for s in runs.values() for n in s})
        raise PlotError(
            f"No metric matches {list(metrics)}. Available: {', '.join(available[:40]) or 'none recorded'}."
        )
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise PlotError("Plotting needs matplotlib, which this ThetaIDE environment does not have.") from exc

    # One column sized for the chat sidebar (about 480 px wide), drawn at 2x so it stays sharp on HiDPI
    # screens and when opened full size.
    figure = Figure(figsize=(4.8, 2.7 * len(names) + (0.3 if title else 0)), dpi=200, layout="constrained")
    FigureCanvasAgg(figure)
    run_ids = list(runs)[:MAX_RUNS]
    for index, name in enumerate(names):
        axes = figure.add_subplot(len(names), 1, index + 1)
        for run_id in run_ids:
            if name not in runs[run_id]:
                continue
            xs, ys = _points(*runs[run_id][name], smooth)
            if xs:
                axes.plot(xs, ys, label=run_id if len(run_ids) > 1 else None, linewidth=1.4)
        axes.set_title(name, fontsize=10)
        axes.set_xlabel("step", fontsize=8)
        axes.tick_params(labelsize=8)
        axes.grid(True, alpha=0.3)
        if log_y:
            axes.set_yscale("symlog")
        if len(run_ids) > 1:
            axes.legend(fontsize=8)
    if title:
        figure.suptitle(title, fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    key = json.dumps([run_ids, names, title, smooth, log_y, time.time()]).encode()
    path = out_dir / f"metrics-{hashlib.sha1(key).hexdigest()[:12]}.png"
    figure.savefig(path, facecolor="white")
    return path, names
