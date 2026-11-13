"""Watch training runs for problems while they happen: NaN/inf, divergence, stalled learning, crashes.

The checks are local and free: they re-read a run's metrics only when its files changed and
reuse the same analysis the assistant's tools use. No model is called until the user presses
Ask AI on an alert, so watching never spends tokens. Each problem is reported once per run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from .analysis import summarize_series

INTERVAL_MS = 30_000
# Learning that levelled off is only worth a warning on a metric that should keep improving,
# and only once there is enough history to tell a plateau from noise.
FAILED = frozenset({"failed", "error", "interrupted"})
OBJECTIVE = re.compile(r"reward|return|acc|score|success|win_rate", re.IGNORECASE)
MIN_POINTS_FOR_STALL = 30


class MetricSourceLike(Protocol):
    def stamp(self) -> tuple: ...

    def load(self) -> dict[str, tuple[list[Any], list[Any]]]: ...


@dataclass(frozen=True)
class WatchTarget:
    run_id: str
    label: str
    status: str
    active: bool
    source: MetricSourceLike | None = None


@dataclass(frozen=True)
class Alert:
    run_id: str
    label: str
    kind: str  # "nonfinite" | "diverging" | "stalled" | "failed"
    metric: str
    message: str
    severity: str  # "error" | "warning"

    @property
    def key(self) -> tuple[str, str, str]:
        return self.run_id, self.kind, self.metric

    def prompt(self) -> str:
        """What Ask AI sends: a concrete question with the evidence the watcher found."""
        if self.kind == "failed":
            return f"Run {self.run_id} ({self.label}) just failed. Why, and how do I fix it?"
        ask = (
            "Is it worth letting it continue, and what should I change?"
            if self.kind == "stalled"
            else "What is going wrong, should I stop it, and what should I change?"
        )
        return f"While run {self.run_id} ({self.label}) is training: {self.message} {ask}"


def _step(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def inspect(run_id: str, label: str, series: dict[str, tuple[list[Any], list[Any]]]) -> list[Alert]:
    """Problems visible in one run's metrics right now, most serious first."""
    alerts = []
    for name, (steps, values) in sorted(series.items()):
        summary = summarize_series(name, steps, values)
        note = summary.get("note", "")
        if "first_nonfinite_step" in summary:
            alerts.append(
                Alert(
                    run_id,
                    label,
                    "nonfinite",
                    name,
                    f"{name} became NaN/inf at step {_step(summary['first_nonfinite_step'])} "
                    f"({summary['nonfinite_count']} bad values so far).",
                    "error",
                )
            )
        elif note.startswith("magnitude grew"):
            alerts.append(
                Alert(
                    run_id,
                    label,
                    "diverging",
                    name,
                    f"{name} is diverging: it went from {summary['first']:g} to {summary['last']:g}.",
                    "error",
                )
            )
        elif note.startswith("levelled off") and OBJECTIVE.search(name) and summary["points"] >= MIN_POINTS_FOR_STALL:
            alerts.append(
                Alert(
                    run_id,
                    label,
                    "stalled",
                    name,
                    f"{name} has stopped improving at about {summary['last']:g} "
                    f"(best {summary['max']:g}) for the second half of the run so far.",
                    "warning",
                )
            )
    return sorted(alerts, key=lambda a: a.severity != "error")


class RunWatcher:
    """Turns repeated polls into new alerts: unchanged runs are skipped and each problem is reported once."""

    def __init__(self) -> None:
        self.reported: set[tuple[str, str, str]] = set()
        self._stamps: dict[str, tuple] = {}
        self._was_active: set[str] = set()

    def check(self, targets: Iterable[WatchTarget]) -> list[Alert]:
        """Safe on any thread: targets carry snapshots, not live IDE objects."""
        found: list[Alert] = []
        active = set()
        for target in targets:
            if target.active and target.source is not None:
                active.add(target.run_id)
                try:
                    stamp = target.source.stamp()
                    if stamp == self._stamps.get(target.run_id):
                        continue
                    self._stamps[target.run_id] = stamp
                    found += inspect(target.run_id, target.label, target.source.load())
                except Exception:  # a half-written CSV: try again on the next poll
                    self._stamps.pop(target.run_id, None)
            elif target.run_id in self._was_active and target.status in FAILED:
                found.append(
                    Alert(
                        target.run_id,
                        target.label,
                        "failed",
                        "",
                        "The run failed.",
                        "error",
                    )
                )
        for gone in self._was_active - active:
            self._stamps.pop(gone, None)
        self._was_active = active
        fresh = [a for a in found if a.key not in self.reported]
        self.reported.update(a.key for a in fresh)
        return fresh
