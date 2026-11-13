"""Turn raw run data into compact, model-friendly evidence.

A 50,000-row metrics.csv or a 20,000-line log does not fit in a prompt and would bury the
signal anyway. These functions reduce them to what diagnosis needs: the error and the lines
around it, per-metric trends and anomalies (NaN, divergence, plateaus), and config diffs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_ERROR_LINE = re.compile(
    r"(Traceback \(most recent call last\)|\b\w*(Error|Exception)\b:|CUDA out of memory|OutOfMemory|"
    r"\bKilled\b|Segmentation fault|\bnan\b loss|RuntimeError|AssertionError|FATAL|CRITICAL)",
    re.IGNORECASE,
)
_WARNING_LINE = re.compile(r"warn(ing)?\b", re.IGNORECASE)  # also UserWarning, DeprecationWarning


def extract_errors(lines: Sequence[str], context: int = 3, limit: int = 5) -> list[dict[str, Any]]:
    """The last `limit` error sites in a log, each with surrounding lines.

    A Python traceback is kept whole (up to 40 lines) because the final frame and the
    exception message are what explain a failure.
    """
    sites: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _ERROR_LINE.search(line):
            i += 1
            continue
        if "Traceback (most recent call last)" in line:
            end = i + 1
            while end < len(lines) and end - i < 40 and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
                end += 1
            end = min(len(lines), end + 1)  # the exception line itself
        else:
            end = min(len(lines), i + context + 1)
        start = max(0, i - context)
        sites.append({"line": i + 1, "excerpt": "\n".join(lines[start:end])})
        i = end
    return sites[-limit:]


def count_warnings(lines: Iterable[str]) -> int:
    return sum(1 for line in lines if _WARNING_LINE.search(line))


def downsample(points: Sequence[tuple[float, float]], n: int = 12) -> list[tuple[float, float]]:
    """At most `n` evenly spaced points, always including the first and last."""
    if len(points) <= n:
        return list(points)
    step = (len(points) - 1) / (n - 1)
    return [points[round(i * step)] for i in range(n)]


def summarize_series(name: str, steps: Sequence[Any], values: Sequence[Any]) -> dict[str, Any]:
    """Shape of one metric: range, trend, and the anomalies that usually explain a bad run."""
    pairs = [
        (float(s) if s is not None else float(i), float(v))
        for i, (s, v) in enumerate(zip(steps, values))
        if v is not None
    ]
    nonfinite = [s for s, v in pairs if not math.isfinite(v)]
    finite = [(s, v) for s, v in pairs if math.isfinite(v)]
    summary: dict[str, Any] = {"metric": name, "points": len(pairs)}
    if nonfinite:
        summary["first_nonfinite_step"] = nonfinite[0]
        summary["nonfinite_count"] = len(nonfinite)
    if not finite:
        return summary
    vals = [v for _s, v in finite]
    summary.update(
        first=round(vals[0], 6),
        last=round(vals[-1], 6),
        min=round(min(vals), 6),
        max=round(max(vals), 6),
        mean=round(sum(vals) / len(vals), 6),
    )
    if len(vals) >= 4:
        quarter = max(1, len(vals) // 4)
        head, tail = sum(vals[:quarter]) / quarter, sum(vals[-quarter:]) / quarter
        scale = max(abs(head), abs(tail), 1e-12)
        change = (tail - head) / scale
        summary["trend"] = "increasing" if change > 0.05 else "decreasing" if change < -0.05 else "flat"
        tail_vals = vals[-quarter:]
        spread = (max(tail_vals) - min(tail_vals)) / scale
        if summary["trend"] == "flat" and spread < 0.01:
            summary["note"] = "plateaued over the last quarter"
        elif summary["trend"] != "flat" and len(vals) >= 8:
            # Learned early, then stalled: the usual shape of a run limited by its hyperparameters.
            middle = vals[len(vals) // 2 : len(vals) // 2 + quarter]
            total = max(vals) - min(vals)
            if total > 0 and abs(tail - sum(middle) / len(middle)) / total < 0.02:
                summary["note"] = "levelled off: under 2% of its total change came in the second half"
        # Against the early level, not the first value: a reward or accuracy that starts at 0 is not divergence.
        early = max(sum(abs(v) for v in vals[:quarter]) / quarter, 1e-6)
        if max(abs(v) for v in vals) > 1e3 * early and abs(vals[-1]) > early * 100:
            summary["note"] = "magnitude grew by over 100x: possible divergence"
    summary["sampled"] = [[round(s, 3), round(v, 6)] for s, v in downsample(finite)]
    return summary


def flatten(config: Any, prefix: str = "") -> dict[str, Any]:
    """Nested config as dotted keys: {"method.lr": 3e-4, ...}."""
    if isinstance(config, Mapping):
        out: dict[str, Any] = {}
        for key, value in config.items():
            out.update(flatten(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): config} if prefix else {"": config}


def diff_configs(configs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Keys whose values differ between runs: {"method.lr": {"run_a": 3e-4, "run_b": 1e-3}}."""
    flat = {name: flatten(cfg or {}) for name, cfg in configs.items()}
    keys = sorted(set().union(*(f.keys() for f in flat.values()))) if flat else []
    diff = {}
    for key in keys:
        values = {name: f.get(key, "<unset>") for name, f in flat.items()}
        if len({repr(v) for v in values.values()}) > 1:
            diff[key] = values
    return diff


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 80] + f"\n… [truncated {len(text) - limit + 80} characters]"
