# Generated from plugins/ai_assistant/tools.py by scripts/sync_mcp_tools.py. Do not edit.
# ruff: noqa: I001
"""Tools the model can call, and the registry that validates and runs them.

Tools reach the IDE only through `IDEBridge`, a narrow protocol. The Qt implementation
lives in bridge.py; tests use an in-memory fake, so every tool is testable without a window.
"""

from __future__ import annotations

import difflib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .analysis import (
    count_warnings,
    diff_configs,
    extract_errors,
    summarize_series,
    truncate,
)
from .figures import PlotError, plot_metrics
from .chat_types import ToolCall, ToolSpec

MAX_RESULT_CHARS = 12_000
# Run statuses that are not finished yet.
PENDING_STATUSES = ("queued", "starting", "running")
# wait_for_runs: default and longest wait, in minutes. A call that times out can simply be repeated.
WAIT_DEFAULT_MINUTES = 10
WAIT_MAX_MINUTES = 60


class IDEBridge(Protocol):
    """Everything the assistant may read from or do to the IDE."""

    def snapshot(self) -> dict[str, Any]:
        """{'current_experiment', 'selected_run', 'active_runs': [...]}"""
        ...

    def runs(self) -> list[dict[str, Any]]:
        """Run summaries, newest first."""
        ...

    def run_config(self, run_id: str) -> dict[str, Any]: ...

    def run_logs(self, run_id: str) -> tuple[list[str], str]:
        """(pipeline output lines, explanation when there are none)"""
        ...

    def run_metrics(self, run_id: str) -> dict[str, tuple[list[Any], list[Any]]]:
        """metric name -> (steps, values). NaN and inf are kept: they are evidence."""
        ...

    def experiments(self) -> list[str]: ...

    def read_text(self, path: str) -> str: ...

    def open_file(self, path: str, line: int | None = None) -> str: ...

    def select_experiment(self, path: str) -> str: ...

    def describe_launch(
        self, path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> str:
        """What the user approves; ToolError when the launch could not work (e.g. it does not validate)."""
        ...

    def launch(
        self, path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> dict[str, Any] | str:
        """Queue the experiment at `path` (the selected one if empty) and start the queue."""
        ...

    def replicate_metrics(self, run_id: str) -> dict[str, dict[str, Any]]:
        """For a run with several seeds: '<method> · seed <n>' -> {status, metrics}. Empty otherwise."""
        ...

    # True once the user asked to stop; long waits check it.
    should_stop: Callable[[], bool]
    poll_seconds: float

    def stop_run(self, run_id: str) -> str: ...

    def config_files(self, group: str = "") -> dict[str, list[str]]:
        """Config YAML files by top-level group (experiment, model, env, agent, paradigms, ...)."""
        ...

    def existing_text(self, path: str) -> str | None:
        """Current contents of a writable workspace file, or None if it does not exist yet."""
        ...

    def write_text(self, path: str, content: str) -> str: ...

    def is_experiment(self, path: str) -> bool: ...

    def validate_experiment(self, path: str) -> dict[str, Any]:
        """Compose and validate exactly as run_pipeline.py would, without running it."""
        ...

    def hub_search(
        self,
        query: str = "",
        kind: str = "",
        installed: bool | None = None,
        refresh: bool = False,
    ) -> dict: ...

    def hub_describe_install(self, component_id: str, version: str | None = None) -> str: ...

    def hub_install(self, component_id: str, version: str | None = None) -> str: ...

    # Where the assistant saves the charts it draws.
    figures_dir: Path

    def run_figures(self, run_id: str) -> list[dict[str, Any]]:
        """Image files a run produced (its results folder and results/plots), newest first."""
        ...

    def figure_path(self, path: str) -> Path:
        """An existing image file in the workspace (or figures_dir); ToolError otherwise."""
        ...


class ToolError(Exception):
    """A failure the model should see and can correct (unknown run id, bad path)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    # Changes IDE state (starts or stops compute, writes files): needs the user's confirmation.
    mutating: bool = False
    # What the user approves: called with the same arguments; raises ToolError if the call cannot work.
    preview: Callable[..., str] | None = None
    # Waits on the network or long work: run on the worker thread, not the GUI thread.
    background: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    # Images shown to the user under the tool line, as (absolute path, caption). The model sees only content.
    figures: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Shown:
    """What a tool handler returns to also show images: the model gets `value`, the user sees `figures`."""

    value: Any
    figures: tuple[tuple[str, str], ...]


def _schema(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self, *, include_mutating: bool = True) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values() if include_mutating or not t.mutating]

    def runs_in_background(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.background)

    def is_mutating(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.mutating)

    def _arguments(self, tool: Tool, call: ToolCall) -> dict[str, Any]:
        if "_unparsed_arguments" in call.arguments:
            raise ToolError("Arguments were not valid JSON; send a JSON object.")
        missing = [p for p in tool.parameters.get("required", []) if p not in call.arguments]
        if missing:
            raise ToolError(f"Missing required argument(s): {', '.join(missing)}")
        allowed = set(tool.parameters.get("properties", {}))
        return {k: v for k, v in call.arguments.items() if k in allowed}

    def preview(self, call: ToolCall) -> str:
        """What the user is asked to approve. Raises ToolError when the call would fail anyway,
        so the user is never asked to approve something that cannot happen."""
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"Unknown tool {call.name!r}.")
        arguments = self._arguments(tool, call)
        if tool.preview is None:
            return f"{call.name}({json.dumps(arguments)})"
        return tool.preview(**arguments)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run a call. Every failure becomes an error result the model can read and recover from."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                f"Unknown tool {call.name!r}. Available: {', '.join(self._tools)}",
                is_error=True,
            )
        try:
            arguments = self._arguments(tool, call)
            value = tool.handler(**arguments)
            figures: tuple[tuple[str, str], ...] = ()
            if isinstance(value, Shown):
                value, figures = value.value, value.figures
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:  # a tool bug must not end the conversation
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)
        text = value if isinstance(value, str) else json.dumps(value, indent=1, default=str)
        return ToolResult(truncate(text, MAX_RESULT_CHARS), figures=figures)


# ── The ThetaIDE tool set ────────────────────────────────────────────────────


RUN_ID = {
    "type": "string",
    "description": "A run id from list_runs, or 'selected' (the run open in the Training Monitor) or 'latest'.",
}


def build_ide_tools(bridge: IDEBridge) -> ToolRegistry:
    registry = ToolRegistry()

    def resolve(run_id: str) -> str:
        ref = (run_id or "").strip()
        if ref in ("", "selected", "current"):
            ref = bridge.snapshot().get("selected_run") or "latest"
        if ref == "latest":
            runs = bridge.runs()
            if not runs:
                raise ToolError("There are no runs in this workspace yet.")
            return runs[0]["id"]
        if not any(r["id"] == ref for r in bridge.runs()):
            raise ToolError(f"No run with id {ref!r}. Call list_runs to see valid ids.")
        return ref

    def list_runs(limit: int = 20, status: str = "") -> dict:
        runs = bridge.runs()
        if status:
            runs = [r for r in runs if r.get("status") == status]
        return {"total": len(runs), "runs": runs[: max(1, min(int(limit), 100))]}

    def get_run_config(run_id: str = "selected") -> dict:
        rid = resolve(run_id)
        return {"run_id": rid, "config": bridge.run_config(rid)}

    def get_run_logs(run_id: str = "selected", tail: int = 120) -> dict:
        rid = resolve(run_id)
        lines, note = bridge.run_logs(rid)
        if not lines:
            return {
                "run_id": rid,
                "lines": 0,
                "note": note or "No output was captured.",
            }
        tail = max(10, min(int(tail), 400))
        return {
            "run_id": rid,
            "lines": len(lines),
            "warnings": count_warnings(lines),
            "errors": extract_errors(lines),
            "tail": "\n".join(lines[-tail:]),
        }

    def get_run_metrics(run_id: str = "selected", metrics: list[str] | None = None) -> dict:
        rid = resolve(run_id)
        series = bridge.run_metrics(rid)
        if not series:
            return {"run_id": rid, "note": "No metrics were recorded for this run."}
        names = [m for m in series if not metrics or any(f in m for f in metrics)]
        return {
            "run_id": rid,
            "available": sorted(series),
            "summaries": [summarize_series(n, *series[n]) for n in names[:30]],
        }

    def compare_runs(run_ids: list[str]) -> dict:
        if not isinstance(run_ids, list) or len(run_ids) < 2:
            raise ToolError("Pass at least two run ids.")
        ids = [resolve(r) for r in run_ids[:6]]
        configs = {rid: bridge.run_config(rid) for rid in ids}
        finals: dict[str, dict[str, Any]] = {}
        for rid in ids:
            for name, (steps, values) in bridge.run_metrics(rid).items():
                s = summarize_series(name, steps, values)
                if "last" in s:
                    finals.setdefault(name, {})[rid] = {
                        "last": s["last"],
                        "max": s["max"],
                        "min": s["min"],
                    }
        shared = {k: v for k, v in finals.items() if len(v) > 1}
        summaries = {r["id"]: r for r in bridge.runs() if r["id"] in ids}
        result: dict[str, Any] = {
            "runs": summaries,
            "config_differences": diff_configs(configs),
            "metrics": shared,
        }
        seeds = {rid: across_seeds(bridge.replicate_metrics(rid)) for rid in ids}
        by_metric: dict[str, dict[str, Any]] = {}
        for rid, stats in seeds.items():
            for name, stat in stats.items():
                by_metric.setdefault(name, {})[rid] = stat
        if by_metric:
            result["across_seeds"] = {k: v for k, v in sorted(by_metric.items()) if len(v) > 1} or by_metric
            result["note"] = (
                "across_seeds: each seed's final value, then their mean and sample std. With few seeds, treat a "
                "difference smaller than about two standard deviations as inconclusive."
            )
        return result

    def list_experiments() -> dict:
        return {
            "current": bridge.snapshot().get("current_experiment"),
            "experiments": bridge.experiments(),
        }

    def read_file(path: str) -> str:
        return bridge.read_text(path)

    def open_file(path: str, line: int | None = None) -> str:
        return bridge.open_file(path, line)

    def select_experiment(path: str) -> str:
        return bridge.select_experiment(path)

    def launch_experiment(
        path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> Any:
        return bridge.launch(path, as_list(overrides, "overrides"), as_list(seeds, "seeds"), str(label or ""))

    def launch_preview(
        path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> str:
        return bridge.describe_launch(path, as_list(overrides, "overrides"), as_list(seeds, "seeds"), str(label or ""))

    def wait_for_runs(run_ids: list[str], timeout_minutes: float = WAIT_DEFAULT_MINUTES) -> dict:
        if not isinstance(run_ids, list) or not run_ids:
            raise ToolError("Pass the run ids to wait for (from launch_experiment or list_runs).")
        ids = list(dict.fromkeys(resolve(r) for r in run_ids[:12]))
        try:
            minutes = float(timeout_minutes)
        except (TypeError, ValueError):
            minutes = WAIT_DEFAULT_MINUTES
        minutes = max(0.0, min(minutes, WAIT_MAX_MINUTES))
        start = time.monotonic()
        while True:
            status = {r["id"]: r for r in bridge.runs() if r["id"] in ids}
            pending = [rid for rid in ids if (status.get(rid) or {}).get("status") in PENDING_STATUSES]
            waited = time.monotonic() - start
            if not pending or waited >= minutes * 60 or bridge.should_stop():
                break
            time.sleep(min(bridge.poll_seconds, max(0.0, minutes * 60 - waited)))
        runs = []
        for rid in ids:
            run = status.get(rid) or {}
            entry: dict[str, Any] = {"id": rid, "status": run.get("status"), "label": run.get("label")}
            if run.get("status") not in PENDING_STATUSES:
                entry["final"] = final_values(bridge.run_metrics(rid))
            runs.append(entry)
        result: dict[str, Any] = {"finished": not pending, "waited_seconds": round(waited), "runs": runs}
        if pending:
            result["still_running"] = pending
            result["note"] = "Not finished yet: call wait_for_runs again to keep waiting."
        failed = [r["id"] for r in runs if r["status"] not in (*PENDING_STATUSES, "completed")]
        if failed:
            result["did_not_complete"] = failed
            result["hint"] = "get_run_logs explains a failed run."
        if len(ids) > 1 and not pending:
            result["next"] = "compare_runs compares them (across seeds, when they had several)."
        return result

    def stop_run(run_id: str = "selected") -> str:
        return bridge.stop_run(resolve(run_id))

    def list_configs(group: str = "") -> dict:
        return bridge.config_files(group)

    def list_figures(run_id: str = "selected") -> dict:
        rid = resolve(run_id)
        figures = bridge.run_figures(rid)
        if not figures:
            return {"run_id": rid, "figures": [], "note": "This run has no image files. plot_metrics can draw one."}
        return {"run_id": rid, "figures": figures}

    def show_figure(path: str, caption: str = "") -> Shown:
        image = bridge.figure_path(path)
        return Shown(
            f"Shown to the user in the chat: {path}. You cannot see the image; describe it only from data you "
            "have read (get_run_metrics), not by guessing.",
            ((str(image), caption or image.name),),
        )

    def plot_metrics_tool(
        run_ids: list[str] | None = None,
        metrics: list[str] | None = None,
        title: str = "",
        smooth: int = 0,
        log_y: bool = False,
    ) -> Shown:
        ids = [resolve(r) for r in (run_ids or ["selected"])[:6]]
        series = {rid: bridge.run_metrics(rid) for rid in ids}
        if not any(series.values()):
            raise ToolError(f"No metrics were recorded for {', '.join(ids)}.")
        try:
            path, panels = plot_metrics(
                series,
                metrics or [],
                bridge.figures_dir,
                title=title,
                smooth=max(0, min(int(smooth or 0), 500)),
                log_y=bool(log_y),
            )
        except PlotError as exc:
            raise ToolError(str(exc)) from exc
        caption = title or f"{', '.join(panels)} · {', '.join(ids)}"
        drawn = [name for names in panels.values() for name in names]
        finals = {
            name: {rid: summarize_series(name, *series[rid][name]).get("last") for rid in ids if name in series[rid]}
            for name in drawn
        }
        return Shown(
            {
                "shown_to_user": path.name,
                "panels": {panel: names for panel, names in panels.items()},
                "metrics": drawn,
                "runs": ids,
                "final_values": finals,
                "note": "The user now sees this chart. You cannot see it: call get_run_metrics for trends.",
            },
            ((str(path), caption),),
        )

    def replaced(path: str, old_text: str, new_text: str) -> tuple[str, str]:
        current = bridge.existing_text(path)
        if current is None:
            raise ToolError(f"{path} does not exist. Use write_file to create it.")
        count = current.count(old_text) if old_text else 0
        if count != 1:
            raise ToolError(
                f"old_text must appear exactly once in {path}; it appears {count} times. "
                "Read the file and include more surrounding lines."
            )
        return current, current.replace(old_text, new_text)

    def write_preview(path: str, content: str) -> str:
        return file_diff(path, bridge.existing_text(path), content)

    def edit_preview(path: str, old_text: str, new_text: str) -> str:
        return file_diff(path, *replaced(path, old_text, new_text))

    def search_hub(
        query: str = "",
        kind: str = "",
        installed: bool | None = None,
        refresh: bool = False,
    ) -> dict:
        return bridge.hub_search(query, kind, installed, refresh)

    def install_preview(component_id: str, version: str | None = None) -> str:
        return bridge.hub_describe_install(component_id, version)

    def install_from_hub(component_id: str, version: str | None = None) -> str:
        return bridge.hub_install(component_id, version)

    def checked(path: str, message: str) -> Any:
        """A write's result; for an experiment config, with the pipeline's verdict on it."""
        if not bridge.is_experiment(path):
            return message
        try:
            report = bridge.validate_experiment(path)
        except ToolError as exc:
            return f"{message}\nCould not validate it: {exc}"
        hint = "" if report["valid"] else " Fix the errors with edit_file; the result is validated again."
        return {"result": message + hint, "validation": report}

    def write_file(path: str, content: str) -> Any:
        return checked(path, bridge.write_text(path, content))

    def edit_file(path: str, old_text: str, new_text: str) -> Any:
        return checked(path, bridge.write_text(path, replaced(path, old_text, new_text)[1]))

    def validate_experiment(path: str) -> dict:
        return bridge.validate_experiment(path)

    registry.register(
        Tool(
            "list_runs",
            "List training runs in the workspace, newest first, with status, experiment, method, "
            "environment, seed and start time.",
            _schema(
                {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum runs to return (default 20).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Only runs with this status, e.g. 'failed'.",
                    },
                }
            ),
            list_runs,
        )
    )
    registry.register(
        Tool(
            "get_run_config",
            "The full configuration a run was launched with: method, model, hyperparameters, environment.",
            _schema({"run_id": RUN_ID}),
            get_run_config,
        )
    )
    registry.register(
        Tool(
            "get_run_logs",
            "A run's pipeline output: detected errors and tracebacks with context, a warning count, and the "
            "last lines. Use first when asked why a run failed.",
            _schema(
                {
                    "run_id": RUN_ID,
                    "tail": {
                        "type": "integer",
                        "description": "Lines from the end (default 120).",
                    },
                }
            ),
            get_run_logs,
        )
    )
    registry.register(
        Tool(
            "get_run_metrics",
            "Summaries of a run's logged metrics (loss, reward, lr, ...): first/last/min/max, trend, NaN/inf, "
            "divergence and plateau notes, and a downsampled curve.",
            _schema(
                {
                    "run_id": RUN_ID,
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Substrings to filter metric names, e.g. ['loss', 'reward']. Omit for all.",
                    },
                }
            ),
            get_run_metrics,
        )
    )
    registry.register(
        Tool(
            "compare_runs",
            "Compare 2-6 runs: which config values differ and how their final metrics compare.",
            _schema({"run_ids": {"type": "array", "items": {"type": "string"}}}, ["run_ids"]),
            compare_runs,
        )
    )
    registry.register(
        Tool(
            "list_experiments",
            "Experiment config files that can be launched, and which one is selected.",
            _schema(),
            list_experiments,
        )
    )
    registry.register(
        Tool(
            "read_file",
            "Read a text file inside the workspace (experiment YAML, component source, a log).",
            _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative or absolute path.",
                    }
                },
                ["path"],
            ),
            read_file,
        )
    )
    registry.register(
        Tool(
            "open_file",
            "Open a workspace file in the user's editor, optionally at a line.",
            _schema({"path": {"type": "string"}, "line": {"type": "integer"}}, ["path"]),
            open_file,
        )
    )
    registry.register(
        Tool(
            "select_experiment",
            "Select an experiment config in the experiment tree so it can be launched.",
            _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "A path from list_experiments.",
                    }
                },
                ["path"],
            ),
            select_experiment,
        )
    )
    registry.register(
        Tool(
            "list_configs",
            "Config YAML files by group: experiment, model, env, agent, paradigms, data, ... Use it to find the "
            "components an experiment can reference (e.g. model/basics/dnn.yaml is the model 'dnn').",
            _schema(
                {
                    "group": {
                        "type": "string",
                        "description": "Only this group, e.g. 'model'. Omit for all.",
                    }
                }
            ),
            list_configs,
        )
    )
    registry.register(
        Tool(
            "write_file",
            "Create a workspace file, or replace one entirely. Paths starting with a config group "
            "(experiment/, model/, env/, ...) are inside in/config. The user sees a diff and must approve it. "
            "An experiment config is validated by the training pipeline right after it is written.",
            _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "e.g. experiment/dnn_study/baseline.yaml",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new file contents.",
                    },
                },
                ["path", "content"],
            ),
            write_file,
            mutating=True,
            preview=write_preview,
            background=True,
        )
    )
    registry.register(
        Tool(
            "edit_file",
            "Replace one exact, unique snippet in an existing workspace file. Prefer this to write_file for "
            "small changes. The user sees a diff and must approve it.",
            _schema(
                {
                    "path": {"type": "string"},
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace; must occur once.",
                    },
                    "new_text": {"type": "string"},
                },
                ["path", "old_text", "new_text"],
            ),
            edit_file,
            mutating=True,
            preview=edit_preview,
            background=True,
        )
    )
    registry.register(
        Tool(
            "validate_experiment",
            "Compose and validate an experiment config exactly as run_pipeline.py would, without running it: "
            "returns valid/errors, notices, the resolved paradigm/env/seed, and each method's effective settings.",
            _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "e.g. experiment/dnn/baseline.yaml",
                    }
                },
                ["path"],
            ),
            validate_experiment,
            background=True,
        )
    )
    registry.register(
        Tool(
            "search_hub",
            "Search the Theta Component Hub registry for installable components: methods/agents, models, "
            "environments, data modules, evaluators, plotters, experiments and suites. Shows which are installed.",
            _schema(
                {
                    "query": {
                        "type": "string",
                        "description": "Words to match in id, name, description, tags.",
                    },
                    "kind": {
                        "type": "string",
                        "description": "e.g. method, model, env, data, experiment, suite.",
                    },
                    "installed": {
                        "type": "boolean",
                        "description": "true: only installed; false: only not installed.",
                    },
                    "refresh": {
                        "type": "boolean",
                        "description": "Re-download the registry index first.",
                    },
                }
            ),
            search_hub,
            background=True,
        )
    )
    registry.register(
        Tool(
            "install_from_hub",
            "Download and install a Component Hub component (with its component and pip dependencies), or update "
            "an installed one. The user sees what will be installed and must approve.",
            _schema(
                {
                    "component_id": {
                        "type": "string",
                        "description": "An id from search_hub.",
                    },
                    "version": {
                        "type": "string",
                        "description": "A specific release; omit for the latest.",
                    },
                },
                ["component_id"],
            ),
            install_from_hub,
            mutating=True,
            preview=install_preview,
            background=True,
        )
    )
    registry.register(
        Tool(
            "list_figures",
            "Image files (plots, figures) a run produced, newest first. Show one with show_figure.",
            _schema({"run_id": RUN_ID}),
            list_figures,
        )
    )
    registry.register(
        Tool(
            "show_figure",
            "Show an image file from the workspace (a PNG/JPG/SVG plot) to the user inline in the chat. "
            "You do not see the image yourself.",
            _schema(
                {
                    "path": {"type": "string", "description": "Workspace-relative path, e.g. from list_figures."},
                    "caption": {"type": "string", "description": "One line shown under the image."},
                },
                ["path"],
            ),
            show_figure,
        )
    )
    registry.register(
        Tool(
            "plot_metrics",
            "Draw logged metrics as line charts and show the chart to the user inline in the chat: one panel "
            "per metric, one line per run, and one line per method when a run trained several (it logs "
            "'<method>:<metric>', e.g. 'ppo:train/reward'; pass 'train/reward' to overlay every method). "
            "Use it when the user asks to see, plot or compare curves.",
            _schema(
                {
                    "run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-6 run ids (or 'selected'/'latest'); default the selected run.",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metric names or substrings (e.g. 'loss', 'reward'); default the first six.",
                    },
                    "title": {"type": "string"},
                    "smooth": {"type": "integer", "description": "Moving-average window in points; 0 = raw."},
                    "log_y": {"type": "boolean", "description": "Symmetric-log y axis for values spanning decades."},
                }
            ),
            plot_metrics_tool,
        )
    )
    registry.register(
        Tool(
            "launch_experiment",
            "Queue and run an experiment; the queue starts it when a worker is free. Starts compute: the user is "
            "asked to confirm. To compare variants (e.g. a change toggled on and off), launch the same experiment "
            "once per variant with a label and the overrides that select it, and the same seeds for each; then "
            "wait_for_runs and compare_runs. The settings are validated before the user is asked.",
            _schema(
                {
                    "path": {
                        "type": "string",
                        "description": "The experiment config (experiment/<group>/<name>.yaml or <group>/<name>). "
                        "Omit for the experiment selected in the IDE.",
                    },
                    "overrides": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Hydra command-line overrides for this run only, e.g. "
                        "['++methods.ppo.use_new_loss=true', 'total_timesteps=20000']. The files are unchanged.",
                    },
                    "seeds": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Training seeds. Several run as one job with a run per seed, so results "
                        "can be compared across seeds (e.g. [1, 2, 3]).",
                    },
                    "label": {
                        "type": "string",
                        "description": "A short name for this variant, e.g. 'edit-on'. It gets its own results "
                        "folder, so variants never overwrite each other or earlier results.",
                    },
                }
            ),
            launch_experiment,
            mutating=True,
            preview=launch_preview,
            background=True,  # waits for the IDE to validate the experiment
        )
    )
    registry.register(
        Tool(
            "wait_for_runs",
            "Wait until runs finish (or fail), then report each one's status and final metric values. Returns "
            f"early when all are done; otherwise after timeout_minutes (default {WAIT_DEFAULT_MINUTES}, at most "
            f"{WAIT_MAX_MINUTES}): call it again to keep waiting.",
            _schema(
                {
                    "run_ids": {"type": "array", "items": {"type": "string"}},
                    "timeout_minutes": {"type": "number", "description": "How long to wait at most."},
                },
                ["run_ids"],
            ),
            wait_for_runs,
            background=True,
        )
    )
    registry.register(
        Tool(
            "stop_run",
            "Stop a running training run. The user is asked to confirm.",
            _schema({"run_id": RUN_ID}),
            stop_run,
            mutating=True,
        )
    )
    return registry


def as_list(value: Any, name: str) -> list:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ToolError(f"{name} must be a list.")
    return value


def last_finite(values: list[Any]) -> float | None:
    for value in reversed(values):
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def final_values(series: dict[str, tuple[list[Any], list[Any]]], limit: int = 25) -> dict[str, float | None]:
    """Each metric's last finite value: what a finished run ended at."""
    return {name: last_finite(values) for name, (_steps, values) in sorted(series.items())[:limit]}


def across_seeds(replicates: dict[str, dict[str, Any]], limit: int = 25) -> dict[str, dict[str, Any]]:
    """metric -> {n, mean, std, values per seed} from each seed's final value; only with two or more seeds."""
    if len(replicates) < 2:
        return {}
    per_metric: dict[str, dict[str, float]] = {}
    for seed, replicate in replicates.items():
        for name, (_steps, values) in (replicate.get("metrics") or {}).items():
            value = last_finite(values)
            if value is not None:
                per_metric.setdefault(name, {})[seed] = value
    out: dict[str, dict[str, Any]] = {}
    for name, by_seed in sorted(per_metric.items())[:limit]:
        values = list(by_seed.values())
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        out[name] = {
            "n": len(values),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "per_seed": {seed: round(v, 6) for seed, v in by_seed.items()},
        }
    return out


def file_diff(path: str, old: str | None, new: str) -> str:
    """A unified diff of a proposed write; a new file is shown as all additions."""
    if old == new:
        raise ToolError(f"{path} already has exactly this content; nothing to change.")
    lines = difflib.unified_diff(
        (old or "").splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="/dev/null" if old is None else f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)
