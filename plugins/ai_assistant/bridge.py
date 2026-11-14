"""IDEBridge over the running ThetaIDE window: run records, logs, metrics.csv files and actions.

Every method runs on the GUI thread (the panel marshals tool calls there), so it may touch
the window directly. File access is confined to the workspace.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from .figures import find_images, is_image
from .tools import ToolError
from .watch import WatchTarget

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 200_000
# Never written by the assistant, even inside the workspace: VCS, environments, IDE state.
PROTECTED_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".thetaide"}
MAX_METRIC_ROWS = 20_000
# How long a launch waits for the IDE to validate a newly selected experiment.
SELECT_TIMEOUT_SECONDS = 180


def read_metrics(path: Path) -> dict[str, tuple[list[Any], list[Any]]]:
    """Numeric columns of a Lightning metrics.csv against its step column.

    Unlike the plot viewer's reader this keeps NaN and inf: a loss that turned NaN is
    exactly what a failure diagnosis needs to see.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(row for _i, row in zip(range(MAX_METRIC_ROWS), csv.DictReader(stream)))
    if not rows:
        return {}
    position = next((c for c in ("transitions", "step", "epoch") if c in rows[0]), None)
    columns: dict[str, tuple[list[Any], list[Any]]] = {}
    for name in rows[0]:
        if name in ("transitions", "step", "epoch") or name is None:
            continue
        steps, values, numeric = [], [], False
        for i, row in enumerate(rows):
            text = (row.get(name) or "").strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                numeric = False
                break
            numeric = True
            pos = (row.get(position) or "").strip() if position else ""
            try:
                steps.append(float(pos) if pos else float(i))
            except ValueError:
                steps.append(float(i))
            values.append(value)
        if numeric and values:
            columns[name] = (steps, values)
    return columns


class MetricSource:
    """Reads one run's metrics: the newest metrics.csv per agent folder, else the run record's points."""

    def __init__(self, folder: Path | None, points: list[dict[str, Any]]):
        self.folder = folder
        self.points = points

    def files(self) -> list[tuple[str, Path]]:
        if self.folder is None or not self.folder.is_dir():
            return []
        agents = [p for p in sorted(self.folder.iterdir()) if p.is_dir()]
        found = []
        for agent in agents:
            versions = sorted(agent.glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
            if versions:
                found.append((f"{agent.name}:" if len(agents) > 1 else "", versions[-1]))
        return found

    def stamp(self) -> tuple:
        """Changes whenever the data does, so an unchanged run is not re-read."""
        stamps: list[Any] = []
        for _prefix, path in self.files():
            try:
                stat = path.stat()
                stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
        return (*stamps, len(self.points))

    def load(self) -> dict[str, tuple[list[Any], list[Any]]]:
        series: dict[str, tuple[list[Any], list[Any]]] = {}
        for prefix, path in self.files():
            try:
                for name, data in read_metrics(path).items():
                    series[prefix + name] = data
            except (OSError, csv.Error):
                continue
        if not series:  # simulated or not yet on disk: the run record's own points
            keys = {k for p in self.points for k in p if k != "step"}
            for key in sorted(keys):
                steps = [p.get("step") for p in self.points if p.get(key) is not None]
                values = [p.get(key) for p in self.points if p.get(key) is not None]
                if values and all(isinstance(v, (int, float)) for v in values):
                    series[key] = (steps, [float(v) for v in values])
        return series


def ide_feature(owner: Any, name: str, what: str) -> Any:
    """A window attribute this plugin relies on. Installed from the Hub, the plugin may meet an IDE
    build that renamed or lacks one; that should fail one tool with a clear reason, not crash."""
    feature = getattr(owner, name, None)
    if feature is None:
        raise ToolError(f"This ThetaIDE build cannot {what} (no window.{name}). Update ThetaIDE or do it by hand.")
    return feature


class WindowBridge:
    def __init__(self, window: Any, workspace: Path):
        from .hub import HubAccess

        self.window = window
        self.workspace = Path(workspace).resolve()
        self.hub = HubAccess(window)  # the panel points its on_gui/cancelled at the worker plumbing
        # Runs a callable on the GUI thread. Background tools (writes, validation, Hub) call the
        # bridge from the worker thread and use this for anything that touches widgets.
        self.on_gui = lambda fn: fn()
        self.compose = self._compose_with_backend
        # True when the assistant's user pressed Stop; long waits (launch, wait_for_runs) check it. The panel
        # points it at its worker; calls from outside agents never see it set.
        self.should_stop = lambda: False
        self.poll_seconds = 5.0  # how often wait_for_runs looks at run statuses
        # Charts the assistant draws; the panel points this at <data dir>/assistant/figures.
        self.figures_dir = self.workspace / ".thetaide" / "assistant" / "figures"

    # ── Reading ─────────────────────────────────────────────────────────────

    def _runs(self) -> list[dict]:
        return list(getattr(self.window, "runs", []) or [])

    def _run(self, run_id: str) -> dict:
        run = next((r for r in self._runs() if r.get("id") == run_id), None)
        if run is None:
            raise ToolError(f"No run with id {run_id!r}.")
        return run

    @staticmethod
    def summary(run: dict) -> dict[str, Any]:
        config = run.get("config") or {}
        backend = run.get("backend") or {}
        return {
            "id": run.get("id"),
            "experiment": backend.get("experiment_id") or config.get("name"),
            "group": backend.get("group"),
            "status": run.get("status"),
            "started": run.get("created"),
            "method": config.get("method_name") or config.get("agent"),
            "model": config.get("model") or None,
            "environment": config.get("env"),
            "seed": config.get("seed"),
            "simulated": bool(run.get("simulated")),
            "description": run.get("description") or None,
            "label": run.get("label") or None,
            "overrides": run.get("overrides") or None,
        }

    def snapshot(self) -> dict[str, Any]:
        selected = getattr(self.window, "selected", None) or {}
        active = getattr(self.window, "active_runs", []) or []
        runs = self._runs()
        return {
            "current_experiment": getattr(self.window, "current_experiment", None),
            "selected_run": selected.get("id"),
            "active_runs": [r.get("id") for r in active],
            "total_runs": len(runs),
            "recent_runs": [self.summary(r) for r in runs[:5]],
        }

    def runs(self) -> list[dict[str, Any]]:
        return [self.summary(r) for r in self._runs()]

    def _run_folder(self, run: dict) -> Path | None:
        folder = getattr(self.window, "run_folder", lambda _run: None)(run)
        return Path(folder) if folder and Path(folder).is_dir() else None

    def run_config(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        result: dict[str, Any] = {"summary": dict(run.get("config") or {})}
        if run.get("overrides") or run.get("label"):
            result["launch"] = {"label": run.get("label"), "overrides": list(run.get("overrides") or [])}
        folder = self._run_folder(run)
        if folder is not None:
            import yaml

            candidates = sorted(folder.glob("*.yaml")) or sorted(folder.glob("*/version_*/hparams.yaml"))
            for path in candidates[:2]:
                try:
                    result[path.relative_to(folder).as_posix()] = yaml.safe_load(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    result[path.name] = f"unreadable: {exc}"
        return result

    def run_logs(self, run_id: str) -> tuple[list[str], str]:
        run = self._run(run_id)
        getter = getattr(self.window, "log_lines_for_run", None)
        if getter is None:
            return [], "This IDE build cannot read run output."
        return getter(run)

    def run_metrics(self, run_id: str) -> dict[str, tuple[list[Any], list[Any]]]:
        series = self.metric_source(self._run(run_id)).load()
        if series:
            return series
        # A run with several seeds keeps each seed's metrics in its own folder: name them per seed.
        return {
            f"{seed}: {name}": data
            for seed, replicate in self.replicate_metrics(run_id).items()
            for name, data in replicate["metrics"].items()
        }

    def replicate_metrics(self, run_id: str) -> dict[str, dict[str, Any]]:
        """For a run launched with several seeds: '<method> · seed <n>' -> {status, metrics}. Empty otherwise.

        The pipeline records each seed in results/executions/<group>/<experiment id>/<execution>/manifest.json;
        the newest execution is the run's."""
        backend = self._run(run_id).get("backend") or {}
        logs_dir = getattr(getattr(self.window, "jobs", None), "logs_dir", None)
        group, experiment_id = backend.get("group"), backend.get("experiment_id")
        if not (logs_dir and group and experiment_id):
            return {}
        results = Path(logs_dir).parent  # <project>/results
        root = results / "executions" / str(group) / str(experiment_id)
        try:
            manifests = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
            manifest = json.loads(manifests[-1].read_text(encoding="utf-8")) if manifests else {}
        except (OSError, ValueError):
            return {}
        replicates: dict[str, dict[str, Any]] = {}
        for entry in manifest.get("runs") or []:
            if not isinstance(entry, dict):
                continue
            seed = f"{entry.get('method')} · seed {entry.get('training_seed')}"
            folder = (
                results.parent
                / str(entry.get("logs", ""))
                / str(entry.get("agent", ""))
                / str(entry.get("version", ""))
            )
            try:
                metrics = read_metrics(folder / "metrics.csv") if (folder / "metrics.csv").is_file() else {}
            except (OSError, csv.Error):
                metrics = {}
            replicates[seed] = {"status": entry.get("status"), "metrics": metrics}
        return replicates

    def metric_source(self, run: dict) -> MetricSource:
        """A snapshot of where a run's metrics live, safe to read from any thread."""
        return MetricSource(self._run_folder(run), [dict(p) for p in run.get("metrics") or []])

    def watch_targets(self, limit: int = 10) -> list[WatchTarget]:
        """Training runs and the most recent others, for the run watcher. GUI thread only."""
        active = {id(r) for r in getattr(self.window, "active_runs", []) or []}
        runs = self._runs()
        chosen = [r for r in runs if id(r) in active] + [r for r in runs[:limit] if id(r) not in active]
        return [
            WatchTarget(
                run_id=str(r.get("id")),
                label=self.summary(r).get("experiment") or str(r.get("id")),
                status=str(r.get("status") or ""),
                active=id(r) in active,
                source=self.metric_source(r) if id(r) in active else None,
            )
            for r in chosen
        ]

    def _config_root(self) -> Path:
        paths = getattr(self.window, "workspace_paths", None)
        if paths is not None:
            return Path(paths.config)
        try:  # older IDE builds without WorkspacePaths
            from frontend.config_tree import find_config_root
        except ImportError:
            return self.workspace / "in" / "config"
        return Path(find_config_root())

    def experiments(self) -> list[str]:
        root = self._config_root()
        folder = next(
            (root / n for n in ("experiment", "experiments") if (root / n).is_dir()),
            None,
        )
        if folder is None:
            return []
        return [
            p.relative_to(root).as_posix()
            for p in sorted(folder.rglob("*.y*ml"))
            if not p.name.startswith("_") and not any(part.startswith(".") for part in p.parts)
        ]

    def _resolve(self, path: str) -> Path:
        """A workspace path. Relative paths that start with a config group (experiment/, model/, ...)
        are inside the config directory; other relative paths are relative to the workspace root."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            root = self._config_root()
            first = candidate.parts[0] if candidate.parts else ""
            in_config = first not in ("", ".", "..") and ((root / first).is_dir() or (root / candidate).is_file())
            candidate = (root if in_config else self.workspace) / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ToolError(f"{path} is outside the workspace ({self.workspace}); only workspace files are allowed.")
        return resolved

    def _inside_workspace(self, path: str) -> Path:
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise ToolError(f"No file at {path}.")
        return resolved

    def config_files(self, group: str = "") -> dict[str, list[str]]:
        root = self._config_root()
        groups: dict[str, list[str]] = {}
        for path in sorted(root.rglob("*.y*ml")):
            rel = path.relative_to(root)
            if any(part.startswith(".") for part in rel.parts) or rel.parts[0] == "hydra":
                continue
            key = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            if not group or key == group:
                groups.setdefault(key, []).append(rel.as_posix())
        if group and not groups:
            raise ToolError(
                f"No config group {group!r}. Groups: {', '.join(sorted(p.name for p in root.iterdir() if p.is_dir()))}"
            )
        return groups

    def overview(self, limit: int = 4000) -> str:
        """One line per config group: `env: gym_classic_control/cartpole, ...`. Goes in the cached part
        of the system prompt, so the model knows what exists without spending tool calls to find out."""
        lines = []
        for group, files in sorted(self.config_files().items()):
            names = [Path(f).with_suffix("").as_posix().removeprefix(f"{group}/") for f in files]
            lines.append(f"{group}: {', '.join(names)}")
        text = "\n".join(lines)
        return text if len(text) <= limit else text[:limit] + "\n… (call list_configs for the rest)"

    def _writable(self, path: str) -> Path:
        resolved = self._resolve(path)
        relative = resolved.relative_to(self.workspace)
        if PROTECTED_PARTS.intersection(relative.parts):
            raise ToolError(
                f"{path} is in a protected folder ({', '.join(sorted(PROTECTED_PARTS))}); it cannot be written."
            )
        if resolved.is_dir():
            raise ToolError(f"{path} is a folder.")
        return resolved

    def existing_text(self, path: str) -> str | None:
        resolved = self._writable(path)
        return self.read_text(str(resolved)) if resolved.exists() else None

    def write_text(self, path: str, content: str) -> str:
        resolved = self._writable(path)
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ToolError(f"The content is over {MAX_WRITE_BYTES} bytes.")
        if resolved.suffix in (".yaml", ".yml"):
            import yaml

            try:
                yaml.safe_load(content)
            except yaml.YAMLError as exc:
                raise ToolError(f"Not written: the YAML does not parse ({exc}).") from None
        existed = resolved.exists()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")
        self._refresh_tree(resolved)
        verb = "Updated" if existed else "Created"
        return f"{verb} {resolved.relative_to(self.workspace).as_posix()} ({content.count(chr(10)) + 1} lines)."

    def _refresh_tree(self, written: Path) -> None:
        """Show a new or changed config in the experiment tree."""
        root = self._config_root().resolve()
        tree = getattr(self.window, "config_tree", None)
        if tree is None or not written.is_relative_to(root):
            return
        try:
            folder = written.parent.relative_to(root).as_posix()
            self.on_gui(lambda: tree.populate(ensure_expanded=folder))
        except Exception:
            pass  # the file is written; the tree refreshes on its next scan

    def read_text(self, path: str) -> str:
        resolved = self._inside_workspace(path)
        if resolved.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"{path} is {resolved.stat().st_size} bytes; only files under {MAX_READ_BYTES} are read.")
        data = resolved.read_bytes()
        if b"\0" in data[:4096]:
            raise ToolError(f"{path} is a binary file.")
        return data.decode("utf-8", errors="replace")

    def run_figures(self, run_id: str) -> list[dict[str, Any]]:
        run = self._run(run_id)
        folders = [f for f in [self._run_folder(run)] if f is not None]
        backend = run.get("backend") or {}
        if backend.get("group") and backend.get("experiment_id"):
            # Plotters write to results/plots/<group>/<result id>; the result id starts with the experiment id.
            plots = self.workspace / "results" / "plots" / str(backend["group"])
            if plots.is_dir():
                folders += sorted(plots.glob(f"{backend['experiment_id']}*"))
        return find_images(folders, self.workspace)

    def figure_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        figures = Path(self.figures_dir).resolve()
        if candidate.is_absolute() and candidate.resolve().is_relative_to(figures):
            resolved = candidate.resolve()  # the assistant's own charts may live outside the workspace
        else:
            resolved = self._resolve(path)
        if not resolved.is_file():
            raise ToolError(f"No file at {path}.")
        if not is_image(resolved):
            raise ToolError(f"{path} is not an image (expected PNG, JPG, GIF, BMP, SVG or WebP).")
        return resolved

    # ── Validation (called on the worker thread) ────────────────────────────

    def _experiment_name(self, path: str) -> str | None:
        """The Hydra name of an experiment config (group/name), or None for any other file."""
        resolved = self._resolve(path)
        folder = (self._config_root() / "experiment").resolve()
        if resolved.suffix not in (".yaml", ".yml") or not resolved.is_relative_to(folder):
            return None
        if resolved.name.startswith("_"):
            return None  # shared _base files are included by experiments, not run on their own
        return resolved.relative_to(folder).with_suffix("").as_posix()

    def is_experiment(self, path: str) -> bool:
        return self._experiment_name(path) is not None

    def validate_experiment(self, path: str) -> dict[str, Any]:
        name = self._experiment_name(path)
        if name is None:
            raise ToolError(f"{path} is not an experiment config (experiment/<group>/<name>.yaml, not a _base file).")
        if not self._resolve(path).is_file():
            raise ToolError(f"No experiment at {path}.")
        return summarize_report(self.compose(name))

    def _compose_with_backend(self, name: str, overrides: list[str] | None = None) -> dict[str, Any]:
        """Compose through the window's backend: its single worker keeps Hydra calls from overlapping
        the IDE's own validation. The callback arrives on the GUI thread; this thread waits for it."""
        backend = getattr(self.window, "backend", None)
        if backend is None or not hasattr(backend, "compose"):
            raise ToolError("Validation needs the training pipeline, which is not available in this session.")
        done, out = threading.Event(), {}

        def finished(data, error):
            out.update(data=data, error=error)
            done.set()

        self.on_gui(lambda: backend.compose(name, list(overrides or []), finished))
        if not done.wait(180):
            raise ToolError(f"Validating {name} timed out after 3 minutes.")
        if out["error"] or not isinstance(out["data"], dict):
            raise ToolError(f"The training pipeline could not run for {name}: {out['error'] or 'no result'}")
        return out["data"]

    # ── Component Hub (called on the worker thread) ─────────────────────────

    def hub_search(
        self,
        query: str = "",
        kind: str = "",
        installed: bool | None = None,
        refresh: bool = False,
    ) -> dict:
        return self.hub.search(query, kind, installed, refresh)

    def hub_describe_install(self, component_id: str, version: str | None = None) -> str:
        return self.hub.describe_install(component_id, version)

    def hub_install(self, component_id: str, version: str | None = None) -> str:
        return self.hub.install(component_id, version)

    # ── Acting ──────────────────────────────────────────────────────────────

    def open_file(self, path: str, line: int | None = None) -> str:
        resolved = self._inside_workspace(path)
        service = getattr(self.window, "editor_service", None)
        if service is None or not service.open_file(resolved, line=line):
            raise ToolError(f"The editor could not open {path}.")
        return f"Opened {resolved.relative_to(self.workspace)}" + (f" at line {line}." if line else ".")

    def select_experiment(self, path: str) -> str:
        root = self._config_root()
        resolved = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not resolved.is_file() or not resolved.is_relative_to(root.resolve()):
            raise ToolError(f"{path} is not an experiment config. Call list_experiments for valid paths.")
        rel = resolved.relative_to(root.resolve()).as_posix()
        ide_feature(self.window, "on_config_file_opened", "select experiments")(resolved, rel)
        return (
            f"Selected {rel}. The IDE validates it in the background; call launch_experiment once the user "
            "wants it run (it reports if validation has not finished)."
        )

    def launch_current(self) -> str:
        window = self.window
        name = getattr(window, "current_experiment", None)
        if not name:
            raise ToolError("No experiment is selected. Use select_experiment first.")
        missing = window.check_missing_components() if hasattr(window, "check_missing_components") else []
        if missing:
            raise ToolError(
                f"Cannot launch {name}: missing components {', '.join(missing)} (install via Component Hub)."
            )
        if not getattr(window, "compose_valid", False):
            reason = getattr(window, "compose_error", None) or "validation has not finished yet; try again shortly"
            raise ToolError(f"Cannot launch {name}: {reason}.")
        ide_feature(window, "launch_training", "launch runs")()
        return f"Queued {name}. It starts when a worker slot is free; watch it in the Training Monitor."

    @staticmethod
    def launch_overrides(overrides: list[str] | None, seeds: list[int] | None) -> list[str]:
        """The command-line overrides for a launch: the caller's, then the seeds as a variability panel."""
        result = [str(o).strip() for o in overrides or [] if str(o).strip()]
        for item in result:
            if item.lstrip("+").startswith("experiment_id="):
                raise ToolError("Do not override experiment_id; give the variant a label instead.")
        if seeds:
            if not all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2**32 for s in seeds):
                raise ToolError("seeds must be non-negative integers.")
            if len(set(seeds)) != len(seeds):
                raise ToolError("seeds must not repeat.")
            if len(seeds) > 1:
                result.append(f"++variability.training_seeds=[{','.join(str(s) for s in seeds)}]")
            else:
                result.append(f"++seed={seeds[0]}")
        return result

    def describe_launch(
        self, path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> str:
        """What the user approves: the experiment, its variant, and that it composes. Raises ToolError (so the
        user is never asked) when the launch could not work. Called on the worker thread."""
        name = self._launch_target(path)
        extra = self.launch_overrides(overrides, seeds)
        if (extra or label) and not hasattr(self.window, "queue_current"):
            raise ToolError("This ThetaIDE build cannot launch with overrides, seeds or a label. Update ThetaIDE.")
        report = summarize_report(self.compose(name, extra))
        if not report["valid"]:
            raise ToolError(f"{name} would not run with these settings:\n" + "\n".join(report.get("errors", [])))
        lines = [f"Experiment: {name}"]
        if label:
            lines.append(f"Variant: {label}")
        lines += [f"Override: {o}" for o in extra]
        lines.append("✓ Validated by the training pipeline with these settings.")
        return "\n".join(lines)

    def _launch_target(self, path: str) -> str:
        if not path:
            name = self.on_gui(lambda: getattr(self.window, "current_experiment", None))
            if not name:
                raise ToolError("No experiment is selected. Pass the experiment's path.")
            return name
        # Accept a config path (experiment/g/x.yaml, g/x.yaml) or a Hydra name (g/x).
        bare = path.strip().removeprefix("experiment/")
        bare = bare if bare.endswith((".yaml", ".yml")) else f"{bare}.yaml"
        name = self._experiment_name(path) if path.endswith((".yaml", ".yml")) else None
        name = name or self._experiment_name(f"experiment/{bare}")
        if name is None or not (self._config_root() / "experiment" / f"{name}.yaml").is_file():
            raise ToolError(f"{path} is not an experiment config. Call list_experiments for valid paths.")
        return name

    def launch(
        self, path: str = "", overrides: list[str] | None = None, seeds: list[int] | None = None, label: str = ""
    ) -> dict[str, Any] | str:
        """Queue an experiment, optionally as a labelled variant with extra overrides and seeds, and start the
        queue. Called on the worker thread: selecting an experiment means waiting for the IDE to validate it."""
        name = self._launch_target(path)
        extra = self.launch_overrides(overrides, seeds)
        window = self.window
        if not hasattr(window, "queue_current"):
            if extra or label:
                raise ToolError("This ThetaIDE build cannot launch with overrides, seeds or a label. Update ThetaIDE.")
            if path:
                self.on_gui(lambda: self.select_experiment(f"experiment/{name}.yaml"))
                self._await_selection(name)
            return self.on_gui(self.launch_current)
        if self.on_gui(lambda: getattr(window, "current_experiment", None)) != name:
            self.on_gui(lambda: self.select_experiment(f"experiment/{name}.yaml"))
        self._await_selection(name)
        try:
            from src.app.services.config_service import LaunchError
        except ImportError:  # only outside ThetaIDE (tests): queue_current raises it, so the IDE has it
            LaunchError = RuntimeError  # noqa: N806

        def queue() -> dict:
            try:
                run = window.queue_current(start=True, extra_overrides=extra, label=label, strict=True)
            except LaunchError as exc:
                raise ToolError(f"Cannot launch {name}: {exc}") from exc
            if run is None:
                raise ToolError(f"Cannot launch {name}; see the IDE's console.")
            return run

        run = self.on_gui(queue)
        backend = run.get("backend") or {}
        return {
            "run_id": run.get("id"),
            "experiment": name,
            "results": f"{backend.get('group')}/{backend.get('experiment_id')}",
            "label": label or None,
            "overrides": extra,
            "status": run.get("status"),
            "next": "Call wait_for_runs with this run_id (and any others) to wait for the results.",
        }

    def _await_selection(self, name: str) -> None:
        """Wait until the IDE has validated `name` as the selected experiment."""
        window = self.window
        deadline = time.monotonic() + SELECT_TIMEOUT_SECONDS

        def state() -> tuple:
            timer = getattr(window, "compose_timer", None)
            pending = bool(timer is not None and timer.isActive())
            return (
                getattr(window, "current_experiment", None),
                pending,
                bool(getattr(window, "compose_valid", False)),
                getattr(window, "compose_error", None),
            )

        while True:
            current, pending, valid, error = self.on_gui(state)
            if current == name and not pending and (valid or error):
                if not valid:
                    raise ToolError(f"Cannot launch {name}: {error}")
                return
            if self.should_stop():
                raise ToolError("Stopped before the experiment was launched.")
            if time.monotonic() > deadline:
                raise ToolError(f"ThetaIDE did not finish validating {name} within {SELECT_TIMEOUT_SECONDS} s.")
            time.sleep(0.2)

    def stop_run(self, run_id: str) -> str:
        run = self._run(run_id)
        if run.get("status") not in ("starting", "running", "queued"):
            raise ToolError(f"Run {run_id} is not running (status: {run.get('status')}).")
        if run.get("simulated"):
            ide_feature(self.window, "finish", "stop demo runs")("stopped", run=run)
            return f"Stopped demo run {run_id}."
        job_id = (run.get("backend") or {}).get("job_id")
        if not job_id:
            raise ToolError(f"Run {run_id} was started outside ThetaIDE and cannot be stopped from here.")
        jobs = ide_feature(self.window, "jobs", "stop runs")
        ide_feature(jobs, "cancel", "stop runs")(job_id)
        return f"Cancelling run {run_id}; its processes are being terminated."


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """The parts of a compose report the model needs to fix a config: errors first, then what resolved."""
    import json

    out: dict[str, Any] = {
        "experiment": report.get("experiment"),
        "valid": bool(report.get("valid")),
    }
    errors = report.get("errors") or []
    if errors:
        out["errors"] = [f"[{e.get('stage')}] {str(e.get('message', '')).strip()[:1500]}" for e in errors]
    if report.get("notices"):
        out["notices"] = list(report["notices"])[:10]
    config = report.get("config") or {}
    env = config.get("env")
    resolved = {k: config.get(k) for k in ("experiment_id", "paradigm", "seed", "total_timesteps") if k in config}
    if env is not None:
        resolved["env"] = env.get("name", env) if isinstance(env, dict) else env
    if resolved:
        out["resolved"] = resolved
    methods = {name: (plan or {}).get("settings", plan) for name, plan in (report.get("methods") or {}).items()}
    if methods:
        text = json.dumps(methods, default=str)
        out["methods"] = methods if len(text) <= 3000 else text[:3000] + "…"
    return out
