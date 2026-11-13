"""Run a scenario through the real Agent and tools, and grade what happened.

Only the IDE is faked (ScenarioBridge serves the scenario's workspace from memory); the agent
loop, tool registry, log and metric analysis, and provider adapter are the production code.
Actions are never carried out: an approval request is recorded and declined.
"""

from __future__ import annotations

import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent import Agent, TextDelta, ToolFinished, ToolStarted, TurnFinished
from ..providers.base import ProviderAdapter
from ..providers.chat_types import LLMError, ToolCall, Usage
from ..tools import ToolError, build_ide_tools
from .scenarios import Scenario

# The workspace's stock configs, as list_configs and read_file see them.
BASE_FILES = {
    "env/gym_classic_control/cartpole.yaml": (
        "# CartPole-v1: an episode ends after 500 steps, so 500 is the best possible return.\n"
        "id: CartPole-v1\nmax_episode_steps: 500\nreward_threshold: 475.0\n"
    ),
    "model/basics/mlp.yaml": "_target_: models.MLP\nhidden_sizes: [64, 64]\nactivation: tanh\n",
    "model/basics/dnn.yaml": "_target_: models.DNN\nhidden_sizes: [512, 256]\ndropout: 0.1\n",
    "paradigms/online_rl.yaml": "name: online_rl\nevaluator: episodic_return\neval_every: 2048\n",
    "paradigms/supervised.yaml": "name: supervised\nevaluator: classification\nval_split: 0.1\n",
}


class ScenarioBridge:
    """IDEBridge over a scenario's in-memory workspace."""

    def __init__(self, workspace: dict[str, Any]):
        self.workspace = workspace
        self.actions: list[tuple] = []
        self._files: dict[str, str] | None = None

    def snapshot(self) -> dict[str, Any]:
        runs = self.workspace["runs"]
        return {
            "current_experiment": f"experiment/demo/{runs[0]['experiment']}.yaml" if runs else None,
            "selected_run": self.workspace.get("selected"),
            "active_runs": [r["id"] for r in runs if r["status"] == "running"],
        }

    def runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.workspace["runs"]]

    def run_config(self, run_id: str) -> dict[str, Any]:
        return {"summary": self.workspace["configs"].get(run_id, {})}

    def run_logs(self, run_id: str) -> tuple[list[str], str]:
        lines = self.workspace["logs"].get(run_id, [])
        return list(lines), "" if lines else "No output was captured for this run."

    def run_metrics(self, run_id: str) -> dict[str, tuple[list[Any], list[Any]]]:
        return {k: (list(s), list(v)) for k, (s, v) in self.workspace["metrics"].get(run_id, {}).items()}

    # Charts drawn during an eval go to a scratch folder; scenario runs have no image files of their own.
    figures_dir = Path(tempfile.gettempdir()) / "thetaide-ai-eval-figures"

    def run_figures(self, run_id: str) -> list[dict[str, Any]]:
        return []

    def figure_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_file() and candidate.resolve().is_relative_to(self.figures_dir.resolve()):
            return candidate
        raise ToolError(f"No image at {path}. Call list_figures, or plot_metrics to draw one.")

    @property
    def files(self) -> dict[str, str]:
        """Every config the overview advertises, readable like the real workspace's in/config.

        Experiment files are the runs' configs; a scenario's own "files" entries override or add.
        """
        if self._files is None:
            import yaml

            files = dict(BASE_FILES)
            for r in self.workspace["runs"]:
                config = self.workspace["configs"].get(r["id"])
                if config is not None:
                    files.setdefault(
                        f"experiment/demo/{r['experiment']}.yaml",
                        yaml.safe_dump(config, sort_keys=False),
                    )
            files.update(self.workspace.get("files", {}))
            self._files = files
        return self._files

    @staticmethod
    def _normal(path: str) -> str:
        return path.strip().lstrip("./").removeprefix("in/config/")

    def experiments(self) -> list[str]:
        return sorted(p for p in self.files if p.startswith("experiment/"))

    def config_files(self, group: str = "") -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for path in sorted(self.files):
            if "/" in path:
                groups.setdefault(path.split("/", 1)[0], []).append(path)
        if group:
            if group not in groups:
                raise ToolError(f"No config group {group!r}. Groups: {', '.join(sorted(groups))}")
            return {group: groups[group]}
        return groups

    def overview(self) -> str:
        return "\n".join(
            f"{g}: {', '.join(f.split('/', 1)[1].removesuffix('.yaml') for f in fs)}"
            for g, fs in sorted(self.config_files().items())
        )

    def read_text(self, path: str) -> str:
        key = self._normal(path)
        for candidate in (key, key + ".yaml"):
            if candidate in self.files:
                return self.files[candidate]
        raise ToolError(f"No file {path!r} in this workspace. Call list_configs to see what exists.")

    def existing_text(self, path: str) -> str | None:
        return self.files.get(self._normal(path))

    def is_experiment(self, path: str) -> bool:
        return path.startswith("experiment/")

    def validate_experiment(self, path: str) -> dict[str, Any]:
        return {"experiment": path, "valid": True}

    def hub_search(
        self,
        query: str = "",
        kind: str = "",
        installed: bool | None = None,
        refresh: bool = False,
    ):
        return {"matches": 0, "components": []}

    # Actions: recorded only. The agent never gets this far without approval, and the harness declines.
    def _act(self, *action: Any) -> str:
        self.actions.append(action)
        return "Done."

    def open_file(self, path: str, line: int | None = None) -> str:
        return self._act("open", path, line)

    def select_experiment(self, path: str) -> str:
        return self._act("select", path)

    def launch_current(self) -> str:
        return self._act("launch")

    def stop_run(self, run_id: str) -> str:
        return self._act("stop", run_id)

    def write_text(self, path: str, content: str) -> str:
        return self._act("write", path)

    def hub_describe_install(self, component_id: str, version: str | None = None) -> str:
        return f"Install {component_id}"

    def hub_install(self, component_id: str, version: str | None = None) -> str:
        return self._act("install", component_id)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Outcome:
    scenario: str
    answer: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    tool_errors: int = 0
    approvals_requested: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    round_trips: int = 0
    seconds: float = 0.0
    error: str = ""
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.error and all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        """Fraction of checks passed; 0 when the run itself failed."""
        if self.error or not self.checks:
            return 0.0
        return sum(c.passed for c in self.checks) / len(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "score": round(self.score, 3),
            "error": self.error or None,
            "checks": [vars(c) for c in self.checks],
            "tools": [c.name for c in self.calls],
            "calls": [{"name": c.name, "arguments": c.arguments} for c in self.calls],
            "tool_errors": self.tool_errors,
            "approvals_requested": self.approvals_requested,
            "round_trips": self.round_trips,
            "usage": vars(self.usage),
            "seconds": round(self.seconds, 2),
            "answer": self.answer,
        }


def grade(scenario: Scenario, outcome: Outcome) -> list[Check]:
    called = {c.name for c in outcome.calls}
    answer = outcome.answer
    checks = []
    for group in scenario.must_call:
        hit = sorted(called & group)
        checks.append(
            Check(
                f"called {' or '.join(sorted(group))}",
                bool(hit),
                ", ".join(hit) or "not called",
            )
        )
    forbidden = sorted(called & scenario.must_not_call)
    checks.append(Check("took no unrequested action", not forbidden, ", ".join(forbidden)))
    for group in scenario.must_mention:
        hit = next((p for p in group if re.search(p, answer, re.IGNORECASE)), None)
        checks.append(Check(f"mentions {' | '.join(group)}", hit is not None, hit or "missing"))
    for pattern in scenario.must_not_mention:
        found = re.search(pattern, answer, re.IGNORECASE)
        checks.append(Check(f"avoids {pattern}", found is None, found.group(0) if found else ""))
    checks.append(Check("gave an answer", bool(answer.strip())))
    return checks


def run_scenario(provider: ProviderAdapter, scenario: Scenario, timeout: float = 300.0) -> Outcome:
    """One scenario, one fresh agent. Provider failures are recorded, not raised, so a suite keeps going."""
    bridge = ScenarioBridge(scenario.workspace)
    outcome = Outcome(scenario.id)

    def decline(call: ToolCall, preview: str) -> bool:
        outcome.approvals_requested.append(call.name)
        return False

    agent = Agent(
        provider,
        build_ide_tools(bridge),
        snapshot=bridge.snapshot,
        overview=bridge.overview,
        confirm=decline,
        max_steps=scenario.max_steps,
    )
    cancel = threading.Event()
    timer = threading.Timer(timeout, cancel.set)
    timer.daemon = True
    text: list[str] = []
    started = time.monotonic()
    timer.start()
    try:
        for event in agent.run(scenario.prompt, cancel):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, ToolStarted):
                outcome.calls.append(event.call)
            elif isinstance(event, ToolFinished):
                outcome.tool_errors += event.result.is_error
            elif isinstance(event, TurnFinished):
                outcome.usage, outcome.round_trips = event.usage, event.steps
                outcome.answer = event.message.content
    except Exception as exc:  # provider errors, and adapter bugs: recorded against this scenario
        timed_out = isinstance(exc, LLMError) and cancel.is_set()
        outcome.error = f"timed out after {timeout:.0f}s" if timed_out else f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
        outcome.seconds = time.monotonic() - started
    outcome.answer = outcome.answer or "".join(text)  # the final reply; partial text if the turn failed
    outcome.checks = grade(scenario, outcome)
    return outcome
