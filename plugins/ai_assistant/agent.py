"""The tool-calling loop: ask the model, run the tools it calls, feed back results, repeat.

The loop is synchronous and Qt-free. The panel runs it on a worker thread and consumes the
events it yields; tests drive it with a scripted fake provider.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .providers.base import ProviderAdapter
from .providers.chat_types import (
    Completion,
    Message,
    StreamCancelled,
    TextDelta,
    ToolCall,
    Usage,
)
from .tools import ToolError, ToolRegistry, ToolResult

SYSTEM_PROMPT = """\
You are the AI assistant inside ThetaIDE, a desktop IDE for configuring, launching and monitoring \
machine-learning training runs (reinforcement, supervised and unsupervised learning).

You help the user understand and improve their runs. Ground every claim in the workspace data your \
tools return: run configs, pipeline logs, and logged metrics. Do not guess at values you can look up.

How to work:
- Why did a run fail: call get_run_logs first, then get_run_config. Quote the decisive error line and \
explain the cause in plain terms, then give a concrete fix (which config key to change, to what).
- Comparing runs: call compare_runs; relate the config differences to the metric differences.
- Suggesting hyperparameters: look at get_run_metrics (trend, NaN, divergence, plateau) and the config, \
then propose specific values with a one-line reason each. Prefer small, testable changes.
- Run ids: 'selected' means the run open in the Training Monitor; 'latest' the newest run.
- launch_experiment and stop_run start or stop compute. Only call them when the user asks for that action.
- Comparing variants (e.g. a change toggled on and off): make the change selectable by a config key, then \
launch the same experiment once per variant with launch_experiment(path, overrides, seeds, label), using the same \
seeds for every variant (three or more when runs are cheap enough). wait_for_runs until they finish, then \
compare_runs: report the mean and spread across seeds, and call a difference inconclusive when the spreads overlap.
- Writing files: write_file and edit_file show the user a diff to approve or reject. Before writing a config, \
read real examples so the schema is right: list_configs, then read_file on the components you reference and on \
an existing experiment if there is one. Do not invent keys.
- Experiment configs live in experiment/<group>/<name>.yaml. Their conventions:
  * first line `# @package _global_`, then a `defaults:` list such as `- /paradigms: supervised` (or online_rl, \
offline_rl, unsupervised) and `- _self_`; a group may share settings in experiment/<group>/_base.yaml, which \
experiments include with `- /experiment/<group>/_base`;
  * top-level keys from config.yaml (experiment_id, seed, total_timesteps, paradigm, env, ...);
  * `methods:` maps a method name to its settings; `model:` names a model config by its file name \
(model/basics/dnn.yaml is `model: dnn`), plus hyperparameters such as epochs or lr.
  Writing an experiment validates it with the real pipeline; if the result is invalid, fix the reported errors \
before telling the user it is ready. validate_experiment re-checks any experiment. Once it is valid, offer to \
select it (select_experiment).
- Missing components: if an experiment needs a method, model, environment or data module that list_configs does \
not show, search_hub for it and offer install_from_hub (the user approves it). After installing, read the new \
config before referencing it.
- Figures: plot_metrics draws metric curves and shows them in the chat; list_figures and show_figure show images \
a run already produced. Use them when the user asks to see or plot something, or when a curve explains your \
answer better than numbers. You cannot see images, so base what you say on data you read.
- If the user rejects a change, ask what they would like instead; do not retry the same change.
- Be concise. Use short Markdown: headings only when the answer has several parts, code spans for config keys.
"""

OVERVIEW = """
## Workspace configs
Config files under in/config by group (names without .yaml). Experiments reference components by these \
paths. Read a file before relying on its keys.
{overview}"""

STATE = """## Current IDE state (as of this message)
{snapshot}"""


@dataclass(frozen=True)
class ToolStarted:
    call: ToolCall
    needs_confirmation: bool


@dataclass(frozen=True)
class ToolFinished:
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class TurnFinished:
    """The model answered without asking for more tools; `usage` totals every step of the turn."""

    message: Message
    usage: Usage
    steps: int


AgentEvent = TextDelta | ToolStarted | ToolFinished | TurnFinished

DECLINED = "The user declined this action. Do not retry it; ask what they would like instead."

# Consecutive failed tool calls before nudging the model, and before taking its tools away.
NUDGE_AFTER_FAILURES = 3
STOP_AFTER_FAILURES = 5
NUDGE = (
    "Note: {n} tool calls in a row have failed. Stop trying variations of the same call. Either change approach "
    "based on what the errors say, or stop and explain the problem to the user."
)
WRAP_UP = """

## Wrap up now
{reason} Tools are unavailable for this reply. Answer the user directly: what you did, what worked, what failed \
and why (quote the key error), and what you need from them or suggest as the next step."""


class Agent:
    def __init__(
        self,
        provider: ProviderAdapter,
        tools: ToolRegistry,
        *,
        snapshot: Callable[[], dict[str, Any]] = dict,
        overview: Callable[[], str] = str,
        confirm: Callable[[ToolCall, str], bool] = lambda call, preview: False,
        execute: Callable[[ToolCall], ToolResult] | None = None,
        preview: Callable[[ToolCall], str] | None = None,
        max_steps: int = 8,
        history_chars: int = 120_000,
    ):
        self.provider = provider
        self.tools = tools
        self.snapshot = snapshot
        self.overview = overview
        self.confirm = confirm
        self.execute = execute or tools.execute
        self.preview = preview or tools.preview
        self.max_steps = max_steps
        self.history_chars = history_chars
        self.history: list[Message] = []

    def reset(self) -> None:
        self.history.clear()

    def system_prompt(self) -> tuple[str, str]:
        """(stable, per-message). The stable part (rules, tools' context, the workspace's configs) only
        changes when config files do, so providers can serve it from their prompt cache; the live IDE
        state goes last so it never invalidates that prefix."""
        try:
            overview = self.overview()
        except Exception as exc:
            overview = f"(unavailable: {exc})"
        try:
            state = json.dumps(self.snapshot(), indent=1, default=str)
        except Exception as exc:
            state = f"(unavailable: {exc})"
        stable = SYSTEM_PROMPT + (OVERVIEW.format(overview=overview) if overview else "")
        return stable, STATE.format(snapshot=state)

    def context_window(self) -> list[Message]:
        """The most recent history that fits the budget, starting at a user turn.

        Cutting anywhere else could separate a tool result from the call that asked for it,
        which every provider rejects.
        """
        total, start = 0, len(self.history)
        for i in range(len(self.history) - 1, -1, -1):
            m = self.history[i]
            total += len(m.content) + sum(len(json.dumps(c.arguments)) for c in m.tool_calls)
            if total > self.history_chars and start < len(self.history):
                break
            if m.role == "user":
                start = i
        return self.history[start:]

    def run(self, user_text: str, cancel: threading.Event | None = None) -> Iterator[AgentEvent]:
        """Answer one user message, yielding events as they happen.

        History is committed only for completed steps, so a cancelled or failed turn leaves
        the conversation in a state every provider will accept on the next message.
        """
        cancel = cancel or threading.Event()
        checkpoint = len(self.history)
        self.history.append(Message("user", user_text))
        system = self.system_prompt()
        specs = self.tools.specs()
        usage = Usage()
        failures = 0  # consecutive failed tool calls
        budget = self.max_steps  # steps that changed something or had a failure
        round_trips = 0
        wrap_up = ""
        try:
            while True:
                round_trips += 1
                completion: Completion | None = None
                prompt = (system[0], system[1] + WRAP_UP.format(reason=wrap_up)) if wrap_up else system
                for event in self.provider.stream(self.context_window(), specs, prompt, allow_tools=not wrap_up):
                    if cancel.is_set():
                        raise StreamCancelled("Stopped.")
                    if isinstance(event, TextDelta):
                        yield event
                    else:
                        completion = event
                if completion is None:
                    raise StreamCancelled("The response ended without completing.")
                usage += completion.usage
                message = completion.message
                if wrap_up and message.tool_calls:  # a model that ignores tool_choice=none: keep only its text
                    message = Message(
                        "assistant",
                        message.content or "I stopped here; see the errors above.",
                    )
                self.history.append(message)
                if not message.tool_calls:
                    yield TurnFinished(message, usage, round_trips)
                    return
                costly = False
                for call in message.tool_calls:
                    needs = self.tools.is_mutating(call.name)
                    yield ToolStarted(call, needs)
                    if cancel.is_set():
                        raise StreamCancelled("Stopped.")
                    result = self._run_tool(call, needs)
                    costly |= needs or result.is_error
                    if result.is_error and result.content != DECLINED:
                        failures += 1
                        if failures >= NUDGE_AFTER_FAILURES:
                            result = ToolResult(
                                result.content + "\n\n" + NUDGE.format(n=failures),
                                is_error=True,
                            )
                    elif not result.is_error:
                        failures = 0
                    self.history.append(
                        Message(
                            "tool",
                            result.content,
                            tool_call_id=call.id,
                            name=call.name,
                            is_error=result.is_error,
                        )
                    )
                    yield ToolFinished(call, result)
                budget -= costly
                if failures >= STOP_AFTER_FAILURES:
                    wrap_up = f"{failures} tool calls in a row have failed."
                elif budget <= 0 or round_trips >= self.max_steps * 3:
                    wrap_up = "You have used this message's tool budget."
        except BaseException:
            self._rollback(checkpoint)
            raise

    def _run_tool(self, call: ToolCall, needs_confirmation: bool) -> ToolResult:
        """Execute a call; one that changes anything runs only after the user approves its preview."""
        if needs_confirmation:
            try:
                preview = self.preview(call)
            except ToolError as exc:  # it would fail anyway: tell the model, don't bother the user
                return ToolResult(str(exc), is_error=True)
            if not self.confirm(call, preview):
                return ToolResult(DECLINED, is_error=True)
        return self.execute(call)

    def _rollback(self, checkpoint: int) -> None:
        """Drop a dangling assistant tool request (and anything after a failed turn's first step)."""
        while len(self.history) > checkpoint:
            last = self.history[-1]
            if last.role == "tool" or (last.role == "assistant" and last.tool_calls):
                self.history.pop()
                continue
            if last.role == "user" and len(self.history) == checkpoint + 1:
                self.history.pop()  # nothing useful happened; let the user resend
            break
