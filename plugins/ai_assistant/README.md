# AI Assistant

A [Theta Hub](https://github.com/ThetaIDE/ThetaHub) plugin for ThetaIDE. Install it from **Component Hub → Plugins → AI Assistant**; the Hub also installs its one Python dependency, `keyring`.

The **AI Assistant** adds an **AI** tab to the sidebar. It is a chat with a large language model that can read this workspace's runs: why a run failed, how two runs differ, and what to change next. With your approval it can also launch and stop runs.

## Choose a provider

Pick a provider and model at the top of the panel:

| Provider | Key | Notes |
| --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` or keychain | **Server URL** also accepts any OpenAI-compatible server (vLLM, LM Studio). |
| Anthropic | `ANTHROPIC_API_KEY` or keychain | |
| Google Gemini | `GEMINI_API_KEY` or keychain | |
| OpenRouter | `OPENROUTER_API_KEY` or keychain | One key for models from many vendors. Model ids look like `anthropic/claude-sonnet-4.5`; press **↻** for the full list. |
| Ollama (local) | none | Runs on your machine at `http://localhost:11434`. No data leaves the computer. |

The model box accepts any model id. Press **↻** to list the models your key can use.

### API keys

When the chosen provider has no key, the panel asks for one right under the model picker: paste it and press **Enter** (or **Save**). A **Get a key ↗** link opens the provider's key page. To change or remove a key later, use the **⋯** menu, which also shows where the current key comes from. You can also use **Settings → Plugins → AI Assistant**.

Keys go into the OS keychain through Python's `keyring` package: macOS Keychain, Windows Credential Locker, or the Secret Service (GNOME Keyring or KWallet) on Linux. They are never written to `settings.toml`, the plugin's JSON storage or any other file, and the UI shows only the last four characters.

Window managers such as i3 or sway often have KWallet or GNOME Keyring installed but not running. The plugin starts it over D-Bus when it needs it, so saving works without extra setup. The first save may show the keychain's own prompt to create or unlock a wallet.

If there is no keychain at all, the plugin offers to use the key **for this session only**: it stays in memory until ThetaIDE closes and is never written to disk. To avoid entering it each time, set the provider's environment variable from the table above.

### What leaves your computer

The first time you send a message to a cloud provider, a notice lists what that provider will receive: your messages, the run configs, log excerpts, metric summaries and file contents the assistant reads, and the names of your config files. Nothing is sent unless you press **Send**. If you cancel, your message stays in the box. The notice appears once for each provider and server. Ollama, and any server on `localhost`, never shows it, because nothing leaves the machine. For OpenRouter, the notice also explains that it forwards each request to the company serving the chosen model, whose terms apply too.

### AI can make mistakes

A line under the message box says so. The assistant's diagnoses, comparisons and suggested hyperparameters can be wrong, and its file changes can be too. Check important answers against the logs and metrics, and read each diff before you approve it.

## Chats

Press **☰** to show your saved chats in a list beside the conversation. Click a chat to reopen it. Right-click (or press **Delete**) to delete one. **＋** starts a new chat. Chats are saved in `.thetaide/assistant/chats` in the workspace.

The **Why did it fail?**, **Compare last two** and **Tune it** chips under the conversation send a ready-made question about the selected run.

## What it can see and do

| Tool | What it does |
| --- | --- |
| `list_runs` | Runs in this workspace, newest first, with status, method, environment and seed |
| `get_run_config` | The configuration a run was launched with, including the saved YAML in its results folder |
| `get_run_logs` | Errors and full tracebacks from the pipeline output, a warning count, and the last lines |
| `get_run_metrics` | Each metric in `metrics.csv` reduced to its range, trend, NaN/inf, and notes on divergence, plateaus, or learning that levelled off, plus a downsampled curve |
| `compare_runs` | Which config values differ between 2–6 runs, and how their final metrics compare |
| `list_experiments`, `list_configs`, `read_file` | Experiment configs, component configs (models, environments, agents, paradigms) and other text files. Access is limited to the workspace. |
| `write_file`, `edit_file` | Create or change a workspace file, such as a new experiment config. **You approve or reject each change in a dialog that shows its diff.** YAML must parse before anything is saved, and `.git`, `.venv` and `.thetaide` are never written. |
| `plot_metrics` | Draw logged metrics as line charts, one panel per metric and one line per run (up to six), with optional smoothing and a log axis. **The chart appears in the chat.** |
| `list_figures`, `show_figure` | Find the images a run produced (its results folder and `results/plots`) and show one in the chat. Only image files inside the workspace can be shown. |
| `open_file`, `select_experiment` | Open a file in your editor; select an experiment in the tree |
| `validate_experiment` | Compose and validate an experiment exactly as `run_pipeline.py` would, without running it: the errors, the resolved paradigm, environment and seed, and each method's effective settings. Writing an experiment file runs this automatically, so the assistant fixes errors before it calls the file ready. |
| `search_hub` | Search the Theta Component Hub for methods, models, environments, data modules, evaluators and suites, and see which are installed |
| `install_from_hub` | Download and install (or update) a Hub component with its component and pip dependencies, through the same installer as the Hub UI. **You approve each install.** The dialog shows the version, source repository and pip packages. |
| `launch_experiment`, `stop_run` | Start or stop compute. **Each one asks for your confirmation first.** |

Paths that start with a config group (`experiment/`, `model/`, …) refer to `in/config`. Other relative paths are relative to the workspace root. The **Launch and stop runs without asking** setting never covers file changes or Hub installs.

### Figures in the chat

Ask to *plot the reward of my last two runs* or *show the confusion matrix* and the image appears in the chat, scaled to the panel. Click it, or **open**, to see it full size in your image viewer. Charts the assistant draws are saved under `.thetaide/assistant/figures` and are deleted with their chat. The model does not see the images; it describes curves from the metric data it reads. Charts are drawn with matplotlib, which ThetaIDE already includes.

A run id may be `selected` (the run open in the Training Monitor) or `latest`. **Ask AI about this run** in the Training Monitor footer, and the launcher command **AI: Explain the selected run**, both diagnose the selected run.

**Staying unstuck.** Settings set the **action steps per message** limit (default 8). Only steps that change something or contain a failed call count toward it; steps that only read successfully are free, up to three times the limit. After three tool failures in a row, the assistant is told to stop retrying variations. After five, or when the budget runs out, it gets one final reply with tools disabled. In that reply it must explain what it did, what failed and why, and what it needs from you. Each adapter supports this through `allow_tools=False` (OpenAI and Anthropic `tool_choice`, Gemini `toolConfig`; Ollama omits the tools).

## Chats and tool activity

Each code block in a reply has a **⧉ Copy** link under it, for pasting a suggested config. Each tool call shows as a line in the chat. Click it to see the exact arguments the model sent and the result it got back. Click again to hide them.

Chats are saved after every reply to `.thetaide/assistant/chats/` in the workspace, one JSON file per chat, written atomically. The last chat reopens when the IDE starts. The **☰** chat list shows them. The 100 most recent chats are kept. Chats hold run data from your workspace but never API keys. A saved chat can be continued with any provider, since history is stored in the provider-neutral format.

## Cost and prompt caching

The assistant's instructions start with a compact list of every config in `in/config` (`env: gym_classic_control/cartpole, …`), so the model knows what exists without spending tool calls to find out. The instructions are sent in two parts. The stable part holds the rules and that config list, and changes only when config files do. The live IDE state comes last, so it never invalidates the stable part.

- **Anthropic:** cache breakpoints go on the tools and stable instructions, and on the conversation so far. Each step of a tool loop re-reads the earlier steps from cache.
- **OpenAI and Gemini:** these cache repeated prompt prefixes automatically, and the stable-first order maximises hits.

The status line reports tokens and an estimated cost, for example `12,400 in (11,800 cached) / 350 out tokens · 2 steps · ≈ $0.0081 (chat ≈ $0.042)`. The chat total is saved with the chat.

The estimate uses list prices from `pricing.py`, checked in November 2026. Cached input is charged at the cached rate. Cache writes count as ordinary input, so the first message of a chat runs slightly low. Models missing from the table get no estimate rather than a guess, and Ollama is free. OpenRouter reports what each request cost, so its figure is used instead of the table. Prices change, so you can override or add them in the plugin's `prices` setting (US dollars per million tokens: input, cached input, output), for example `{"my-finetune": [3.0, 0.3, 12.0]}`. This is an estimate; your provider's billing page is the actual cost.

## Watching training runs

While runs train, the assistant checks their metrics every 30 seconds and flags:

- **NaN or inf** in any metric, with the step where it first appeared
- **Divergence:** a metric that grew more than 100× over its early level
- **Stalled learning:** a reward, return, accuracy or score that has levelled off for the second half of the run so far (only after 30 points of history; a flat loss is not flagged)
- **Crashes** of runs it was watching

Each problem appears once, as a row above the chat and as an IDE toast (a system notification when ThetaIDE is in the background). **Ask AI** opens the panel and asks about that run with the evidence filled in: what is going wrong, whether to stop it, and what to change. If a reply is already streaming, the question waits in the message box instead.

Watching is free: it reads `metrics.csv` locally, re-reads a file only when it changed, and does the reading off the GUI thread. No model is called until you press **Ask AI**. Turn it off in **Settings → Plugins → AI Assistant**.

## Measuring answer quality

Unit tests prove the plumbing works. Evals measure whether the answers are any good. Each scenario is a small workspace held as data (runs, configs, logs, metrics), plus a question and checks that need no human judgement:

| Scenario | The model must |
| --- | --- |
| `diagnose_oom` | read the logs, name the CUDA out-of-memory error and point at `batch_size` |
| `diagnose_nan_divergence` | find the NaN or divergence in the metrics and tie it to the learning rate |
| `compare_two_runs` | find that only `n_steps` differs and say which value did better |
| `tune_plateau` | notice that reward levelled off and propose concrete changes |
| `unknown_run_no_hallucination` | say that run `r99` does not exist instead of inventing a cause |
| `explain_without_acting` | report on a live run without stopping or relaunching it |

Every scenario fails if the model launches, stops, writes or installs anything it was not asked to. The harness declines every approval request, so nothing is ever carried out. The agent loop, tools, analysis and provider adapter are the production code; only the IDE is replaced by the scenario's data.

Run these from a `theta-plugins` checkout with the IDE's Python (`../ThetaIDE/.venv/bin/python`):

```bash
python -m plugins.ai_assistant.evals                      # every provider with a key
python -m plugins.ai_assistant.evals -p anthropic -p openai --repeat 3
python -m plugins.ai_assistant.evals -p ollama -m qwen3 --json results.json
```

The scorecard lists, per provider and model, how many scenarios passed, tokens in and out (with cached), time taken, the tools each scenario used, and every failed check. `--json` also saves the full answers. Models are not deterministic, so use `--repeat` before comparing pass rates. The command exits non-zero if anything failed. The CI tests check the grader itself with scripted good and bad models, so they need no key.

To add a scenario, append a `Scenario` to `evals/scenarios.py` and an ideal answer to `GOOD_ANSWERS` in `test_ai_assistant.py`. The tests check that every required tool works on the scenario's data and that the ideal answer passes.

## Design

The source is `plugins/ai_assistant/` in [theta-plugins](https://github.com/ThetaIDE/theta-plugins):

- `providers/`: one `ProviderAdapter` interface, with an adapter for each provider. Each adapter maps the shared `Message`, `ToolSpec` and `ToolCall` types to its vendor's wire format, and parses that vendor's streaming format (SSE or NDJSON) back into `TextDelta` and `Completion` events. The adapters use the standard library's HTTP client, not vendor SDKs. A `Transport` protocol lets tests replace the network.
- `providers/transport.py`: maps HTTP failures to typed errors (`AuthenticationError`, `RateLimitError`, `ProviderUnavailableError`, `BadRequestError`). Rate limits and outages are retried with exponential backoff and jitter, and a server's `Retry-After` is honoured. Only opening a stream is retried. Once text has reached the user, a failure is reported instead of replayed, because a replay would duplicate output.
- `evals/`: the answer-quality scenarios, the harness that grades them, and the command-line runner.
- `agent.py`: the tool-calling loop. It is synchronous and has no Qt. It yields events and commits history only for completed steps, so a cancelled or failed turn never leaves a tool call without its result.
- `tools.py` and `analysis.py`: the tool set, built on a narrow `IDEBridge` protocol. Logs and metrics are reduced to evidence before they reach the model.
- `watch.py`: the run watcher. It is Qt-free; the panel snapshots runs on the GUI thread and checks them on a background thread.
- `bridge.py`, `panel.py`, `settings_widget.py`: the Qt side. The agent runs on a `QThread`, and tool calls and confirmation dialogs are sent back to the GUI thread.

Tests in `test_ai_assistant.py` (not shipped in the archive) replay each provider's real stream format through a fake transport. They also cover retries, keychain handling, every tool, and the agent loop, including the same tool-calling conversation through all four adapters. No API key or network access is needed.

## Developing and releasing

```bash
# from theta-plugins, with the IDE's Python
QT_QPA_PLATFORM=offscreen ../ThetaIDE/.venv/bin/python -m pytest plugins/ai_assistant -q
../ThetaIDE/.venv/bin/python scripts/package_plugins.py --only ai_assistant
```

The package uses only ThetaIDE's plugin API (`Plugin`, `PluginContext`) and a few `window` attributes. A missing attribute fails the one tool that needs it with a clear message, so an older or newer IDE degrades instead of breaking. Publishing follows ThetaIDE's `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`.
