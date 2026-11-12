"""AI Assistant plugin: provider adapters, retries, key storage, tools and the agent loop.

No test touches the network or a real keychain. Providers get a FakeTransport that replays
recorded-shape streams, so CI needs no API keys.

Run from the theta-plugins checkout with the IDE's Python (its editable install provides
`frontend` and `src.app`):

    QT_QPA_PLATFORM=offscreen ../ThetaIDE/.venv/bin/python -m pytest plugins/ai_assistant -q
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

try:  # Qt WebEngine (the IDE's terminal) must load before any QApplication exists
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass

PLUGINS_REPO = Path(__file__).resolve().parents[2]  # theta-plugins: makes `plugins.ai_assistant` importable
if str(PLUGINS_REPO) not in sys.path:
    sys.path.insert(0, str(PLUGINS_REPO))

from plugins.ai_assistant.agent import (
    DECLINED,
    Agent,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)
from plugins.ai_assistant.analysis import diff_configs, extract_errors, summarize_series
from plugins.ai_assistant.bridge import read_metrics
from plugins.ai_assistant.keys import KeyStore, KeyStoreError, mask
from plugins.ai_assistant.providers import (
    PROVIDERS,
    AuthenticationError,
    BadRequestError,
    Completion,
    Message,
    ProviderConfig,
    ProviderUnavailableError,
    RateLimitError,
    StreamCancelled,
    TextDelta,
    ToolCall,
    ToolSpec,
    Usage,
    create_provider,
)
from plugins.ai_assistant.providers.gemini import gemini_schema
from plugins.ai_assistant.providers.transport import (
    RetryPolicy,
    call_with_retries,
    error_for_status,
    iter_sse,
)
from plugins.ai_assistant.tools import Tool, ToolRegistry, build_ide_tools

# ── Fakes ────────────────────────────────────────────────────────────────────


class FakeTransport:
    """Replays one scripted response per request: a list of lines, or an exception to raise."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def post_stream(self, url, headers, payload, timeout):
        self.requests.append({"url": url, "headers": headers, "payload": json.loads(json.dumps(payload))})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return iter(response)

    def get_json(self, url, headers, timeout):
        self.requests.append({"url": url, "headers": headers})
        return self.responses.pop(0)


def sse(*events: dict, done: bool = False, named: bool = False) -> list[str]:
    lines = []
    for event in events:
        if named:
            lines.append(f"event: {event['type']}")
        lines += [f"data: {json.dumps(event)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


# Per-provider streams in each API's real wire shape: (a tool call to get_run_logs) and (a text answer).


def openai_tool_call():
    return sse(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "get_run_logs", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"run_id": '}}]}}]},
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"r1"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 7}},
        done=True,
    )


def openai_text(text="CUDA ran out of memory."):
    half = len(text) // 2
    return sse(
        {"choices": [{"delta": {"content": text[:half]}}]},
        {"choices": [{"delta": {"content": text[half:]}, "finish_reason": "stop"}]},
        done=True,
    )


def anthropic_tool_call():
    return sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 50}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Checking logs."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_run_logs",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"run_id"'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": ': "r1"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
        named=True,
    )


def anthropic_text(text="CUDA ran out of memory."):
    return sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 80}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
        named=True,
    )


def gemini_tool_call():
    return sse(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_run_logs",
                                    "args": {"run_id": "r1"},
                                },
                                "thoughtSignature": "sig==",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 5},
        }
    )


def gemini_text(text="CUDA ran out of memory."):
    return sse(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": text[:5]}]}}]},
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": text[5:]}]},
                    "finishReason": "STOP",
                }
            ]
        },
    )


def ollama_tool_call():
    return [
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_run_logs",
                                "arguments": {"run_id": "r1"},
                            }
                        }
                    ],
                },
                "done": False,
            }
        ),
        json.dumps(
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 50,
                "eval_count": 5,
            }
        ),
    ]


def ollama_text(text="CUDA ran out of memory."):
    return [
        json.dumps({"message": {"role": "assistant", "content": text[:5]}, "done": False}),
        json.dumps(
            {
                "message": {"role": "assistant", "content": text[5:]},
                "done": True,
                "done_reason": "stop",
            }
        ),
    ]


STREAMS = {
    "openai": (openai_tool_call, openai_text),
    "anthropic": (anthropic_tool_call, anthropic_text),
    "gemini": (gemini_tool_call, gemini_text),
    "ollama": (ollama_tool_call, ollama_text),
    # OpenRouter speaks OpenAI's protocol and sends ": OPENROUTER PROCESSING" keep-alive comments.
    "openrouter": (
        lambda: [": OPENROUTER PROCESSING", "", *openai_tool_call()],
        lambda text="CUDA ran out of memory.": [": OPENROUTER PROCESSING", "", *openai_text(text)],
    ),
}


def adapter(pid, *responses, **config):
    transport = FakeTransport(*responses)
    cfg = ProviderConfig(
        model=config.pop("model", "m"),
        api_key=config.pop("api_key", "sk-test-1234"),
        **config,
    )
    return create_provider(pid, cfg, transport=transport, sleep=lambda s: None), transport


def collect(events):
    events = list(events)
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    completion = events[-1]
    assert isinstance(completion, Completion)
    return text, completion


class FakeBridge:
    """An in-memory workspace: two runs, one failed with a CUDA OOM and a NaN loss."""

    def __init__(self):
        self.actions: list[tuple] = []
        self.run_records = [
            {
                "id": "r2",
                "experiment": "ppo_cartpole",
                "status": "failed",
                "method": "ppo",
            },
            {
                "id": "r1",
                "experiment": "ppo_cartpole",
                "status": "completed",
                "method": "ppo",
            },
        ]

    def snapshot(self):
        return {
            "current_experiment": "experiment/ppo_cartpole.yaml",
            "selected_run": "r2",
            "active_runs": [],
        }

    def runs(self):
        return list(self.run_records)

    def run_config(self, run_id):
        return {
            "method": {"lr": 3e-4 if run_id == "r1" else 3e-2, "batch_size": 64},
            "seed": 1,
        }

    def run_logs(self, run_id):
        if run_id == "r1":
            return ["epoch 1", "epoch 2", "done"], ""
        return [
            "Epoch 0: loss=0.69",
            "UserWarning: something minor",
            "Traceback (most recent call last):",
            '  File "train.py", line 10, in <module>',
            "    step()",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
            "process exited 1",
        ], ""

    def run_metrics(self, run_id):
        if run_id == "r1":
            return {
                "train/loss": (
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28],
                )
            }
        return {"train/loss": ([0, 1, 2, 3], [0.69, 4.0, math.nan, math.nan])}

    def experiments(self):
        return ["experiment/ppo_cartpole.yaml"]

    def read_text(self, path):
        return "method: ppo\n"

    def open_file(self, path, line=None):
        self.actions.append(("open", path, line))
        return f"Opened {path}"

    def select_experiment(self, path):
        self.actions.append(("select", path))
        return f"Selected {path}"

    def launch_current(self):
        self.actions.append(("launch",))
        return "Queued"

    def stop_run(self, run_id):
        self.actions.append(("stop", run_id))
        return f"Stopping {run_id}"

    files = {"experiment/demo/base.yaml": "seed: 1\nmethods:\n  clf:\n    model: mlp\n"}

    def config_files(self, group=""):
        return {"model": ["model/basics/dnn.yaml"]}

    def existing_text(self, path):
        return self.files.get(path)

    def write_text(self, path, content):
        self.actions.append(("write", path))
        self.files = {**self.files, path: content}
        return f"Wrote {path}"

    validations = {}

    def is_experiment(self, path):
        return path.startswith("experiment/") and not path.split("/")[-1].startswith("_")

    def validate_experiment(self, path):
        return self.validations.get(path, {"experiment": path, "valid": True})

    def hub_search(self, query="", kind="", installed=None, refresh=False):
        return {
            "matches": 1,
            "components": [{"id": "ppo", "kind": "method", "installed": False}],
        }

    def hub_describe_install(self, component_id, version=None):
        if component_id == "dnn":
            from plugins.ai_assistant.tools import ToolError

            raise ToolError("DNN 1.0 is already installed; nothing to do.")
        return f"PPO ({component_id}) 1.2.0\nInstalls Python packages with pip: gymnasium"

    def hub_install(self, component_id, version=None):
        self.actions.append(("install", component_id))
        return f"Installed {component_id}."


# ── Provider adapters ────────────────────────────────────────────────────────


def test_every_provider_is_registered():
    assert set(PROVIDERS) == {"openai", "anthropic", "gemini", "ollama", "openrouter"}


@pytest.mark.parametrize("pid", sorted(STREAMS))
def test_each_adapter_parses_a_streamed_tool_call(pid):
    tool_stream, _ = STREAMS[pid]
    provider, _transport = adapter(pid, tool_stream())
    _text, completion = collect(
        provider.stream(
            [Message("user", "why?")],
            [ToolSpec("get_run_logs", "d", {"type": "object"})],
        )
    )
    [call] = completion.message.tool_calls
    assert call.name == "get_run_logs"
    assert call.arguments == {"run_id": "r1"}
    assert call.id


@pytest.mark.parametrize("pid", sorted(STREAMS))
def test_each_adapter_streams_text_incrementally(pid):
    _, text_stream = STREAMS[pid]
    provider, _ = adapter(pid, text_stream())
    events = list(provider.stream([Message("user", "hi")], []))
    deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(deltas) >= 1
    assert "".join(d.text for d in deltas) == "CUDA ran out of memory."
    assert events[-1].message.content == "CUDA ran out of memory."


def test_openai_payload_shape():
    provider, transport = adapter("openai", openai_text())
    history = [
        Message("user", "why"),
        Message(
            "assistant",
            "",
            tool_calls=[ToolCall("call_a", "get_run_logs", {"run_id": "r1"})],
        ),
        Message("tool", "OOM", tool_call_id="call_a", name="get_run_logs"),
    ]
    list(
        provider.stream(
            history,
            [ToolSpec("get_run_logs", "Logs", {"type": "object"})],
            system="sys",
        )
    )
    request = transport.requests[0]
    assert request["url"] == "https://api.openai.com/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer sk-test-1234"
    messages = request["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[2]["tool_calls"][0]["function"] == {
        "name": "get_run_logs",
        "arguments": '{"run_id": "r1"}',
    }
    assert messages[3] == {"role": "tool", "tool_call_id": "call_a", "content": "OOM"}
    assert request["payload"]["tools"][0]["type"] == "function"
    assert request["payload"]["stream"] is True


def test_openai_base_url_supports_compatible_servers():
    provider, transport = adapter("openai", openai_text(), base_url="http://localhost:8000/v1/")
    list(provider.stream([Message("user", "hi")], []))
    assert transport.requests[0]["url"] == "http://localhost:8000/v1/chat/completions"


def test_anthropic_merges_tool_results_into_one_user_turn():
    provider, transport = adapter("anthropic", anthropic_text())
    calls = [ToolCall("t1", "a", {}), ToolCall("t2", "b", {"x": 1})]
    history = [
        Message("user", "go"),
        Message("assistant", "Looking.", tool_calls=calls),
        Message("tool", "ra", tool_call_id="t1", name="a"),
        Message("tool", "rb", tool_call_id="t2", name="b", is_error=True),
    ]
    list(provider.stream(history, [ToolSpec("a", "A", {"type": "object"})], system="sys"))
    request = transport.requests[0]
    payload = request["payload"]
    assert request["headers"]["x-api-key"] == "sk-test-1234"
    assert request["headers"]["anthropic-version"]
    assert payload["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user"]
    assert payload["messages"][1]["content"][2] == {
        "type": "tool_use",
        "id": "t2",
        "name": "b",
        "input": {"x": 1},
    }
    results = payload["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True


def test_anthropic_overloaded_error_event_is_retryable():
    stream = sse(
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        },
        named=True,
    )
    provider, _ = adapter("anthropic", stream)
    with pytest.raises(ProviderUnavailableError):
        list(provider.stream([Message("user", "hi")], []))


def test_anthropic_truncated_stream_is_an_error():
    stream = anthropic_text()[:-3]  # drop message_delta and message_stop
    provider, _ = adapter("anthropic", stream[: len(stream) - 3])
    with pytest.raises(ProviderUnavailableError):
        list(provider.stream([Message("user", "hi")], []))


def test_gemini_keeps_key_out_of_url_and_round_trips_thought_signature():
    provider, transport = adapter("gemini", gemini_tool_call(), gemini_text(), model="gemini-x")
    _, completion = collect(provider.stream([Message("user", "why")], []))
    call = completion.message.tool_calls[0]
    assert call.extra == {"thoughtSignature": "sig=="}
    history = [
        Message("user", "why"),
        completion.message,
        Message("tool", "OOM", tool_call_id=call.id, name=call.name),
    ]
    list(provider.stream(history, [], system="sys"))
    request = transport.requests[1]
    assert "sk-test" not in request["url"]
    assert request["url"].endswith("/models/gemini-x:streamGenerateContent?alt=sse")
    assert request["headers"]["x-goog-api-key"] == "sk-test-1234"
    contents = request["payload"]["contents"]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["thoughtSignature"] == "sig=="
    assert contents[2]["parts"][0]["functionResponse"] == {
        "name": "get_run_logs",
        "response": {"content": "OOM"},
    }
    assert request["payload"]["systemInstruction"] == {"parts": [{"text": "sys"}]}


def test_gemini_schema_drops_unsupported_keywords():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string", "default": "a", "title": "X"}},
    }
    assert gemini_schema(schema) == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }


def test_ollama_needs_no_key_and_uses_local_server():
    provider, transport = adapter("ollama", ollama_text(), api_key=None)
    list(provider.stream([Message("user", "hi")], []))
    assert transport.requests[0]["url"] == "http://localhost:11434/api/chat"
    assert "authorization" not in transport.requests[0]["headers"]


@pytest.mark.parametrize("pid", ["openai", "anthropic", "gemini", "openrouter"])
def test_missing_key_fails_before_any_request(pid):
    provider, transport = adapter(pid, api_key=None)
    with pytest.raises(AuthenticationError, match="no API key"):
        list(provider.stream([Message("user", "hi")], []))
    assert transport.requests == []


def test_list_models():
    provider, transport = adapter("ollama", {"models": [{"name": "qwen3"}, {"name": "llama3.1"}]}, api_key=None)
    assert provider.list_models() == ["llama3.1", "qwen3"]
    assert transport.requests[0]["url"].endswith("/api/tags")


# ── Errors and retries ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "status, error",
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (400, BadRequestError),
        (404, BadRequestError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (529, ProviderUnavailableError),
    ],
)
def test_error_for_status(status, error):
    assert isinstance(error_for_status(status, '{"error": {"message": "nope"}}', {}, "X"), error)


def test_error_message_is_extracted_from_body():
    error = error_for_status(
        400,
        '{"error": {"type": "invalid_request_error", "message": "model not found"}}',
        {},
        "Anthropic",
    )
    assert "model not found" in str(error)


def test_rate_limit_is_retried_honouring_retry_after():
    sleeps = []
    provider, transport = adapter(
        "openai",
        error_for_status(429, "slow down", {"retry-after": "7"}, "OpenAI"),
        openai_text(),
    )
    provider._sleep = sleeps.append
    retries = []
    provider.on_retry = lambda attempt, wait, error: retries.append(attempt)
    text, _ = collect(provider.stream([Message("user", "hi")], []))
    assert text == "CUDA ran out of memory."
    assert len(transport.requests) == 2
    assert sleeps and sleeps[0] >= 7
    assert retries == [1]


def test_auth_errors_are_not_retried():
    provider, transport = adapter("openai", error_for_status(401, "bad key", {}, "OpenAI"), openai_text())
    with pytest.raises(AuthenticationError):
        list(provider.stream([Message("user", "hi")], []))
    assert len(transport.requests) == 1


def test_retries_give_up_after_max_attempts():
    calls = []

    def flaky():
        calls.append(1)
        raise ProviderUnavailableError("down")

    with pytest.raises(ProviderUnavailableError):
        call_with_retries(flaky, RetryPolicy(max_attempts=3), sleep=lambda s: None)
    assert len(calls) == 3


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay=1, max_delay=5)
    error = ProviderUnavailableError("x")
    assert policy.delay(0, error) <= 1
    assert 2 <= policy.delay(2, error) * 2  # 4 * jitter in [0.5, 1]
    assert policy.delay(10, error) <= 5


def test_sse_framing_ignores_comments_and_joins_multiline_data():
    lines = [": keepalive", "event: x", "data: a", "data: b", "", "data: c"]
    assert list(iter_sse(lines)) == [("x", "a\nb"), ("message", "c")]


# ── Key storage ──────────────────────────────────────────────────────────────


class FakeKeychain:
    def __init__(self):
        self.store = {}

    def get_keyring(self):
        return self

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, value):
        self.store[(service, user)] = value

    def delete_password(self, service, user):
        self.store.pop((service, user))


def test_keys_live_in_the_keychain_not_in_settings():
    keychain = FakeKeychain()
    keys = KeyStore(backend=keychain, env={})
    keys.set("openai", "  sk-abc123456  ")
    assert keychain.store == {("ThetaIDE", "openai-api-key"): "sk-abc123456"}
    assert keys.get("openai") == "sk-abc123456"
    assert keys.source("openai") == "keychain"
    keys.delete("openai")
    keys.delete("openai")  # idempotent
    assert keys.get("openai") is None


def test_environment_variable_is_the_fallback():
    keys = KeyStore(backend=FakeKeychain(), env={"ANTHROPIC_API_KEY": "sk-env"})
    assert keys.get("anthropic") == "sk-env"
    assert keys.source("anthropic") == "environment"


def test_insecure_keyring_backend_refuses_to_store():
    class PlaintextKeyring:  # stands in for keyrings.alt.file.PlaintextKeyring
        pass

    PlaintextKeyring.__module__ = "keyrings.alt.file"
    backend = SimpleNamespace(
        get_keyring=lambda: PlaintextKeyring(),
        set_password=None,
        get_password=lambda *a: None,
    )
    keys = KeyStore(backend=backend, env={})
    assert not keys.is_secure()
    with pytest.raises(KeyStoreError, match="OPENAI_API_KEY"):
        keys.set("openai", "sk-x")


def test_locked_keychain_does_not_crash():
    def locked(*_):
        raise RuntimeError("locked")

    backend = SimpleNamespace(get_keyring=lambda: None, get_password=locked)
    assert KeyStore(backend=backend, env={}).get("openai") is None


def test_mask_shows_only_last_four():
    assert mask("sk-1234567890abcd") == "••••••••abcd"
    assert "1234" not in mask("sk-1234567890abcd")


# ── Analysis ─────────────────────────────────────────────────────────────────


def test_extract_errors_keeps_the_whole_traceback():
    _bridge_lines, _ = FakeBridge().run_logs("r2")
    [site] = extract_errors(_bridge_lines)
    assert "Traceback" in site["excerpt"]
    assert "CUDA out of memory" in site["excerpt"]


def test_summarize_flags_nan_and_trend():
    summary = summarize_series("loss", [0, 1, 2, 3], [0.69, 4.0, math.nan, math.nan])
    assert summary["first_nonfinite_step"] == 2
    assert summary["nonfinite_count"] == 2
    falling = summarize_series("loss", list(range(8)), [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28])
    assert falling["trend"] == "decreasing"
    assert falling["first"] == 1.0 and falling["last"] == 0.28


def test_summarize_detects_plateau_and_downsamples():
    values = [1.0] * 200
    summary = summarize_series("reward", list(range(200)), values)
    assert summary["note"] == "plateaued over the last quarter"
    assert len(summary["sampled"]) == 12


def test_summarize_flags_learning_that_levelled_off():
    rising_then_flat = [min(i, 10) * 10.0 for i in range(40)]
    assert summarize_series("reward", list(range(40)), rising_then_flat)["note"].startswith("levelled off")
    still_learning = [float(i) for i in range(40)]
    assert "note" not in summarize_series("reward", list(range(40)), still_learning)


def test_summarize_divergence_is_measured_against_the_early_level():
    exploding = summarize_series("loss", list(range(8)), [1.0, 0.9, 3.0, 40.0, 900.0, 2e4, 3e5, 4e5])
    assert exploding["note"].startswith("magnitude grew")
    accuracy_from_zero = summarize_series("acc", list(range(8)), [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.96, 0.97])
    assert "note" not in accuracy_from_zero


def test_diff_configs_reports_only_differences():
    diff = diff_configs(
        {
            "a": {"m": {"lr": 1, "bs": 64}},
            "b": {"m": {"lr": 2, "bs": 64}, "extra": True},
        }
    )
    assert diff == {"extra": {"a": "<unset>", "b": True}, "m.lr": {"a": 1, "b": 2}}


def test_read_metrics_keeps_nan(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("step,train/loss,note\n0,0.5,x\n1,nan,y\n2,,z\n")
    series = read_metrics(path)
    steps, values = series["train/loss"]
    assert steps == [0.0, 1.0]
    assert values[0] == 0.5 and math.isnan(values[1])
    assert "note" not in series


# ── Tools ────────────────────────────────────────────────────────────────────


def run_tool(registry, name, **arguments):
    return registry.execute(ToolCall("c1", name, arguments))


def test_tool_specs_are_valid_json_schema_objects():
    registry = build_ide_tools(FakeBridge())
    for spec in registry.specs():
        assert spec.parameters["type"] == "object"
        assert set(spec.parameters["required"]) <= set(spec.parameters["properties"])
    assert registry.is_mutating("launch_experiment") and registry.is_mutating("stop_run")
    assert not registry.is_mutating("get_run_logs")


def test_get_run_logs_resolves_selected_run_and_extracts_errors():
    result = run_tool(build_ide_tools(FakeBridge()), "get_run_logs")
    data = json.loads(result.content)
    assert not result.is_error
    assert data["run_id"] == "r2"
    assert "CUDA out of memory" in data["errors"][0]["excerpt"]
    assert data["warnings"] == 1


def test_get_run_metrics_and_compare_runs():
    registry = build_ide_tools(FakeBridge())
    metrics = json.loads(run_tool(registry, "get_run_metrics", run_id="r2").content)
    assert metrics["summaries"][0]["first_nonfinite_step"] == 2
    comparison = json.loads(run_tool(registry, "compare_runs", run_ids=["r1", "r2"]).content)
    assert comparison["config_differences"] == {"method.lr": {"r1": 3e-4, "r2": 3e-2}}
    assert set(comparison["metrics"]["train/loss"]) == {"r1", "r2"}


def test_latest_alias_and_list_filter():
    registry = build_ide_tools(FakeBridge())
    assert json.loads(run_tool(registry, "get_run_config", run_id="latest").content)["run_id"] == "r2"
    listed = json.loads(run_tool(registry, "list_runs", status="completed").content)
    assert [r["id"] for r in listed["runs"]] == ["r1"]


@pytest.mark.parametrize(
    "name, arguments, message",
    [
        ("get_run_logs", {"run_id": "nope"}, "No run with id 'nope'"),
        ("compare_runs", {}, "Missing required argument"),
        ("compare_runs", {"run_ids": ["r1"]}, "at least two"),
        ("no_such_tool", {}, "Unknown tool"),
        ("get_run_logs", {"_unparsed_arguments": "{bad"}, "not valid JSON"),
    ],
)
def test_tool_failures_are_reported_to_the_model(name, arguments, message):
    result = run_tool(build_ide_tools(FakeBridge()), name, **arguments)
    assert result.is_error
    assert message in result.content


def test_unexpected_tool_exception_becomes_an_error_result():
    registry = ToolRegistry()
    registry.register(
        Tool(
            "boom",
            "d",
            {"type": "object", "properties": {}, "required": []},
            lambda: 1 / 0,
        )
    )
    result = run_tool(registry, "boom")
    assert result.is_error and "ZeroDivisionError" in result.content


def test_unknown_arguments_are_dropped():
    registry = build_ide_tools(FakeBridge())
    assert not run_tool(registry, "list_experiments", surprise=1).is_error


# ── Agent loop ───────────────────────────────────────────────────────────────


class ScriptedProvider:
    """Returns one scripted assistant Message per call and records what it was sent."""

    def __init__(self, *replies, fail_at=None):
        self.replies = list(replies)
        self.calls = []
        self.config = ProviderConfig(model="scripted")
        self.fail_at = fail_at

    def stream(self, messages, tools, system="", allow_tools=True):
        from plugins.ai_assistant.providers.chat_types import system_text

        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "system": system_text(system),
                "system_parts": system,
                "allow_tools": allow_tools,
            }
        )
        if self.fail_at == len(self.calls):
            raise ProviderUnavailableError("boom")
        reply = self.replies.pop(0)
        if reply.content:
            yield TextDelta(reply.content)
        yield Completion(reply, "stop", Usage(10, 2))


def call(name, **arguments):
    return ToolCall(f"id_{name}", name, arguments)


def test_agent_runs_tools_then_answers():
    bridge = FakeBridge()
    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=[call("get_run_logs", run_id="r2")]),
        Message("assistant", "It ran out of GPU memory; halve batch_size."),
    )
    agent = Agent(provider, build_ide_tools(bridge), snapshot=bridge.snapshot)
    events = list(agent.run("Why did my run fail?"))
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolStarted", "ToolFinished", "TextDelta", "TurnFinished"]
    assert "CUDA out of memory" in events[1].result.content
    assert events[-1].usage == Usage(20, 4) and events[-1].steps == 2
    # The second request carried the tool result back, and the IDE state was in the system prompt.
    sent = provider.calls[1]["messages"]
    assert [m.role for m in sent] == ["user", "assistant", "tool"]
    assert sent[2].tool_call_id == "id_get_run_logs"
    assert "ppo_cartpole" in provider.calls[0]["system"]


def test_mutating_tool_requires_confirmation():
    bridge = FakeBridge()
    asked = []
    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=[call("stop_run", run_id="r2")]),
        Message("assistant", "OK, left it running."),
    )
    agent = Agent(
        provider,
        build_ide_tools(bridge),
        confirm=lambda c, preview: asked.append(c.name) or False,
    )
    events = list(agent.run("stop it"))
    started = next(e for e in events if isinstance(e, ToolStarted))
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert started.needs_confirmation
    assert asked == ["stop_run"]
    assert bridge.actions == []
    assert finished.result.is_error and finished.result.content == DECLINED


def test_approved_action_runs():
    bridge = FakeBridge()
    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=[call("launch_experiment")]),
        Message("assistant", "Queued."),
    )
    list(Agent(provider, build_ide_tools(bridge), confirm=lambda c, preview: True).run("run it"))
    assert bridge.actions == [("launch",)]


def test_read_only_tools_never_ask():
    def never(c, preview):
        raise AssertionError("asked")

    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=[call("open_file", path="a.yaml", line=3)]),
        Message("assistant", "Opened."),
    )
    bridge = FakeBridge()
    list(Agent(provider, build_ide_tools(bridge), confirm=never).run("open it"))
    assert bridge.actions == [("open", "a.yaml", 3)]


def test_read_only_steps_are_free_up_to_a_hard_cap_then_the_model_wraps_up():
    reads = [Message("assistant", "", tool_calls=[call("list_runs")]) for _ in range(9)]
    provider = ScriptedProvider(*reads, Message("assistant", "Here is what I found so far."))
    events = list(Agent(provider, build_ide_tools(FakeBridge()), max_steps=3).run("loop"))
    assert len(provider.calls) == 10  # 3 x max_steps successful reads, then one wrap-up reply
    assert all(c["allow_tools"] for c in provider.calls[:9])
    wrap = provider.calls[9]
    assert wrap["allow_tools"] is False and "used this message's tool budget" in wrap["system"]
    assert events[-1].message.content == "Here is what I found so far." and events[-1].steps == 10


def test_steps_that_change_something_use_the_budget():
    edits = [Message("assistant", "", tool_calls=[call("open_file", path="a"), call("stop_run")]) for _ in range(2)]
    provider = ScriptedProvider(*edits, Message("assistant", "Stopped it."))
    list(
        Agent(
            provider,
            build_ide_tools(FakeBridge()),
            confirm=lambda c, p: True,
            max_steps=2,
        ).run("x")
    )
    assert [c["allow_tools"] for c in provider.calls] == [True, True, False]


def failing_reads(n):
    return [Message("assistant", "", tool_calls=[call("get_run_logs", run_id=f"bad{i}")]) for i in range(n)]


def test_repeated_failures_get_a_nudge_then_tools_are_taken_away():
    provider = ScriptedProvider(
        *failing_reads(5),
        Message("assistant", "I can't find that run; which one did you mean?"),
    )
    events = list(Agent(provider, build_ide_tools(FakeBridge()), max_steps=20).run("logs?"))
    results = [e.result for e in events if isinstance(e, ToolFinished)]
    assert ["tool calls in a row have failed" in r.content for r in results] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert provider.calls[-1]["allow_tools"] is False
    assert "5 tool calls in a row have failed" in provider.calls[-1]["system"]
    assert events[-1].message.content.startswith("I can't find that run")


def test_a_success_resets_the_failure_count():
    replies = [
        *failing_reads(2),
        Message("assistant", "", tool_calls=[call("list_runs")]),
        *failing_reads(2),
        Message("assistant", "done"),
    ]
    events = list(Agent(ScriptedProvider(*replies), build_ide_tools(FakeBridge()), max_steps=20).run("x"))
    assert not any("in a row have failed" in e.result.content for e in events if isinstance(e, ToolFinished))


def test_user_rejections_are_not_counted_as_failures():
    rejects = [Message("assistant", "", tool_calls=[call("launch_experiment")]) for _ in range(5)]
    provider = ScriptedProvider(*rejects, Message("assistant", "ok"))
    list(
        Agent(
            provider,
            build_ide_tools(FakeBridge()),
            confirm=lambda c, p: False,
            max_steps=20,
        ).run("x")
    )
    assert all("in a row have failed" not in m.content for c in provider.calls for m in c["messages"])


def test_wrap_up_drops_tool_calls_from_a_model_that_ignores_tool_choice():
    provider = ScriptedProvider(
        *failing_reads(5),
        Message("assistant", "Sorry.", tool_calls=[call("list_runs")]),
    )
    agent = Agent(provider, build_ide_tools(FakeBridge()), max_steps=20)
    events = list(agent.run("x"))
    assert events[-1].message.content == "Sorry." and not events[-1].message.tool_calls
    assert not agent.history[-1].tool_calls


@pytest.mark.parametrize(
    "pid, check",
    [
        ("openai", lambda p: p["tool_choice"] == "none" and p["tools"]),
        ("anthropic", lambda p: p["tool_choice"] == {"type": "none"} and p["tools"]),
        (
            "gemini",
            lambda p: p["toolConfig"]["functionCallingConfig"]["mode"] == "NONE" and p["tools"],
        ),
        ("ollama", lambda p: "tools" not in p),
    ],
)
def test_each_adapter_can_forbid_tool_calls(pid, check):
    _, text_stream = STREAMS[pid]
    provider, transport = adapter(pid, text_stream(), text_stream())
    specs = [ToolSpec("a", "A", {"type": "object"})]
    list(provider.stream([Message("user", "hi")], specs, allow_tools=False))
    list(provider.stream([Message("user", "hi")], specs))
    assert check(transport.requests[0]["payload"])
    second = transport.requests[1]["payload"]
    assert "tool_choice" not in second and "toolConfig" not in second and "tools" in second


def test_failed_turn_rolls_back_dangling_tool_calls():
    provider = ScriptedProvider(Message("assistant", "", tool_calls=[call("list_runs")]), fail_at=2)
    agent = Agent(provider, build_ide_tools(FakeBridge()))
    agent.history = [Message("user", "earlier"), Message("assistant", "earlier answer")]
    with pytest.raises(ProviderUnavailableError):
        list(agent.run("now"))
    assert [(m.role, m.content) for m in agent.history] == [
        ("user", "earlier"),
        ("assistant", "earlier answer"),
    ]


def test_cancel_stops_the_stream():
    cancel = threading.Event()
    cancel.set()
    agent = Agent(
        ScriptedProvider(Message("assistant", "long answer")),
        build_ide_tools(FakeBridge()),
    )
    with pytest.raises(StreamCancelled):
        list(agent.run("hi", cancel))
    assert agent.history == []


def test_context_window_starts_at_a_user_turn():
    agent = Agent(ScriptedProvider(), ToolRegistry(), history_chars=50)
    agent.history = [
        Message("user", "x" * 40),
        Message("assistant", "", tool_calls=[call("list_runs")]),
        Message("tool", "y" * 40, tool_call_id="id_list_runs"),
        Message("user", "recent"),
        Message("assistant", "answer"),
    ]
    window = agent.context_window()
    assert window[0].role == "user" and window[0].content == "recent"


@pytest.mark.parametrize("pid", sorted(STREAMS))
def test_agent_works_identically_over_every_provider(pid):
    """The same tool-calling conversation through each real adapter and its wire format."""
    tool_stream, text_stream = STREAMS[pid]
    provider, transport = adapter(
        pid,
        tool_stream(),
        text_stream(),
        api_key=None if pid == "ollama" else "sk-test-1234",
    )
    bridge = FakeBridge()
    events = list(Agent(provider, build_ide_tools(bridge), snapshot=bridge.snapshot).run("Why did r1 fail?"))
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert [e.call.name for e in finished] == ["get_run_logs"]
    assert json.loads(finished[0].result.content)["run_id"] == "r1"
    assert events[-1].message.content == "CUDA ran out of memory."
    assert len(transport.requests) == 2
    assert "r1" in json.dumps(transport.requests[1]["payload"])  # the tool result went back to the model


# ── Qt panel (threading and main-thread tool marshaling) ─────────────────────


def test_panel_streams_a_turn_on_a_worker_thread(monkeypatch, tmp_path):
    pytest.importorskip("PyQt6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QCoreApplication, QThread, QUrl
    from PyQt6.QtWidgets import QApplication

    from plugins.ai_assistant import panel as panel_module

    app = QApplication.instance() or QApplication([])
    main_thread = QThread.currentThread()
    seen_threads = []

    class Context:
        window = SimpleNamespace(runs=[], selected=None, active_runs=[], current_experiment=None)

        def __init__(self):
            self.settings = {}

        def get_workspace_dir(self):
            return tmp_path

        def get_data_dir(self):
            return tmp_path / ".thetaide"

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def set_setting(self, key, value):
            self.settings[key] = value

    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=[call("list_runs")]),
        Message(
            "assistant",
            "You have no runs yet. Start with:\n\n```yaml\nseed: 1\nmethods: {}\n```",
        ),
    )
    monkeypatch.setattr(panel_module, "create_provider", lambda *a, **k: provider)
    keys = KeyStore(backend=FakeKeychain(), env={})
    context = Context()
    panel = panel_module.AssistantPanel(context, keys)
    sharing_asked = []
    monkeypatch.setattr(
        panel,
        "_confirm_data_sharing",
        lambda name, host: sharing_asked.append(host) or True,
    )
    original = panel.tools.execute

    def recording_execute(c):
        seen_threads.append(QThread.currentThread())
        return original(c)

    panel.tools.execute = recording_execute
    try:
        panel.ask("How many runs?")
        for _ in range(500):
            QCoreApplication.processEvents()
            if not panel.busy:
                break
            QThread.msleep(5)
        assert not panel.busy
        panel._render()
        text = panel.transcript.toPlainText()
        assert "You have no runs yet." in text
        assert "list_runs" in text
        assert seen_threads == [main_thread]  # tools ran on the GUI thread
        # Before the first message to a cloud provider, the user was told what leaves the computer; once.
        assert sharing_asked == [panel.destination()] and panel.destination() == "api.openai.com"
        assert context.settings["data_sharing_ok"] == ["openai@api.openai.com"]
        assert panel._data_sharing_allowed() and len(sharing_asked) == 1

        # Each complete code block in a reply gets a Copy link.
        assert "⧉ Copy" in text
        panel._on_link(QUrl("copy:0"))
        assert QApplication.clipboard().text() == "seed: 1\nmethods: {}"

        # Clicking a tool line shows exactly what the model sent and got back; clicking again hides it.

        index = next(i for i, e in enumerate(panel.entries) if e["kind"] == "tool")
        assert '"total": 0' not in panel.transcript.toPlainText()
        panel._on_link(QUrl(f"tool:{index}"))
        assert '"total": 0' in panel.transcript.toPlainText() and "Arguments" in panel.transcript.toPlainText()
        panel._on_link(QUrl(f"tool:{index}"))
        assert '"total": 0' not in panel.transcript.toPlainText()

        # The chat was saved after the turn and reopens in a new panel (as after an IDE restart).
        saved = panel.store.list()
        assert [c.title for c in saved] == ["How many runs?"]
        assert context.settings["last_chat"] == saved[0].id
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
    reopened = panel_module.AssistantPanel(context, keys)
    try:
        assert "You have no runs yet." in reopened.transcript.toPlainText()
        assert reopened.disclaimer.text().startswith("AI can make mistakes")
        assert [m.role for m in reopened.history] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        # Declining the notice sends nothing and keeps the message in the box.
        reopened.new_chat()
        context.settings["data_sharing_ok"] = []
        monkeypatch.setattr(reopened, "_confirm_data_sharing", lambda name, host: False)
        reopened.ask("Compare my runs")
        assert not reopened.busy and reopened.entries == []
        assert reopened.input.toPlainText() == "Compare my runs" and "Not sent" in reopened.status.text()
        context.settings["base_urls"] = {"openai": "http://localhost:8000/v1"}  # local server: nothing leaves
        assert reopened.destination() is None and reopened._data_sharing_allowed()
        assert reopened.history == [] and context.settings["last_chat"] is None
        assert len(reopened.store.list()) == 1  # starting a new chat keeps the old one
    finally:
        reopened.shutdown()
        reopened.deleteLater()
        app.processEvents()


def build_archive(destination: Path) -> Path:
    """The release archive, built by the same function scripts/package_plugins.py uses."""
    import importlib.util

    script = PLUGINS_REPO / "scripts" / "package_plugins.py"
    spec = importlib.util.spec_from_file_location("package_plugins", script)
    packager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packager)
    archive = destination / "ai_assistant.zip"
    packager.package_component(Path(__file__).resolve().parent, archive)
    return archive


def test_release_archive_matches_its_manifest(tmp_path):
    """Every declared file ships, tests do not, and the embedded manifest is the Hub's contract."""
    import zipfile

    with zipfile.ZipFile(build_archive(tmp_path)) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(".theta_component.json"))
    payload = manifest["payload"]
    assert set(payload["implementations"]) | set(payload["resources"]) <= names
    assert not any(Path(n).name.startswith("test_") or "__pycache__" in n for n in names)
    shipped_python = {n for n in names if n.endswith(".py")}
    assert shipped_python == set(payload["implementations"]), "every shipped module must be declared"
    assert manifest["id"] == "ai_assistant" and manifest["kind"] == "plugin"
    assert (manifest["storage_version"], manifest["api_version"]) == (2, 1)
    assert manifest["capabilities"] == [
        {
            "id": "ai_assistant",
            "kind": "plugin",
            "entry_point": "__init__.py:AIAssistantPlugin",
        }
    ]
    assert manifest["dependencies"] == {"components": [], "pip": ["keyring>=24"]}
    plugin = json.loads((Path(__file__).resolve().parent / "plugin.json").read_text())
    assert manifest["version"] == plugin["version"] and manifest["entry_point"] == plugin["entry_point"]


def test_hub_install_mounts_in_the_ide_and_its_tools_read_the_real_window(monkeypatch, tmp_path):
    """The Hub path end to end: build the archive, install it with ThetaIDE's installer into a fresh
    workspace, and mount the installed copy (not this source tree) in a real Window."""
    pytest.importorskip("PyQt6.QtWidgets")
    import os
    import subprocess
    from unittest.mock import patch

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841  (must outlive the window)
    from frontend.app import Window

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = build_archive(tmp_path)
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.app.components",
            "--root",
            str(workspace),
            "install",
            str(archive),
            "--no-dependencies",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    root = workspace / "src" / "plugins" / "app" / "installed" / "ai_assistant"
    assert (root / "__init__.py").is_file() and (root / "README.md").is_file()
    assert not (root / "test_ai_assistant.py").exists()

    # The installed copy is a different module from this test's import: keep both off the real keychain.
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setattr(KeyStore, "backend", property(lambda self: FakeKeychain()))

    class HeadlessTerminalPanel(QWidget):
        def __init__(self, *, parent=None, **_kwargs):
            super().__init__(parent)
            self.terminals = []

        def configure(self, **_kwargs):
            pass

        def set_font_size(self, _size):
            pass

        def apply_theme(self, _theme):
            pass

    class InertBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    with (
        patch("frontend.app.TerminalPanel", HeadlessTerminalPanel),
        patch("frontend.app.LocalBackend", InertBackend),
        patch.object(Window, "init_default_experiment", lambda self: None),
    ):
        window = Window(tmp_path / "data", workspace=workspace)
    try:
        manager = window.plugin_manager
        manifest = manager.manifests["ai_assistant"]
        assert not manifest.is_core and manifest.extra.get("origin") != "bundled"  # a Hub plugin, uninstallable
        assert manager.enable_plugin("ai_assistant")
        plugin = manager.instances["ai_assistant"]
        module = sys.modules[type(plugin).__module__]
        assert Path(module.__file__).resolve() == (root / "__init__.py").resolve()  # the installed copy runs
        assert "ai_assistant" in window.tabs.tabs
        assert window.command_registry.get_command("ai_assistant.explain_run") is not None
        slot = window.extension_slots.get_slot("monitor.footer")
        assert "ai_assistant.ask" in [action_id for action_id, _ in slot.get_items()]

        panel = manager.instances["ai_assistant"].panel
        window.runs = [
            {
                "id": "demo1",
                "status": "completed",
                "simulated": True,
                "config": {"name": "demo"},
                "metrics": [
                    {"step": 1, "reward": 1.0, "loss": 0.5},
                    {"step": 2, "reward": 2.0, "loss": 0.4},
                ],
            }
        ]
        listed = json.loads(run_tool(panel.tools, "list_runs").content)
        assert listed["runs"][0]["id"] == "demo1"
        metrics = json.loads(run_tool(panel.tools, "get_run_metrics", run_id="demo1").content)
        assert {s["metric"] for s in metrics["summaries"]} == {"loss", "reward"}
        outside = run_tool(panel.tools, "read_file", path="/etc/passwd")
        assert outside.is_error and "outside the workspace" in outside.content

        # Run watching through the real window: its active runs, and the IDE's own toast with Ask AI.
        live = {
            "id": "demo2",
            "status": "running",
            "simulated": True,
            "config": {"name": "demo"},
            "metrics": [{"step": 1, "loss": 0.5}, {"step": 2, "loss": math.nan}],
        }
        window.runs = [live, *window.runs]
        window.active_runs = [live]
        monkeypatch.setattr(window.run_notifier, "in_front", lambda: True)
        alerts = panel.watcher.check(panel.bridge.watch_targets())
        assert [(a.run_id, a.kind) for a in alerts] == [("demo2", "nonfinite")]
        panel._on_alerts(alerts)
        toast = window.run_notifier.toasts.toasts[-1]
        assert len(panel.alert_rows()) == 1
        asked = []
        monkeypatch.setattr(panel, "ask", asked.append)
        toast.action.emit()
        assert asked == [alerts[0].prompt()] and panel.alert_rows() == []
        window.active_runs = []

        manager.disable_plugin("ai_assistant")
        assert "ai_assistant" not in manager.instances and "ai_assistant" not in window.tabs.tabs
        assert window.command_registry.get_command("ai_assistant.open") is None
    finally:
        window._quitting = True
        window.close()


# ── File writes need approval ────────────────────────────────────────────────


def writing_agent(bridge, *calls, approve):
    asked = []
    replies = [Message("assistant", "", tool_calls=[c]) for c in calls] + [Message("assistant", "Done.")]

    def confirm(c, preview):
        asked.append((c.name, preview))
        return approve

    agent = Agent(ScriptedProvider(*replies), build_ide_tools(bridge), confirm=confirm)
    events = list(agent.run("make an experiment"))
    return asked, [e.result for e in events if isinstance(e, ToolFinished)]


def test_new_file_is_written_only_after_approving_its_diff():
    bridge = FakeBridge()
    content = "# @package _global_\nmethods:\n  clf:\n    model: dnn\n"
    asked, [result] = writing_agent(
        bridge,
        call("write_file", path="experiment/demo/dnn.yaml", content=content),
        approve=True,
    )
    [(name, preview)] = asked
    assert name == "write_file"
    assert preview.startswith("--- /dev/null\n+++ b/experiment/demo/dnn.yaml")
    assert "+    model: dnn" in preview
    assert not result.is_error
    assert bridge.files["experiment/demo/dnn.yaml"] == content


def test_rejected_write_changes_nothing():
    bridge = FakeBridge()
    asked, [result] = writing_agent(
        bridge,
        call("write_file", path="experiment/demo/x.yaml", content="a: 1"),
        approve=False,
    )
    assert len(asked) == 1
    assert result.is_error and result.content == DECLINED
    assert ("write", "experiment/demo/x.yaml") not in bridge.actions


def test_edit_shows_a_diff_of_just_the_change():
    bridge = FakeBridge()
    edit = call(
        "edit_file",
        path="experiment/demo/base.yaml",
        old_text="model: mlp",
        new_text="model: dnn",
    )
    asked, [result] = writing_agent(bridge, edit, approve=True)
    preview = asked[0][1]
    assert "-    model: mlp" in preview and "+    model: dnn" in preview
    assert "seed: 1" not in preview.replace(" seed: 1", "")  # context lines only, unchanged
    assert bridge.files["experiment/demo/base.yaml"].endswith("model: dnn\n")
    assert not result.is_error


@pytest.mark.parametrize(
    "tool_call, message",
    [
        (
            call(
                "edit_file",
                path="experiment/demo/base.yaml",
                old_text="nope",
                new_text="x",
            ),
            "exactly once",
        ),
        (
            call(
                "edit_file",
                path="experiment/demo/missing.yaml",
                old_text="a",
                new_text="b",
            ),
            "does not exist",
        ),
        (
            call(
                "write_file",
                path="experiment/demo/base.yaml",
                content=FakeBridge.files["experiment/demo/base.yaml"],
            ),
            "nothing to change",
        ),
    ],
)
def test_impossible_writes_fail_without_asking_the_user(tool_call, message):
    bridge = FakeBridge()
    asked, [result] = writing_agent(bridge, tool_call, approve=True)
    assert asked == []
    assert result.is_error and message in result.content
    assert not any(a[0] == "write" for a in bridge.actions)


@pytest.fixture
def workspace_bridge(tmp_path, monkeypatch):
    from plugins.ai_assistant.bridge import WindowBridge

    config = tmp_path / "in" / "config"
    (config / "model" / "basics").mkdir(parents=True)
    (config / "model" / "basics" / "dnn.yaml").write_text("architecture: dnn\n")
    (config / "experiment").mkdir()
    (config / "hydra").mkdir()
    (config / "hydra" / "x.yaml").write_text("a: 1\n")
    (tmp_path / ".git").mkdir()
    refreshed = []
    window = SimpleNamespace(
        config_tree=SimpleNamespace(populate=lambda ensure_expanded=None: refreshed.append(ensure_expanded))
    )
    bridge = WindowBridge(window, tmp_path)
    monkeypatch.setattr(bridge, "_config_root", lambda: config)
    bridge.refreshed = refreshed
    return bridge


def test_bridge_writes_config_group_paths_into_the_config_dir(workspace_bridge, tmp_path):
    message = workspace_bridge.write_text("experiment/dnn/baseline.yaml", "seed: 1")
    written = tmp_path / "in" / "config" / "experiment" / "dnn" / "baseline.yaml"
    assert written.read_text() == "seed: 1\n"
    assert message.startswith("Created in/config/experiment/dnn/baseline.yaml")
    assert workspace_bridge.refreshed == ["experiment/dnn"]
    assert workspace_bridge.existing_text("experiment/dnn/baseline.yaml") == "seed: 1\n"
    assert workspace_bridge.existing_text("experiment/dnn/other.yaml") is None


@pytest.mark.parametrize(
    "path, content, message",
    [
        ("/etc/evil.yaml", "a: 1", "outside the workspace"),
        ("../escape.txt", "x", "outside the workspace"),
        (".git/config", "x", "protected"),
        ("experiment/bad.yaml", "a: [1, 2", "does not parse"),
    ],
)
def test_bridge_refuses_unsafe_writes(workspace_bridge, tmp_path, path, content, message):
    from plugins.ai_assistant.tools import ToolError

    with pytest.raises(ToolError, match=message):
        workspace_bridge.write_text(path, content)
    assert not (tmp_path / "in" / "config" / "experiment" / "bad.yaml").exists()


def test_list_configs_finds_component_configs(workspace_bridge):
    assert workspace_bridge.config_files() == {"model": ["model/basics/dnn.yaml"]}  # hydra internals hidden
    assert workspace_bridge.config_files("model") == {"model": ["model/basics/dnn.yaml"]}
    assert workspace_bridge.read_text("model/basics/dnn.yaml") == "architecture: dnn\n"


def test_approval_dialog_rejects_by_default():
    pytest.importorskip("PyQt6.QtWidgets")
    import os
    import sys

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841

    from plugins.ai_assistant.approval import ApprovalDialog, diff_html

    dialog = ApprovalDialog(
        "Edit file?",
        "change <b>x</b>",
        "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        is_diff=True,
    )
    assert dialog.reject_button.isDefault() and not dialog.approve_button.isDefault()
    assert "rgba(46, 160, 67" in diff_html("+b") and "rgba(248, 81, 73" in diff_html("-a")
    assert "&lt;tag&gt;" in diff_html("+<tag>")
    dialog.deleteLater()


# ── Component Hub ────────────────────────────────────────────────────────────


def test_hub_install_needs_approval_and_shows_what_it_installs():
    bridge = FakeBridge()
    registry = build_ide_tools(bridge)
    assert registry.runs_in_background("install_from_hub") and registry.runs_in_background("search_hub")
    assert registry.is_mutating("install_from_hub") and not registry.is_mutating("search_hub")
    asked, [result] = writing_agent(bridge, call("install_from_hub", component_id="ppo"), approve=True)
    assert asked[0][0] == "install_from_hub" and "pip: gymnasium" in asked[0][1]
    assert bridge.actions == [("install", "ppo")] and not result.is_error

    bridge = FakeBridge()
    _asked, [result] = writing_agent(bridge, call("install_from_hub", component_id="ppo"), approve=False)
    assert bridge.actions == [] and result.content == DECLINED

    bridge = FakeBridge()
    asked, [result] = writing_agent(bridge, call("install_from_hub", component_id="dnn"), approve=True)
    assert asked == [] and "already installed" in result.content and bridge.actions == []


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class FakeHubClient:
    """The surface of frontend.hub.HubClient that the assistant uses; installs finish on another thread."""

    def __init__(self, outcome=(True, "Installation complete."), finish=True):
        self.installFinished = FakeSignal()
        self.indexLoaded = FakeSignal()
        self.outcome, self.finish = outcome, finish
        self.fetches = 0
        self.components = []
        self.installed_with = None

    def _component(self, cid, kind, installed=None, pip=()):
        return SimpleNamespace(
            id=cid,
            name=cid.upper(),
            kind=kind,
            version="1.0",
            description=f"{cid} component",
            tags=[kind],
            author=SimpleNamespace(name="Theta"),
            license="MIT",
            repository=None,
            releases={"1.0": {}},
            is_installed=installed is not None,
            installed_version=installed,
            has_update=False,
            dependencies={"pip": list(pip)} if pip else {},
        )

    def fetch_index_sync(self, force=False, _publish=True):
        self.fetches += 1
        return [
            self._component("ppo", "method", pip=["gymnasium"]),
            self._component("dnn", "model", installed="1.0"),
        ]

    def refresh_local_components(self):
        pass

    def search(self, query="", kind=None):
        return [c for c in self.components if query in c.id and (kind is None or c.kind == kind)]

    def get_component(self, cid):
        return next((c for c in self.components if c.id == cid), None)

    def install_async(self, component, version=None):
        self.installed_with = (component.id, version)
        if self.finish:
            threading.Timer(0.05, lambda: self.installFinished.emit(component.id, *self.outcome)).start()


def hub_access(client, **kwargs):
    from plugins.ai_assistant.hub import HubAccess

    return HubAccess(SimpleNamespace(hub_client=client), **kwargs)


def test_hub_search_loads_the_index_once_and_reports_install_state():
    client = FakeHubClient()
    hub = hub_access(client)
    result = hub.search("")
    assert {c["id"]: c["installed"] for c in result["components"]} == {
        "ppo": False,
        "dnn": "1.0",
    }
    assert hub.search("", installed=False)["components"][0]["id"] == "ppo"
    assert hub.search("", kind="model")["components"][0]["id"] == "dnn"
    assert client.fetches == 1
    hub.search("", refresh=True)
    assert client.fetches == 2


def test_hub_describe_install_lists_dependencies_and_refuses_no_ops():
    from plugins.ai_assistant.tools import ToolError

    hub = hub_access(FakeHubClient())
    text = hub.describe_install("ppo")
    assert "PPO (ppo) 1.0" in text and "pip: gymnasium" in text and "third-party code" in text
    with pytest.raises(ToolError, match="already installed"):
        hub.describe_install("dnn")
    with pytest.raises(ToolError, match="No Hub component"):
        hub.describe_install("nope")
    with pytest.raises(ToolError, match="no version 9.9"):
        hub.describe_install("ppo", "9.9")


def test_hub_install_waits_for_the_hub_client_to_finish():
    client = FakeHubClient()
    message = hub_access(client).install("ppo")
    assert message.startswith("Installed PPO 1.0.")
    assert client.installed_with == ("ppo", None)
    assert client.installFinished.slots == []  # listener removed


def test_hub_install_failure_is_reported_to_the_model():
    from plugins.ai_assistant.tools import ToolError

    with pytest.raises(ToolError, match="pip install failed"):
        hub_access(FakeHubClient(outcome=(False, "pip install failed (code 1)"))).install("ppo")


def test_stop_ends_the_wait_but_not_the_install():
    client = FakeHubClient(finish=False)
    message = hub_access(client, cancelled=lambda: True).install("ppo")
    assert "keeps installing in the background" in message
    assert client.installFinished.slots == []


def test_hub_unavailable():
    from plugins.ai_assistant.hub import HubAccess
    from plugins.ai_assistant.tools import ToolError

    with pytest.raises(ToolError, match="not available"):
        HubAccess(SimpleNamespace()).search()


def test_bridge_uses_the_window_workspace_not_the_ide_data_folder(tmp_path):
    """Regression: PluginContext.get_workspace_dir() is <root>/.thetaide in a real session, which put
    every config outside the sandbox. The window's WorkspacePaths is the authority."""
    from plugins.ai_assistant.bridge import WindowBridge
    from plugins.ai_assistant.panel import workspace_root

    config = tmp_path / "in" / "config"
    (config / "agent" / "sb3").mkdir(parents=True)
    (config / "agent" / "sb3" / "ppo.yaml").write_text("lr: 0.0003\n")
    (config / "config.yaml").write_text("seed: 1\n")
    window = SimpleNamespace(workspace_paths=SimpleNamespace(root=tmp_path, config=config))
    context = SimpleNamespace(window=window, get_workspace_dir=lambda: tmp_path / ".thetaide")
    bridge = WindowBridge(window, workspace_root(context))
    for path in (
        "agent/sb3/ppo.yaml",
        "in/config/agent/sb3/ppo.yaml",
        str(config / "agent/sb3/ppo.yaml"),
    ):
        assert bridge.read_text(path) == "lr: 0.0003\n"
    assert bridge.read_text("config.yaml") == "seed: 1\n"
    assert bridge.config_files("agent") == {"agent": ["agent/sb3/ppo.yaml"]}


# ── Validating experiments ───────────────────────────────────────────────────


def test_writing_an_experiment_returns_the_pipeline_verdict():
    bridge = FakeBridge()
    bridge.validations = {
        "experiment/demo/dnn.yaml": {
            "valid": False,
            "errors": ["[compose] Could not find 'model/dnnn'"],
        }
    }
    _asked, [result] = writing_agent(
        bridge,
        call("write_file", path="experiment/demo/dnn.yaml", content="a: 1"),
        approve=True,
    )
    data = json.loads(result.content)
    assert data["validation"]["errors"] == ["[compose] Could not find 'model/dnnn'"]
    assert "Fix the errors with edit_file" in data["result"]


def test_non_experiment_writes_are_not_validated():
    bridge = FakeBridge()
    _asked, [result] = writing_agent(
        bridge,
        call("write_file", path="model/basics/x.yaml", content="a: 1"),
        approve=True,
    )
    assert result.content == "Wrote model/basics/x.yaml"


def test_summarize_report():
    from plugins.ai_assistant.bridge import summarize_report

    report = {
        "valid": False,
        "experiment": "dnn/baseline",
        "errors": [{"stage": "validation", "message": "eval_episodes must be 0 for supervised"}],
        "notices": ["using default site"],
        "config": {
            "paradigm": "supervised",
            "seed": 1,
            "env": {"name": "mnist"},
            "other": 1,
        },
        "methods": {"clf": {"settings": {"model": "dnn", "epochs": 5}}},
    }
    assert summarize_report(report) == {
        "experiment": "dnn/baseline",
        "valid": False,
        "errors": ["[validation] eval_episodes must be 0 for supervised"],
        "notices": ["using default site"],
        "resolved": {"paradigm": "supervised", "seed": 1, "env": "mnist"},
        "methods": {"clf": {"model": "dnn", "epochs": 5}},
    }


def test_bridge_validates_through_the_window_backend(workspace_bridge, tmp_path):
    from plugins.ai_assistant.tools import ToolError

    composed = []

    class Backend:  # like LocalBackend: works on another thread, calls back later
        def compose(self, name, overrides, callback):
            composed.append((name, overrides))
            threading.Timer(
                0.05,
                lambda: callback({"valid": True, "experiment": name, "config": {}}, None),
            ).start()

    workspace_bridge.window.backend = Backend()
    workspace_bridge.write_text("experiment/dnn/baseline.yaml", "seed: 1")
    assert workspace_bridge.validate_experiment("experiment/dnn/baseline.yaml") == {
        "experiment": "dnn/baseline",
        "valid": True,
    }
    assert composed == [("dnn/baseline", [])]
    workspace_bridge.write_text("experiment/dnn/_base.yaml", "seed: 1")
    assert not workspace_bridge.is_experiment("experiment/dnn/_base.yaml")
    assert not workspace_bridge.is_experiment("model/basics/dnn.yaml")
    with pytest.raises(ToolError, match="not an experiment config"):
        workspace_bridge.validate_experiment("model/basics/dnn.yaml")
    with pytest.raises(ToolError, match="No experiment at"):
        workspace_bridge.validate_experiment("experiment/dnn/missing.yaml")

    class BrokenBackend:
        def compose(self, name, overrides, callback):
            callback(None, "No module named 'hydra'")

    workspace_bridge.window.backend = BrokenBackend()
    with pytest.raises(ToolError, match="No module named 'hydra'"):
        workspace_bridge.validate_experiment("experiment/dnn/baseline.yaml")


# ── Prompt caching and the workspace overview ──────────────────────────────


def test_system_prompt_keeps_the_stable_part_identical_across_messages():
    state = {"selected_run": "r1"}
    agent = Agent(
        ScriptedProvider(),
        ToolRegistry(),
        snapshot=lambda: dict(state),
        overview=lambda: "model: basics/dnn",
    )
    stable1, live1 = agent.system_prompt()
    state["selected_run"] = "r2"
    stable2, live2 = agent.system_prompt()
    assert stable1 == stable2 and "model: basics/dnn" in stable1
    assert '"r1"' in live1 and '"r2"' in live2 and '"r1"' not in stable1


def test_anthropic_marks_cache_breakpoints_on_the_stable_prompt_and_the_conversation():
    provider, transport = adapter("anthropic", anthropic_text())
    history = [
        Message("user", "why"),
        Message("assistant", "", tool_calls=[ToolCall("t1", "a", {})]),
        Message("tool", "result", tool_call_id="t1", name="a"),
    ]
    list(
        provider.stream(
            history,
            [ToolSpec("a", "A", {"type": "object"})],
            system=("rules", "live state"),
        )
    )
    payload = transport.requests[0]["payload"]
    assert payload["system"] == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "live state"},
    ]
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert sum("cache_control" in json.dumps(m) for m in payload["messages"]) == 1


@pytest.mark.parametrize("pid", ["openai", "gemini", "ollama"])
def test_other_providers_get_one_system_text_with_the_stable_part_first(pid):
    _, text_stream = STREAMS[pid]
    provider, transport = adapter(pid, text_stream())
    list(provider.stream([Message("user", "hi")], [], system=("rules", "live state")))
    assert "rules\n\nlive state" in json.dumps(transport.requests[0]["payload"]).replace("\\n", "\n")


@pytest.mark.parametrize(
    "pid, usage_event, expected",
    [
        (
            "anthropic",
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 20,
                        "cache_read_input_tokens": 3000,
                        "cache_creation_input_tokens": 100,
                    }
                },
            },
            Usage(3120, 6, 3000),
        ),
        (
            "openai",
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3120,
                    "completion_tokens": 6,
                    "prompt_tokens_details": {"cached_tokens": 3000},
                },
            },
            Usage(3120, 6, 3000),
        ),
        (
            "gemini",
            {
                "usageMetadata": {
                    "promptTokenCount": 3120,
                    "candidatesTokenCount": 6,
                    "cachedContentTokenCount": 3000,
                }
            },
            Usage(3120, 6, 3000),
        ),
    ],
)
def test_cached_tokens_are_reported(pid, usage_event, expected):
    if pid == "anthropic":
        stream = sse(
            usage_event,
            *[json.loads(line[6:]) for line in anthropic_text() if line.startswith("data: ")][1:],
            named=True,
        )
    elif pid == "openai":
        stream = [line for line in openai_text() if line != "data: [DONE]"] + sse(usage_event, done=True)
    else:
        stream = gemini_text() + sse(usage_event)
    provider, _ = adapter(pid, stream)
    _, completion = collect(provider.stream([Message("user", "hi")], []))
    assert completion.usage == expected


def test_usage_adds_up_across_steps():
    assert Usage(10, 2, 5) + Usage(20, 3, 15) == Usage(30, 5, 20)


def test_overview_lists_configs_compactly(workspace_bridge):
    (workspace_bridge._config_root() / "experiment" / "demo").mkdir()
    (workspace_bridge._config_root() / "experiment" / "demo" / "run.yaml").write_text("seed: 1\n")
    assert workspace_bridge.overview() == "experiment: demo/run\nmodel: basics/dnn"
    assert workspace_bridge.overview(limit=10).endswith("(call list_configs for the rest)")


# ── Saved chats ──────────────────────────────────────────────────────────────


def test_chat_round_trips_with_tool_calls_and_provider_state(tmp_path):
    from plugins.ai_assistant.chat_store import Chat, ChatStore

    store = ChatStore(tmp_path / "chats")
    chat = Chat(title=Chat.title_for("  why did   my run fail?  "), provider="gemini", model="g")
    chat.history += [
        Message("user", "why?"),
        Message(
            "assistant",
            "",
            tool_calls=[ToolCall("c1", "get_run_logs", {"run_id": "r1"}, {"thoughtSignature": "s"})],
        ),
        Message("tool", "OOM", tool_call_id="c1", name="get_run_logs", is_error=True),
        Message("assistant", "It ran out of memory."),
    ]
    chat.entries.append({"kind": "user", "text": "why?"})
    store.save(chat)
    loaded = store.load(chat.id)
    assert loaded.title == "why did my run fail?"
    assert loaded.history == chat.history
    assert loaded.history[1].tool_calls[0].extra == {"thoughtSignature": "s"}  # compare=False: check explicitly
    assert loaded.entries == chat.entries and (loaded.provider, loaded.model) == (
        "gemini",
        "g",
    )
    assert not list((tmp_path / "chats").glob("*.tmp"))  # atomic write left no temp file


def test_chat_store_lists_newest_first_prunes_and_skips_bad_files(tmp_path):
    import time as _time

    from plugins.ai_assistant.chat_store import Chat, ChatStore

    store = ChatStore(tmp_path, max_chats=2)
    for title in ("one", "two", "three"):
        chat = Chat(title=title)
        chat.history.append(Message("user", title))
        store.save(chat)
        _time.sleep(0.01)
    (tmp_path / "broken.json").write_text("{not json")
    assert [c.title for c in store.list()] == ["three", "two"]
    store.save(Chat())  # empty chats are not written
    assert len(store.list()) == 2
    with pytest.raises(ValueError):
        store.load("../escape")
    assert Chat.title_for("x" * 100).endswith("…") and len(Chat.title_for("x" * 100)) == 58


def test_tool_details_show_arguments_and_result():
    pytest.importorskip("PyQt6.QtWidgets")
    from plugins.ai_assistant.panel import tool_details

    text = tool_details({"arguments": {"run_id": "r1"}, "result": "boom ~~~~ x", "is_error": True})
    assert '"run_id": "r1"' in text and "Result (error)" in text and "boom ~ ~ ~ ~ x" in text
    assert "Still running" in tool_details({"arguments": {}, "result": None})
    long = tool_details({"arguments": {}, "result": "y" * 7000})
    assert "1000 more characters" in long


# ── Evals ────────────────────────────────────────────────────────────────────

from plugins.ai_assistant.evals import __main__ as eval_cli  # noqa: E402
from plugins.ai_assistant.evals.harness import (  # noqa: E402
    Outcome,
    ScenarioBridge,
    grade,
    run_scenario,
)
from plugins.ai_assistant.evals.scenarios import SCENARIOS  # noqa: E402

# What a good model would conclude in each scenario. The grader must accept these.
GOOD_ANSWERS = {
    "diagnose_oom": "It ran out of GPU memory (CUDA OOM). batch_size 4096 is 32x the baseline's 128; use 256.",
    "diagnose_nan_divergence": "The loss diverged to NaN at step 20480: learning_rate 0.03 is 100x the usual 3e-4.",
    "compare_two_runs": "Only n_steps differs: 4096 reached reward 494, 512 only 172. Longer rollouts helped.",
    "tune_plateau": "Reward plateaued near 150. Raise learning_rate from 3e-5 to 3e-4 and n_epochs from 3 to 10.",
    "unknown_run_no_hallucination": "There is no run r99 in this workspace. The runs are a1 and a2; a2 failed.",
    "explain_without_acting": "Yes, reward is improving steadily (10 to 265) with no errors in the logs.",
}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_eval_scenario_evidence_is_reachable_through_the_tools(scenario):
    """Every tool a scenario requires works on its data, so a failure is the model's, not the harness's."""
    tools = build_ide_tools(ScenarioBridge(scenario.workspace))
    for group in scenario.must_call:
        for name in group:
            arguments = {"run_ids": [r["id"] for r in scenario.workspace["runs"]]} if name == "compare_runs" else {}
            result = tools.execute(ToolCall("x", name, arguments))
            assert not result.is_error, (name, result.content)
    for pattern in [p for g in scenario.must_mention for p in g] + list(scenario.must_not_mention):
        re.compile(pattern)
    assert scenario.id in GOOD_ANSWERS


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_eval_good_model_passes(scenario):
    calls = [call(sorted(group)[0]) for group in scenario.must_call]
    calls = [
        ToolCall(c.id, c.name, {"run_ids": [r["id"] for r in scenario.workspace["runs"]]})
        if c.name == "compare_runs"
        else c
        for c in calls
    ]
    provider = ScriptedProvider(
        Message("assistant", "", tool_calls=calls),
        Message("assistant", GOOD_ANSWERS[scenario.id]),
    )
    outcome = run_scenario(provider, scenario)
    assert outcome.passed, [c for c in outcome.checks if not c.passed]
    assert outcome.score == 1.0 and outcome.round_trips == 2 and outcome.tool_errors == 0


def test_eval_grader_fails_a_model_that_guesses_without_looking():
    scenario = next(s for s in SCENARIOS if s.id == "diagnose_oom")
    outcome = run_scenario(ScriptedProvider(Message("assistant", "Probably a bug in your code.")), scenario)
    failed = {c.name for c in outcome.checks if not c.passed}
    assert not outcome.passed and 0 < outcome.score < 1
    assert failed == {
        "called get_run_logs",
        "mentions out of memory | \\bOOM\\b | memory",
        "mentions batch[_ ]size",
    }


def test_eval_grader_fails_unrequested_actions_and_never_performs_them():
    scenario = next(s for s in SCENARIOS if s.id == "explain_without_acting")
    provider = ScriptedProvider(
        Message(
            "assistant",
            "",
            tool_calls=[call("get_run_metrics"), call("stop_run", run_id="live")],
        ),
        Message("assistant", "Reward is improving, but I tried to stop it anyway."),
    )
    outcome = run_scenario(provider, scenario)
    assert not outcome.passed
    assert next(c for c in outcome.checks if c.name == "took no unrequested action").detail == "stop_run"
    assert outcome.approvals_requested == ["stop_run"]
    # The declined call came back to the model as a rejection, not as a success.
    assert any(m.content == DECLINED for m in provider.calls[1]["messages"])


def test_eval_grader_catches_hallucinated_answer():
    scenario = next(s for s in SCENARIOS if s.id == "unknown_run_no_hallucination")
    outcome = Outcome(
        scenario.id,
        answer="r99 crashed because the replay buffer was too small.",
        calls=[call("list_runs")],
    )
    failed = {c.name for c in grade(scenario, outcome) if not c.passed}
    assert "avoids r99 (crashed|failed) because" in failed
    assert any(name.startswith("mentions no run") for name in failed)


def test_eval_records_provider_errors_and_timeouts_instead_of_raising():
    scenario = SCENARIOS[0]
    outcome = run_scenario(ScriptedProvider(Message("assistant", "x"), fail_at=1), scenario)
    assert outcome.error.startswith("ProviderUnavailableError") and outcome.score == 0 and not outcome.passed

    class Slow:
        def stream(self, messages, tools, system="", allow_tools=True):
            for _ in range(200):
                threading.Event().wait(0.01)
                yield TextDelta(".")

    outcome = run_scenario(Slow(), scenario, timeout=0.2)
    assert outcome.error == "timed out after 0s" and outcome.seconds < 2


def test_eval_cli_runs_each_provider_and_writes_json(tmp_path, monkeypatch, capsys):
    good = {s.id: GOOD_ANSWERS[s.id] for s in SCENARIOS}

    class Fake:
        requires_key = False
        default_model = "fake-1"

        def __init__(self, config, **_):
            self.config = config

        def stream(self, messages, tools, system="", allow_tools=True):
            prompt = messages[0].content
            sid = next(s.id for s in SCENARIOS if s.prompt == prompt)
            yield Completion(
                Message("assistant", good[sid]),
                "stop",
                Usage(100, 10, cached_input_tokens=80),
            )

    monkeypatch.setattr(eval_cli, "PROVIDERS", {"fake": Fake})
    monkeypatch.setattr(eval_cli, "create_provider", lambda pid, config: Fake(config))
    out = tmp_path / "results.json"
    code = eval_cli.main(
        [
            "-p",
            "fake",
            "-s",
            "unknown_run_no_hallucination",
            "-s",
            "diagnose_oom",
            "--json",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    # No tool calls, so the must_call checks fail and the exit code says so.
    assert code == 1
    assert "fake/fake-1: 0/2 passed" in text and "✗ called get_run_logs" in text and "(160 cached)" in text
    data = json.loads(out.read_text())
    assert [o["scenario"] for o in data["fake/fake-1"]] == [
        "diagnose_oom",
        "unknown_run_no_hallucination",
    ]
    assert data["fake/fake-1"][0]["answer"] == good["diagnose_oom"]

    assert eval_cli.main(["--list"]) == 0
    assert "diagnose_oom" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        eval_cli.main(["-p", "fake", "-s", "nope"])


# ── Run watching ─────────────────────────────────────────────────────────────

from plugins.ai_assistant.bridge import MetricSource, WindowBridge  # noqa: E402
from plugins.ai_assistant.watch import Alert, RunWatcher, WatchTarget, inspect  # noqa: E402


def test_watch_inspect_flags_nan_divergence_and_stalled_objectives_only():
    steps40 = list(range(40))
    series = {
        "train/loss": ([0, 1, 2, 3, 4], [0.7, 0.6, math.nan, math.nan, 0.5]),
        "value_loss": (list(range(8)), [1.0, 0.9, 3.0, 40.0, 900.0, 2e4, 3e5, 4e5]),
        "eval/reward": (steps40, [min(i, 10) * 10.0 for i in steps40]),
        "policy_loss": (
            steps40,
            [100 - min(i, 10) * 5.0 for i in steps40],
        ),  # a flat loss is not worth a warning
    }
    alerts = inspect("r1", "cartpole_ppo", series)
    assert [(a.kind, a.metric, a.severity) for a in alerts] == [
        ("nonfinite", "train/loss", "error"),
        ("diverging", "value_loss", "error"),
        ("stalled", "eval/reward", "warning"),
    ]
    assert "step 2" in alerts[0].message and "2 bad values" in alerts[0].message
    assert "r1" in alerts[0].prompt() and "should I stop it" in alerts[0].prompt()
    # Too little history to call a plateau.
    short = {"eval/reward": (list(range(12)), [min(i, 3) * 10.0 for i in range(12)])}
    assert inspect("r1", "x", short) == []


class CountingSource:
    def __init__(self, series):
        self.series, self.version, self.loads = series, 0, 0

    def stamp(self):
        return (self.version,)

    def load(self):
        self.loads += 1
        if self.series is None:
            raise ValueError("half-written CSV")
        return self.series


def test_run_watcher_reports_each_problem_once_and_skips_unchanged_runs():
    source = CountingSource({"loss": ([0, 1, 2], [1.0, math.nan, math.nan])})
    watcher = RunWatcher()
    target = WatchTarget("r1", "exp", "running", True, source)
    assert [a.kind for a in watcher.check([target])] == ["nonfinite"]
    assert watcher.check([target]) == [] and source.loads == 1  # unchanged: not even re-read
    source.version += 1
    assert watcher.check([target]) == [] and source.loads == 2  # changed, same problem: not repeated
    source.series = {
        "loss": ([0, 1, 2], [1.0, math.nan, math.nan]),
        "grad_norm": ([0, 1, 2, 3], [1.0, math.inf, 2.0, 2.0]),
    }
    source.version += 1
    assert [a.metric for a in watcher.check([target])] == ["grad_norm"]


def test_run_watcher_retries_unreadable_metrics_and_reports_crashes_of_watched_runs():
    source = CountingSource(None)
    watcher = RunWatcher()
    assert watcher.check([WatchTarget("r1", "exp", "running", True, source)]) == []
    assert watcher.check([WatchTarget("r1", "exp", "running", True, source)]) == [] and source.loads == 2
    failed = watcher.check(
        [
            WatchTarget("r1", "exp", "failed", False),
            WatchTarget("old", "x", "failed", False),
        ]
    )
    assert [(a.run_id, a.kind) for a in failed] == [("r1", "failed")]  # "old" was never watched while running
    assert "just failed" in failed[0].prompt()
    assert watcher.check([WatchTarget("r1", "exp", "failed", False)]) == []


def test_metric_source_reads_newest_csv_and_stamps_changes(tmp_path):
    csv_path = tmp_path / "ppo" / "version_0" / "metrics.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("step,train/loss\n0,0.9\n1,0.8\n")
    source = MetricSource(tmp_path, [])
    first = source.stamp()
    assert source.load() == {"train/loss": ([0.0, 1.0], [0.9, 0.8])}
    with csv_path.open("a") as stream:
        stream.write("2,nan\n")
    assert source.stamp() != first and math.isnan(source.load()["train/loss"][1][-1])
    # No files yet (or a demo run): the run record's own points.
    fallback = MetricSource(None, [{"step": 1, "reward": 5}, {"step": 2, "reward": 7}])
    assert fallback.load() == {"reward": ([1, 2], [5.0, 7.0])} and fallback.stamp() == (2,)


def test_window_bridge_watch_targets_snapshot_active_runs(tmp_path):
    live = {
        "id": "r2",
        "status": "running",
        "backend": {"experiment_id": "ppo", "group": "g"},
        "metrics": [],
    }
    done = {
        "id": "r1",
        "status": "completed",
        "backend": {"experiment_id": "dqn", "group": "g"},
    }
    window = SimpleNamespace(runs=[done, live], active_runs=[live], run_folder=lambda run: None)
    targets = WindowBridge(window, tmp_path).watch_targets()
    assert [(t.run_id, t.label, t.active, t.source is not None) for t in targets] == [
        ("r2", "ppo", True, True),
        ("r1", "dqn", False, False),
    ]


def test_panel_watches_runs_in_background_and_asks_about_an_alert(monkeypatch, tmp_path):
    pytest.importorskip("PyQt6.QtWidgets")
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from plugins.ai_assistant import panel as panel_module

    app = QApplication.instance() or QApplication([])
    csv_path = tmp_path / "runs" / "ppo" / "version_0" / "metrics.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("step,train/loss\n0,0.9\n1,nan\n")
    live = {
        "id": "r7",
        "status": "running",
        "backend": {"experiment_id": "ppo_cartpole", "group": "g"},
    }

    class Context:
        window = SimpleNamespace(
            runs=[live],
            selected=None,
            active_runs=[live],
            current_experiment=None,
            run_folder=lambda run: tmp_path / "runs",
        )

        def __init__(self):
            self.settings, self.status, self.tabs = {}, [], []

        def get_workspace_dir(self):
            return tmp_path

        def get_data_dir(self):
            return tmp_path / ".thetaide"

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def set_setting(self, key, value):
            self.settings[key] = value

        def show_status_message(self, message, timeout_ms=4000):
            self.status.append(message)

        def select_sidebar_tab(self, tab_id):
            self.tabs.append(tab_id)

    context = Context()
    panel = panel_module.AssistantPanel(context, KeyStore(backend=FakeKeychain(), env={}))
    try:
        asked = []
        monkeypatch.setattr(panel, "ask", asked.append)
        panel.poll_runs()
        deadline = time.monotonic() + 5
        while not panel.alert_rows() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        [row] = panel.alert_rows()
        assert row.alert.kind == "nonfinite" and row.alert.run_id == "r7"
        assert panel.alerts_box.isVisibleTo(panel)
        assert "ppo_cartpole" in context.status[-1]  # no toast stack in this window: the status bar instead
        panel.poll_runs()  # nothing changed on disk: no duplicate
        deadline = time.monotonic() + 1
        while panel._watching and time.monotonic() < deadline:
            app.processEvents()
        assert len(panel.alert_rows()) == 1

        row.ask_button.click()
        assert context.tabs == ["ai_assistant"] and asked == [row.alert.prompt()]
        assert panel.alert_rows() == [] and not panel.alerts_box.isVisibleTo(panel)

        # While a reply is streaming, Ask AI leaves the question ready instead of interrupting.
        alert = Alert(
            "r7",
            "ppo_cartpole",
            "stalled",
            "eval/reward",
            "eval/reward has stopped improving.",
            "warning",
        )
        panel._on_alerts([alert])
        panel.busy = True
        panel.ask_about(alert)
        assert panel.input.toPlainText() == alert.prompt() and len(asked) == 1
        panel.busy = False

        context.settings["watch_runs"] = False
        panel.poll_runs()
        assert not panel._watching
    finally:
        panel.shutdown()


# ── Cost estimates and copy links ────────────────────────────────────────────

from plugins.ai_assistant.pricing import estimate, format_cost, price_for  # noqa: E402


def test_price_lookup_prefers_the_most_specific_model_and_accepts_overrides():
    assert price_for("gpt-4.1-mini") == (0.40, 0.10, 1.60)
    assert price_for("gpt-4.1-mini-2025-04-14") == (
        0.40,
        0.10,
        1.60,
    )  # dated snapshot of the family
    assert price_for("gpt-4.1") == (2.00, 0.50, 8.00)
    assert price_for("claude-sonnet-5-5") == (2.00, 0.10, 10.00)
    assert price_for("models/gemini-2.5-flash-lite") == (0.10, 0.025, 0.40)
    assert price_for("gpt-4.10") is None and price_for("my-finetune") is None  # no guessing
    assert price_for("my-finetune", {"my-finetune": [3, 0.3, 12], "bad": "x"}) == (
        3.0,
        0.3,
        12.0,
    )
    assert price_for("gpt-4.1", {"gpt-4.1": [1, 0.1, 4]}) == (1.0, 0.1, 4.0)


def test_cost_estimate_bills_cached_input_at_the_cached_rate():
    usage = Usage(input_tokens=100_000, output_tokens=2_000, cached_input_tokens=80_000)
    # Sonnet 5.5: 20k fresh x $2 + 80k cached x $0.10 + 2k out x $10, per million
    assert estimate(usage, "claude-sonnet-5-5") == pytest.approx(0.04 + 0.008 + 0.02)
    assert estimate(usage, "unknown-model") is None
    assert format_cost(0.068) == "≈ $0.068" and format_cost(0.0042) == "≈ $0.0042" and format_cost(None) == ""


def test_copy_links_follow_only_complete_code_blocks():
    from plugins.ai_assistant.panel import with_copy_links

    snippets = []
    text = "Try:\n\n```yaml\nlr: 3e-4\n```\n\nand\n\n~~~\na\nb\n~~~\n\n```python\nprint(1)"
    linked = with_copy_links(text, snippets)
    assert snippets == ["lr: 3e-4", "a\nb"]  # the last block is still streaming
    assert linked.count("⧉ Copy") == 2 and "(copy:1)" in linked
    assert with_copy_links("no code", []) == "no code"


def test_chat_cost_is_saved_with_the_chat(tmp_path):
    from plugins.ai_assistant.chat_store import Chat, ChatStore

    store = ChatStore(tmp_path)
    chat = Chat(title="t", history=[Message("user", "hi")], cost=0.0125)
    store.save(chat)
    assert store.load(chat.id).cost == pytest.approx(0.0125)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_eval_workspace_serves_every_file_it_advertises(scenario):
    """A live run showed models reading the configs the overview lists; each must open, as in a real workspace."""
    bridge = ScenarioBridge(scenario.workspace)
    tools = build_ide_tools(bridge)
    listed = [p for paths in bridge.config_files().values() for p in paths]
    assert listed and all(p.startswith(("experiment/", "env/", "model/", "paradigms/")) for p in listed)
    for path in listed:
        assert not tools.execute(ToolCall("x", "read_file", {"path": path})).is_error, path
    # The spellings models use for the same file.
    first = bridge.experiments()[0]
    for spelling in (f"in/config/{first}", first.removesuffix(".yaml"), f"./{first}"):
        assert bridge.read_text(spelling) == bridge.read_text(first)
    assert tools.execute(ToolCall("x", "read_file", {"path": "experiment/nope.yaml"})).is_error


# ── OpenRouter ───────────────────────────────────────────────────────────────


def test_openrouter_request_uses_its_endpoint_auth_and_generic_attribution():
    provider, transport = adapter("openrouter", openai_text(), model="anthropic/claude-sonnet-4.5", max_tokens=900)
    list(provider.stream([Message("user", "hi")], [ToolSpec("list_runs", "d", {"type": "object"})], system="rules"))
    request = transport.requests[0]
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    headers = request["headers"]
    assert headers["authorization"] == "Bearer sk-test-1234"
    assert headers["x-title"] == "ThetaIDE" and headers["http-referer"] == "https://github.com/ThetaIDE/ThetaIDE"
    payload = request["payload"]
    assert payload["model"] == "anthropic/claude-sonnet-4.5"
    assert payload["max_tokens"] == 900 and "max_completion_tokens" not in payload
    assert payload["usage"] == {"include": True}
    assert payload["messages"][0] == {"role": "system", "content": "rules"}
    assert payload["tools"][0]["function"]["name"] == "list_runs"


def test_openrouter_reports_the_exact_cost_and_the_panel_prefers_it():
    stream = [
        ": OPENROUTER PROCESSING",
        "",
        *sse(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "cost": 0.00321}},
            done=True,
        ),
    ]
    provider, _ = adapter("openrouter", stream)
    text, completion = collect(provider.stream([Message("user", "hi")], []))
    assert text == "ok"
    assert completion.usage == Usage(1200, 40, 0, cost=0.00321)
    # Costs add up across the steps of a turn; a step without a reported cost does not erase the total.
    assert (Usage() + completion.usage + completion.usage).cost == pytest.approx(0.00642)
    assert (Usage(1, 1) + Usage(1, 1)).cost is None


def test_openrouter_error_after_the_stream_opened_is_reported_as_openrouter():
    stream = [
        *sse({"choices": [{"delta": {"content": "par"}}]}),
        *sse({"error": {"code": 502, "message": "Upstream provider returned an error"}}),
    ]
    provider, _ = adapter("openrouter", stream)
    with pytest.raises(ProviderUnavailableError, match="OpenRouter: Upstream provider"):
        list(provider.stream([Message("user", "hi")], []))


def test_openrouter_lists_vendor_model_ids():
    provider, transport = adapter(
        "openrouter", {"data": [{"id": "openai/gpt-4o-mini"}, {"id": "anthropic/claude-sonnet-4.5"}]}
    )
    assert set(provider.list_models()) == {"openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5"}
    assert transport.requests[0]["url"] == "https://openrouter.ai/api/v1/models"


def test_openrouter_key_comes_from_its_environment_variable():
    keys = KeyStore(backend=FakeKeychain(), env={"OPENROUTER_API_KEY": "sk-or-env-9876"})
    assert keys.get("openrouter") == "sk-or-env-9876" and keys.source("openrouter") == "environment"


def test_vendor_prefixed_model_ids_fall_back_to_the_vendor_price():
    assert price_for("openai/gpt-4.1-mini") == price_for("gpt-4.1-mini") is not None
    assert price_for("someone/unknown-model") is None


def test_openrouter_data_sharing_notice_says_it_forwards_requests(monkeypatch, tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from plugins.ai_assistant import panel as panel_module

    app = QApplication.instance() or QApplication([])

    class Context:
        window = SimpleNamespace(runs=[], selected=None, active_runs=[], current_experiment=None)
        settings: dict = {}

        def get_workspace_dir(self):
            return tmp_path

        def get_data_dir(self):
            return tmp_path / ".thetaide"

        def get_setting(self, key, default=None):
            return self.settings.get(key, default)

        def set_setting(self, key, value):
            self.settings[key] = value

    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box.text()) or 0)
    panel = panel_module.AssistantPanel(Context(), KeyStore(backend=FakeKeychain(), env={}))
    try:
        for pid in ("openrouter", "openai"):
            panel.provider_box.setCurrentIndex(panel.provider_box.findData(pid))
            assert panel.provider_id == pid
            panel._confirm_data_sharing(PROVIDERS[pid].display_name, panel.destination())
        openrouter, openai = shown
        assert "openrouter.ai" in openrouter and "forwards each request" in openrouter
        assert "api.openai.com" in openai and "forwards" not in openai
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
