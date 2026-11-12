"""Rough cost of a reply from its token counts, for the status line. An estimate, never a bill.

Prices are list prices in US dollars per million tokens: (input, cached input, output). They
change; the table records when it was last checked, unknown models get no estimate rather
than a guess, and the plugin setting "prices" overrides or extends it, e.g.
    {"my-finetune": [3.0, 0.3, 12.0]}
Cache writes (Anthropic bills them at 1.25x input) are counted as ordinary input, so an
estimate for the first message of a chat runs slightly low.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .providers.chat_types import Usage

PRICES_CHECKED = "2026-11"

# Matched by longest prefix, so "gpt-4.1-mini" wins over "gpt-4.1" and dated snapshots match their family.
PRICES: dict[str, tuple[float, float, float]] = {
    # Anthropic (checked against Anthropic's model table, November 2026)
    "claude-fable-5-2": (8.00, 0.20, 40.00),
    "claude-fable-5-1": (10.00, 1.00, 50.00),
    "claude-fable-5": (10.00, 1.00, 50.00),
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-opus-5": (5.00, 0.50, 25.00),
    "claude-opus-4": (5.00, 0.50, 25.00),
    "claude-sonnet-5-5": (2.00, 0.10, 10.00),
    "claude-sonnet-5": (2.00, 0.20, 10.00),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
    "claude-haiku-4-5": (1.00, 0.10, 5.00),
    # OpenAI (list prices as last known; check openai.com/api/pricing)
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4o": (2.50, 1.25, 10.00),
    # Google Gemini (prompts up to 200k tokens; check ai.google.dev/pricing)
    "gemini-2.5-flash-lite": (0.10, 0.025, 0.40),
    "gemini-2.5-flash": (0.30, 0.075, 2.50),
    "gemini-2.5-pro": (1.25, 0.31, 10.00),
}

FREE_PROVIDERS = frozenset({"ollama"})


def price_for(model: str, overrides: Mapping[str, Any] | None = None) -> tuple[float, float, float] | None:
    table: dict[str, tuple[float, float, float]] = dict(PRICES)
    for key, value in (overrides or {}).items():
        try:
            prices = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            continue
        if len(prices) == 3:
            table[str(key)] = prices  # type: ignore[assignment]
    model = (model or "").strip().removeprefix("models/")
    matches = [key for key in table if model == key or model.startswith(key + "-") or model.startswith(key + "@")]
    return table[max(matches, key=len)] if matches else None


def estimate(usage: Usage, model: str, overrides: Mapping[str, Any] | None = None) -> float | None:
    """US dollars for this usage, or None when the model's price is unknown."""
    prices = price_for(model, overrides)
    if prices is None:
        return None
    rate_in, rate_cached, rate_out = prices
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    return ((usage.input_tokens - cached) * rate_in + cached * rate_cached + usage.output_tokens * rate_out) / 1e6


def format_cost(dollars: float | None) -> str:
    if dollars is None:
        return ""
    if dollars < 0.01:
        return f"≈ ${dollars:.4f}"
    return f"≈ ${dollars:.3f}" if dollars < 1 else f"≈ ${dollars:.2f}"
