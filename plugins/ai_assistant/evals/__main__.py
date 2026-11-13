"""Run the eval scenarios against real providers and print a scorecard.

    python -m plugins.ai_assistant.evals                       # every provider with a key
    python -m plugins.ai_assistant.evals -p anthropic -p openai --repeat 3
    python -m plugins.ai_assistant.evals -p ollama -m qwen3 -s diagnose_oom --json out.json

Keys come from the OS keychain or the provider's environment variable, as in the IDE. This
calls paid APIs: six scenarios cost a few cents per provider with the default models.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from ..keys import KeyStore
from ..providers import PROVIDERS, ProviderConfig, create_provider
from .harness import Outcome, run_scenario
from .scenarios import SCENARIOS


def available_providers(keys: KeyStore) -> list[str]:
    return [pid for pid, cls in PROVIDERS.items() if not cls.requires_key or keys.get(pid)]


def scorecard(results: dict[str, list[Outcome]]) -> str:
    lines = []
    for label, outcomes in results.items():
        passed = sum(o.passed for o in outcomes)
        tokens_in = sum(o.usage.input_tokens for o in outcomes)
        cached = sum(o.usage.cached_input_tokens for o in outcomes)
        tokens_out = sum(o.usage.output_tokens for o in outcomes)
        seconds = sum(o.seconds for o in outcomes)
        lines += [
            "",
            f"{label}: {passed}/{len(outcomes)} passed, mean score {sum(o.score for o in outcomes) / len(outcomes):.2f}",
            f"  tokens {tokens_in:,} in ({cached:,} cached) / {tokens_out:,} out, {seconds:.0f}s",
        ]
        for o in outcomes:
            mark = "PASS" if o.passed else "FAIL"
            tools = ", ".join(c.name for c in o.calls) or "no tools"
            lines.append(f"  {mark} {o.scenario:<32} {o.score:4.2f}  {o.seconds:5.1f}s  [{tools}]")
            if o.error:
                lines.append(f"       error: {o.error}")
            for check in o.checks:
                if not check.passed:
                    lines.append(f"       ✗ {check.name}" + (f" ({check.detail})" if check.detail else ""))
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_assistant.evals", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-p",
        "--provider",
        action="append",
        choices=sorted(PROVIDERS),
        help="repeatable",
    )
    parser.add_argument("-m", "--model", help="model id (default: each provider's default)")
    parser.add_argument("--base-url", help="server URL, e.g. an OpenAI-compatible server")
    parser.add_argument(
        "-s",
        "--scenario",
        action="append",
        help="scenario id (repeatable; default all)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per scenario; models are not deterministic",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per scenario")
    parser.add_argument("--json", type=Path, help="write every outcome, with full answers, here")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        for s in SCENARIOS:
            print(f"{s.id:<32} {s.description}")
        return 0
    scenarios = [s for s in SCENARIOS if not args.scenario or s.id in args.scenario]
    unknown = set(args.scenario or ()) - {s.id for s in scenarios}
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(sorted(unknown))}; see --list")

    if args.model and len(args.provider or ()) != 1:
        parser.error("--model needs exactly one --provider")
    keys = KeyStore()
    providers = args.provider or available_providers(keys)
    if not providers:
        parser.error("no provider has a key; set e.g. ANTHROPIC_API_KEY or pass -p ollama")

    results: dict[str, list[Outcome]] = {}
    for pid in providers:
        cls = PROVIDERS[pid]
        config = ProviderConfig(
            model=args.model or cls.default_model,
            api_key=keys.get(pid),
            base_url=args.base_url,
        )
        provider = create_provider(pid, config)
        label = f"{pid}/{config.model}"
        outcomes = results.setdefault(label, [])
        for scenario in scenarios:
            for _ in range(args.repeat):
                print(f"{label}: {scenario.id} …", file=sys.stderr, flush=True)
                outcomes.append(run_scenario(provider, scenario, timeout=args.timeout))

    print(scorecard(results))
    if args.json:
        args.json.write_text(
            json.dumps(
                {label: [o.to_dict() for o in outs] for label, outs in results.items()},
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")
    return 0 if all(o.passed for outs in results.values() for o in outs) else 1


if __name__ == "__main__":
    sys.exit(main())
