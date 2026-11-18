"""Standalone config-to-metric catalog; deliberately has no ML dependencies."""
from collections import Counter
from itertools import combinations
import re

KIND_ROLES = {
    "logic": "the symbolic expert, which scores actions with differentiable logic rules over the symbolic state",
    "neural": "the neural-network expert, which scores actions directly from the observation",
}
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

GLOBAL_METRICS = {
    "entropy": ("Routing entropy", "Mean entropy of the routing weights in nats. Larger values mean routing is spread across more experts."),
    "entropy_normalized": ("Routing entropy (normalized)", "Routing entropy divided by log(number of experts): 0 selects one expert, 1 weights all experts equally. Zero for a single expert."),
    "mixed_share": ("Mixed routing share", "Fraction of sampled observations where no expert receives at least 90% of the routing weight."),
}
EXPERT_METRICS = {
    "weight_mean": ("mean weight", "Mean routing weight the gate assigns to this expert across sampled observations."),
    "weight_std": ("weight std", "Population standard deviation of this expert's routing weight across sampled observations."),
    "dominant_share": ("dominant share", "Fraction of sampled observations where this expert has the largest routing weight. Ties select the expert listed first."),
    "exclusive_share": ("exclusive share", "Fraction of sampled observations where this expert receives at least 90% of the routing weight."),
}
PAIR_METRICS = {
    "top_action": ("top-action agreement", "Fraction of sampled observations where the two experts choose the same highest-probability action."),
    "js_divergence": ("JS divergence", "Mean Jensen-Shannon divergence between the two experts' action distributions, in nats. Zero means identical distributions."),
}
COVERAGE = ("rule coverage", "Fraction of sampled observations where this logic expert's strongest rule score exceeds logic_confidence_floor, "
            "so it expresses an action preference. Otherwise it abstains and contributes the default action (withhold, noop, or uniform).")


def _kind(value):
    return re.sub(r"[^a-z0-9_]+", "_", str(value).lower()).strip("_") or "module"


def experts(module_types):
    """(metric name, kind, number) per module: a type is numbered from 1 only when it repeats."""
    kinds = [_kind(kind) for kind in module_types]
    totals, seen, result = Counter(kinds), Counter(), []
    for kind in kinds:
        seen[kind] += 1
        number = seen[kind] if totals[kind] > 1 else None
        result.append((kind if number is None else f"{kind}_{number}", kind, number))
    return result


def expert_names(module_types):
    return [name for name, _, _ in experts(module_types)]


def _title(kind, number):
    return kind.replace("_", " ").capitalize() + (f" {number}" if number else "")


def _expert(kind, number):
    return kind.replace("_", " ").capitalize() + " expert" + (f" {number}" if number else "")


def _role(kind, number):
    role = KIND_ROLES.get(kind, f"the {kind.replace('_', ' ')} expert")
    if number:
        role += f" ({ORDINALS.get(number, f'#{number}')} {kind.replace('_', ' ')} module in model order)"
    return f"{_expert(kind, number)}: {role}."


def module_types(config):
    modules = config.get("modules")
    symbolic = config.get("symbolic") or config.get("symbolic_actor")
    neural = config.get("neural") or config.get("neural_actor")
    if modules is None and (symbolic is not None or neural is not None):
        modules = []
        if symbolic is not None:
            kind = (symbolic.get("type") or symbolic.get("name") or symbolic.get("reasoner")) if isinstance(symbolic, dict) else None
            modules.append({"type": kind or config.get("reasoner", "nsfr")})
        if neural is not None:
            modules.append({"module_type": "neural"})
    if not modules:
        rules = config.get("rules", "default")
        rules = rules.split(",") if isinstance(rules, str) else rules
        return ["neural"] + ["logic"] * len(rules)
    types = [item.get("module_type", item.get("type", "neural")) for item in modules]
    return ["logic" if kind in ("nsfr", "neumann") else kind for kind in types]


def routing_outputs(types, *, agreement=True):
    """(name, label, description) for every routing summary that ``routing_metrics`` emits."""
    outputs = [(key, label, text) for key, (label, text) in GLOBAL_METRICS.items()]
    listed = experts(types)
    for name, kind, number in listed:
        outputs += [(f"{name}/{key}", f"{_expert(kind, number)} · {label}", f"{text} {_role(kind, number)}")
                    for key, (label, text) in EXPERT_METRICS.items()]
    for kind, count in Counter(kind for _, kind, _ in listed).items():
        if count > 1:
            noun = kind.replace("_", " ")
            outputs.append((f"{kind}/weight_mean", f"All {noun} experts · mean weight",
                            f"Mean total routing weight assigned to all {count} {noun} experts across sampled observations."))
    if agreement:
        for (first, kind_a, number_a), (second, kind_b, number_b) in combinations(listed, 2):
            a, b = _title(kind_a, number_a), _title(kind_b, number_b)
            outputs += [(f"agreement/{first}__{second}/{key}", f"{a} vs {b.lower()} · {label}",
                         f"{text} Compares the {_expert(kind_a, number_a).lower()} with the {_expert(kind_b, number_b).lower()}.")
                        for key, (label, text) in PAIR_METRICS.items()]
    return outputs


def describe(config, context):
    types = module_types(config)
    outputs = routing_outputs(types)
    outputs += [(f"{name}/coverage", f"{_expert(kind, number)} · {COVERAGE[0]}", f"{COVERAGE[1]} {_role(kind, number)}")
                for name, kind, number in experts(types) if kind == "logic"]
    if config.get("blend_function") == "gumbel_softmax" and config.get("actor_mode", "hybrid") == "hybrid":
        outputs += [("soft_gate/" + name, "Soft gate · " + label,
                     "Computed from the soft routing probabilities before hard Gumbel selection. " + text)
                    for name, label, text in routing_outputs(types, agreement=False)]
    return [{"name": f"blend/{phase}/{name}", "label": label,
             "description": f"{'Training' if phase == 'train' else 'Evaluation'} observations. {text}",
             "group": f"Routing · {phase.title()}", "requires": ["model_observations.v1"]}
            for phase in ("train", "eval") for name, label, text in outputs]
