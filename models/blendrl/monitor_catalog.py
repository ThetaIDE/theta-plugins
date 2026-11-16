"""Standalone config-to-metric catalog; deliberately has no ML dependencies."""
from itertools import combinations
import re


def expert_names(module_types):
    return [f'{re.sub(r"[^a-z0-9_]+", "_", str(kind).lower()).strip("_") or "module"}_{index}'
            for index, kind in enumerate(module_types)]


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


def routing_output_names(types, *, agreement=True):
    names = ["entropy", "entropy_normalized", "mixed_share"]
    experts = expert_names(types)
    for expert in experts:
        names += [f"{expert}/{key}" for key in ("weight_mean", "weight_std", "dominant_share", "exclusive_share")]
    names += [f"{kind}/weight_mean" for kind in ("neural", "logic") if kind in types]
    if agreement:
        for first, second in combinations(experts, 2):
            names += [f"agreement/{first}__{second}/{key}" for key in ("top_action", "js_divergence")]
    return names


def metric_description(name):
    leaf = name.rsplit("/", 1)[-1]
    descriptions = {
        "entropy": "Mean entropy of the routing weights in nats. Larger values mean routing is spread across more experts.",
        "entropy_normalized": "Routing entropy divided by log(number of experts): 0 selects one expert, 1 weights all experts equally. Zero for a single expert.",
        "mixed_share": "Fraction of sampled observations where no expert receives at least 90% of the routing weight.",
        "weight_mean": "Mean routing weight assigned to this expert across sampled observations.",
        "weight_std": "Population standard deviation of this expert's routing weight across sampled observations.",
        "dominant_share": "Fraction of sampled observations where this expert has the largest routing weight. Ties select the first expert.",
        "exclusive_share": "Fraction of sampled observations where this expert receives at least 90% of the routing weight.",
        "top_action": "Fraction of sampled observations where the two experts choose the same highest-probability action.",
        "js_divergence": "Mean Jensen-Shannon divergence between the two experts' action distributions, in nats. Zero means identical distributions.",
        "coverage": "Fraction of sampled observations where this logic expert produces an active action output.",
    }
    text = descriptions[leaf]
    if name.removeprefix("soft_gate/") in ("neural/weight_mean", "logic/weight_mean"):
        text = "Mean total routing weight assigned to all experts of this type across sampled observations."
    if name.startswith("soft_gate/"):
        text = "Computed from the soft routing probabilities before hard Gumbel selection. " + text
    return text


def describe(config, context):
    types = module_types(config)
    names = routing_output_names(types)
    names += [f"logic_{index}/coverage" for index, kind in enumerate(types) if kind == "logic"]
    if config.get("blend_function") == "gumbel_softmax" and config.get("actor_mode", "hybrid") == "hybrid":
        names += ["soft_gate/" + name for name in routing_output_names(types, agreement=False)]
    return [{"name": f"blend/{phase}/{name}", "label": name.replace("/", " · ").replace("_", " "),
             "description": f"{'Training' if phase == 'train' else 'Evaluation'} observations. " + metric_description(name),
             "group": f"Routing · {phase.title()}", "requires": ["model_observations.v1"]}
            for phase in ("train", "eval") for name in names]
