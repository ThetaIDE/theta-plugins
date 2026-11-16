"""Architecture-owned BlendRL expert and gate interventions on fixed states."""
import numpy as np
import torch

from src.app.api.monitoring import Result
from src.app.api.monitoring import action_difference, validate_params


class BlendReliance:
    @staticmethod
    def validate(params):
        validate_params(params, {"action", "reference_action"}, {"action": (0, 10000), "reference_action": (0, 10000)})

    def __init__(self, action=0, reference_action=1):
        self.validate({"action": action, "reference_action": reference_action})
        self.action, self.reference_action = action, reference_action

    def setup(self, context):
        model = context.access.target(context.spec["target"])
        if not callable(getattr(model, "monitoring_interventions", None)):
            raise ValueError("Blend reliance requires a BlendRL model exposing monitoring_interventions(obs)")
        if context.access.score_kind != "log_probabilities":
            raise ValueError("Blend reliance v1 requires an actor policy probability adapter")

    def collect(self, context):
        self.setup(context)
        model = context.access.target(context.spec["target"])
        baseline = context.access.scores(context.observations)
        reference = action_difference(baseline, self.action, self.reference_action)
        action_meanings = getattr(getattr(model, "env", None), "get_action_meanings", None)
        action_names = list(action_meanings()) if callable(action_meanings) else [f"action {i}" for i in range(baseline.shape[1])]
        needs_interventions = any(context.wants_output(key) for key in (
            "mean_absolute_effect", "max_action_flip_rate", "intervention_effects", "interventions"))
        variants, metadata = model.monitoring_interventions(context.observations) if needs_interventions else ({}, {})
        arrays, rows = {"baseline_scores": reference.cpu().numpy()}, []
        for index, (name, probabilities) in enumerate(variants.items()):
            changed = action_difference(probabilities.clamp_min(1e-12).log(), self.action, self.reference_action)
            effect = (reference - changed).cpu().numpy()
            arrays[f"effect_{index}"] = effect
            rows.append({"intervention": name, "mean_absolute_effect": float(np.abs(effect).mean()),
                         "mean_signed_effect": float(effect.mean()),
                         "standard_error": float(effect.std(ddof=1) / np.sqrt(len(effect))),
                         "action_flip_rate": float((baseline.argmax(-1) != probabilities.argmax(-1)).float().mean())})
        visuals = {}
        if context.wants_output("routing"):
            from .monitor_catalog import expert_names
            _, responses = model.collect_monitoring(context.observations, include_experts=True)
            experts = expert_names(model.module_types)
            arrays["routing_weights"] = responses["weights"].cpu().numpy()
            arrays["policy_probabilities"] = responses["action_probabilities"].cpu().numpy()
            layers = [{"id": "Routing weights", "array": "routing_weights", "feature_ids": experts}]
            for index, name in enumerate(experts):
                arrays[f"expert_{index}_probabilities"] = responses["expert_probabilities"][:, index].cpu().numpy()
                layers.append({"id": f"{name} action probabilities", "array": f"expert_{index}_probabilities",
                               "feature_ids": action_names})
            layers.append({"id": "Blended policy probabilities", "array": "policy_probabilities",
                           "feature_ids": action_names})
            for layer in layers:
                layer.update(unit="probability", value_range=[0., 1.],
                             value_semantics="Probability on fixed probe states",
                             coverage=f"All {len(layer['feature_ids'])} named columns; fixed cohort")
            visuals["routing"] = {"schema": "theta.activation-matrix/v1", "layers": layers,
                                  "sample_ids": "sample_ids"}
        if context.wants_output("intervention_effects"):
            arrays["intervention_effects"] = np.stack([arrays[f"effect_{i}"] for i in range(len(rows))], axis=1)
            visuals["intervention_effects"] = {"schema": "theta.activation-matrix/v1", "sample_ids": "sample_ids",
                "layers": [{"id": "Expert and gate counterfactuals", "array": "intervention_effects",
                            "feature_ids": [row["intervention"] for row in rows], "unit": "log-probability contrast",
                            "value_semantics": "Unmodified minus intervened action contrast",
                            "coverage": f"All {len(rows)} explicit interventions; all fixed probes; no held-out split"}]}
        if visuals:
            arrays["sample_ids"] = context.probes.sample_ids
        graph = None
        if context.wants_output("interventions"):
            graph = {"schema": "theta.intervention-graph/v1", "cohort": "all fixed probes; no held-out split",
                     "nodes": [{"id": row["intervention"], "label": row["intervention"],
                                "effect": row["mean_absolute_effect"], "selected": False, **row} for row in rows]
                               + [{"id": "behavior", "label": f"log P({action_names[self.action]}) − log P({action_names[self.reference_action]})"}],
                     "edges": [{"source": row["intervention"], "target": "behavior",
                                "kind": "measured_total_intervention_effect", "effect": row["mean_absolute_effect"],
                                "mean_signed_effect": row["mean_signed_effect"], "action_flip_rate": row["action_flip_rate"],
                                "baseline": "unmodified policy", "test": row["intervention"],
                                "cohort": "all fixed probes; no discovery/held-out split"} for row in rows],
                     "interpretation": "Fixed-state expert/gate counterfactuals against the unmodified policy, averaged over all fixed probes. These are not synaptic connections, held-out circuit validation, or rollout reward effects. Probe samples may be correlated."}
            visuals["interventions"] = {"graph": graph}
        details = {"interventions": rows, "architecture": metadata, "baseline": "unmodified BlendRL policy",
                   "cohort": "all fixed probes; no discovery/held-out split", "action": self.action,
                   "reference_action": self.reference_action, "action_names": action_names,
                   "interpretation": "Fixed-state policy interventions; these do not measure rollout reward effects."}
        if graph is not None:
            details["graph"] = graph
        metrics = {"mean_absolute_effect": float(np.mean([row["mean_absolute_effect"] for row in rows])),
                   "max_action_flip_rate": float(max(row["action_flip_rate"] for row in rows))} if rows else {}
        return Result(metrics, arrays, details, visuals)
