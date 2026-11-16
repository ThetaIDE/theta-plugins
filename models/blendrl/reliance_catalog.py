def describe(config, context):
    outputs = [{"name": f"interp/{config['instance']}/{key}", "label": key.replace("_", " "),
             "group": f"BlendRL causal reliance · {config['instance']}",
             "requires": ["probe_forward.v1", "module_interventions.v1", "post_update.v1", "blendrl.expert_interventions.v1", "policy_probabilities.v1"]}
            for key in ("mean_absolute_effect", "max_action_flip_rate")]

    for key, label, kind, schema, axes, semantics in (
        ("routing", "Expert routing and action probabilities", "matrix", "theta.activation-matrix/v1",
         ["probe_sample", "expert_or_action"], "Gate/expert/policy probabilities on fixed probes; probability units"),
        ("intervention_effects", "Expert/gate intervention effects", "matrix", "theta.activation-matrix/v1",
         ["probe_sample", "intervention"], "Signed change in action log-probability contrast; unmodified policy baseline"),
        ("interventions", "Expert/gate reliance map", "intervention_graph", "theta.intervention-graph/v1",
         ["intervention", "behavior"], "Mean absolute effect on action log-probability contrast over all fixed probes; not held-out or rollout reward"),
    ):
        outputs.append({"name": f"interp/{config['instance']}/{key}", "label": label, "kind": kind,
                        "group": f"BlendRL causal reliance · {config['instance']}",
                        "requires": outputs[0]["requires"], "visual_schema": "theta.visual-output/v1",
                        "artifact_schema": schema, "axes": axes, "value_semantics": semantics,
                        "retention": "snapshots"})
    return outputs
