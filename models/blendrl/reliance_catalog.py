INTERVENTIONS = ("Interventions: <expert>_uniform_output replaces that expert's action distribution with a uniform one; "
                 "<expert>_removed_renormalized zeroes its gate weight and renormalizes the others; uniform_gate weights "
                 "every expert equally; mean_gate applies the average gate weights to every state.")
CONTRAST = "the action contrast log P(action) − log P(reference_action) on the fixed probe states"


def describe(config, context):
    requires = ["probe_forward.v1", "module_interventions.v1", "post_update.v1", "blendrl.expert_interventions.v1", "policy_probabilities.v1"]
    group = f"BlendRL causal reliance · {config['instance']}"
    outputs = [{"name": f"interp/{config['instance']}/{key}", "label": label, "description": description,
                "group": group, "requires": requires}
               for key, label, description in (
                   ("mean_absolute_effect", "Mean absolute intervention effect",
                    f"Absolute change in {CONTRAST}, unmodified policy minus intervened policy, averaged over probes and "
                    f"over all expert and gate interventions. Larger values mean the policy depends more on its experts "
                    f"and gate. {INTERVENTIONS}"),
                   ("max_action_flip_rate", "Max action flip rate",
                    f"Largest fraction of fixed probe states whose highest-probability action changes under any single "
                    f"expert or gate intervention. {INTERVENTIONS}"))]

    for key, label, kind, schema, axes, semantics, description in (
        ("routing", "Expert routing and action probabilities", "matrix", "theta.activation-matrix/v1",
         ["probe_sample", "expert_or_action"], "Gate/expert/policy probabilities on fixed probes; probability units",
         "Heatmaps on the fixed probe states: the gate's routing weight for each expert, each expert's action "
         "probabilities, and the blended policy's action probabilities."),
        ("intervention_effects", "Expert/gate intervention effects", "matrix", "theta.activation-matrix/v1",
         ["probe_sample", "intervention"], "Signed change in action log-probability contrast; unmodified policy baseline",
         f"Probe × intervention heatmap of the signed change in {CONTRAST}, unmodified minus intervened. {INTERVENTIONS}"),
        ("interventions", "Expert/gate reliance map", "intervention_graph", "theta.intervention-graph/v1",
         ["intervention", "behavior"], "Mean absolute effect on action log-probability contrast over all fixed probes; not held-out or rollout reward",
         f"Graph linking each intervention to the action contrast, weighted by its mean absolute effect over all fixed "
         f"probes. Hover a node for the signed effect and action-flip rate. Not a rollout reward effect. {INTERVENTIONS}"),
    ):
        outputs.append({"name": f"interp/{config['instance']}/{key}", "label": label, "description": description,
                        "kind": kind, "group": group, "requires": requires, "visual_schema": "theta.visual-output/v1",
                        "artifact_schema": schema, "axes": axes, "value_semantics": semantics,
                        "retention": "snapshots"})
    return outputs
