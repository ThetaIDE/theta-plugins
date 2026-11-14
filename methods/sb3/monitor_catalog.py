"""Individual outputs transported by this runner; no SB3/torch imports."""

DESCRIPTIONS = {
    "episode_reward": "Total reward for each completed training episode.",
    "episode_length": "Number of environment steps in each completed training episode.",
    "reward": "Mean episode return. Training uses the recent episode window; evaluation uses the evaluation episodes.",
    "length": "Mean episode length over the recent training episode window.",
    "reward_std": "Standard deviation of evaluation episode returns; this is episode spread, not uncertainty across training seeds.",
    "fps": "Training throughput in environment steps per second.",
    "time_elapsed": "Elapsed training wall-clock time in seconds.",
    "entropy_loss": "Negative policy entropy used in the optimization objective. More negative values indicate more action randomness.",
    "policy_gradient_loss": "PPO clipped policy-gradient objective for the latest training update.",
    "policy_loss": "Policy optimization objective for the latest training update.",
    "value_loss": "Mean squared error between value predictions and target returns.",
    "loss": "Training optimization loss reported by the selected algorithm.",
    "actor_loss": "SAC policy objective combining entropy regularization and predicted action value.",
    "critic_loss": "SAC critic error against target action values.",
    "ent_coef_loss": "Objective used to learn SAC's entropy regularization coefficient.",
    "learning_rate": "Current optimizer learning rate.",
    "n_updates": "Cumulative number of optimization updates reported by the algorithm.",
    "approx_kl": "Approximate KL divergence between the old and updated policies; measures the size of a policy update.",
    "clip_fraction": "Fraction of PPO probability ratios outside the clipping interval.",
    "explained_variance": "Fraction of return variance explained by value predictions. One is perfect; zero is no better than a constant prediction; negative is worse.",
    "clip_range": "PPO policy probability-ratio clipping range around one.",
    "clip_range_vf": "PPO value-prediction clipping range in reward units.",
    "ent_coef": "Current SAC entropy regularization coefficient.",
    "std": "Mean standard deviation of the policy's continuous-action distribution.",
}


def describe(config, context):
    agent = config
    algorithm = str(agent.get("algorithm", "PPO")).rsplit("/", 1)[-1].upper()
    outputs = []

    def add(name, group):
        leaf = name.rsplit("/", 1)[-1]
        mode = next((m for m in ("greedy", "sampled") if leaf.startswith(m + "_")), None)
        description = DESCRIPTIONS[leaf[len(mode) + 1:] if mode else leaf]
        if mode:
            description = ("Deterministic highest-probability actions. " if mode == "greedy" else
                           "Stochastic actions sampled from the policy. ") + description
        outputs.append({"name": name, "label": leaf.replace("_", " "), "group": group,
                        "description": description})

    for name in ("episode_reward", "episode_length", "reward", "length"):
        add("train/" + name, "Training episodes")
    for name in ("reward", "reward_std"):
        add("eval/" + name, "Evaluation")
    for mode in ("greedy", "sampled"):
        for suffix in ("reward", "reward_std"):
            add(f"eval/{mode}_{suffix}", "Evaluation")
    for name in ("fps", "time_elapsed"):
        add("time/" + name, "Timing")
    losses = {
        "PPO": ["entropy_loss", "policy_gradient_loss", "value_loss", "loss"],
        "A2C": ["entropy_loss", "policy_loss", "value_loss"],
        "DQN": ["loss"],
        "SAC": ["actor_loss", "critic_loss"],
    }.get(algorithm, [])
    if algorithm == "SAC" and str(agent.get("ent_coef", "auto")).startswith("auto"):
        losses.append("ent_coef_loss")
    for name in losses:
        add("losses/train/" + name, "Optimization losses")
    diagnostics = ["learning_rate", "n_updates"]
    if algorithm == "PPO":
        diagnostics += ["approx_kl", "clip_fraction", "explained_variance", "clip_range"]
        if agent.get("clip_range_vf") is not None:
            diagnostics.append("clip_range_vf")
    elif algorithm == "A2C":
        diagnostics.append("explained_variance")
    elif algorithm == "SAC":
        diagnostics.append("ent_coef")
    # Selected Theta PPO/A2C policies have no log_std. Native continuous policies
    # and SAC's SDE policy can additionally emit their action standard deviation.
    if (context.get("model") or {}).get("sb3_native_policy") is True and algorithm in ("PPO", "A2C"):
        diagnostics.append("std")
    if algorithm == "SAC" and agent.get("use_sde"):
        diagnostics.append("std")
    for name in diagnostics:
        add("diagnostics/" + name, "Optimizer diagnostics")
    return outputs
