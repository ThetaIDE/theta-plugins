"""Read-only scheduling hints; importing this module requires no ML backend."""


from src.app.api.dynamic import MutableParameter, Replay

PPO_MUTABLE_PARAMETERS = {
    "agent.learning_rate": MutableParameter(0, exclusive_minimum=True),
    "agent.ent_coef": MutableParameter(0),
}


def replay_controller(spec, *, rollout, duration):
    if not hasattr(spec, "get") or set(spec) - {"api_version", "start", "source", "schedule"}:
        raise ValueError("Invalid dynamic PPO configuration fields")
    if spec.get("api_version") != 1:
        raise ValueError("Dynamic PPO requires api_version 1")
    start = spec.get("start", 0)
    if (type(start) is not int or start < 0 or start % rollout
            or type(duration) is not int or duration <= 0 or duration % rollout):
        raise ValueError("Dynamic PPO requires rollout-aligned start and duration")
    if bool(spec.get("source")) != bool(start):
        raise ValueError("A source checkpoint is required exactly when the starting step is positive")
    return start, Replay(spec.get("schedule", []), PPO_MUTABLE_PARAMETERS,
                         rollout=rollout, start=start, stop=start + duration)


def ppo_preflight(cfg, method_name, *, paths):
    spec = cfg.get("theta_dynamic")
    if spec is None:
        return {}
    import yaml
    method = (cfg.get("methods") or {}).get(method_name) or {}
    agent = method.get("agent") or cfg.get("agent") or {}
    model = method.get("model") or cfg.get("model") or {}
    selector = agent if isinstance(agent, str) else agent.get("name") or agent.get("algorithm")
    profile = paths.resolve_config(f"agent/{selector}.yaml")
    options = yaml.safe_load(profile.read_text()) if profile.is_file() else {}
    options.update(agent if hasattr(agent, "items") else {})
    try:
        if model and (not hasattr(model, "get") or model.get("sb3_native_policy") is not True):
            raise ValueError("Dynamic v1 requires native SB3 PPO")
        if options.get("learning_rate_schedule", "constant") != "constant":
            raise ValueError("Dynamic PPO requires a constant learning-rate schedule")
        env = cfg.get("env") or {}
        rollout = options.get("n_steps", 2048) * (env.get("num_envs", 1) if hasattr(env, "get") else 1)
        replay_controller(spec, rollout=rollout, duration=cfg.get("total_timesteps"))
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        return {"block": [str(exc)]}
    return {}


def resource_hints(cfg, *, paths):
    model = cfg.get("model") or {}
    native = model.get("sb3_native_policy", False) if hasattr(model, "get") else False
    # The adapter fixes Theta model policies to CPU. Native policies may use a GPU.
    return {"accelerator": bool(native)}


def ppo_tuning_profile(cfg, *, agent, paths):
    """Declare PPO's budget and optional dynamic capabilities."""
    env = cfg.get("env") or {}
    model = cfg.get("model") or {}
    quantum = agent.get("n_steps", 2048) * (env.get("num_envs", 1) if hasattr(env, "get") else 1)
    if type(quantum) is not int or quantum <= 0:
        raise ValueError("PPO requires a positive integral rollout size")
    dynamic = ((not model or (hasattr(model, "get") and model.get("sb3_native_policy") is True))
               and agent.get("learning_rate_schedule", "constant") == "constant")
    return {
        "api_version": 1,
        "unit": "transitions",
        "quantum": quantum,
        "budget_path": "total_timesteps",
        "metrics": ("eval/reward",),
        "protected_paths": ("agent.n_steps", "model.sb3_native_policy"),
        "mutable_parameters": PPO_MUTABLE_PARAMETERS if dynamic else {},
        "transfer_format": "theta.sb3-ppo-training/v1" if dynamic else None,
    }
