"""Safe PPO mutation and native training-state transfer."""
import json
from pathlib import Path

import gymnasium
import stable_baselines3 as sb3
import torch
from omegaconf import OmegaConf

from src.app.api.dynamic import Boundary, validate_changes
from .execution_hooks import PPO_MUTABLE_PARAMETERS as PARAMETERS, replay_controller


class Session:
    api_version = 1
    effective_boundary = "after_rollout_optimizer_update"

    def __init__(self, model, cfg, artifacts, seed, context):
        self.model = model
        self.artifacts = artifacts
        self.context = context
        self.spec = cfg.theta_dynamic
        if self.spec.get("api_version") != 1 or type(model) is not sb3.PPO:
            raise ValueError("Dynamic v1 requires native SB3 PPO")
        if cfg.agent.get("learning_rate_schedule", "constant") != "constant":
            raise ValueError("Dynamic PPO requires a constant learning-rate schedule")
        if model.get_vec_normalize_env() is not None:
            raise ValueError("Dynamic PPO checkpoint v1 requires an unnormalized environment")
        self.start, self.controller = replay_controller(
            self.spec, rollout=model.n_steps * model.n_envs, duration=cfg.total_timesteps)
        self.history = []
        self.effective_updates = []
        self.identity = dict(
            format="theta.sb3-ppo-training/v1",
            sb3=sb3.__version__, torch=str(torch.__version__), gym=gymnasium.__version__,
            policy=type(model.policy).__qualname__,
            optimizer=type(model.policy.optimizer).__qualname__,
            device=str(model.device), model=OmegaConf.to_container(cfg.get("model") or OmegaConf.create({}), resolve=True),
            env=OmegaConf.to_container(cfg.env, resolve=True),
            agent={k: v for k, v in OmegaConf.to_container(cfg.agent, resolve=True).items()
                   if f"agent.{k}" not in PARAMETERS and k != "seed"},
            observation=self.space_identity(model.observation_space),
            action=self.space_identity(model.action_space),
        )
        self.source = self.spec.get("source")
        self.recipient_seed = int(seed)
        self.reset_seed = (self.recipient_seed + self.start) % 2**32
        if self.start:
            if not self.source:
                raise ValueError("A continued PPO segment requires a source checkpoint")
            self.restore(Path(self.source))
            model.set_random_seed(self.reset_seed)
        elif self.source:
            raise ValueError("A source checkpoint requires a positive starting step")

    @staticmethod
    def space_identity(space):
        fields = {"description": repr(space)}
        for name in ("low", "high", "nvec", "start"):
            value = getattr(space, name, None)
            if value is not None:
                fields[name] = value.tolist() if hasattr(value, "tolist") else value
        return fields

    def restore(self, path):
        state = torch.load(path, weights_only=True, map_location=self.model.device)
        if (not isinstance(state, dict) or set(state) != {"identity", "step", "policy", "optimizer", "updates", "parameters"}
                or state["identity"] != self.identity or state["step"] != self.start
                or type(state["updates"]) is not int or state["updates"] <= 0
                or not state["optimizer"].get("state")):
            raise ValueError("Incompatible or incomplete PPO training checkpoint")
        parameters = validate_changes(state["parameters"], PARAMETERS)
        if parameters.keys() != PARAMETERS.keys():
            raise ValueError("Incomplete PPO checkpoint parameters")
        self.model.policy.load_state_dict(state["policy"], strict=True)
        self.model.policy.optimizer.load_state_dict(state["optimizer"])
        self.model.num_timesteps = self.start
        self.model._n_updates = state["updates"]
        self.apply(parameters)

    def apply(self, changes):
        model = self.model
        if "agent.learning_rate" in changes:
            rate = changes["agent.learning_rate"]
            model.learning_rate = rate
            model._setup_lr_schedule()
            for group in model.policy.optimizer.param_groups:
                group["lr"] = rate
            if model.lr_schedule(1.0) != rate or any(
                group["lr"] != rate for group in model.policy.optimizer.param_groups
            ):
                raise RuntimeError("PPO learning rate schedule and optimizer disagree")
        if "agent.ent_coef" in changes:
            model.ent_coef = changes["agent.ent_coef"]

    def parameters(self):
        return {"agent.learning_rate": float(self.model.learning_rate),
                "agent.ent_coef": float(self.model.ent_coef)}

    def boundary(self, metrics=None):
        self.context.check_cancelled()
        step = self.model.num_timesteps
        changes = self.controller.update(Boundary(step, metrics or {}, self.parameters()))
        if changes:
            self.apply(validate_changes(changes, PARAMETERS))
            self.history.append({"step": step, "parameters": changes})
            self.record()

    def record(self):
        path = self.artifacts / "dynamic_schedule.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(api_version=1, schedule=self.history, optimizer_updates=self.effective_updates,
            source=self.source, recipient_seed=self.recipient_seed, reset_seed=self.reset_seed,
            reset_policy="recipient_seed_plus_start" if self.source else "initial_reset"), indent=2))
        temporary.replace(path)

    def checkpoint(self):
        self.record()
        path = self.artifacts / "ppo_training.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(dict(identity=self.identity, step=self.model.num_timesteps,
                        policy=self.model.policy.state_dict(), optimizer=self.model.policy.optimizer.state_dict(),
                        updates=self.model._n_updates, parameters=self.parameters()), temporary)
        temporary.replace(path)
        return path
