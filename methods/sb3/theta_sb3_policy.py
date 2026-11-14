"""SB3 on-policy adapter for Theta's PyTorch actor-critic models."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO, A2C, DQN
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy


def model_optimizer_groups(model, base_lr, weight_decay=0.0):
    """Use model-provided groups, preserving their ratios under SB3 schedules."""
    configure = getattr(model, "configure_optimizer_groups", None)
    if not callable(configure):
        return model.parameters()
    groups = configure(base_lr=base_lr, weight_decay=weight_decay)
    if base_lr <= 0:
        raise ValueError("Model optimizer groups require a positive initial SB3 learning rate.")
    for group in groups:
        group["theta_lr_scale"] = float(group.get("lr", base_lr)) / base_lr
    return groups


class ModelLearningRateSchedule:
    """Apply SB3's schedule proportionally to model-defined parameter groups."""

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
            for group in optimizer.param_groups:
                group["lr"] *= group.get("theta_lr_scale", 1.0)


class ThetaPPO(ModelLearningRateSchedule, PPO):
    pass


class ThetaA2C(ModelLearningRateSchedule, A2C):
    pass


class ThetaDQN(ModelLearningRateSchedule, DQN):
    pass


class ModelEnvironment:
    """Describe the simulator to a model without handing it the training environment."""

    def __init__(self, observation_space, action_space, env_name, action_meanings):
        if not isinstance(action_space, spaces.Discrete) or not isinstance(observation_space, spaces.Box):
            raise ValueError("Theta SB3 actor-critic policies currently require Box observations and discrete actions.")
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.name = env_name
        self._action_meanings = action_meanings

    def reset(self):
        return np.zeros((1, *self.observation_space.shape), dtype=self.observation_space.dtype)

    def get_action_meanings(self):
        if not self._action_meanings or len(self._action_meanings) != self.n_actions:
            raise ValueError("The environment must provide one symbolic action name per action through "
                             "env.action_meanings or its get_action_meanings() method.")
        return list(self._action_meanings)


def build_selected_model(observation_space, action_space, theta_config, theta_model, env_name, action_meanings, algorithm):
    from src.app.core.model_registry import build_model

    cfg = OmegaConf.create(theta_config)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    for key in ("name", "architecture", "sb3_native_policy"):
        model_cfg.pop(key, None)
    environment = ModelEnvironment(observation_space, action_space, env_name, action_meanings)
    model = build_model(theta_model, env=environment, device="cpu", cfg=cfg, **model_cfg)
    validate = getattr(model, "validate_engine_compatibility", None)
    if callable(validate):
        validate("sb3", algorithm)
    return model


class ThetaActorCriticPolicy(ActorCriticPolicy):
    """Delegate SB3 PPO/A2C actions and values to the selected model."""

    def __init__(self, *args, theta_config, theta_model, env_name, action_meanings=None, **kwargs):
        self.theta_config = theta_config
        self.theta_model = theta_model
        self.theta_env_name = env_name
        self.theta_action_meanings = action_meanings
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.theta_model_module = build_selected_model(self.observation_space, self.action_space, self.theta_config,
                                                       self.theta_model, self.theta_env_name, self.theta_action_meanings,
                                                       "on_policy")
        for method in ("get_action_probs", "get_action_and_value"):
            if not callable(getattr(self.theta_model_module, method, None)):
                raise ValueError(f"Model {self.theta_model!r} cannot be used with PPO/A2C: implement {method}(obs).")
        base_lr = lr_schedule(1)
        groups = model_optimizer_groups(self.theta_model_module, base_lr, self.optimizer_kwargs.get("weight_decay", 0.0))
        self.optimizer = self.optimizer_class(groups, lr=base_lr, **self.optimizer_kwargs)

    def get_distribution(self, obs):
        probs = self.theta_model_module.get_action_probs(obs)
        if probs.ndim != 2 or probs.shape[1] != self.action_space.n:
            raise ValueError("Model get_action_probs(obs) must return [batch, number of actions] probabilities.")
        if not torch.isfinite(probs).all() or (probs < 0).any() or (probs.sum(dim=1) <= 0).any():
            raise ValueError("Model get_action_probs(obs) returned invalid probabilities.")
        return self.action_dist.proba_distribution(action_logits=torch.log(probs.clamp_min(1e-12)))

    def forward(self, obs, deterministic=False):
        if deterministic:
            actions = self.get_distribution(obs).get_actions(deterministic=True)
            result = self.theta_model_module.get_action_and_value(obs, action=actions)
        else:
            result = self.theta_model_module.get_action_and_value(obs)
        return result.action, result.value, result.logprob

    def evaluate_actions(self, obs, actions):
        result = self.theta_model_module.get_action_and_value(obs, action=actions.long().flatten())
        return result.value, result.logprob, result.entropy

    def predict_values(self, obs):
        return self.theta_model_module.get_action_and_value(obs).value


class ThetaQNetwork(BasePolicy):
    def __init__(self, observation_space, action_space, theta_config, theta_model, env_name, action_meanings):
        super().__init__(observation_space, action_space)
        self.theta_model_module = build_selected_model(observation_space, action_space, theta_config,
                                                       theta_model, env_name, action_meanings, "dqn")
        if not callable(getattr(self.theta_model_module, "get_q_values", None)):
            raise ValueError(f"Model {theta_model!r} cannot be used with DQN: implement get_q_values(obs).")

    def forward(self, obs):
        values = self.theta_model_module.get_q_values(obs)
        if values.ndim != 2 or values.shape[1] != self.action_space.n:
            raise ValueError("Model get_q_values(obs) must return [batch, number of actions] Q-values.")
        return values

    def _predict(self, obs, deterministic=True):
        return self(obs).argmax(dim=1).reshape(-1)


class ThetaDQNPolicy(DQNPolicy):
    def __init__(self, *args, theta_config, theta_model, env_name, action_meanings=None, **kwargs):
        self.theta_config = theta_config
        self.theta_model = theta_model
        self.theta_env_name = env_name
        self.theta_action_meanings = action_meanings
        super().__init__(*args, **kwargs)

    def make_q_net(self):
        return ThetaQNetwork(self.observation_space, self.action_space, self.theta_config, self.theta_model,
                             self.theta_env_name, self.theta_action_meanings).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        base_lr = lr_schedule(1)
        groups = model_optimizer_groups(self.q_net.theta_model_module, base_lr, self.optimizer_kwargs.get("weight_decay", 0.0))
        self.optimizer = self.optimizer_class(groups, lr=base_lr, **self.optimizer_kwargs)
