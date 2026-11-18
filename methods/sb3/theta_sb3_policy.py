"""SB3 on-policy adapter for Theta's PyTorch actor-critic models."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO, A2C, DQN
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.dqn.policies import DQNPolicy

# Opt-in model protocols. A model lists the ones it implements in ``sb3_protocols``.
#   rollout_context.v1  get_action_and_value(obs, action=None, *, phase, context=None); phase is
#                       "rollout" (may return aux["rollout_context"], [batch, width]), "update"
#                       (receives the stored context) or "bootstrap" (value of the next state).
#   training_progress.v1  on_rollout_start(num_timesteps, total_timesteps),
#                         on_env_step(dones, infos) after every environment step, and optionally
#                         training_metrics() -> {name: number} at each rollout end.
#   auxiliary_losses.v1   compute_auxiliary_losses(batch, output) -> {name: weighted, signed loss}.
# Only ThetaPPO implements them; other algorithms reject a model that declares any.
MODEL_PROTOCOLS = frozenset({"rollout_context.v1", "training_progress.v1", "auxiliary_losses.v1"})


def model_protocols(model):
    declared = frozenset(getattr(model, "sb3_protocols", ()) or ())
    unknown = sorted(declared - MODEL_PROTOCOLS)
    if unknown:
        raise ValueError(f"Model declares unknown SB3 protocols {unknown}; this adapter provides {sorted(MODEL_PROTOCOLS)}.")
    return declared


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


class ThetaRolloutSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    contexts: torch.Tensor | None


class ThetaRolloutBuffer(RolloutBuffer):
    """A rollout buffer that also keeps each step's model-provided rollout context."""

    context_source = None

    def reset(self) -> None:
        super().reset()
        self.contexts = None

    def add(self, *args, **kwargs) -> None:
        take = getattr(self.context_source, "take_rollout_context", None)
        context = take() if callable(take) else None
        if context is not None:
            context = torch.as_tensor(context).detach().cpu().numpy().astype(np.float32).reshape(self.n_envs, -1)
            if self.contexts is None:
                self.contexts = np.zeros((self.buffer_size, self.n_envs, context.shape[1]), dtype=np.float32)
            self.contexts[self.pos] = context
        elif self.contexts is not None:
            raise ValueError("The model returned a rollout context for some rollout steps but not others.")
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.generator_ready and self.contexts is not None:
            self.contexts = self.swap_and_flatten(self.contexts)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        samples = super()._get_samples(batch_inds, env)
        contexts = None if self.contexts is None else self.to_torch(self.contexts[batch_inds])
        return ThetaRolloutSamples(*samples, contexts=contexts)


class TrainingProgressCallback(BaseCallback):
    """Tell a training_progress.v1 model about rollout starts and environment steps."""

    def __init__(self, model):
        super().__init__()
        self.theta_model = model

    def _on_rollout_start(self) -> None:
        self.theta_model.on_rollout_start(int(self.num_timesteps), int(self.model._total_timesteps))

    def _on_step(self) -> bool:
        self.theta_model.on_env_step(np.asarray(self.locals["dones"]), self.locals.get("infos") or [])
        return True

    def _on_rollout_end(self) -> None:
        report = getattr(self.theta_model, "training_metrics", None)
        for name, value in (report() if callable(report) else {}).items():
            self.logger.record(f"train/{name}", float(value))


class ThetaPPO(ModelLearningRateSchedule, PPO):
    """PPO that additionally serves the opt-in model protocols.

    A model declaring none of them trains with SB3's own ``PPO.train``. Otherwise ``train``
    follows SB3 2.9's PPO update exactly, adding the stored rollout context to each
    re-evaluation and the model's auxiliary losses to each minibatch loss.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", ThetaRolloutBuffer)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy.theta_protocols_served = True
        if isinstance(self.rollout_buffer, ThetaRolloutBuffer):
            self.rollout_buffer.context_source = self.policy

    def _theta_protocols(self):
        model = getattr(self.policy, "theta_model_module", None)
        return model_protocols(model) if model is not None else frozenset()

    def _init_callback(self, callback, progress_bar=False):
        if "training_progress.v1" in self._theta_protocols():
            callbacks = [] if callback is None else list(callback) if isinstance(callback, list) else [callback]
            callback = [TrainingProgressCallback(self.policy.theta_model_module), *callbacks]
        return super()._init_callback(callback, progress_bar)

    def train(self) -> None:
        protocols = self._theta_protocols()
        if not protocols & {"rollout_context.v1", "auxiliary_losses.v1"}:
            return super().train()
        model = self.policy.theta_model_module
        auxiliary = "auxiliary_losses.v1" in protocols

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        auxiliary_losses = {}

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                output = self.policy.evaluate_with_context(rollout_data.observations, actions, rollout_data.contexts)
                values, log_prob, entropy = output.value.flatten(), output.logprob, output.entropy
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if auxiliary:
                    batch = {"observations": rollout_data.observations, "actions": actions,
                             "old_values": rollout_data.old_values, "old_log_prob": rollout_data.old_log_prob,
                             "advantages": rollout_data.advantages, "returns": rollout_data.returns,
                             "contexts": rollout_data.contexts}
                    for name, term in (model.compute_auxiliary_losses(batch, output) or {}).items():
                        if not isinstance(term, torch.Tensor) or term.ndim != 0:
                            raise ValueError(f"Auxiliary loss {name!r} must be a scalar tensor.")
                        loss = loss + term
                        auxiliary_losses.setdefault(name, []).append(term.item())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        for name, terms in auxiliary_losses.items():
            self.logger.record(f"train/aux_loss/{name}", np.mean(terms))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


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
    from src.app.api.models import build_model

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

    theta_protocols_served = False
    _rollout_context = None

    def _protocols(self):
        protocols = model_protocols(self.theta_model_module)
        if protocols and not self.theta_protocols_served:
            raise ValueError(f"Model {self.theta_model!r} uses SB3 protocols {sorted(protocols)}, "
                             "which only Theta's PPO provides; select sb3/ppo.")
        return protocols

    def forward(self, obs, deterministic=False):
        context = "rollout_context.v1" in self._protocols()
        phase = {"phase": "rollout"} if context else {}
        if deterministic:
            actions = self.get_distribution(obs).get_actions(deterministic=True)
            result = self.theta_model_module.get_action_and_value(obs, action=actions, **phase)
        else:
            result = self.theta_model_module.get_action_and_value(obs, **phase)
        self._rollout_context = (result.aux or {}).get("rollout_context") if context else None
        return result.action, result.value, result.logprob

    def take_rollout_context(self):
        """The rollout context of the latest forward step, handed to the rollout buffer once."""
        context, self._rollout_context = self._rollout_context, None
        return context

    def evaluate_with_context(self, obs, actions, context=None):
        """The model's full re-evaluation of stored actions, with their rollout context."""
        if "rollout_context.v1" in self._protocols():
            return self.theta_model_module.get_action_and_value(obs, action=actions, phase="update", context=context)
        return self.theta_model_module.get_action_and_value(obs, action=actions)

    def evaluate_actions(self, obs, actions):
        result = self.evaluate_with_context(obs, actions.long().flatten())
        return result.value, result.logprob, result.entropy

    def predict_values(self, obs):
        if "rollout_context.v1" in self._protocols():
            return self.theta_model_module.get_action_and_value(obs, phase="bootstrap").value
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
