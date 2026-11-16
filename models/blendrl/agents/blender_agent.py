import os
import pickle
import random
import sys
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from torch.distributions.categorical import Categorical

from src.app.api.packages import provider_export
get_nsfr_model = provider_export('nsfr', 'get_nsfr_model')
load_module = provider_export('nsfr', 'load_module')
from src.app.api.models import register_model
from src.app.api.models import ActionResult


def _plain(value):
    """A configuration node as plain Python values."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return value


def _softor(xs, dim=0, gamma=0.015):
    """Smooth logical OR used to merge action scores within this plugin."""
    if not torch.is_tensor(xs):
        xs = torch.stack(xs, dim=dim)
    result = gamma * torch.logsumexp(xs / gamma, dim=dim)
    maximum = result.max()
    return result / maximum if maximum > 1.0 else result


def _build_neural_agent(env_name, n_actions, device, arch_name=None, hidden_sizes=None, num_in_features=None,
                        model_options=None):
    from src.app.api.models import build_model

    return build_model(
        arch_name or "dnn",
        env=env_name,
        n_actions=n_actions,
        device=device,
        hidden_sizes=hidden_sizes or [64, 64],
        obs_dim=num_in_features,
        **(model_options or {}),
    )


def _instantiate(factory, **kwargs):
    target = factory.__init__ if isinstance(factory, type) else factory
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return factory(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return factory(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return factory(**accepted)


class BlenderActor(nn.Module):
    """
    BlendeRL actor that combines heterogeneous policy modules.

    Args:
        env: environment
        policy_modules: list of policy modules (NSFR, CEW, Neural, etc.)
        module_types: list of types for each module
        blender: blending policy
        actor_mode: actor mode, one of ["hybrid", "logic", "neural"]
        blender_mode: blender mode, one of ["logic", "neural"]
        blend_function: blending function, one of ["softmax", "gumbel_softmax"]
        device: device
    """

    def __init__(
        self,
        env,
        policy_modules,
        module_types,
        blender,
        actor_mode,
        blender_mode,
        blend_function,
        device=None,
        explain=False,
        logic_confidence_floor=0.05,
    ):
        """
        Initialize a BlendeRL agent.
        """
        super().__init__()
        self.env = env
        self.policy_modules = nn.ModuleList(policy_modules)
        self.module_types = module_types
        self.blender = blender
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.device = device
        self.explain = explain
        self.logic_confidence_floor = float(logic_confidence_floor)
        if not 0 <= self.logic_confidence_floor < 1:
            raise ValueError("logic_confidence_floor must be in [0, 1).")

        # Build mappings for logic-based blender if needed
        self.blender_id_to_pred_indices = self._build_blender_id_dict()

    def _build_blender_id_dict(self):
        """
        Initialize a dictionary that maps blender mode id to predicate indices.
        Returns:
            blender_id_to_pred_indices: dictionary that maps blender mode id to predicate indices
        """
        if self.blender_mode == "neural":
            return {}

        blender_mode_names = [f"agent_{i}" for i in range(len(self.policy_modules))]
        # A neural + logic blend names its modes neural_agent and logic_agent, as its blender rules do.
        if len(self.policy_modules) == 2 and set(self.module_types) == {"logic", "neural"}:
            blender_mode_names = [f"{kind}_agent" for kind in self.module_types]

        blender_id_to_pred_indices = {i: [] for i in range(len(blender_mode_names))}

        if hasattr(self.blender, "get_prednames"):
            for j, pred_name in enumerate(self.blender.get_prednames()):
                for i, mode_name in enumerate(blender_mode_names):
                    if mode_name in pred_name:
                        blender_id_to_pred_indices[i].append(j)
        return blender_id_to_pred_indices

    def _map_logic_output(self, q, module):
        action_names = self.env.get_action_meanings()

        if hasattr(module, "prednames"):
            mapped_q = torch.zeros(q.size(0), len(action_names), device=q.device)
            for idx, action_name in enumerate(action_names):
                if action_name in module.prednames:
                    pred_idx = module.prednames.index(action_name)
                    mapped_q[:, idx] = q[:, pred_idx]
        elif q.size(1) == len(action_names):
            mapped_q = q
        else:
            return q

        # Truth scores are evidence, not a categorical distribution. Smooth logic
        # produces small positive scores even for unmatched rules. Normalizing
        # those scores makes a one-action ruleset certain in every state.
        floor = self.logic_confidence_floor if hasattr(module, "prednames") else None
        evidence = (mapped_q if floor is None else
                    ((mapped_q.clamp(0, 1) - floor) / (1 - floor)).clamp_min(0))
        sum_q = evidence.sum(dim=-1, keepdim=True)
        confidence = ((sum_q > 1e-6).to(mapped_q.dtype) if floor is None else
                      evidence.max(dim=-1, keepdim=True).values)
        active_mask = confidence > 1e-6
        if getattr(self, "_capture_experts", False):
            self._last_logic_active = active_mask.detach()

        if "withhold" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("withhold")] = 1.0
        elif "noop" in action_names:
            default_probs = torch.zeros_like(mapped_q)
            default_probs[:, action_names.index("noop")] = 1.0
        else:
            default_probs = torch.full_like(mapped_q, 1.0 / len(action_names))

        normalized_q = evidence / sum_q.clamp_min(1e-6)
        return confidence * normalized_q + (1 - confidence) * default_probs

    def get_explanation(self, neural_state, logic_state, action):
        """
        Get the explanation of the blending weights.
        """
        # TODO: Update for heterogeneous modules
        return None, None, self.to_blender_policy_distribution(neural_state, logic_state)[0].detach().cpu().numpy()

    def compute_action_probs_hybrid(self, neural_state, logic_state):
        """
        Compute action probabilities by blending all modules.
        """
        batch_size = neural_state.size(0)
        module_probs = []

        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                probs = module.get_action_probs(neural_state)
            elif m_type == "logic":
                self._last_logic_active = None
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
                if getattr(self, "_capture_experts", False) and self._last_logic_active is not None:
                    self._logic_coverage[i] = self._last_logic_active
            else:
                # Custom / plugin submodule (e.g. CEW or other continuous/relational modules)
                sub_inp = (
                    neural_state
                    if (
                        neural_state.ndim == 2
                        and hasattr(module, "n_inputs")
                        and neural_state.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else neural_state)
                )
                if hasattr(module, "get_action_probs"):
                    probs = self._map_logic_output(module.get_action_probs(sub_inp), module)
                else:
                    probs = module(sub_inp)
            module_probs.append(probs)

        if getattr(self, "_capture_experts", False):
            self._expert_probs = torch.stack(module_probs, dim=1).detach()

        # weights size: B * N_modules
        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        self.w_policy = weights[0]

        action_probs = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_probs in enumerate(module_probs):
            action_probs += weights[:, i].unsqueeze(1) * m_probs.to(neural_state.device)

        return action_probs, weights

    def compute_action_probs_logic(self, logic_state):
        """
        Compute action probabilities using only non-neural modules.
        """
        # Determine expected neural input size from blender if possible
        n_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
            n_in = self.blender.network[0].in_features
        elif hasattr(self.blender, "fc"):  # logic blender
            n_in = 1  # dummy

        dummy_neural = torch.zeros(logic_state.size(0), n_in).to(logic_state.device)
        weights = self.to_blender_policy_distribution(dummy_neural, logic_state)

        # Mask without mutating softmax output, which autograd needs intact.
        mask = weights.new_tensor([0.0 if kind == "neural" else 1.0 for kind in self.module_types])
        weights = weights * mask

        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]

        action_probs = torch.zeros(logic_state.size(0), self.env.n_actions, device=logic_state.device)
        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if m_type == "neural":
                continue
            elif m_type == "logic":
                probs = self._map_logic_output(module.get_action_probs(logic_state), module)
                action_probs += weights[:, i].unsqueeze(1) * probs
            else:
                sub_inp = (
                    dummy_neural
                    if (
                        dummy_neural.ndim == 2
                        and hasattr(module, "n_inputs")
                        and dummy_neural.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else dummy_neural)
                )
                if hasattr(module, "get_action_probs"):
                    probs = self._map_logic_output(module.get_action_probs(sub_inp), module)
                else:
                    probs = module(sub_inp)
                action_probs += weights[:, i].unsqueeze(1) * probs

        return action_probs, weights

    def compute_action_probs_neural(self, neural_state, logic_state=None):
        """
        Compute action probabilities using only neural modules.
        """
        # Determine expected logic input size from blender
        l_in = 1
        if hasattr(self.blender, "network") and len(self.blender.network) > 0:
            l_in = self.blender.network[0].in_features

        dummy_logic = torch.zeros(neural_state.size(0), l_in).to(neural_state.device)
        weights = self.to_blender_policy_distribution(neural_state, logic_state if logic_state is not None else dummy_logic)

        mask = weights.new_tensor([1.0 if kind == "neural" else 0.0 for kind in self.module_types])
        weights = weights * mask

        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-12)
        self.w_policy = weights[0]

        action_probs = torch.zeros(neural_state.size(0), self.env.n_actions, device=neural_state.device)
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                probs = module.get_action_probs(neural_state)
                action_probs += weights[:, i].unsqueeze(1) * probs

        return action_probs, weights

    def to_blender_policy_distribution(self, neural_state, logic_state):
        """
        Merge policies using the blender function.
        """
        if self.blender_mode in ("logic", "symbolic"):
            policy_probs = self.blender(logic_state)
            batch_size = policy_probs.size(0)
            mode_probs = []
            n_modes = len(self.policy_modules)
            for i in range(n_modes):
                indices = torch.tensor(self.blender_id_to_pred_indices.get(i, []), device=policy_probs.device)
                if len(indices) == 0:
                    mode_probs.append(torch.zeros(batch_size, 1, device=policy_probs.device))
                    continue
                indices = indices.expand(batch_size, -1)
                gathered = torch.gather(policy_probs, 1, indices)
                merged = _softor(gathered, dim=1)
                mode_probs.append(merged)

            probs = torch.stack(mode_probs, dim=1).squeeze(-1)
            logits = torch.logit(probs, eps=0.01)
        else:
            # Neural blender
            if callable(getattr(self.blender, "get_action_probs", None)):
                probs = self.blender.get_action_probs(neural_state)
                logits = torch.log(probs.clamp_min(1e-12))
            else:
                logits = self.blender(neural_state)

        if getattr(self, "_capture_experts", False):
            self._soft_gate = torch.softmax(logits, dim=1).detach()
        if self.blend_function == "softmax":
            return torch.softmax(logits, dim=1)
        else:
            return F.gumbel_softmax(logits, dim=1, hard=True)

    def forward(self, neural_state, logic_state):
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state, logic_state)

    def get_q_values(self, neural_state, logic_state=None):
        """Compute Q-values by blending Q-values from all modules."""
        if logic_state is None:
            if neural_state.ndim == 2:
                raise ValueError("BlendRL needs a model-built symbolic observation for vector Q-value inputs.")
            else:
                logic_state = neural_state
        batch_size = neural_state.size(0)
        module_q_values = []
        active = [
            (self.actor_mode != "neural" or kind == "neural")
            and (self.actor_mode != "logic" or kind != "neural")
            for kind in self.module_types
        ]
        if not any(active):
            raise ValueError(f"Actor mode {self.actor_mode!r} has no compatible policy modules.")

        for i, module in enumerate(self.policy_modules):
            m_type = self.module_types[i]
            if not active[i]:
                module_q_values.append(neural_state.new_zeros(batch_size, self.env.n_actions))
                continue
            if m_type == "neural":
                if hasattr(module, "get_q_values"):
                    try:
                        q = module.get_q_values(neural_state, logic_obs=logic_state)
                    except TypeError:
                        q = module.get_q_values(neural_state)
                elif hasattr(module, "forward"):
                    try:
                        q = module(neural_state, logic_obs=logic_state)
                    except TypeError:
                        q = module(neural_state)  # Assuming forward returns Q-values for Q-networks
                else:
                    q = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
            elif m_type == "logic":
                # logic (NSFR / Neumann)
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(logic_state)
                else:
                    # Logic modules usually return probs, treat as Q-values [0, 1]
                    q = self._map_logic_output(module.get_action_probs(logic_state), module)
            else:
                # Custom / plugin submodule (e.g. CEW or other continuous/relational modules)
                sub_inp = (
                    neural_state
                    if (
                        neural_state.ndim == 2
                        and hasattr(module, "n_inputs")
                        and neural_state.shape[1] == module.n_inputs
                    )
                    else (logic_state if logic_state is not None else neural_state)
                )
                if hasattr(module, "get_q_values"):
                    q = module.get_q_values(sub_inp)
                else:
                    q = module(sub_inp)
            module_q_values.append(q)

        weights = self.to_blender_policy_distribution(neural_state, logic_state)
        if self.actor_mode != "hybrid":
            weights = weights * weights.new_tensor(active)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

        q_values = torch.zeros(batch_size, self.env.n_actions, device=neural_state.device)
        for i, m_q in enumerate(module_q_values):
            q_values = q_values + weights[:, i].unsqueeze(1) * m_q.to(neural_state.device)

        return q_values


def build_blender_module(
    env,
    blender_rules,
    device,
    train=True,
    blender_mode="logic",
    reasoner="nsfr",
    explain=False,
    out_size=2,
    architecture="cnn",
    blender_model=None,
):
    if blender_mode == "symbolic":
        blender_mode = "logic"
    assert blender_mode in ["logic", "neural"]
    blender_model_name = None
    if isinstance(blender_model, str):
        blender_model_name = blender_model
    elif isinstance(blender_model, dict) or hasattr(blender_model, "items"):
        model_spec = dict(blender_model)
        blender_model_name = (
            model_spec.get("name")
            or model_spec.get("architecture")
            or model_spec.get("type")
        )

    if blender_mode == "logic":
        selected_reasoner = blender_model_name or reasoner
        model_params = dict(blender_model) if hasattr(blender_model, "items") else {}
        selected_rules = model_params.get("rules", blender_rules)
        if selected_reasoner == "nsfr":
            get_blender_nsfr_model = provider_export('nsfr', 'get_blender_nsfr_model')

            return get_blender_nsfr_model(env.name, selected_rules, device, train=train, explain=explain)
        elif selected_reasoner == "neumann":
            get_blender_neumann_model = provider_export('neumann', 'get_blender_neumann_model')
            get_neumann_model = provider_export('neumann', 'get_neumann_model')

            return get_blender_neumann_model(env.name, selected_rules, device, train=train, explain=explain)
        if blender_model is not None:
            from src.app.api.models import build_model

            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            return build_model(
                blender_model,
                env=env,
                device=device,
                n_actions=out_size,
                obs_dim=int(np.prod(obs.shape[1:])),
            )

    if blender_mode == "neural":
        if blender_model is not None:
            from src.app.api.models import build_model

            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            return build_model(
                blender_model,
                env=env,
                device=device,
                n_actions=out_size,
                obs_dim=int(np.prod(obs.shape[1:])),
            )
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return _build_neural_agent(env.name, out_size, device,
                                   arch_name="nature_cnn" if architecture == "cnn" else "dnn",
                                   num_in_features=int(np.prod(obs.shape[1:])))


@register_model("blendrl")
class BlenderActorCritic(nn.Module):
    """
    BlendeRL actor-critic that supports heterogeneous policy modules.
    Implements CompositeModelProtocol, AuxiliaryLossProtocol, and OptimizerGroupsProtocol.
    """

    def get_routing_weights(self, obs, logic_obs=None):
        if hasattr(self, "actor") and hasattr(self.actor, "to_blender_policy_distribution"):
            if logic_obs is None:
                logic_obs = self.build_logic_observation(obs)
            return self.actor.to_blender_policy_distribution(obs, logic_obs)
        return None

    def get_monitoring_adapter(self, algorithm):
        from src.app.api.monitoring import ModelAccess
        return ModelAccess(self, "get_q_values" if algorithm.lower() == "dqn" else "get_action_probs",
                           "q_values" if algorithm.lower() == "dqn" else "log_probabilities",
                           frozenset({"probe_forward.v1", "module_interventions.v1", "blendrl.expert_interventions.v1"}))

    def get_monitor_specs(self):
        """Architecture-owned provider; opt in by enabling this named instance."""
        return {"blend_reliance": {"component": "blendrl/reliance", "enabled": False}}

    @torch.no_grad()
    def monitoring_interventions(self, obs):
        """Explicit expert/gate counterfactuals for the model-owned monitor."""
        if self.actor.actor_mode != "hybrid" or self.blend_function != "softmax":
            raise ValueError("Blend reliance v1 requires hybrid mode and softmax blending")
        self.actor._capture_experts = True
        self.actor._expert_probs = None
        self.actor._logic_coverage = {}
        try:
            _, weights = self.actor(obs, self.build_logic_observation(obs))
            experts = self.actor._expert_probs
            if experts is None or experts.shape[1] < 2:
                raise ValueError("Blend reliance requires at least two experts")
            variants = {}
            from ..monitor_catalog import expert_names
            for index, name in enumerate(expert_names(self.module_types)):
                changed = experts.clone()
                changed[:, index] = 1.0 / experts.shape[-1]
                variants[name + "_uniform_output"] = (weights.unsqueeze(-1) * changed).sum(1)
                gates = weights.clone()
                gates[:, index] = 0
                denominator = gates.sum(-1, keepdim=True)
                fallback = torch.ones_like(gates)
                fallback[:, index] = 0
                fallback /= fallback.sum(-1, keepdim=True)
                gates = torch.where(denominator > 1e-12, gates / denominator.clamp_min(1e-12), fallback)
                variants[name + "_removed_renormalized"] = (gates.unsqueeze(-1) * experts).sum(1)
            variants["uniform_gate"] = experts.mean(1)
            variants["mean_gate"] = (weights.mean(0)[None, :, None] * experts).sum(1)
            return variants, {"module_types": list(self.module_types), "gate": "softmax",
                              "expert_baseline": "uniform action distribution", "removal": "renormalize remaining gates"}
        finally:
            self.actor._capture_experts = False
            self.actor._expert_probs = None
            self.actor._logic_coverage = {}

    monitoring_namespace = "blend"

    def monitoring_metadata(self):
        return {"module_types": np.asarray(self.module_types, dtype=str)}

    @torch.no_grad()
    def collect_monitoring(self, obs, *, include_experts=False):
        """Summarize routing on an explicitly provided observation batch."""
        from ..monitoring import routing_metrics

        logic_obs = self.build_logic_observation(obs)
        self.actor._capture_experts = True
        self.actor._expert_probs = None
        self.actor._logic_coverage = {}
        try:
            _, weights = self.actor(obs, logic_obs)
            probs = self.actor._expert_probs
            metrics = routing_metrics(weights, self.module_types, expert_probs=probs)
            for i, active in self.actor._logic_coverage.items():
                metrics[f"blend/logic_{i}/coverage"] = active.float().mean().item()
            if self.blend_function == "gumbel_softmax" and self.actor.actor_mode == "hybrid":
                gate = routing_metrics(self.actor._soft_gate, self.module_types)
                metrics.update({key.replace("blend/", "blend/soft_gate/", 1): value
                                for key, value in gate.items()})
            responses = {"states": obs.detach(), "weights": weights.detach()}
            if include_experts:
                responses["expert_probabilities"] = probs.detach()
                responses["action_probabilities"] = (weights.unsqueeze(-1) * probs).sum(1).detach()
            return metrics, responses
        finally:
            self.actor._capture_experts = False
            self.actor._expert_probs = None
            self.actor._logic_coverage = {}

    def validate_engine_compatibility(self, engine, algorithm):
        if engine != "sb3":
            return
        if self.blend_function != "softmax":
            raise ValueError("BlendRL with SB3 currently requires blend_function=softmax.")
        if self.get_cfg("blend_ent_coef") not in (None, 0, 0.0):
            raise ValueError("BlendRL with SB3 does not yet support blend entropy auxiliary losses.")

    def clone_topology_to(self, target) -> None:
        """Sync topology-changing policy modules into this plugin's target model."""
        source_modules = getattr(self, "policy_modules", None) or []
        target_modules = getattr(target, "policy_modules", None) or []
        for source, destination in zip(source_modules, target_modules):
            if hasattr(source, "clone_topology_to"):
                source.clone_topology_to(destination)
            elif hasattr(destination, "clone_topology_from"):
                destination.clone_topology_from(source)

    def compute_auxiliary_losses(self, batch, output) -> dict[str, torch.Tensor]:
        losses = {}
        be = getattr(output, "blend_entropy", None)
        if be is None and hasattr(output, "aux"):
            be = output.aux.get("blend_entropy")
        if isinstance(be, torch.Tensor):
            losses["blend_entropy"] = be.mean()
        return losses

    def configure_optimizer_groups(
        self, base_lr: float, weight_decay: float = 0.0, cfg: Any = None
    ) -> list[dict[str, Any]]:
        groups = []
        def rate(name):
            value = self.get_cfg(name)
            return base_lr if value is None else float(value)

        logic_lr = rate("logic_lr")
        blender_lr = rate("blender_lr")
        used_ids = set()

        if hasattr(self, "actor") and hasattr(self.actor, "policy_modules"):
            for m, m_type in zip(self.actor.policy_modules, getattr(self, "module_types", [])):
                lr = base_lr if m_type == "neural" else logic_lr
                m_params = [p for p in m.parameters() if p.requires_grad and id(p) not in used_ids]
                if m_params:
                    used_ids.update(id(p) for p in m_params)
                    groups.append({"params": m_params, "lr": lr, "weight_decay": weight_decay})

        if hasattr(self, "actor") and hasattr(self.actor, "blender") and self.actor.blender is not None:
            b_params = [p for p in self.actor.blender.parameters() if p.requires_grad and id(p) not in used_ids]
            if b_params:
                used_ids.update(id(p) for p in b_params)
                groups.append({"params": b_params, "lr": blender_lr, "weight_decay": weight_decay})

        rem = [p for p in self.parameters() if p.requires_grad and id(p) not in used_ids]
        if rem:
            groups.append({"params": rem, "lr": base_lr, "weight_decay": weight_decay})
        return groups

    def __init__(
        self,
        env,
        rules="default",
        actor_mode="hybrid",
        blender_mode="neural",
        blend_function="softmax",
        reasoner="nsfr",
        device="cpu",
        architecture=None,
        rng=None,
        explain=False,
        modules=None,
        cfg=None,  # For accessing other agent hyperparams
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env
        self.cfg = cfg
        self.actor_mode = actor_mode
        self.blender_mode = blender_mode
        self.blend_function = blend_function
        self.reasoner = reasoner
        self.architecture = architecture
        self.explain = explain

        hidden_sizes = self.get_cfg("hidden_sizes", [64, 64])
        if hidden_sizes is not None:
            hidden_sizes = list(hidden_sizes)

        self.policy_modules = nn.ModuleList()
        self.module_types = []

        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        neural_in_features = obs.shape[-1]

        def _get_val(obj, key, fallback=None):
            if isinstance(obj, dict):
                return obj.get(key, fallback)
            return getattr(obj, key, fallback) if hasattr(obj, key) else fallback

        # 1. Parse modules from argument, modules list, or hierarchical neural/symbolic blocks
        modules_list = modules if modules is not None else self.get_cfg("modules", None)
        symbolic_cfg = self.get_cfg("symbolic", None) or self.get_cfg("symbolic_actor", None)
        neural_cfg = self.get_cfg("neural", None) or self.get_cfg("neural_actor", None)
        blender = self.get_cfg("blender", blender_mode)
        if not isinstance(blender, str) or blender not in ("neural", "symbolic"):
            raise ValueError("blender must be neural or symbolic")
        self.blender_mode = "logic" if blender == "symbolic" else blender
        self.blender_model = self.get_cfg("blender_model", None) or None
        self.blend_function = self.get_cfg("blend_function", self.blend_function)
        blender_reasoner = self.reasoner or "nsfr"

        if modules_list is None and (symbolic_cfg is not None or neural_cfg is not None):
            modules_list = []
            if symbolic_cfg is not None:
                if isinstance(symbolic_cfg, str):
                    s_dict = {"type": self.reasoner or "nsfr", "rules": symbolic_cfg}
                else:
                    s_dict = dict(symbolic_cfg) if hasattr(symbolic_cfg, "items") else {}
                    s_type = _get_val(
                        s_dict, "type", _get_val(s_dict, "name", _get_val(s_dict, "reasoner", self.reasoner or "nsfr"))
                    )
                    s_dict["type"] = s_type
                    if "rules" not in s_dict and s_type in ("nsfr", "neumann"):
                        s_dict["rules"] = self.get_cfg("rules", "default")
                modules_list.append(s_dict)

            if neural_cfg is not None:
                if isinstance(neural_cfg, str):
                    n_dict = {"module_type": "neural", "architecture": neural_cfg}
                else:
                    n_dict = dict(neural_cfg) if hasattr(neural_cfg, "items") else {}
                    arch = _get_val(
                        n_dict,
                        "architecture",
                        _get_val(n_dict, "type", _get_val(n_dict, "name", self.architecture or "dnn")),
                    )
                    n_dict["architecture"] = arch
                    n_dict["module_type"] = "neural"
                modules_list.append(n_dict)

        self.module_cfgs = list(modules_list) if modules_list else []

        blender_rules = None
        if modules_list:
            for m_cfg in modules_list:
                m_type = _get_val(m_cfg, "module_type", _get_val(m_cfg, "type", "neural"))
                if m_type == "nsfr" or m_type == "neumann":
                    m_rules = _get_val(m_cfg, "rules", self.get_cfg("rules", "default"))
                    if self.reasoner == "neumann" or m_type == "neumann":
                        get_neumann_model = provider_export('neumann', 'get_neumann_model')

                        m = get_neumann_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    else:
                        m = get_nsfr_model(env.name, m_rules, device=device, train=True, explain=self.explain)
                    self.policy_modules.append(m)
                    self.module_types.append("logic")
                    if blender_rules is None:
                        blender_rules = m_rules
                elif m_type not in ("nsfr", "neumann", "neural"):
                    # Dynamically instantiate from MODEL_REGISTRY without hardcoded classes
                    from src.app.api.models import build_model

                    m = build_model(
                        m_cfg,
                        env=env,
                        device=device,
                        obs_dim=int(np.prod(obs.shape[1:])),
                        n_actions=env.n_actions,
                    ).to(device)
                    self.policy_modules.append(m)
                    self.module_types.append(m_type)
                elif m_type == "neural":
                    arch = _get_val(m_cfg, "architecture", _get_val(m_cfg, "type", self.architecture))
                    if arch == "neural" or not arch:
                        arch = self.architecture or "dnn"
                    m_hidden = _get_val(m_cfg, "hidden_sizes", hidden_sizes)
                    m = _build_neural_agent(
                        env.name,
                        env.n_actions,
                        device,
                        arch_name=arch,
                        hidden_sizes=m_hidden,
                        num_in_features=neural_in_features,
                        model_options={k: v for k, v in dict(m_cfg).items()
                                       if k not in {"type", "name", "architecture", "module_type", "hidden_sizes"}},
                    )
                    self.policy_modules.append(m)
                    self.module_types.append("neural")
                    self.architecture = arch
        else:
            # One ruleset name, or a list of them.
            if isinstance(rules, list):
                rulesets = rules
            else:
                rulesets = [rules]

            # Add Neural module first
            self.policy_modules.append(
                _build_neural_agent(
                    env.name,
                    env.n_actions,
                    device,
                    arch_name=self.architecture,
                    hidden_sizes=hidden_sizes,
                    num_in_features=neural_in_features,
                )
            )
            self.module_types.append("neural")

            # Add Logic modules
            for r in rulesets:
                if self.reasoner == "neumann":
                    get_neumann_model = provider_export('neumann', 'get_neumann_model')

                    la = get_neumann_model(env.name, r, device=device, train=True, explain=self.explain)
                else:
                    la = get_nsfr_model(env.name, r, device=device, train=True, explain=self.explain)
                self.policy_modules.append(la)
                self.module_types.append("logic")

        out_size = len(self.policy_modules)

        # Use first logic module's rules for blender if logic-based
        if blender_rules is None:
            blender_rules = rulesets[0] if "rulesets" in locals() else (rules if isinstance(rules, str) else rules[0])

        self.blender = build_blender_module(
            env,
            blender_rules,
            device,
            blender_mode=self.blender_mode,
            reasoner=blender_reasoner,
            train=True,
            explain=self.explain,
            out_size=out_size,
            architecture=self.architecture if self.architecture else "cnn",
            blender_model=self.blender_model,
        )

        # Load logic critic (MLP)
        from src.app.api.packages import environment_resource
        try:
            mlp_module_path = environment_resource(env.name, "mlp.py")
        except FileNotFoundError:
            mlp_module_path = None
        if mlp_module_path is not None:
            module = load_module(mlp_module_path)
            mlp_cls = getattr(module, "StandardMLP", getattr(module, "MLP", None))
            if mlp_cls:
                self.logic_critic = _instantiate(
                    mlp_cls,
                    device=device,
                    out_size=1,
                    logic=True,
                    hidden_sizes=hidden_sizes,
                    num_in_features=neural_in_features,
                )
            else:
                self.logic_critic = None
        else:
            self.logic_critic = None

        self.actor = BlenderActor(
            env,
            self.policy_modules,
            self.module_types,
            self.blender,
            self.actor_mode,
            self.blender_mode,
            self.blend_function,
            device=device,
            # Checkpoints predating this setting retain their action mapping.
            # Newly composed component profiles explicitly enable calibration.
            logic_confidence_floor=self.get_cfg("logic_confidence_floor", None),
        )

    def get_cfg(self, key, default=None):
        """Helper to get a config value from cfg.model, cfg.agent, or cfg (supports dot-paths)."""
        if self.cfg is None:
            return default

        def _get_nested(root, k):
            if root is None:
                return None, False
            if hasattr(root, k):
                val = getattr(root, k)
                if val is not None:
                    return val, True
            if isinstance(root, dict) and k in root:
                return root[k], True
            if "." in k:
                parts = k.split(".")
                curr = root
                for p in parts:
                    if curr is None:
                        return None, False
                    if hasattr(curr, p):
                        curr = getattr(curr, p)
                    elif isinstance(curr, dict) and p in curr:
                        curr = curr[p]
                    else:
                        return None, False
                return curr, True
            return None, False

        for ns in ["model", "agent"]:
            if hasattr(self.cfg, ns):
                val, found = _get_nested(getattr(self.cfg, ns), key)
                if found:
                    return val
            elif isinstance(self.cfg, dict) and ns in self.cfg:
                val, found = _get_nested(self.cfg[ns], key)
                if found:
                    return val

        val, found = _get_nested(self.cfg, key)
        if found:
            return val
        return default

    def forward(self, neural_state, logic_state=None, action=None):
        return self.get_action_and_value(neural_state, logic_state, action=action)

    def build_logic_observation(self, neural_state):
        """Consume symbolic states; expert, gate and critic construction stay here."""
        if self.actor_mode == "neural" and self.blender_mode == "neural":
            return neural_state
        if getattr(self, "_symbolic_state_builder", None) is None:
            selection = self.get_cfg("symbolic_state", None)
            if selection is None:
                raise ValueError("BlendRL's logic actor and symbolic blender need a symbolic state builder: set model.symbolic_state.")
            from src.app.api.packages import load_selected
            loader = load_selected("datamodule", _plain(selection))
            if not callable(loader):
                raise ValueError("The symbolic state builder must be callable.")
            self._symbolic_state_builder = loader
        return self._symbolic_state_builder(neural_state)

    def get_action_probs(self, neural_state):
        logic_state = self.build_logic_observation(neural_state)
        return self.actor(neural_state, logic_state)[0]

    def get_action_and_value(self, neural_state, logic_state=None, action=None):
        if logic_state is None:
            logic_state = self.build_logic_observation(neural_state)
        action_probs, blending_weights = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        blend_dist = Categorical(blending_weights)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)

        blended_value = self.get_value(neural_state, logic_state, blending_weights=blending_weights)

        return ActionResult(
            action=action,
            logprob=logprob,
            entropy=dist.entropy(),
            value=blended_value,
            aux={
                "blend_entropy": blend_dist.entropy(),
                "blending_weights": blending_weights,
            },
        )

    def get_q_values(self, neural_state, logic_state=None):
        if logic_state is None:
            logic_state = self.build_logic_observation(neural_state)
        if self.get_cfg("blend_q_values", True):
            return self.actor.get_q_values(neural_state, logic_state)
        # blend_q_values: false reports the neural module's own Q-values.
        for i, module in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                if hasattr(module, "get_q_values"):
                    return module.get_q_values(neural_state)
                elif hasattr(module, "actor"):
                    x = neural_state.float().reshape(neural_state.shape[0], -1)
                    hidden = module.network(x)
                    return module.actor(hidden)
                elif hasattr(module, "forward"):
                    return module(neural_state)
        return self.actor.get_q_values(neural_state, logic_state)

    def get_value(self, neural_state, logic_state=None, blending_weights=None):
        if logic_state is None:
            logic_state = self.build_logic_observation(neural_state)
        if blending_weights is None:
            _, blending_weights = self.actor(neural_state, logic_state)

        neural_value = self.get_neural_value(neural_state).squeeze(1)
        # A missing symbolic critic must not turn the gate into a value scale.
        # Share the available state critic until a dedicated one is provided.
        logic_value = (self.get_logic_value(logic_state).squeeze(1)
                       if self.logic_critic is not None and self.actor_mode != "neural" else neural_value)

        # Weighted value blending
        # w_neural_sum * V_neural + w_logic_sum * V_logic
        neural_weight_sum = 0
        logic_weight_sum = 0
        for i, m_type in enumerate(self.module_types):
            if m_type == "neural":
                neural_weight_sum += blending_weights[:, i]
            else:
                logic_weight_sum += blending_weights[:, i]

        blended_value = (neural_weight_sum * neural_value + logic_weight_sum * logic_value).unsqueeze(1)
        return blended_value

    def get_neural_value(self, neural_state):
        # Find the first neural module to use its critic
        for i, m in enumerate(self.policy_modules):
            if self.module_types[i] == "neural":
                return m.get_value(neural_state)
        return torch.zeros(neural_state.size(0), 1, device=neural_state.device)

    def get_logic_value(self, logic_state):
        if self.logic_critic:
            if hasattr(self.logic_critic, "num_in_features"):
                flat_size = np.prod(logic_state.shape[1:])
                if flat_size != self.logic_critic.num_in_features and logic_state.ndim > 2:
                    if logic_state.shape[-1] == self.logic_critic.num_in_features:
                        logic_state = logic_state[:, 0, :]
            return self.logic_critic(logic_state)
        return torch.zeros(logic_state.size(0), 1, device=logic_state.device)

    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)
