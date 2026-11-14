"""CEWModel — self-organizing neuro-fuzzy Q-function.

Encapsulates all structural self-organization (CLIP, ECM, Wang-Mendel,
Mamdani antecedent stabilization, FYD pruning) so that any offline RL
agent (CQL, IQL, …) can use CEW by setting model: cew without bespoke
agent code.

The model sets `_request_optimizer_rebind = True` on itself whenever
the fuzzy rule topology changes, which OfflineAgentBase detects and
reacts to by rebinding its optimizer and syncing the target network.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.app.api.models import register_model
from src.app.api.models import DynamicTopologyProtocol, ExtraStateProtocol, HasModelCallbacks
from .cew_utils import (
    MultiFLC,
    rule_creation,
    run_CLIP,
    run_ECM,
    run_FYD,
    stabilize_antecedents,
)


@register_model("cew")
class CEWModel(nn.Module, DynamicTopologyProtocol, ExtraStateProtocol, HasModelCallbacks):
    """Self-organizing fuzzy Q-network (Clinical Expert Weighting).

    Can be used standalone (agent: cql/iql, model: cew) or as a
    symbolic sub-module inside BlendRL (module_type: cew).

    Args:
        n_inputs:  Number of observation features.
        n_actions: Number of discrete actions.
        cql_alpha: CQL regularisation coefficient.
        lr:        Learning rate passed to MultiFLC.
        ecm_dthr:  ECM clustering distance threshold.
        eps:       CLIP eps parameter.
        kappa:     CLIP kappa parameter.
        fyd:       Whether to apply FYD pruning.
        fyd_top_k: Top-k cutoff for FYD (None = Kneedle heuristic).
        stabilize: Whether to run Mamdani autoencoder stabilization.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        cql_alpha: float = 1.0,
        lr: float = 3e-4,
        ecm_dthr: float = 0.1,
        eps: float = 0.1,
        kappa: float = 0.6,
        fyd: bool = False,
        fyd_top_k: int | None = None,
        stabilize: bool = True,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.cql_alpha = cql_alpha
        self.lr = lr
        self.ecm_dthr = ecm_dthr
        self.eps = eps
        self.kappa = kappa
        self.fyd = fyd
        self.fyd_top_k = fyd_top_k
        self.stabilize = stabilize

        # Internal fuzzy model — uninitialized until first self_organize()
        self._flc: MultiFLC | None = None
        self.rules: list | None = None
        self.antecedents: list | None = None
        self.is_organized: bool = False

        # Signal to OfflineAgentBase that the optimizer needs rebinding
        self._request_optimizer_rebind: bool = False

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action. Returns zeros until organized."""
        if self._flc is None:
            return torch.zeros(obs.shape[0], self.n_actions, device=obs.device)
        first_p = next(self._flc.parameters(), None)
        if first_p is not None and first_p.device != obs.device:
            self._flc.to(obs.device)
            self._device = obs.device
        return self._flc(obs)

    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.forward(obs)
        return torch.softmax(q, dim=-1)

    # ── Self-organization ─────────────────────────────────────────────────

    def self_organize(self, obs: torch.Tensor) -> bool:
        """Run full structural self-organization pipeline on observation sample.

        Steps:
            1. CLIP  — partition feature space into Gaussian membership functions
            2. ECM   — evolving clustering over obs to find centroids
            3. WM    — Wang-Mendel rule creation from centroids + antecedents
            4. Stabilize — refine antecedent centers/sigmas via Mamdani autoencoder
            5. FYD   — prune infrequent / indiscernible rules (optional)
            6. Rebuild MultiFLC if topology changed

        Returns:
            True if the rule topology changed (optimizer rebind needed).
        """
        obs_np = obs.detach().cpu().numpy()
        if len(obs_np.shape) > 2:
            obs_np = obs_np.reshape(obs_np.shape[0], -1)

        mins = obs_np.min(axis=0)
        maxes = obs_np.max(axis=0)

        # 1. CLIP
        new_antecedents = run_CLIP(obs_np, mins, maxes, eps=self.eps, kappa=self.kappa)

        # 2. ECM
        clusters = run_ECM(obs_np, [], self.ecm_dthr)
        reduced_X = np.array([c.center for c in clusters])

        # 3. Wang-Mendel
        new_antecedents, new_rules = rule_creation(reduced_X, new_antecedents)

        # 4. Mamdani antecedent stabilization
        if self.stabilize and len(new_rules) > 0:
            new_antecedents = stabilize_antecedents(obs_np, new_antecedents, new_rules, "cpu")

        # 5. FYD pruning (optional)
        if self.fyd and len(new_rules) > 0:
            new_rules, new_antecedents = run_FYD(new_rules, obs_np, new_antecedents, top_k=self.fyd_top_k)

        # 6. Check if topology changed
        topology_changed = self._topology_changed(new_rules, new_antecedents)
        if not topology_changed:
            return False

        self.rules = new_rules
        self.antecedents = new_antecedents
        self._rebuild(device=obs.device)
        self._request_optimizer_rebind = True
        self.is_organized = True
        return True

    def _topology_changed(self, new_rules: list, new_antecedents: list) -> bool:
        """Return True if rule count or antecedent count differs from current topology."""
        if self._flc is None:
            return True
        if len(new_rules) != len(self.rules or []):
            return True
        current_ant = sum(len(p) for p in (self.antecedents or []))
        new_ant = sum(len(p) for p in new_antecedents)
        return current_ant != new_ant

    def _rebuild(self, device: torch.device | str | None = None) -> None:
        """Instantiate a new MultiFLC from the current rules and antecedents."""
        if device is None:
            first_p = next(self.parameters(), None)
            device = first_p.device if first_p is not None else getattr(self, "_device", torch.device("cpu"))
        self._device = device
        self._flc = MultiFLC(
            n_inputs=self.n_inputs,
            n_outputs=self.n_actions,
            antecedents=self.antecedents,
            rules=self.rules,
            learning_rate=self.lr,
            cql_alpha=self.cql_alpha,
        ).to(device)
        # Register as sub-module so parameters() / state_dict() sees it
        self._modules["_flc"] = self._flc

    def parameters(self, recurse: bool = True):
        if self._flc is not None:
            return self._flc.parameters(recurse=recurse)
        return iter([])

    # ── Topology-aware weight copy (for target network sync) ───────────────

    def clone_topology_from(self, source: CEWModel) -> None:
        """Match topology to source then copy weights. Used to sync target network."""
        if source.rules is None:
            return
        self.rules = source.rules
        self.antecedents = source.antecedents
        first_p = next(self.parameters(), None)
        target_device = (
            first_p.device
            if first_p is not None
            else getattr(self, "_device", getattr(source, "_device", torch.device("cpu")))
        )
        self._rebuild(device=target_device)
        self._flc.load_state_dict(source._flc.state_dict())
        self.is_organized = True

    # ── Protocol implementations ─────────────────────────────────────────

    def has_topology_changed(self) -> bool:
        """Return True if model requests an optimizer rebind due to topology mutation."""
        return bool(getattr(self, "_request_optimizer_rebind", False))

    def reset_topology_changed(self) -> None:
        """Reset topology change flag after optimizer rebinding."""
        self._request_optimizer_rebind = False

    def clone_topology_to(self, target: nn.Module) -> None:
        """Synchronize this model's architecture and weights to target network."""
        if hasattr(target, "clone_topology_from"):
            target.clone_topology_from(self)

    def get_callbacks(self) -> list:
        """Return the self-organization callback for this model."""
        from .cew_callback import CEWSelfOrganizationCallback

        return [CEWSelfOrganizationCallback(self)]

    # ── Checkpoint support ────────────────────────────────────────────────

    def extra_state(self) -> dict:
        """Extra state to persist alongside nn.Module state_dict."""
        return {"rules": self.rules, "antecedents": self.antecedents}

    def load_extra_state(self, state: dict) -> None:
        """Restore rules/antecedents and rebuild topology."""
        self.rules = state.get("rules")
        self.antecedents = state.get("antecedents")
        if self.rules is not None and self.antecedents is not None:
            self._rebuild()
            self.is_organized = True
