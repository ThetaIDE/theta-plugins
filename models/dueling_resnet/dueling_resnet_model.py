"""Dueling Residual Network (Dueling ResNet) Policy & Value Network.

Modular component plugin for Theta-IDE.
Features:
  - Deep residual representations with LayerNorm, GELU activations, and skip connections.
  - Dueling Q-Value decomposition: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a'))).
  - Compatible with online RL (PPO), offline RL (CQL, IQL), and composite architectures (BlendRL).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.app.api.models import register_model
from src.app.api.models import ActionResult


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize linear layers with orthogonal weights and constant bias."""
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = layer_init(nn.Linear(dim, dim))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = layer_init(nn.Linear(dim, dim))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return residual + out


@register_model("dueling_resnet", "resnet")
class DuelingResNetMLP(nn.Module):
    """Scaled Dueling ResNet Neural Policy / Q-Network."""

    def __init__(
        self,
        device=None,
        hidden_sizes=(512, 512, 256, 128),
        has_softmax: bool = False,
        has_sigmoid: bool = False,
        out_size: int = 2,
        as_dict: bool = False,
        logic: bool = False,
        use_dueling: bool = True,
        dropout: float = 0.05,
        num_in_features: int | None = None,
        obs_dim: int | None = None,
        n_actions: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.logic = logic
        self.use_dueling = use_dueling

        in_feat = num_in_features if num_in_features is not None else (obs_dim if obs_dim is not None else 4)
        self.num_in_features = int(in_feat)

        n_acts = n_actions if n_actions is not None else out_size
        self.out_size = int(n_acts)

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [512, 512, 256, 128]
        else:
            hidden_sizes = list(hidden_sizes)

        # ── Feature Extractor Backbone ──────────────────────────────
        embed_dim = hidden_sizes[0]
        self.input_proj = nn.Sequential(
            layer_init(nn.Linear(self.num_in_features, embed_dim)),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        layers = []
        curr_dim = embed_dim
        for next_dim in hidden_sizes[1:]:
            if next_dim == curr_dim:
                layers.append(ResBlock(curr_dim, dropout=dropout))
            else:
                layers.append(
                    nn.Sequential(
                        nn.LayerNorm(curr_dim),
                        layer_init(nn.Linear(curr_dim, next_dim)),
                        nn.GELU(),
                        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    )
                )
                curr_dim = next_dim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        final_dim = curr_dim

        # ── Dueling Value Head V(s) ───────────────────────────────────
        self.value_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            layer_init(nn.Linear(final_dim, max(64, final_dim // 2))),
            nn.GELU(),
            layer_init(nn.Linear(max(64, final_dim // 2), 1), std=1.0),
        )

        # ── Dueling Advantage Head A(s, a) ───────────────────────────
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(final_dim),
            layer_init(nn.Linear(final_dim, max(64, final_dim // 2))),
            nn.GELU(),
            layer_init(nn.Linear(max(64, final_dim // 2), self.out_size), std=0.01),
        )

        # Backward-compatible references
        self.critic = self.value_head
        self.actor = self.advantage_head
        self.network = nn.Sequential(self.input_proj, self.backbone)

        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()
        self.sigmoid = nn.Sigmoid() if has_sigmoid else nn.Identity()

        if self.device is not None:
            self.to(self.device)

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.float().reshape(x.shape[0], -1)
        if flat.shape[-1] < self.num_in_features:
            pad = torch.zeros(
                (flat.shape[0], self.num_in_features - flat.shape[-1]), dtype=flat.dtype, device=flat.device
            )
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > self.num_in_features:
            flat = flat[:, : self.num_in_features]
        return flat

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._flat(x)
        emb = self.input_proj(flat)
        return self.backbone(emb)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        if self.use_dueling:
            val = self.value_head(features)
            adv = self.advantage_head(features)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.advantage_head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return self.softmax(q)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        q = self.get_q_values(x)
        return torch.softmax(q, dim=-1)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        features = self.extract_features(x)
        return self.value_head(features)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        features = self.extract_features(x)
        v = self.value_head(features)
        a = self.advantage_head(features)
        q = v + (a - a.mean(dim=-1, keepdim=True)) if self.use_dueling else a
        probs = torch.softmax(q, dim=-1)
        dist = Categorical(probs=probs)
        if action is None:
            action = dist.sample()
        return ActionResult(
            action=action,
            logprob=dist.log_prob(action),
            entropy=dist.entropy(),
            value=v,
        )

    def act(self, x: torch.Tensor, logic_state=None, epsilon: float = 0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)


# Aliases
DuelingResNet = DuelingResNetMLP
