"""Nature CNN Policy & Value Network (Mnih et al., 2015).

Modular component plugin for Theta-IDE.
Implements the canonical 3-stage convolutional feature extractor followed by actor-critic
and Q-value heads for visual reinforcement learning tasks (e.g. Atari 2600).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.app.api.models import register_model
from src.app.api.models import ActionResult


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize convolutional and linear layers with orthogonal weights and constant bias."""
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@register_model("nature_cnn", "nature_cnn_actor", "cnn", "cnn_actor")
class NatureCNNActor(nn.Module):
    """Canonical Nature CNN feature extractor and Actor-Critic / Q-Network heads."""

    def __init__(
        self,
        out_size: int = 2,
        n_actions: int | None = None,
        in_channels: int = 4,
        features_dim: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.n_actions = n_actions if n_actions is not None else out_size
        self.out_size = self.n_actions
        self.in_channels = in_channels
        self.features_dim = features_dim

        # Standard Nature CNN Convolutional Backbone
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, features_dim)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(features_dim, self.n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(features_dim, 1), std=1.0)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is float32 and normalized to [0, 1]."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        # Reshape flat vectors if needed (e.g. from 4*84*84 flat replay buffers)
        if x.dim() == 2 and x.shape[1] == self.in_channels * 84 * 84:
            x = x.reshape(-1, self.in_channels, 84, 84)

        if x.max() > 1.0:
            x = x / 255.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action probabilities."""
        hidden = self.network(self._preprocess(x))
        logits = self.actor(hidden)
        return torch.softmax(logits, dim=-1)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for all actions."""
        hidden = self.network(self._preprocess(x))
        return self.actor(hidden)

    def get_value(self, x: torch.Tensor, logic_state=None) -> torch.Tensor:
        """Compute state value V(s)."""
        hidden = self.network(self._preprocess(x))
        return self.critic(hidden)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None) -> ActionResult:
        hidden = self.network(self._preprocess(x))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=self.critic(hidden),
        )


# Backward compatibility aliases
NatureCNN = NatureCNNActor
CNNActor = NatureCNNActor
