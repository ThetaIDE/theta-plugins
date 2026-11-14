"""Standard Deep Neural Network (DNN / MLP) Policy & Value Network.

Modular component plugin for Theta-IDE wrapping standard PyTorch linear and normalization layers.
Supports configurable hidden sizes, LayerNorm, GELU/ReLU/Tanh activations, and optional Dueling Q-value heads.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.app.core.model_registry import register_model
from src.app.core.types import ActionResult


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize linear layers with orthogonal weights and constant bias."""
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


@register_model("dnn")
class DNNModel(nn.Module):
    """Configurable Multi-Layer Perceptron (DNN) for actor-critic policy and Q-value approximation."""

    def __init__(
        self,
        n_actions: int = 2,
        out_size: int | None = None,
        num_in_features: int = 4,
        hidden_sizes: tuple[int, ...] | list[int] = (64, 64),
        dueling: bool | None = None,
        activation: str | None = None,
        dropout: float = 0.0,
        shared_features: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n_actions = out_size if out_size is not None else n_actions
        self.num_in_features = num_in_features

        if hidden_sizes is None or len(hidden_sizes) == 0:
            h_sizes = [64, 64]
        else:
            h_sizes = list(hidden_sizes)

        max_width = max(h_sizes) if h_sizes else 64
        self.dueling = (max_width >= 128) if dueling is None else bool(dueling)

        # Activation resolution
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
                       "elu": nn.ELU, "silu": nn.SiLU, "sigmoid": nn.Sigmoid,
                       "leaky_relu": nn.LeakyReLU, "identity": nn.Identity}
        if activation is not None and activation not in activations:
            raise ValueError(f"Unsupported DNN activation {activation!r}.")
        act_fn = activations[activation] if activation is not None else (nn.GELU if max_width >= 128 else nn.ReLU)

        layers: list[nn.Module] = []
        last_size = num_in_features
        for size in h_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        self.network = nn.Sequential(*layers)
        # PPO can use independent policy/value features. Keep the legacy shared
        # default so existing checkpoints retain their topology.
        self.value_network = None if shared_features else copy.deepcopy(self.network)

        if self.dueling:
            val_dim = max(32, last_size // 2)
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(last_size, val_dim)),
                act_fn(),
                layer_init(nn.Linear(val_dim, 1), std=1.0),
            )
            self.advantage_head = nn.Sequential(
                layer_init(nn.Linear(last_size, val_dim)),
                act_fn(),
                layer_init(nn.Linear(val_dim, self.n_actions), std=0.01),
            )
        else:
            self.value_head = layer_init(nn.Linear(last_size, 1), std=1.0)
            self.advantage_head = layer_init(nn.Linear(last_size, self.n_actions), std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract categorical action probabilities."""
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]

        hidden = self.network(x)
        if self.dueling:
            adv = self.advantage_head(hidden)
            logits = adv - adv.mean(dim=-1, keepdim=True)
        else:
            logits = self.advantage_head(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for offline or Q-learning methods."""
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]

        hidden = self.network(x)
        val = self.value_head(self.value_network(x) if self.value_network is not None else hidden)
        adv = self.advantage_head(hidden)
        if self.dueling:
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return adv

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        hidden = self.network(x) if self.value_network is None else self.value_network(x)
        return self.value_head(hidden)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None) -> ActionResult:
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]

        hidden = self.network(x)
        val = self.value_head(self.value_network(x) if self.value_network is not None else hidden)
        adv = self.advantage_head(hidden)
        logits = (adv - adv.mean(dim=-1, keepdim=True)) if self.dueling else adv
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=val,
        )
