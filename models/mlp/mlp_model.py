from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.app.core.model_registry import register_model
from src.app.core.types import ActionResult


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


from src.app.core.provider_loader import provider_object
NatureCNNActor = provider_object('nature_cnn', 'nature_cnn_model.py:NatureCNNActor')

# Backward compatibility alias
NeuralBlenderActor = NatureCNNActor


@register_model("standard_mlp_actor", "standard_mlp", "neural_blender_mlp")
class StandardMLPActor(nn.Module):
    def __init__(self, num_in_features, out_size=2, hidden_sizes=[256, 256]):
        super().__init__()
        self.num_in_features = num_in_features
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        act_fn = nn.GELU if max_width >= 128 else nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            last_size = size
        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(last_size, out_size), std=0.01)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        if x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        hidden = self.network(x)
        logits = self.actor(hidden)
        return logits


# Backward compatibility alias
NeuralBlenderMLP = StandardMLPActor


# Backward compatibility alias
CNNActor = NatureCNNActor


@register_model("value_network", "cnn_value_network")
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


@register_model("mlp", "mlpqnetwork")
class MLPQNetwork(nn.Module):
    def __init__(self, n_actions, num_in_features=4, hidden_sizes=[64, 64], dueling=None, activation=None, dropout=0.0,
                 hidden_dims=None):
        super().__init__()
        self.n_actions = n_actions
        self.num_in_features = num_in_features
        if hidden_dims is not None:
            hidden_sizes = hidden_dims
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        self.dueling = (max_width >= 128) if dueling is None else dueling

        if activation is None:
            act_fn = nn.GELU if max_width >= 128 else nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        self.network = nn.Sequential(*layers)

        if self.dueling:
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(last_size, max(32, last_size // 2))),
                act_fn(),
                layer_init(nn.Linear(max(32, last_size // 2), 1), std=1.0),
            )
            self.advantage_head = nn.Sequential(
                layer_init(nn.Linear(last_size, max(32, last_size // 2))),
                act_fn(),
                layer_init(nn.Linear(max(32, last_size // 2), n_actions), std=0.01),
            )
        else:
            self.head = layer_init(nn.Linear(last_size, n_actions), std=1.0)
            self.value_head = layer_init(nn.Linear(last_size, 1), std=1.0)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        if self.dueling:
            val = self.value_head(feat)
            adv = self.advantage_head(feat)
            return val + (adv - adv.mean(dim=-1, keepdim=True))
        return self.head(feat)

    def get_q_values(self, x):
        return self.forward(x)

    def get_action_probs(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_value(self, x, logic_state=None):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        return self.value_head(feat)

    def get_action_and_value(self, x, action=None):
        x = x.float().reshape(x.shape[0], -1)
        if hasattr(self, "num_in_features") and x.shape[-1] < self.num_in_features:
            pad = torch.zeros((x.shape[0], self.num_in_features - x.shape[-1]), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        elif hasattr(self, "num_in_features") and x.shape[-1] > self.num_in_features:
            x = x[:, : self.num_in_features]
        feat = self.network(x)
        val = self.value_head(feat)
        if self.dueling:
            adv = self.advantage_head(feat)
            logits = val + (adv - adv.mean(dim=-1, keepdim=True))
        else:
            logits = self.head(feat)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return ActionResult(
            action=action,
            logprob=probs.log_prob(action),
            entropy=probs.entropy(),
            value=val,
        )

    def act(self, x, logic_state=None, epsilon=0.0):
        probs = self.get_action_probs(x)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action)


@register_model("mlp_value_network")
class MLPValueNetwork(nn.Module):
    def __init__(self, num_in_features=4, hidden_sizes=[64, 64], activation=None, dropout=0.0):
        super().__init__()
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 256]
        else:
            hidden_sizes = list(hidden_sizes)

        max_width = max(hidden_sizes) if hidden_sizes else 64
        if activation is None:
            act_fn = nn.GELU if max_width >= 128 else nn.Tanh
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        else:
            act_fn = nn.Tanh

        layers = []
        last_size = num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            if size >= 128:
                layers.append(nn.LayerNorm(size))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_size = size
        layers.append(layer_init(nn.Linear(last_size, 1), std=1.0))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        return self.network(x)


@register_model("q_network", "cnn_q_network")
class QNetwork(nn.Module):
    def __init__(self, n_actions=18):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions), std=1.0),
        )

    def forward(self, x):
        return self.network(x / 255.0)


# Standard architecture aliases (decoupled from BlendRL legacy naming)
NatureCNN = NeuralBlenderActor
StandardMLP = NeuralBlenderMLP


"""Feed-forward classifiers for supervised learning on tabular data.

The registry's `dnn` is MLPQNetwork, an RL Q-network built around n_actions and
a duelling head, so it is the wrong shape for a labelled dataset even though the
name suggests otherwise. These are plain classifiers: features in, logits out.
"""

import torch
import torch.nn as nn

from src.app.core.model_registry import register_model


@register_model("mlp_classifier", "tabular_classifier", "classifier")
class MLPClassifier(nn.Module):
    """Multi-layer perceptron over a flat feature vector.

    `n_classes=1` emits a single logit per sample, which is what the shipped
    ClassificationEvalProtocol expects for a binary task; a larger value emits
    one logit per class.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        n_classes: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(str(activation).lower(), nn.ReLU)

        sizes = list(hidden_sizes) or [128, 64]
        layers: list[nn.Module] = []
        last = input_dim
        for size in sizes:
            layers += [nn.Linear(last, size), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, n_classes)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        """Logits, shaped (B,) for a binary task and (B, n_classes) otherwise."""
        logits = self.head(self.body(x.reshape(x.size(0), -1)))
        return logits.squeeze(-1) if self.n_classes == 1 else logits
