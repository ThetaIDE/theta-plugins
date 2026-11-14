"""Ground CartPole rule predicates against Gymnasium's four observation values.

The observation order is cart position, cart velocity, pole angle and pole
angular velocity. The two logic objects both contain this same observation.
"""

import torch


def build_symbolic_observation(observation):
    """Represent the CartPole state as the agent and environment rule objects."""
    if observation.ndim != 2 or observation.shape[-1] != 4:
        raise ValueError("CartPole symbolic observations require batches of four state values.")
    return observation.unsqueeze(1).expand(-1, 2, -1)


def _prob(condition):
    return torch.where(condition, 0.99, 0.01)


def leaning_left(agent):
    return _prob(agent[..., 2] < 0)


def leaning_right(agent):
    return _prob(agent[..., 2] > 0)


def moving_left(agent):
    return _prob(agent[..., 1] < 0)


def moving_right(agent):
    return _prob(agent[..., 1] > 0)


def is_stable(env):
    return _prob(torch.abs(env[..., 2]) < 0.05)


def not_stable(env):
    return _prob(torch.abs(env[..., 2]) >= 0.05)


def true(env):
    return torch.ones(env.shape[0], device=env.device)
