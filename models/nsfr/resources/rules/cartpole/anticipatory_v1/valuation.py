"""Self-contained CartPole predicates for the anticipatory_v1 ruleset.

The simulator is Gymnasium CartPole-v1. ruleset.yaml selects this module
without modifying the environment's predicates or symbolic identity.
"""

import torch


def build_symbolic_observation(observation):
    """Expose the four raw Gymnasium state values as agent/environment objects."""
    if observation.ndim != 2 or observation.shape[-1] != 4:
        raise ValueError("CartPole symbolic observations require batches of four state values.")
    return observation.unsqueeze(1).expand(-1, 2, -1)


def _projected_lean(agent, horizon):
    """Predict pole lean, with small cart position/velocity feedback terms.

    State order: cart position (m), cart velocity (m/s), pole angle (rad),
    pole angular velocity (rad/s). The two horizon choices are candidate
    control heuristics, rather than exact forecasts of nonlinear dynamics.
    Positive scores call for a right push; negative scores call for left.
    Cart feedback coefficients approximate normalized upright LQR feedback.
    """
    return agent[..., 2] + horizon * agent[..., 3] + 0.025 * agent[..., 0] + 0.05 * agent[..., 1]


def _prob(condition):
    return torch.where(condition, 0.99, 0.01)


def short_fall_left(agent):
    return _prob(_projected_lean(agent, 0.20) < 0)


def short_fall_right(agent):
    return _prob(_projected_lean(agent, 0.20) > 0)


def long_fall_left(agent):
    return _prob(_projected_lean(agent, 0.30) < 0)


def long_fall_right(agent):
    return _prob(_projected_lean(agent, 0.30) > 0)


def true(env):
    return torch.ones(env.shape[0], device=env.device)
