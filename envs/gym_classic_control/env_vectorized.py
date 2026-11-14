"""Vectorized Gymnasium Classic Control environment wrapper for Theta-IDE."""
from __future__ import annotations

from typing import Any
import gymnasium as gym
import torch

from src.app.core.env_vectorized import ActionCount, VectorizedBaseEnv

GYM_MAP = {
    "cartpole": "CartPole-v1",
    "mountaincar": "MountainCar-v0",
    "pendulum": "Pendulum-v1",
    "acrobot": "Acrobot-v1",
}


class VectorizedEnv(VectorizedBaseEnv):
    """Vectorized wrapper for standard Gymnasium classic control benchmarks."""

    name = "gym_classic_control"

    def __init__(
        self,
        name: str | None = None,
        env_id: str | None = None,
        n_envs: int = 1,
        seed: int | None = None,
        mode: str = "ppo",
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        resolved_name = (name or "cartpole").split("/")[-1].lower()
        self.env_id = env_id or GYM_MAP.get(resolved_name, f"{resolved_name.capitalize()}-v1")
        self.n_envs = max(1, int(n_envs))
        self.seed = seed

        self.venv = gym.make_vec(self.env_id, num_envs=self.n_envs, vectorization_mode="sync")

        obs_space = getattr(self.venv, "single_observation_space", None) or getattr(
            self.venv, "observation_space", None
        )
        self.observation_space = getattr(obs_space, "shape", (4,))
        self.n_features = self.observation_space[0] if len(self.observation_space) > 0 else 4

        act_space = getattr(self.venv, "single_action_space", None) or getattr(self.venv, "action_space", None)
        if hasattr(act_space, "n"):
            self._n_actions = int(act_space.n)
        elif hasattr(act_space, "shape"):
            self._n_actions = int(act_space.shape[0])
        else:
            self._n_actions = 2

        self.pred2action = {f"action_{i}": i for i in range(self._n_actions)}

    def reset(self, seed: int | None = None) -> torch.Tensor:
        seed_to_use = seed if seed is not None else self.seed
        obs, _ = self.venv.reset(seed=seed_to_use)
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, actions, is_mapped: bool = False) -> tuple[torch.Tensor, Any, Any, Any, Any]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, rewards, terminations, truncations, infos = self.venv.step(actions)
        return (
            torch.as_tensor(obs, dtype=torch.float32),
            rewards,
            terminations,
            truncations,
            infos,
        )

    @property
    def n_actions(self) -> ActionCount:
        return ActionCount(getattr(self, "_n_actions", 2))

    @n_actions.setter
    def n_actions(self, val: int):
        self._n_actions = int(val)

    def close(self):
        if hasattr(self, "venv") and hasattr(self.venv, "close"):
            self.venv.close()
