"""Vectorized Gymnasium MuJoCo continuous-control environments."""
from __future__ import annotations

from typing import Any
import gymnasium as gym
import torch

from src.app.core.env_vectorized import ActionCount, VectorizedBaseEnv

ENV_IDS = {
    "ant": "Ant-v5",
    "half_cheetah": "HalfCheetah-v5",
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
    "humanoid": "Humanoid-v5",
}


class VectorizedEnv(VectorizedBaseEnv):
    """Vectorized wrapper for continuous-action MuJoCo robot tasks."""

    name = "mujoco"

    def __init__(self, name: str | None = None, env_id: str | None = None, n_envs: int = 1, seed: int | None = None, mode: str = "ppo", **kwargs):
        super().__init__()
        self.mode = mode
        task = (name or "ant").split("/")[-1].lower()
        self.env_id = env_id or ENV_IDS.get(task, task)
        self.n_envs = max(1, int(n_envs))
        self.seed = seed
        self.venv = gym.make_vec(self.env_id, num_envs=self.n_envs, vectorization_mode="sync")
        obs_space = getattr(self.venv, "single_observation_space", self.venv.observation_space)
        self.observation_space = obs_space.shape
        self.n_features = int(obs_space.shape[0])
        act_space = getattr(self.venv, "single_action_space", self.venv.action_space)
        self._n_actions = int(act_space.shape[0])
        self.pred2action = {f"action_{i}": i for i in range(self._n_actions)}

    def reset(self, seed: int | None = None) -> torch.Tensor:
        obs, _ = self.venv.reset(seed=self.seed if seed is None else seed)
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, actions, is_mapped: bool = False) -> tuple[torch.Tensor, Any, Any, Any, Any]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, rewards, terminations, truncations, infos = self.venv.step(actions)
        return torch.as_tensor(obs, dtype=torch.float32), rewards, terminations, truncations, infos

    @property
    def n_actions(self) -> ActionCount:
        return ActionCount(self._n_actions)

    @n_actions.setter
    def n_actions(self, val: int):
        self._n_actions = int(val)

    def close(self):
        self.venv.close()
