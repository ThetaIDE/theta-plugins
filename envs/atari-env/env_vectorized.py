"""Vectorized Atari 2600 environment wrapper for NeSyRL / Theta-IDE."""
from typing import Any, Sequence
import ale_py
import gymnasium as gym

try:
    gym.register_envs(ale_py)
except Exception:
    pass
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)
import numpy as np
import torch

from src.app.core.env_vectorized import ActionCount, VectorizedBaseEnv


GAME_MAP = {
    "breakout": "ALE/Breakout-v5",
    "pong": "ALE/Pong-v5",
    "space_invaders": "ALE/SpaceInvaders-v5",
    "seaquest": "ALE/Seaquest-v5",
    "beam_rider": "ALE/BeamRider-v5",
    "qbert": "ALE/Qbert-v5",
    "mspacman": "ALE/MsPacman-v5",
}


class VectorizedEnv(VectorizedBaseEnv):
    """Vectorized Atari environment with standard preprocessing (grayscale, 84x84, 4-frame stack)."""
    name = "atari"

    def __init__(
        self,
        mode: str = "ppo",
        n_envs: int = 4,
        env_id: str | None = None,
        name: str | None = None,
        seed: int | None = None,
        frame_stack: int = 4,
        grayscale: bool = True,
        screen_size: int = 84,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.n_envs = max(1, int(n_envs))
        self.name = name or "atari"

        # Resolve ALE env_id
        if not env_id:
            game_stem = (self.name.split("/")[-1] if "/" in self.name else self.name).lower()
            env_id = GAME_MAP.get(game_stem, f"ALE/{game_stem.capitalize()}-v5")

        self.env_id = env_id
        self.seed = seed

        def make_atari_env(rank: int):
            def _thunk():
                env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
                env = AtariPreprocessing(
                    env,
                    screen_size=screen_size,
                    grayscale_obs=grayscale,
                    frame_skip=4,
                    scale_obs=False,
                )
                if frame_stack > 1:
                    env = FrameStackObservation(env, stack_size=frame_stack)
                if seed is not None:
                    env.action_space.seed(seed + rank)
                return env
            return _thunk

        # Build SyncVectorEnv
        self.venv = gym.vector.SyncVectorEnv([make_atari_env(i) for i in range(self.n_envs)])

        obs_space = getattr(self.venv, "single_observation_space", None) or getattr(self.venv, "observation_space", None)
        self.observation_space = getattr(obs_space, "shape", (frame_stack, screen_size, screen_size))

        act_space = getattr(self.venv, "single_action_space", None) or getattr(self.venv, "action_space", None)
        self._n_actions = int(act_space.n) if hasattr(act_space, "n") else 4
        self.pred2action = {f"action_{i}": i for i in range(self._n_actions)}

    def reset(self, seed=None) -> torch.Tensor:
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
        return ActionCount(getattr(self, "_n_actions", 4))

    @n_actions.setter
    def n_actions(self, val: int):
        self._n_actions = int(val)
