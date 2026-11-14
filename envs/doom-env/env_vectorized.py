from collections.abc import Sequence
from pathlib import Path
import numpy as np
import torch as th

from src.app.api.environments import VectorizedBaseEnv
from .env import DoomSimGym


class VectorizedNudgeEnv(VectorizedBaseEnv):
    """Vectorized batch environment wrapper for DOOM RL simulator."""
    name = "doom"
    pred2action = {
        "forward": 0,
        "backward": 1,
        "turn_left": 2,
        "turn_right": 3,
        "fire": 4,
        "use": 5,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str = "ppo",
        n_envs: int = 1,
        render_mode="rgb_array",
        seed=None,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.n_envs = n_envs
        self.seed = seed

        self.envs = []
        for i in range(n_envs):
            s = (seed + i) if seed is not None else None
            self.envs.append(DoomSimGym(seed=s))

        self.n_actions = 6
        self.n_raw_actions = 6
        self.n_features = 12

    def reset(self, seed=None):
        seed_i = seed if seed is not None else self.seed
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            obs_list.append(th.tensor(obs, dtype=th.float32))
            if seed_i is not None:
                seed_i += 1
        return th.stack(obs_list)

    def step(self, actions, is_mapped: bool = False):
        rewards = []
        truncations = []
        dones = []
        infos = []
        obs_list = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            if hasattr(action, "item"):
                action = action.item()
            obs, reward, done, truncation, info = env.step(int(action))
            obs_list.append(th.tensor(obs, dtype=th.float32))
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
        return (
            th.stack(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncations, dtype=bool),
            infos,
        )

    def get_action_meanings(self):
        return ["forward", "backward", "turn_left", "turn_right", "fire", "use"]

    def close(self):
        for env in self.envs:
            env.close()


VectorizedEnv = VectorizedNudgeEnv
