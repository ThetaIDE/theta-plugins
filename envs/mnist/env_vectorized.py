import numpy as np
import torch
from src.app.api.environments import VectorizedBaseEnv, ActionCount


class VectorizedEnv(VectorizedBaseEnv):
    """Vectorized environment presenting MNIST/Digits samples for RL and supervised benchmarking."""

    def __init__(self, name="mnist", n_envs=1, seed=None, mode="supervised", **kwargs):
        super().__init__()
        self.mode = mode
        self.name = name
        self.n_envs = max(1, int(n_envs))
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        try:
            import sklearn.datasets as skd
            bunch = skd.load_digits()
            self.features = np.asarray(bunch.data, dtype=np.float32)
            self.labels = np.asarray(bunch.target, dtype=np.int64)
            mean = self.features.mean(axis=0, keepdims=True)
            std = self.features.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            self.features = (self.features - mean) / std
        except Exception:
            self.features = np.zeros((100, 64), dtype=np.float32)
            self.labels = np.zeros(100, dtype=np.int64)

        self.num_samples = len(self.features)
        self.indices = np.zeros(self.n_envs, dtype=np.int64)
        self.observation_space = (self.features.shape[1],)
        self.n_actions = ActionCount(10)
        self._step_count = np.zeros(self.n_envs, dtype=np.int64)
        self.max_steps = kwargs.get("max_steps", 100)

    def reset(self):
        self.indices = self.rng.integers(0, self.num_samples, size=self.n_envs)
        self._step_count = np.zeros(self.n_envs, dtype=np.int64)
        obs = self.features[self.indices]
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        actions = np.asarray(actions).reshape(-1)

        current_labels = self.labels[self.indices]
        rewards = (actions == current_labels).astype(np.float32)

        self._step_count += 1
        dones = self._step_count >= self.max_steps

        self.indices = self.rng.integers(0, self.num_samples, size=self.n_envs)
        next_obs = self.features[self.indices]

        term = dones
        trunc = np.zeros_like(dones, dtype=bool)
        infos = [{} for _ in range(self.n_envs)]

        return (
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            term,
            trunc,
            infos,
        )

    def close(self):
        pass
