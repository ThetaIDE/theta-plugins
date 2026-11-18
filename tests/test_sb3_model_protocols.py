"""The sb3-suite PPO adapter's opt-in model protocols (rollout context, progress, auxiliary losses)."""
import pytest

pytest.importorskip('stable_baselines3')

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from package_fixtures import package_module

CARTPOLE = {'env_name': 'cartpole', 'action_meanings': ['left', 'right']}


class ToyActorCritic(nn.Module):
    """A small actor-critic; ``protocols`` selects which adapter protocols it declares."""

    def __init__(self, protocols=(), aux_scale=0.0):
        super().__init__()
        torch.manual_seed(0)
        self.actor = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)
        self.extra = nn.Parameter(torch.ones(()))
        self.sb3_protocols = frozenset(protocols)
        self.aux_scale = aux_scale
        self.progress, self.dones, self.updates, self.bootstraps = [], [], [], 0

    def get_action_probs(self, obs):
        return torch.softmax(self.actor(obs), dim=-1)

    def get_action_and_value(self, obs, action=None, *, phase=None, context=None):
        from src.app.api.models import ActionResult
        dist = Categorical(self.get_action_probs(obs))
        action = dist.sample() if action is None else action
        aux = {}
        if phase == 'rollout':
            aux['rollout_context'] = obs[:, :1].clone()
        elif phase == 'update':
            self.updates.append((obs[:, :1].clone(), context))
        elif phase == 'bootstrap':
            self.bootstraps += 1
        return ActionResult(action=action, logprob=dist.log_prob(action), entropy=dist.entropy(),
                            value=self.critic(obs), aux=aux)

    def on_rollout_start(self, num_timesteps, total_timesteps):
        self.progress.append((num_timesteps, total_timesteps))

    def on_env_step(self, dones, infos):
        self.dones.append(np.asarray(dones).copy())

    def training_metrics(self):
        return {'toy/rollouts': len(self.progress)}

    def compute_auxiliary_losses(self, batch, output):
        if not self.aux_scale:
            return {}
        return {'extra': self.aux_scale * self.extra ** 2}


@pytest.fixture
def adapter(monkeypatch):
    module = package_module('sb3-suite', 'theta_sb3_policy.py')
    built = {}

    def build(observation_space, action_space, theta_config, theta_model, env_name, action_meanings, algorithm):
        built['model'] = ToyActorCritic(**theta_config)
        return built['model']

    monkeypatch.setattr(module, 'build_selected_model', build)
    return module, built


def ppo(module, env, protocols=(), aux_scale=0.0, algorithm=None, **kwargs):
    algorithm = algorithm or module.ThetaPPO
    options = dict(n_steps=16, batch_size=8, n_epochs=2, seed=3)
    options.update(kwargs)
    return algorithm(module.ThetaActorCriticPolicy, env,
                     policy_kwargs={'theta_config': {'protocols': protocols, 'aux_scale': aux_scale},
                                    'theta_model': 'toy', **CARTPOLE}, **options)


def cartpole(n_envs=1):
    from stable_baselines3.common.env_util import make_vec_env
    return make_vec_env('CartPole-v1', n_envs=n_envs, seed=5)


def test_protocol_update_matches_sb3_ppo_exactly(adapter):
    """With no context and no auxiliary terms, the protocol update is SB3's PPO update."""
    module, built = adapter
    results = []
    for protocols in ((), ('auxiliary_losses.v1',)):
        model = ppo(module, cartpole(), protocols)
        model.learn(total_timesteps=48)
        results.append({name: value.detach().clone() for name, value in built['model'].named_parameters()})
    assert results[0].keys() == results[1].keys()
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)


def test_rollout_context_returns_with_its_shuffled_transition(adapter):
    module, built = adapter
    model = ppo(module, cartpole(n_envs=2), ('rollout_context.v1',))
    model.learn(total_timesteps=32)
    toy = built['model']
    assert toy.updates and toy.bootstraps
    for observed, context in toy.updates:
        assert context is not None and context.shape == observed.shape
        torch.testing.assert_close(context, observed)


def test_auxiliary_losses_reach_the_optimizer_and_the_log(adapter, monkeypatch):
    module, built = adapter
    model = ppo(module, cartpole(), ('auxiliary_losses.v1',), aux_scale=5.0)
    recorded = {}
    model.learn(total_timesteps=16)
    toy = built['model']
    assert toy.extra.item() < 1.0

    def record(key, value, exclude=None):
        recorded[key] = value
    monkeypatch.setattr(model.logger, 'record', record)
    model.train()
    assert recorded['train/aux_loss/extra'] > 0


def test_training_progress_sees_rollouts_and_every_environment_step(adapter):
    module, built = adapter
    model = ppo(module, cartpole(n_envs=2), ('training_progress.v1',))
    model.learn(total_timesteps=64)
    toy = built['model']
    assert [step for step, _ in toy.progress] == [0, 32]
    assert all(total == 64 for _, total in toy.progress)
    assert len(toy.dones) == 32 and all(d.shape == (2,) for d in toy.dones)


def test_training_metrics_are_recorded_at_each_rollout_end(adapter):
    from stable_baselines3.common.logger import KVWriter, Logger
    module, _ = adapter
    dumps = []

    class Capture(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            dumps.append(dict(key_values))

    model = ppo(module, cartpole(), ('training_progress.v1',))
    model.set_logger(Logger(None, [Capture()]))
    model.learn(total_timesteps=32)
    assert [dump['train/toy/rollouts'] for dump in dumps if 'train/toy/rollouts' in dump] == [1.0, 2.0]


@pytest.mark.parametrize('choice', ['a2c', 'plain_ppo'])
def test_only_theta_ppo_serves_the_protocols(adapter, choice):
    module, _ = adapter
    from stable_baselines3 import PPO
    algorithm, options = (module.ThetaA2C, {}) if choice == 'a2c' else (PPO, {'batch_size': 8, 'n_epochs': 1})
    model = algorithm(module.ThetaActorCriticPolicy, cartpole(), n_steps=8, seed=3, **options,
                      policy_kwargs={'theta_config': {'protocols': ('rollout_context.v1',)},
                                     'theta_model': 'toy', **CARTPOLE})
    with pytest.raises(ValueError, match="only Theta's PPO provides"):
        model.learn(total_timesteps=8)


def test_unknown_protocols_are_rejected(adapter):
    module, _ = adapter
    with pytest.raises(ValueError, match='unknown SB3 protocols'):
        ppo(module, cartpole(), ('rollout_context.v2',)).learn(total_timesteps=16)
