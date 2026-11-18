"""Optuna pruning through the maintained SB3 runner's evaluation callback."""
from types import SimpleNamespace

import pytest

from src.app.core.execution import RunContext
from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_module
from src.app.workspace import WorkspacePaths
from package_fixtures import install_package


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """The SB3 runner and the Optuna tuner, installed together into this test's workspace."""
    paths = WorkspacePaths(tmp_path)
    install_package(paths, 'sb3-suite', tmp_path.parent / (tmp_path.name + '-archives'))
    install_package(paths, 'optuna', tmp_path.parent / (tmp_path.name + '-archives'))
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    return load_module(get_catalog(paths).package('sb3-suite'), 'sb3_runner.py', paths=paths)


def test_sb3_pruning_reports_evaluation_and_does_not_save_best(runner, tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("stable_baselines3")
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="sb3", storage=storage, direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=1000, intermediate_values={0: 1000}))
    trial = study.ask()
    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "total_timesteps": 16,
        "eval_freq_steps": 8, "eval_episodes": 1,
        "agent": {"name": "ppo", "algorithm": "PPO", "n_steps": 8, "batch_size": 8},
        "model": {"sb3_native_policy": True}, "env": {"env_id": "CartPole-v1"},
        "tuning": {"metric": "eval/reward"},
        "theta_tuning": {"component": "optuna", "pruner": "median", "pruner_startup_trials": 1,
                         "pruner_warmup_steps": 0, "study_name": "sb3", "storage": storage,
                         "optuna_trial_id": trial._trial_id, "trial_number": trial.number},
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(mode=SimpleNamespace(name="RUN"))))
    result = runner.execute(cfg, RunContext(
        'test', WorkspacePaths.for_process(),
        int(cfg.get('seed', 42)), tmp_path / 'src/usr/results/checkpoints' / cfg.group / cfg.experiment_id /
        cfg.agent.name / str(cfg.get('variability_version') or '')))
    assert result.status == "pruned"
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    frozen = study.trials[-1]
    assert frozen.state == optuna.trial.TrialState.PRUNED
    assert 0 < frozen.intermediate_values[0] < 1000
    assert not list((WorkspacePaths.for_process().results / "checkpoints").rglob("best_model.zip"))
