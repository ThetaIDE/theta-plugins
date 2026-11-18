from package_fixtures import package_module

import importlib.util
import json
import optuna
import os
import pytest
import subprocess
import sys
import textwrap
from package_fixtures import IDE
from package_fixtures import sweeper_args
from src.app.pipeline.trial_artifacts import trial_checkpoint_dir



promote_best_trial_checkpoint = package_module('optuna', 'utils.py').promote_best_trial_checkpoint

@pytest.mark.parametrize("extension", ["ckpt", "zip"])
def test_promotion_uses_study_and_trial_identity_not_hydra_job_index(tmp_path, workspace_results, extension):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="study", storage=storage, direction="minimize")
    for value in [10.0, 1.0, 4.0]:
        study.tell(study.ask(), value)
    root = workspace_results / "checkpoints/g/e/m"
    winner = trial_checkpoint_dir(root, "study", 1)
    winner.mkdir(parents=True)
    (winner / f"best_model.{extension}").write_text("winner")
    wrong = root / "0" / "best_model.ckpt"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("wrong")

    assert promote_best_trial_checkpoint("g", "e", "m", storage, "study") == "1"
    assert (root / f"best_model.{extension}").read_text() == "winner"
    assert json.loads((root / "best_trial_summary.json").read_text())["best_trial_id"] == 1
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["trials"][1]["checkpoint"] == str(winner / f"best_model.{extension}")
    assert summary["trials"][0]["checkpoint"] is None
    assert json.loads((root / "sweeps/study/tuning_summary.json").read_text())["study"] == "study"



def test_missing_winner_never_promotes_an_unrelated_checkpoint(tmp_path, workspace_results):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="old", storage=storage)
    study.tell(study.ask(), 1.0)
    root = workspace_results / "checkpoints/g/e/m"
    root.mkdir(parents=True)
    (root / "best_model.ckpt").write_text("existing")
    with pytest.raises(FileNotFoundError, match="Expected one checkpoint"):
        promote_best_trial_checkpoint("g", "e", "m", storage, "old")
    assert (root / "best_model.ckpt").read_text() == "existing"



def test_hydra_sweeps_pass_optuna_trial_numbers_after_resuming(tmp_path, monkeypatch):
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "objective.py"
    script.write_text(
        "from pathlib import Path\n"
        "import hydra\n"
        "from src.app.pipeline.trial_artifacts import trial_checkpoint_dir\n"
        "from src.app.workspace import WorkspacePaths\n"
        f"@hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    Path(f'trial_{cfg.theta_tuning.trial_number}.txt').write_text(cfg.theta_tuning.study_name)\n"
        "    path = trial_checkpoint_dir(WorkspacePaths.for_process().results / 'checkpoints/g/e/m', cfg.theta_tuning.study_name, cfg.theta_tuning.trial_number)\n"
        "    path.mkdir(parents=True, exist_ok=True)\n"
        "    (path / 'best_model.ckpt').write_text(str(cfg.theta_tuning.trial_number))\n"
        "    return -float(cfg.theta_tuning.trial_number)\n"
        "if __name__ == '__main__': main()\n",
        encoding="utf-8",
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root), "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    command = [
        sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "resumed", 1),
        f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)",
        "++tuning.metric=score",
    ]
    for _ in range(2):
        subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert (tmp_path / "trial_0.txt").read_text() == "resumed"
    assert (tmp_path / "trial_1.txt").read_text() == "resumed"
    study = optuna.load_study(study_name="resumed", storage=storage)
    assert [trial.number for trial in study.trials] == [0, 1]
    monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))
    promote_best_trial_checkpoint("g", "e", "m", storage, "resumed")
    assert (paths.results / "checkpoints/g/e/m/best_model.ckpt").read_text() == "1"
    assert json.loads((paths.results / "checkpoints/g/e/m/tuning_summary.json").read_text())["objective_metric"] == "score"



def test_hydra_sweep_marks_invalid_objective_as_failed(tmp_path):
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "bad_objective.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')\n"
        "def main(cfg): return float('nan')\n"
        "if __name__ == '__main__': main()\n",
        encoding="utf-8",
    )
    storage = f"sqlite:///{tmp_path / 'failed.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root), "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "failed", 1),
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)",
         "++group=g", "++experiment_id=e", "++agent.name=m"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    study = optuna.load_study(study_name="failed", storage=storage)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs["theta_last_stop_reason"] == "failure_limit"
    summary = json.loads((paths.results / "checkpoints/g/e/m/tuning_summary.json").read_text())
    assert summary["best_trial_id"] is None
    assert summary["trials"][0]["state"] == "fail"



def test_time_budget_waits_for_running_trial_but_stops_new_batches(tmp_path):
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "slow_objective.py"
    script.write_text(
        "import hydra, time\n"
        f"@hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    time.sleep(1.2)\n"
        "    return 1.0\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'timed.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root),
           "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "timed", 3, timeout_seconds=1),
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="timed", storage=storage)
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.user_attrs["theta_last_stop_reason"] == "time_budget"



def test_sweep_continues_after_allowed_trial_failure(tmp_path):
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "sometimes_fails.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    if cfg.theta_tuning.trial_number == 0: raise ValueError('broken candidate')\n"
        "    return 1.0\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'failures.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root), "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "failures", 2, max_failed_trials=1),
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="failures", storage=storage)
    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.COMPLETE]
    assert study.user_attrs["theta_last_stop_reason"] == "trial_limit"



def test_qmc_sampler_runs_a_real_sweep(tmp_path):
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "objective.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')\n"
        "def main(cfg): return float(cfg.x)\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'qmc.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root), "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "qmc", 2, sampler="QMCSampler"), "++hydra.sweeper.sampler.scramble=true",
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=range(0.1,0.5,0.2)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="qmc", storage=storage)
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert all(any(abs(next(iter(trial.params.values())) - expected) < 1e-9 for expected in (0.1, 0.3, 0.5))
               for trial in study.trials)



def test_interruption_marks_active_trial_and_exports_summary(tmp_path, workspace_results):
    from omegaconf import OmegaConf
    from package_fixtures import package_module
    _optuna_sweeper = package_module('optuna', 'sweeper.py')
    ThetaOptunaSweeper = _optuna_sweeper.ThetaOptunaSweeper

    class InterruptedLauncher:
        def launch(self, jobs, initial_job_idx):
            raise KeyboardInterrupt()

    storage = f"sqlite:///{tmp_path / 'interrupted.db'}"
    sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(seed=1), "minimize", storage,
                                 "interrupted", 1, 1, None, "none", 5, 0, 0, 0)
    sweeper.impl.config = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "m"}})
    sweeper.impl.launcher = InterruptedLauncher()
    with pytest.raises(KeyboardInterrupt):
        sweeper.sweep(["+x=interval(1.0,2.0)"])
    study = optuna.load_study(study_name="interrupted", storage=storage)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs["theta_last_stop_reason"] == "interrupted"
    summary = json.loads((workspace_results / "checkpoints/g/e/m/tuning_summary.json").read_text())
    assert summary["stop_reason"] == "interrupted"



@pytest.mark.skipif(importlib.util.find_spec("lightning") is None, reason="Lightning is not installed")
def test_real_lightning_trial_returns_the_selected_metric(tmp_path, monkeypatch):
    # Managed (adaptive) trial workers run only Theta-owned bootstraps, never a test script.
    from package_fixtures import sweep_workspace
    project = IDE
    paths, search = sweep_workspace(tmp_path / "workspace")
    script = tmp_path / "lightning_trial.py"
    script.write_text(textwrap.dedent(f"""
        from pathlib import Path
        import hydra
        import lightning as L
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.app.core.lightning_builder import finalize_training
        from src.app.pipeline.trial_artifacts import trial_checkpoint_dir
        from src.app.workspace import WorkspacePaths

        class Tiny(L.LightningModule):
            def __init__(self, lr):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)
                self.lr = lr
            def training_step(self, batch, batch_idx):
                loss = self.layer(batch[0]).square().mean()
                self.log('train/loss', loss, on_step=False, on_epoch=True)
                return loss
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=self.lr)
            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)

        @hydra.main(version_base=None, config_path={str(paths.config)!r}, config_name='config')
        def main(cfg):
            model = Tiny(cfg.x)
            trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model)
            directory = trial_checkpoint_dir(WorkspacePaths.for_process().results / 'checkpoints/g/e/m', cfg.theta_tuning.study_name,
                                             cfg.theta_tuning.trial_number)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'best_model.ckpt').write_text('initial checkpoint placeholder')
            return finalize_training(trainer, cfg, str(directory), 1.0, 0, 0)
        if __name__ == '__main__': main()
    """), encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'lightning.db'}"
    env = {**os.environ, "THETA_WORKSPACE": str(paths.root), "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, str(script), "--multirun", search, *sweeper_args(storage, "lightning-smoke", 1),
               "++group=g", "++experiment_id=e",
               "++tuning.metric=train/loss", "++tuning.direction=minimize", "+x=interval(0.01,0.02)"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.setenv("THETA_WORKSPACE", str(paths.root))
    promote_best_trial_checkpoint("g", "e", "m", storage, "lightning-smoke")
    root = paths.results / "checkpoints/g/e/m"
    assert (root / "best_model.ckpt").is_file()
    assert (root / "best_model.ckpt").read_bytes() != b"initial checkpoint placeholder"
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["objective_metric"] == "train/loss"
    assert summary["trials"][0]["value"] > 0
