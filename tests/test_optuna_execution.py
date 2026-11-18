"""Allocation thread bounds, accounting detail and interrupted producer finalization."""
import json

import optuna
import pytest

from src.app.sites import execution
from src.app.sites.storage import write_json
from package_fixtures import package_module
_optuna_component = package_module('optuna', 'component.py')
OptunaTuner = _optuna_component.OptunaTuner
from package_fixtures import package_module
_optuna_utils = package_module('optuna', 'utils.py')
write_study_summary = _optuna_utils.write_study_summary


def test_shutdown_closes_seeds_and_trials_without_changing_completed_work(tmp_path, monkeypatch):
    from package_fixtures import optuna_plan
    record = {"execution_id": "a" * 32, "group": "g", "result_experiment_id": "nested/e",
              "config": {"tuning": {"component": "optuna"}}, "tuner_execution": optuna_plan("a" * 32)}
    manifest = tmp_path / "src/usr/results/executions/g/nested/e/x/manifest.json"
    write_json(manifest, {"runs": [{"status": s} for s in ("completed", "running", "queued")]})
    database = tmp_path / "src/usr/results/optuna" / record["execution_id"] / "study.sqlite"
    database.parent.mkdir(parents=True)
    storage = "sqlite:///" + str(database)
    study = optuna.create_study(storage=storage, study_name="test")
    trial = study.ask()
    study.tell(trial, 1.0)
    study.ask()
    study.set_user_attr("theta_last_stop_reason", "running")
    study.set_user_attr("theta_requested_trials", 64)
    checkpoint = tmp_path / "src/usr/results/checkpoints/g/nested/e/method"
    checkpoint.mkdir(parents=True)
    write_study_summary(study, checkpoint)
    def selected_tuner(cfg, *, paths):
        assert paths.root == tmp_path
        return OptunaTuner()
    monkeypatch.setattr("src.app.pipeline.tuners.get_tuner", selected_tuner)
    execution.close_unfinished_runs(tmp_path, record, "cancelled")
    runs = json.loads(manifest.read_text())["runs"]
    assert [r["status"] for r in runs] == ["completed", "cancelled", "cancelled"]
    summary = json.loads((checkpoint / "tuning_summary.json").read_text())
    assert [t["state"] for t in summary["trials"]] == ["complete", "fail"]
    assert summary["pending_trials"] == 0
    assert summary["stop_reason"] == "interrupted"
    assert [t.state for t in study.get_trials()] == [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL]


def test_allocation_study_storage_is_exclusive(tmp_path, monkeypatch):
    from package_fixtures import package_module
    _optuna_utils = package_module('optuna', 'utils.py')
    get_optuna_storage = _optuna_utils.get_optuna_storage
    monkeypatch.setenv('THETA_SQLITE_ROOT', str(tmp_path / 'scratch'))
    with pytest.raises(ValueError, match='exclusively owned'):
        get_optuna_storage('sqlite:///' + str(tmp_path / 'shared/study.sqlite'))
