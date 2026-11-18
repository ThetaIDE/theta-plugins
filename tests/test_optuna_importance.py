"""Live importance estimates and the shared sweep importance chart."""
import json
import os
import sys
from unittest.mock import Mock

import pytest

optuna = pytest.importorskip("optuna")
pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

from src.app.pipeline.trial_artifacts import trial_checkpoint_dir
from package_fixtures import package_module
_optuna_utils = package_module('optuna', 'utils.py')
write_study_summary = _optuna_utils.write_study_summary

app = QApplication.instance() or QApplication(sys.argv[:1])


def complete(study, value):
    trial = study.ask()
    trial.suggest_float("learning_rate", 0.001, 0.1)
    study.tell(trial, value)



def read_summary(root):
    return json.loads((root / "tuning_summary.json").read_text())



def test_live_importance_finds_the_parameter_that_controls_the_objective(tmp_path):
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    study = optuna.create_study(study_name="importance", direction="maximize")
    study.set_user_attr("theta_last_stop_reason", "running")
    distributions = {"signal": FloatDistribution(-2, 2), "noise": CategoricalDistribution(["a", "b", "c", "d", "e"])}
    for signal in [-2, -1, 0, 1, 2]:
        for noise in ["a", "b", "c", "d", "e"]:
            study.add_trial(optuna.trial.create_trial(params={"signal": signal, "noise": noise},
                            distributions=distributions, value=10 * signal))
    write_study_summary(study, tmp_path)
    summary = read_summary(tmp_path)
    scores = summary["parameter_importance"]
    assert scores["signal"] > scores["noise"]
    assert sum(scores.values()) == pytest.approx(1)
    assert summary["stop_reason"] == "running"
    assert summary["parameter_importance_info"]["status"] == "ready"
    assert summary["parameter_importance_info"]["completed_trials"] == 25
    study_snapshot = trial_checkpoint_dir(tmp_path, "importance", 0).parent / "tuning_summary.json"
    assert json.loads(study_snapshot.read_text())["parameter_importance"] == scores



def test_importance_waits_for_five_completions_and_only_recomputes_when_they_change(tmp_path, monkeypatch):
    import optuna.importance

    estimate = Mock(return_value={"learning_rate": 1.0})
    monkeypatch.setattr(optuna.importance, "get_param_importances", estimate)
    study = optuna.create_study()
    study.set_user_attr("theta_last_stop_reason", "running")
    for value in range(4):
        complete(study, value)
    write_study_summary(study, tmp_path)
    assert read_summary(tmp_path)["parameter_importance_info"]["status"] == "waiting"
    estimate.assert_not_called()
    complete(study, 4)
    write_study_summary(study, tmp_path)
    assert read_summary(tmp_path)["parameter_importance"] == {"learning_rate": 1.0}
    assert estimate.call_count == 1
    running = study.ask()
    running.suggest_float("learning_rate", 0.001, 0.1)
    study.tell(study.ask(), state=optuna.trial.TrialState.PRUNED)
    write_study_summary(study, tmp_path)
    assert estimate.call_count == 1
    study.tell(running, 5)
    write_study_summary(study, tmp_path)
    assert estimate.call_count == 2
    assert read_summary(tmp_path)["parameter_importance_info"]["completed_trials"] == 6
    study.set_user_attr("theta_last_stop_reason", "trial_limit")
    write_study_summary(study, tmp_path)
    assert estimate.call_count == 2  # final snapshot keeps the same deterministic estimate



def test_estimation_failure_does_not_break_snapshots_and_retries_after_completion(tmp_path, monkeypatch):
    import optuna.importance

    estimate = Mock(side_effect=[RuntimeError("estimator unavailable"), {"learning_rate": 1.0}])
    monkeypatch.setattr(optuna.importance, "get_param_importances", estimate)
    study = optuna.create_study()
    for value in range(5):
        complete(study, value)
    write_study_summary(study, tmp_path)
    snapshot = read_summary(tmp_path)
    assert snapshot["parameter_importance_info"]["status"] == "unavailable"
    assert len(snapshot["trials"]) == 5
    complete(study, 6)
    write_study_summary(study, tmp_path)
    assert read_summary(tmp_path)["parameter_importance_info"]["status"] == "ready"



def test_constant_objective_has_an_explanation_instead_of_meaningless_bars(tmp_path):
    study = optuna.create_study()
    for _ in range(5):
        complete(study, 1)
    write_study_summary(study, tmp_path)
    snapshot = read_summary(tmp_path)
    assert snapshot["parameter_importance"] == {}
    assert "variation" in snapshot["parameter_importance_info"]["message"]
