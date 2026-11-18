from types import SimpleNamespace

import pytest

pytest.importorskip("optuna")
pytest.importorskip("lightning")

from package_fixtures import package_module
_optuna_pruning = package_module('optuna', 'pruning.py')
OptunaPruningCallback = _optuna_pruning.OptunaPruningCallback
make_pruner = _optuna_pruning.make_pruner


def test_pruning_requires_an_objective_visible_during_validation():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=1, callback_metrics={}, current_epoch=0)
    callback.on_validation_end(trainer, None)
    with pytest.raises(ValueError, match="val/loss.*validation"):
        callback.on_fit_end(trainer, None)


def test_pruning_rejects_nonfinite_intermediate_metrics():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=1,
                              callback_metrics={"val/loss": float("nan")}, current_epoch=0)
    with pytest.raises(ValueError, match="finite"):
        callback.on_validation_end(trainer, None)


def test_pruning_rejects_multi_process_training():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=2)
    with pytest.raises(ValueError, match="single Lightning process"):
        callback.on_validation_end(trainer, None)


@pytest.mark.parametrize("startup,warmup", [(-1, 0), (0, -1), (True, 0)])
def test_invalid_pruner_counts_are_rejected(startup, warmup):
    with pytest.raises(ValueError, match="nonnegative"):
        make_pruner("median", startup, warmup)




def test_repeated_runs_and_objective_components_can_prune():
    component = package_module('optuna', 'component.py').OptunaTuner()
    cfg = {"tuning": {"options": {"optuna": {"pruner": "median"}}, "direction": "maximize"},
           "variability": {"training_runs": 4}, "objective": {"component": "return_curve"}}
    assert component.pruning_enabled(cfg)
    component.validate_config(cfg)  # no longer rejects repeated runs with pruning
    import json
    from package_fixtures import source
    features = json.loads((source('optuna') / '.theta_component.json').read_text())['capabilities'][0]['tuning_features']
    assert features['pruning_objectives'] and features['pruning_repeated_runs']


def test_coordinator_monitor_prunes_with_a_status_the_ide_recognizes(tmp_path):
    """The IDE's trial coordinator reports objective progress and stops seeds on this exception."""
    import optuna
    monitor_module = package_module('optuna', 'monitor.py')
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name='s', storage=storage, direction='maximize')

    def runtime(trial):
        return {"study_name": "s", "storage": storage, "optuna_trial_id": trial._trial_id,
                "trial_number": trial.number, "pruner": "median", "pruner_startup_trials": 1,
                "pruner_warmup_steps": 0}

    good = study.ask()
    monitor = monitor_module.OptunaTrialMonitor(runtime(good))
    for step in range(3):
        monitor.report(0.5 + step / 10, step)
    study.tell(good, 0.8)
    poor = study.ask()
    monitor = monitor_module.OptunaTrialMonitor(runtime(poor))
    # Median pruning compares the trial's best value so far with earlier trials at that step.
    with pytest.raises(optuna.TrialPruned) as raised:
        monitor.report(0.1, 0)
    assert raised.value.theta_trial_status == "pruned"
