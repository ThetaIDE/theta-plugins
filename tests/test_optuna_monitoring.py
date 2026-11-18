import json
from omegaconf import OmegaConf

def test_tuning_summary_counts_pending_sampled_and_future_trials(tmp_path):
    import optuna
    from package_fixtures import package_module
    _optuna_utils = package_module('optuna', 'utils.py')
    write_study_summary = _optuna_utils.write_study_summary
    study = optuna.create_study()
    study.set_user_attr('theta_requested_trials', 10)
    study.set_user_attr('theta_initial_trials', 0)
    study.set_user_attr('theta_last_stop_reason', 'running')
    trial = study.ask()
    trial.set_user_attr('theta_execution_state', 'pending')
    write_study_summary(study, tmp_path)
    summary = json.loads((tmp_path / 'tuning_summary.json').read_text())
    assert summary['pending_trials'] == 10 and summary['future_trials'] == 9
    assert summary['trials'][0]['state'] == 'pending'
    trial.set_user_attr('theta_execution_state', 'running')
    write_study_summary(study, tmp_path)
    summary = json.loads((tmp_path / 'tuning_summary.json').read_text())
    assert summary['pending_trials'] == 9 and summary['trials'][0]['state'] == 'running'



def test_summary_preserves_recorded_log_version_without_guessing_for_old_trials(tmp_path):
    import optuna
    from package_fixtures import package_module
    _optuna_utils = package_module('optuna', 'utils.py')
    write_study_summary = _optuna_utils.write_study_summary

    study = optuna.create_study()
    old = study.ask()
    study.tell(old, 1.0)
    current = study.ask()
    current.set_user_attr("theta_log_version", "version_sweep_current")
    write_study_summary(study, tmp_path)
    trials = json.loads((tmp_path / "tuning_summary.json").read_text())["trials"]
    assert trials[0]["log_version"] is None
    assert trials[1]["log_version"] == "version_sweep_current"


def test_cancelled_method_closes_only_current_launch_trials_and_seed_records(tmp_path, monkeypatch):
    import optuna
    from src.app.pipeline.variability import finish_trial_runs
    from package_fixtures import package_module
    _optuna_component = package_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="s", storage=storage)
    study.ask()  # An earlier launch's outstanding trial is not owned by this cancellation.
    study.set_user_attr("theta_initial_trials", 1)
    study.ask()
    for key, pid, method in [("current", 11, "m"), ("sibling", 11, "other"), ("earlier", 22, "m")]:
        path = tmp_path / "src/usr/results/executions/g/e" / key / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"owner_pid": pid, "parent": {"method": method}, "runs": [{"status": "running"}]}))
    finish_trial_runs("g", "e", None, "cancelled", owner_pid=11, method="m")
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e"})
    OptunaTuner().cancel(cfg, "m", storage, "s")
    trials = optuna.load_study(study_name="s", storage=storage).trials
    assert [trial.state.name for trial in trials] == ["RUNNING", "PRUNED"]
    for key, expected in [("current", "cancelled"), ("sibling", "running"), ("earlier", "running")]:
        path = tmp_path / "src/usr/results/executions/g/e" / key / "manifest.json"
        assert json.loads(path.read_text())["runs"][0]["status"] == expected
