"""Downloaded cluster sweeps expose study metadata and separate seed feeds."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from frontend.app import Window
from frontend.live_run import LiveRunCache
from src.app.pipeline.tuning_monitor import study_summaries
from src.app.sites.storage import write_json
from package_fixtures import install_package, package_module, optuna_plan
_optuna_utils = package_module('optuna', 'utils.py')
write_study_summary = _optuna_utils.write_study_summary


def snapshot(root):
    # Study summaries come from the run's selected tuner, so the monitored
    # workspace needs the optional Optuna component installed and ready.
    install_package(root, "optuna", root.parent / (root.name + "-archives"))
    execution = "a" * 32
    experiment = "nested/sweep__" + execution
    database = root / "src/usr/results/optuna" / execution / "study.sqlite"
    database.parent.mkdir(parents=True)
    study = optuna.create_study(storage="sqlite:///" + str(database), study_name="s")
    study.set_user_attr("theta_log_method", "m")
    study.set_user_attr("theta_last_stop_reason", "running")
    study.set_user_attr("theta_requested_trials", 64)
    trial = study.ask()
    trial.set_user_attr("theta_log_version", "version_trial")
    manifest = {"parent": {"kind": "trial", "study": "s", "trial_number": 0, "log_version": "version_trial"},
        "runs": [{"method": "m", "agent": "m", "training_seed": seed, "status": status,
                  "version": version, "logs": "/scratch/not-on-this-machine/results/logs/g/old"}
                 for seed, status, version in [(11, "completed", "seed0"), (22, "running", "seed1"), (33, "queued", "seed2")]]}
    path = root / "src/usr/results/executions/g" / experiment / "x/manifest.json"
    write_json(path, manifest)
    for version in ("seed0", "seed1", "seed2"):
        events = root / "src/usr/results/logs/g" / experiment / "m" / version / "events.jsonl"
        events.parent.mkdir(parents=True)
        events.write_text(json.dumps({"type": "start", "run": {"version": version}}) + "\n" +
                          json.dumps({"type": "metrics", "step": 10, "data": {"eval/reward": 123}}) + "\n")
    run = {"id": "r", "status": "running", "backend": {"group": "g", "experiment_id": experiment, "job_id": "remote", "agents": ["m"]},
           "remote": {"execution_id": execution, "config": {"tuning": {"component": "optuna"}},
                      "tuner_execution": optuna_plan(execution)}}
    return run, database, path


def test_remote_monitor_reads_running_seed_with_trial_identity(tmp_path):
    run, database, path = snapshot(tmp_path)
    host = SimpleNamespace(jobs=SimpleNamespace(job=lambda _: None, project_root=tmp_path),
        live_runs_for_job=lambda _: {}, disk_lives=LiveRunCache())
    lives = Window.live_for_run(host, run)
    assert list(lives) == ["m · s · trial 0 · seed 22"]
    assert lives[next(iter(lives))].rows
    manifest = json.loads(path.read_text())
    manifest["runs"][1]["status"] = "completed"
    path.write_text(json.dumps(manifest))
    assert Window.live_for_run(host, run) == {}  # no fallback to a completed/queued feed


def test_monitor_recovers_study_from_closed_synced_database_read_only(tmp_path):
    run, database, _ = snapshot(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    studies = study_summaries(tmp_path, run)
    assert len(studies) == 1
    method, study = studies[0]
    assert method == "m" and study["study"] == "s"
    assert study["trials"][0]["log_version"] == "version_trial"
    assert before == hashlib.sha256(database.read_bytes()).hexdigest()
    assert not list(database.parent.glob("*-wal"))


def test_summary_saves_commit_metadata_for_running_cluster_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("THETA_COMMIT_DIR", str(tmp_path / ".thetaide/commits"))
    study = optuna.create_study(study_name="s")
    study.set_user_attr("theta_log_method", "m")
    root = tmp_path / "src/usr/results/checkpoints/g/e/m"
    root.mkdir(parents=True)
    write_study_summary(study, root)
    commits = [json.loads(p.read_text()) for p in (tmp_path / ".thetaide/commits").glob("*.json")]
    assert len(commits) == 2
    assert {Path(c["path"]).name for c in commits} == {"tuning_summary.json"}


def test_remote_monitor_renders_trial_title_and_study_overview(tmp_path):
    from unittest.mock import Mock
    from PyQt6.QtWidgets import QApplication
    from frontend.monitor.panel import MonitorPanel
    app = QApplication.instance() or QApplication([])
    run, _, _ = snapshot(tmp_path)
    run.update(simulated=False, config={})
    run['remote'].update(site='cluster', state='running', scheduler_state='running')
    host = SimpleNamespace(selected=run, runs=[run],
        jobs=SimpleNamespace(job=lambda _: None, project_root=tmp_path, queue_changed=Mock()),
        live_runs_for_job=lambda _: {}, disk_lives=LiveRunCache())
    host.live_for_run = lambda run, load=True: Window.live_for_run(host, run, load)
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    assert 'Trial 0' in monitor.trial_button.text()
    assert 'seed' in monitor.focused_agent
    assert monitor.charts_view.points
    monitor.focus_sweep('m:s')
    assert monitor.trial_button.text() == ' / Sweep'
    assert monitor.sweep_view.studies[0][1]['study'] == 's'
    monitor.close()


@pytest.mark.parametrize('summary_file', [False, True])
def test_newer_synced_trial_keeps_sweep_navigation_and_compact_title(tmp_path, summary_file):
    from unittest.mock import Mock
    from PyQt6.QtWidgets import QApplication
    from frontend.monitor.panel import MonitorPanel
    app = QApplication.instance() or QApplication([])
    run, database, path = snapshot(tmp_path)
    run.update(simulated=False, config={})
    run['remote'].update(site='cluster', state='running', scheduler_state='running')
    manifest = json.loads(path.read_text())
    manifest['parent'].update(trial_number=19, log_version='version_trial_19')
    path.write_text(json.dumps(manifest))
    if summary_file:
        backend = run['backend']
        summary = (tmp_path / 'src/usr/results/checkpoints' / backend['group'] /
                   backend['experiment_id'] / 'm/tuning_summary.json')
        write_json(summary, {'schema_version': 1, 'component': 'optuna', 'study': 's', 'stop_reason': 'running', 'trials': [
            {'id': 0, 'state': 'complete', 'value': 12, 'parameters': {'rate': 0.1}}]})
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    host = SimpleNamespace(selected=run, runs=[run],
        jobs=SimpleNamespace(job=lambda _: None, project_root=tmp_path, queue_changed=Mock()),
        live_runs_for_job=lambda _: {}, disk_lives=LiveRunCache())
    host.live_for_run = lambda run, load=True: Window.live_for_run(host, run, load)
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    assert monitor.method_button.text() == ' / m'
    assert monitor.trial_button.text() == ' / Trial 19 · seed 22'
    assert 'Sweep' in [a.text() for a in monitor.trial_button.menu().actions()]
    assert 'Trial 19 / Seed 22' in [a.text() for a in monitor.trial_button.menu().actions()]
    monitor.focus_sweep('m:s')
    trials = monitor.sweep_view.studies[0][1]['trials']
    assert [trial['id'] for trial in trials] == [0, 19]
    assert trials[-1]['value'] is None and trials[-1]['parameters'] == {}
    if summary_file:
        assert trials[0]['value'] == 12 and trials[0]['parameters'] == {'rate': 0.1}
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    monitor.close()


def test_new_synced_trial_requires_matching_study_method_and_active_worker(tmp_path):
    run, _, path = snapshot(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['parent']['trial_number'] = 19
    for field, value in [('study', 'other-study'), ('study', 's')]:
        manifest['parent'][field] = value
        if value == 's':
            manifest['runs'][1]['status'] = 'completed'
        path.write_text(json.dumps(manifest))
        assert [t['id'] for _, s in study_summaries(tmp_path, run) for t in s['trials']] == [0]
    manifest['runs'][1].update(status='running', method='another-method')
    path.write_text(json.dumps(manifest))
    assert [t['id'] for _, s in study_summaries(tmp_path, run) for t in s['trials']] == [0]


def test_live_tuning_summary_is_published_as_metrics_without_weights(tmp_path):
    from src.app.sites.publication import Publisher
    from src.app.sites.policy import DEFAULTS
    runtime, work = tmp_path / 'runtime', tmp_path / 'work'
    runtime.mkdir()
    work.mkdir()
    record = {'execution_id': 'a' * 32, 'project_id': 'p', 'group': 'g', 'result_experiment_id': 'e', 'tuner_execution': None}
    (runtime / 'src/usr/results/sites' / record['execution_id']).mkdir(parents=True)
    summary = runtime / 'src/usr/results/checkpoints/g/e/m/tuning_summary.json'
    write_json(summary, {'schema_version': 1, 'component': 'optuna', 'study': 's', 'trials': []})
    (summary.parent / 'model.zip').write_bytes(b'uncommitted weights')
    publisher = Publisher(runtime, work, record, DEFAULTS)
    publisher.tick()
    entries = publisher.manifest['files']
    assert any(e['path'].endswith('tuning_summary.json') and e['category'] == 'metrics' for e in entries)
    assert not any(e['path'].endswith('model.zip') for e in entries)
    publisher.close()
