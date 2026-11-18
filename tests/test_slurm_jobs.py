from package_fixtures import install_package

import pytest
import subprocess
import sys
from omegaconf import OmegaConf
from package_fixtures import fake_bundles
from src.app.sites import execution
from src.app.sites.storage import read_json, workspace_id, write_json
from types import SimpleNamespace



def project(root):
    """A fake control project with Slurm installed."""
    (root / 'src/app').mkdir(parents=True)
    (root / 'src/app/worker.py').write_text('print("original")\n')
    (root / 'src/usr/config').mkdir(parents=True)
    (root / 'src/usr/config/config.yaml').write_text('defaults: [{site: local}]\n')
    (root / 'run_pipeline.py').write_text('# runner\n')
    workspace_id(root)
    install_package(root, 'slurm', root.parent / (root.name + '-archives'))
    return root



def record(root, state='completed'):
    """A finalized (or still running) Slurm receipt with its logs and a checkpoint."""
    project(root)
    execution_id = 'a' * 32
    value = {'schema_version': 1, 'scheduler_component': 'slurm', 'tuner_execution': None,
             'execution_id': execution_id, 'project_id': workspace_id(root),
             'site': 'cluster', 'group': 'demo', 'result_experiment_id': 'nested/test__' + execution_id,
             'state': state, 'scheduler_state': state, 'job_id': '123', 'artifacts_finalized': False}
    write_json(execution.receipt_path(root, execution_id), value)
    logs = root / 'src/usr/results/logs/demo' / value['result_experiment_id'] / 'agent/version_0'
    logs.mkdir(parents=True)
    (logs / 'metrics.csv').write_text('step,reward\n1,2\n')
    (logs / 'events.jsonl').write_text('{"type":"metrics"}\n')
    ckpt = root / 'src/usr/results/checkpoints/demo' / value['result_experiment_id'] / 'model.ckpt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b'weights')
    if state in execution.TERMINAL: execution.finalize(root, execution_id)
    return value, logs, ckpt


CLUSTER = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'scheduler': 'slurm',
                            'python_executable': sys.executable}, 'resources': {}})

def test_unknown_submission_is_not_retried(tmp_path, monkeypatch):
    root = project(tmp_path)
    fake_bundles(monkeypatch, CLUSTER)
    calls = []
    def submit(args, **kw):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 60)
    monkeypatch.setattr(subprocess, 'run', submit)
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'scheduler': 'slurm', 'python_executable': sys.executable}, 'resources': {}})
    with pytest.raises(RuntimeError): execution.submit(root, 'demo/test', [], cfg)
    assert len(calls) == 1
    receipt = read_json(next((root / 'src/usr/results/sites').glob('*/submission.json')))
    assert receipt['state'] == 'submission_unknown'
    assert receipt['result_experiment_id'] == 'test/' + receipt['execution_id']
    script = next((root / 'src/usr/results/sites').glob('*/job.sbatch')).read_text()
    assert 'unset THETA_RESOURCE_POOL' in script
    assert '_run' in script



def test_allocation_worker_runs_frozen_pipeline_and_records_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '321')
    root = project(tmp_path)
    fake_bundles(monkeypatch, CLUSTER)
    script = root / 'run_pipeline.py'
    script.write_text('''import json, os, pathlib, sys
assert os.environ['THETA_IN_ALLOCATION'] == '1'
assert 'THETA_RESOURCE_POOL' not in os.environ
value = next(item.split('=', 1)[1] for item in sys.argv if item.startswith('++experiment_id='))
name = json.loads(value)
path = pathlib.Path('src/usr/results/logs/demo') / name / 'agent/version_0'
path.mkdir(parents=True)
(path / 'metrics.csv').write_text('step,reward\\n1,5\\n')
print('frozen source executed')
''')
    def sbatch(args, **kw):
        assert args[0] == 'sbatch'
        return SimpleNamespace(returncode=0, stdout='321;cluster\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', sbatch)
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'scheduler': 'slurm', 'python_executable': sys.executable}, 'resources': {}})
    result = execution.submit(root, 'demo/test', [], cfg)
    script.write_text('raise RuntimeError("mutable project must not run")')
    monkeypatch.setenv('THETA_RESOURCE_POOL', '/unrelated/workstation-pool')
    code = execution.run(root, result['execution_id'])
    assert code == 0
    receipt = read_json(execution.receipt_path(root, result['execution_id']))
    assert receipt['state'] == 'completed'
    assert not receipt['artifacts_finalized']  # wait for scheduler termination before sealing stdout/stderr
    assert list((root / 'src/usr/results/logs').rglob('metrics.csv'))
    assert 'frozen source executed' in (execution.receipt_path(root, result['execution_id']).parent / 'pipeline.log').read_text()



def test_scheduler_finishing_is_distinct_from_submission_exit(tmp_path, monkeypatch):
    value, _, _ = record(tmp_path, 'running')
    replies = iter(['', '123|TIMEOUT|0:15\n'])
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=next(replies)))
    updated = execution.refresh_status(tmp_path, value)
    assert updated['state'] == 'timeout'
    assert updated['exit_code'] == '0:15'



def test_default_batch_interpreter_preserves_virtualenv_path(tmp_path, monkeypatch):
    real_run = subprocess.run
    root = project(tmp_path)
    fake_bundles(monkeypatch, CLUSTER)
    executable = root / 'venv/bin/python3'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='123\n', stderr=''))
    cfg = OmegaConf.create({'group': 'demo', 'experiment_id': 'test', 'site': {'name': 'cluster', 'scheduler': 'slurm'}, 'resources': {}})
    value = execution.submit(root, 'demo/test', [], cfg)
    script = (execution.receipt_path(root, value['execution_id']).parent / 'job.sbatch').read_text()
    assert str(executable) in script
    assert real_run(['/bin/bash', '-n', str(execution.receipt_path(root, value['execution_id']).parent / 'job.sbatch')]).returncode == 0


@pytest.mark.parametrize("consolidate", [False, True])
def test_slurm_nested_results_use_valid_script_names(tmp_path, tmp_path_factory, monkeypatch, consolidate):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from package_fixtures import install_package
    from src.app.sites.execution import submit

    install_package(tmp_path, "slurm", tmp_path_factory.mktemp("archives"))
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run_pipeline.py").write_text("# pipeline")
    submitted = []
    def sbatch(arguments, **kwargs):
        assert arguments[0] == "sbatch"
        submitted.append(Path(arguments[-1]).read_text())
        return SimpleNamespace(returncode=0, stdout="123\n", stderr="")
    monkeypatch.setattr("subprocess.run", sbatch)
    cfg = OmegaConf.create({"group": "group", "experiment_id": "path/dir/run",
                            "site": {"name": "cluster", "scheduler": "slurm", "python_executable": sys.executable},
                            "paradigm": "online_rl", "consolidate": consolidate, "plots": ["losses"]})
    from package_fixtures import fake_bundles
    fake_bundles(monkeypatch, cfg)
    record = submit(tmp_path, "group/path/dir/run", [], cfg)
    directory = tmp_path / "src/usr/results/sites" / record["execution_id"]
    assert (directory / "job.sbatch").is_file()
    assert len(submitted) == 1
    assert record["result_experiment_id"] == "path/dir/run/" + record["execution_id"]
    assert "#SBATCH --job-name=theta_" in submitted[0]
    assert not (tmp_path / "src/usr/results/plots/group/path/dir/run").exists()
