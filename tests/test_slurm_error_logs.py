from package_fixtures import IDE

import os
import pytest
import shlex
import signal
import subprocess
import sys
import time
from omegaconf import OmegaConf
from package_fixtures import activate_fake_bundle
from package_fixtures import fake_bundles
from src.app.sites import execution
from src.app.sites.storage import read_json
from src.app.sites.storage import workspace_id
from types import SimpleNamespace

def project(root, pipeline):
    (root / 'src/app').mkdir(parents=True)
    (root / 'src/app/worker.py').write_text('# fixture\n')
    (root / 'src/usr/config').mkdir(parents=True)
    (root / 'src/usr/config/config.yaml').write_text('defaults: [{site: local}]\n')
    (root / 'run_pipeline.py').write_text(pipeline)
    workspace_id(root)
    from package_fixtures import install_package
    install_package(root, 'slurm', root.parent / (root.name + '-archives'))
    return root



def cluster_cfg(managed=None):
    cfg = {'group': 'demo', 'experiment_id': 'run',
           'site': {'name': 'cluster', 'scheduler': 'slurm', 'python_executable': sys.executable}, 'resources': {}}
    if managed:
        cfg['site']['execution'] = managed
    return OmegaConf.create(cfg)



def submit(root, monkeypatch, *, managed=None):
    cfg = cluster_cfg(managed)
    fake_bundles(monkeypatch, cfg)
    # Submission is faked; workers and their output pipes are real local processes.
    with monkeypatch.context() as patch:
        patch.setattr(execution.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='321\n', stderr=''))
        return execution.submit(root, 'demo/run', [], cfg)



def test_cluster_pipeline_err_contains_original_nested_exception(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '321')
    pipeline = f'''import os, pathlib, sys
sys.path.insert(0, {str(IDE)!r})
from src.app.pipeline.datasets import _execute_training
print('pipeline started', flush=True)
_execute_training([sys.executable, '-c', "raise RuntimeError('original child failure')"], dict(os.environ), pathlib.Path('src/usr/results/executions/demo/child.log'))
'''
    root = project(tmp_path / 'control', pipeline)
    record = submit(root, monkeypatch)
    assert execution.run(root, record['execution_id']) == 1
    directory = execution.receipt_path(root, record['execution_id']).parent
    assert 'original child failure' in (directory / 'pipeline.err').read_text()
    assert 'CalledProcessError' in (directory / 'pipeline.err').read_text()
    assert 'pipeline started' not in (directory / 'pipeline.err').read_text()
    assert 'pipeline started' in (directory / 'pipeline.log').read_text()
    assert 'original child failure' in (directory / 'pipeline.log').read_text()



def test_batch_term_reaches_supervisor_and_seals_error_logs(tmp_path, monkeypatch):
    ready = tmp_path / 'ready'
    pipeline = f'''import pathlib,signal,sys,time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
pathlib.Path({str(ready)!r}).touch()
print('training started', flush=True)
try:
    while True: time.sleep(.02)
finally:
    print('training cleanup ran', file=sys.stderr, flush=True)
'''
    root = project(tmp_path / 'control', pipeline)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    work = tmp_path / 'work'
    managed = {'schema_version': 1, 'runtime': 'scratch', 'work_root': str(work),
               'scratch_root': str(scratch), 'stage_out_seconds': 4, 'scratch_reserve_bytes': 1}
    activate_fake_bundle(root, 'demo/run', cluster_cfg(managed))
    record = submit(root, monkeypatch, managed=managed)
    directory = execution.receipt_path(root, record['execution_id']).parent
    script = (directory / 'job.sbatch').read_text()
    assert script.splitlines()[-1].startswith('exec ')
    # Use the actual generated shell with the working tree as the framework import root.
    # The test recipe remains frozen in the isolated fixture's source archive.
    harness = f"import sys; sys.path.insert(0, {str(IDE)!r}); from src.app.sites.__main__ import main; sys.exit(main())"
    script = script.replace('-m src.app.sites _run', shlex.join(['-c', harness, '_run']))
    batch = tmp_path / 'batch.sh'
    batch.write_text(script)
    environment = dict(os.environ, SLURM_JOB_ID='321', PYTHONDONTWRITEBYTECODE='1')
    with (tmp_path / 'supervisor.err').open('w') as errors:
        process = subprocess.Popen(['/bin/bash', str(batch)], env=environment, stdout=subprocess.DEVNULL,
                                   stderr=errors, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail('worker did not start: ' + (tmp_path / 'supervisor.err').read_text())
                time.sleep(.02)
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    final = read_json(directory / 'submission.json')
    assert final['state'] == 'cancelled' and final['ended_at']
    assert final['publication_state'] == 'sealed'
    manifest = read_json(work / 'src/usr/results/sites' / record['execution_id'] / 'publication.json')
    assert manifest['sealed']
    item = next(item for item in manifest['files'] if item['path'].endswith('/pipeline.err'))
    text = b''.join((work / chunk['source']).read_bytes()[chunk.get('offset', 0):chunk.get('offset', 0)+chunk.get('length', chunk['size'])]
                    for chunk in item['chunks']).decode()
    assert 'training cleanup ran' in text and 'SIGTERM' in text
    assert 'training cleanup ran' in (tmp_path / 'supervisor.err').read_text()
    assert not list(scratch.iterdir())  # Cleanup happens only after durable sealing.
