import json
import os
import pytest
import shutil
import sqlite3
import subprocess
import sys
import time
from omegaconf import OmegaConf
from package_fixtures import package_module
validate_allocation = package_module('slurm', 'execution.py').validate_allocation
generate_sbatch_header = package_module('slurm', 'headers.py').generate_sbatch_header
from package_fixtures import IDE
from package_fixtures import activate_fake_bundle
from pathlib import Path
from src.app.pipeline.exceptions import ConfigurationError
from src.app.sites import bundle as bundles
from src.app.sites import execution
from src.app.sites import transfer
from src.app.sites.policy import DEFAULTS
from src.app.sites.publication import Publisher
from src.app.sites.storage import read_json
from src.app.sites.storage import workspace_id
from src.app.sites.storage import write_json
from types import SimpleNamespace

def test_threads_do_not_expand_to_allocation_size():
    from package_fixtures import package_module
    worker_threads = package_module('slurm', 'execution.py').worker_threads
    assert worker_threads({}, {"SLURM_CPUS_PER_TASK": "16"}) == 1
    assert worker_threads({"site": {"threads_per_worker": 2}}, {"SLURM_CPUS_PER_TASK": "16"}) == 2
    assert worker_threads({"site": {"threads_per_worker": 8}}, {"SLURM_CPUS_PER_TASK": "2"}) == 2
    assert worker_threads({"site": {"extra_env_vars": {"OMP_NUM_THREADS": "1"}}}, {"SLURM_CPUS_PER_TASK": "16"}) == 1



@pytest.mark.parametrize("batched", [False, True])
def test_scheduler_preserves_canceller_reason_and_time(tmp_path, monkeypatch, batched):
    from package_fixtures import install_package
    install_package(tmp_path, "slurm", tmp_path.parent / (tmp_path.name + "-archives"))
    record = {"execution_id": "a" * 32, "job_id": "123", "state": "running", "project_id": "test",
              "scheduler_component": "slurm"}
    write_json(execution.receipt_path(tmp_path, record["execution_id"]), record)
    replies = iter(["", "123|CANCELLED by 42|0:15|TimeLimit|00:28:00|00:30:00|start|end\n"])
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=next(replies)))
    if batched:
        execution.refresh_many(tmp_path, [record], minimum_seconds=0)
    else:
        record = execution.refresh_status(tmp_path, record)
    assert record["scheduler_state"] == "cancelled"
    assert record["scheduler_state_raw"] == "CANCELLED by 42"
    assert record["scheduler_reason"] == "TimeLimit"
    assert record["scheduler_time_limit"] == "00:30:00"


def cfg(**changes):
    value = {'group': 'g', 'experiment_id': 'e', 'site': {'name': 'cluster', 'scheduler': 'slurm',
             'python_executable': sys.executable, 'resources': {'partition': 'cpu', 'gpus': 0}}}
    value.update(changes)
    return OmegaConf.create(value)



def managed(tmp_path, script=None):
    root, scratch, work = [tmp_path / name for name in ('control', 'scratch', 'work')]
    for path in (root, scratch, work): path.mkdir(parents=True)
    (root / 'run_pipeline.py').write_text(script or '# worker\n')
    workspace_id(root)
    from package_fixtures import install_package
    install_package(root, 'slurm', tmp_path / 'archives')
    install_package(root, 'optuna', tmp_path / 'archives')
    config = cfg()
    config.site.execution = {**DEFAULTS, 'work_root': str(work), 'scratch_root': str(scratch)}
    activate_fake_bundle(root, 'g/e', config)
    return root, scratch, work, config



def fake_submit(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    calls = []
    def run(args, **kwargs):
        if args[0] == 'squeue': return SimpleNamespace(returncode=0, stdout='PENDING\n', stderr='')
        assert args[0] == 'sbatch'
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='123\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    return calls



@pytest.mark.parametrize('resources,flags', [({'gpus': 1, 'gres': 'gpu:h200:8', 'partition': 'gpu'}, []),
    ({'ntasks': 2}, []), ({}, ['--cpus-per-task=8']), ({}, ['-N8']), ({}, ['--requeue']),
    ({}, ['--output=/tmp/leak']), ({}, ['--constraint=node --nodes=8'])])
def test_reject_conflicting_actual_requests(resources, flags):
    config = cfg(resources=resources)
    config.site.extra_sbatch_flags = flags
    with pytest.raises(ConfigurationError): validate_allocation(config)



def test_normalized_request_and_environment(monkeypatch, tmp_path):
    from package_fixtures import install_package
    install_package(tmp_path, 'slurm', tmp_path.parent / (tmp_path.name + '-archives'))
    config = cfg(resources={'cores': 3, 'memory': '6G'})
    resources = validate_allocation(config)
    header = generate_sbatch_header('cpu', 'logs', config)
    assert f'--cpus-per-task={resources["cores"]}' in header
    assert '--ntasks=1' in header and '--gres' not in header
    monkeypatch.setenv('SBATCH_GRES', 'gpu:8')
    with pytest.raises(ConfigurationError, match='SBATCH_GRES'):
        execution.submit(tmp_path, 'g/e', [], config)
    assert not (tmp_path / 'src/usr/results').exists()



@pytest.mark.parametrize('change', [{'tuning': {'options': {'optuna': {'n_jobs': 2}}}}, {'plot_only': True},
                                  {'variability': {'parallel_runs': 2}}])
def test_unsupported_slurm_modes_are_explicit(change):
    with pytest.raises(ConfigurationError): validate_allocation(cfg(**change))



def test_scratch_worker_publishes_without_desktop_and_separates_allocations(tmp_path, monkeypatch):
    script = '''import json, os, pathlib, sqlite3, sys
root = pathlib.Path.cwd()
assert root == pathlib.Path(os.environ['THETA_WORKSPACE'])
assert root == pathlib.Path(os.environ['THETA_OUTPUT_ROOT'])
assert 'THETA_RESOURCE_POOL' not in os.environ
name = json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
p = root / 'src/usr/results/logs/g' / name / 'm/version_0'
p.mkdir(parents=True)
(p / 'metrics.csv').write_text('step,reward\\n1,7\\n')
db = pathlib.Path(os.environ['THETA_SQLITE_ROOT']) / 'study.sqlite'
db.parent.mkdir(parents=True)
connection = sqlite3.connect(db)
connection.execute('PRAGMA journal_mode=WAL')
connection.execute('CREATE TABLE trials(score)')
connection.execute('INSERT INTO trials VALUES (7)')
connection.commit()
print(root)
'''
    root, scratch, work, config = managed(tmp_path, script)
    fake_submit(monkeypatch)
    first = execution.submit(root, 'g/e', [], config)
    second = execution.submit(root, 'g/e', [], config)
    assert first['bundle_sha256'] == second['bundle_sha256']
    assert len(list((root / '.thetaide/bundles').glob('*.zip'))) == 1
    assert not list((root / 'src/usr/results/sites').glob('*/workspace'))
    for record in (first, second):
        # This synthetic producer writes a study without running a real tuner.
        # Declare its state artifact explicitly; ordinary allocations own none.
        OptunaTuner = package_module('optuna', 'component.py').OptunaTuner
        record['tuner_execution'] = OptunaTuner().execution_plan({}, execution_id=record['execution_id']).record()
        write_json(execution.receipt_path(root, record['execution_id']), record)
        assert execution.run(root, record['execution_id']) == 0
        final = read_json(execution.receipt_path(root, record['execution_id']))
        assert final['publication_state'] == 'sealed' and not final['artifacts_finalized']
        assert final['last_durable']['database']['sha256']
        publication = read_json(work / 'src/usr/results/sites' / record['execution_id'] / 'publication.json')
        exported = next(item for item in publication['files'] if item['category'] == 'database')
        copy = tmp_path / (record['execution_id'] + '.sqlite')
        copy.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in exported['chunks']))
        with sqlite3.connect(copy) as database:
            assert database.execute('select score from trials').fetchone() == (7,)
            assert database.execute('pragma journal_mode').fetchone() == ('delete',)
    assert not list(scratch.iterdir())
    assert not list(root.rglob('*-wal')) and not list(work.rglob('*-wal'))



def test_index_never_scans_payload_or_opens_database(tmp_path, monkeypatch):
    root, scratch, work, config = managed(tmp_path)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    record.update(state='completed', scheduler_state='completed')
    write_json(execution.receipt_path(root, record['execution_id']), record)
    files = [{'path': 'src/usr/results/checkpoints/g/e/model.ckpt', 'category': 'checkpoints', 'size': 0,
              'sha256': '0' * 64, 'chunks': []}]
    write_json(work / 'src/usr/results/sites' / record['execution_id'] / 'publication.json',
               {'schema': 'theta.publication/v1', 'sealed': True, 'files': files})
    monkeypatch.setattr(transfer, 'owned_files', lambda *_: pytest.fail('login scanned runtime'))
    monkeypatch.setattr(transfer, 'digest', lambda *_: pytest.fail('login hashed payload'))
    monkeypatch.setattr(sqlite3, 'connect', lambda *_a, **_k: pytest.fail('login opened SQLite'))
    for _ in range(2):
        listing = transfer.index(root, 'results', record['execution_id'])
        assert listing['files'][0]['sha256'] == '0' * 64
        transfer.release(root, listing['generation'])
    assert not list((root / 'src/usr/results/sites/.sync').iterdir())



def test_continuation_history_reconciliation_ownership_and_compatibility(tmp_path, monkeypatch):
    optuna = pytest.importorskip('optuna')
    from src.app.sites.continuation import pin, restore
    root, scratch, work, config = managed(tmp_path)
    config.tuning = {'component': 'optuna', 'options': {'optuna': {'n_jobs': 1, 'n_trials': 2}}}
    fake_submit(monkeypatch)
    parent = execution.submit(root, 'g/e', [], config)
    runtime = scratch / 'prior'
    runtime.mkdir()
    from package_fixtures import install_package
    # A real allocation runtime is the extracted snapshot, including its frozen providers.
    install_package(runtime, 'optuna', tmp_path / 'archives')
    (runtime / 'src/usr/results/sites' / parent['execution_id']).mkdir(parents=True)
    db = runtime / 'src/usr/results/optuna' / parent['execution_id'] / 'study.sqlite'
    db.parent.mkdir(parents=True)
    storage = optuna.storages.RDBStorage('sqlite:///' + str(db))
    study = optuna.create_study(storage=storage, study_name='cpu', direction='maximize')
    trial = study.ask(); study.tell(trial, 7)
    running = study.ask()
    publisher = Publisher(runtime, work, parent, parent['execution_policy'])
    publisher.finish(seconds=10)
    export = next(item for item in publisher.manifest['files'] if item['category'] == 'database')
    parent.update(state='timeout', scheduler_state='timeout')
    write_json(execution.receipt_path(root, parent['execution_id']), parent)
    request = {'execution_id': parent['execution_id'], 'export_sha256': export['sha256'], 'study_name': 'cpu'}
    config.study_continuation = request
    child = execution.submit(root, 'g/e', [], config)
    assert child['execution_id'] != parent['execution_id'] and child['result_experiment_id'] != parent['result_experiment_id']
    assert child['continuation']['training_resume'] is False
    fresh = scratch / 'new'; fresh.mkdir()
    install_package(fresh, 'optuna', tmp_path / 'archives')
    restore(fresh, child)
    url = 'sqlite:///' + str(fresh / 'src/usr/results/optuna' / child['execution_id'] / 'study.sqlite')
    continued = optuna.load_study(storage=url, study_name='cpu')
    assert [trial.state.name for trial in continued.trials] == ['COMPLETE', 'FAIL']
    assert continued.trials[0].value == 7 and continued.ask().number == 2
    with pytest.raises(ValueError, match='owner'): execution.submit(root, 'g/e', [], config)
    modified = dict(child); modified['config'] = {**child['config'], 'objective': {'metric': 'different'}}
    with pytest.raises(ValueError, match='incompatible'): pin(root, modified, request)
    broken = {**request, 'export_sha256': 'f' * 64}
    with pytest.raises(ValueError, match='missing'): pin(root, child, broken)
    # Checkpoint/study dependencies remain pinned even after the new owner terminates.
    child.update(state='completed', scheduler_state='completed')
    write_json(execution.receipt_path(root, child['execution_id']), child)
    parent = execution.finalize(root, parent['execution_id'])
    proof = {'project_id': parent['project_id'], 'files': read_json(execution.receipt_path(root, parent['execution_id']).parent / 'artifacts.json')['files']}
    with pytest.raises(ValueError, match='pins'): transfer.clean_remote(root, parent['execution_id'], proof)
    storage.engine.dispose()



def test_gc_preserves_active_unknown_and_legacy_and_releases_verified_payload(tmp_path, monkeypatch):
    from src.app.sites.maintenance import collect
    root, scratch, work, config = managed(tmp_path)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    lease = root / 'src/usr/results/sites/.sync' / ('b' * 32)
    write_json(lease / '.theta-managed.json', {'project_id': record['project_id'], 'expires_at': time.time() + 7200})
    old = root / 'src/usr/results/sites/.sync/legacy'
    old.mkdir()
    bundle = root / '.thetaide/bundles' / (record['bundle_sha256'] + '.zip')
    assert collect(root, apply=True)['candidates'] == []
    assert bundle.exists() and old.exists() and lease.exists()
    orphan = root / '.thetaide/bundles' / ('c' * 64 + '.zip')
    orphan.write_bytes(b'garbage')
    assert len(collect(root)['candidates']) == 1 and orphan.exists()
    collect(root, apply=True)
    assert not orphan.exists()
    write_json(lease / '.theta-managed.json', {'project_id': record['project_id'], 'expires_at': 0})
    collect(root, apply=True)
    assert not lease.exists() and old.exists()



def test_scheduler_queries_are_batched_cached_and_preserve_training_outcome(tmp_path, monkeypatch):
    from package_fixtures import install_package
    install_package(tmp_path, 'slurm', tmp_path.parent / (tmp_path.name + '-archives'))
    records = []
    for index in range(3):
        record = {'execution_id': f'{index:032x}', 'project_id': workspace_id(tmp_path), 'job_id': str(index + 1),
                  'scheduler_component': 'slurm', 'state': 'failed' if index == 1 else 'running'}
        write_json(execution.receipt_path(tmp_path, record['execution_id']), record)
        records.append(record)
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout='1|RUNNING|\n' if args[0] == 'squeue' else '2|COMPLETED|1:0\n3|TIMEOUT|0:15\n')
    monkeypatch.setattr(subprocess, 'run', scheduler)
    execution.refresh_many(tmp_path, records)
    assert calls == ['squeue', 'sacct']
    assert records[1]['training_state'] == 'failed' and records[1]['scheduler_state'] == 'completed'
    assert records[1]['state'] == 'failed'  # scheduler success cannot overwrite training failure
    execution.refresh_many(tmp_path, records)
    assert len(calls) == 2



PILOT_RECIPES = {
    'cpu_smoke': """# @package _global_
defaults:
  - override /site: cluster/cpu
  - override /env: gym_classic_control/cartpole
  - _self_
paradigm: online_rl
group: pilot
experiment_id: cpu_smoke
seed: 42
total_timesteps: 64
eval_episodes: 1
no_plot: true
monitors: {}
methods:
  cpu:
    agent: {name: sb3/ppo, n_steps: 16, batch_size: 8, n_epochs: 1}
    model: {name: basics/dnn, hidden_sizes: [16, 16]}
""",
    'cpu_study': """# @package _global_
defaults:
  - /experiment/pilot/cpu_smoke
  - _self_
experiment_id: cpu_study
tuning:
  component: optuna
  metric: eval/reward
  direction: maximize
  options:
    optuna: {n_trials: 3, n_jobs: 1, sampler: random, seed: 42, study_name: cpu_pilot, study_mode: new, pruner: none}
methods:
  cpu:
    tune:
      agent.learning_rate: {type: log_float, low: 0.0001, high: 0.001}
""",
}


@pytest.mark.parametrize('recipe', ['cpu_smoke', 'cpu_study'])
def test_real_cpu_pipeline_in_fresh_scratch_with_frozen_providers(tmp_path, monkeypatch, recipe):
    """Train SB3 on CartPole through the managed CPU path, with real packages from THETA_HUB_ROOT."""
    pytest.importorskip('stable_baselines3')
    from package_fixtures import cluster_site, install_package
    from src.app.pipeline.compose import compose_experiment, validate_composed
    from src.app.workspace import WorkspacePaths
    root, scratch, work = [tmp_path / name for name in ('control', 'scratch', 'work')]
    for path in (root, scratch, work): path.mkdir()
    # The control workspace starts empty: no authored or installed packages and no package
    # selections of whoever runs the test. The bundle brings the application.
    paths = WorkspacePaths(root)
    for package in ('sb3-suite', 'gym_classic_control', 'dnn', 'slurm', 'optuna'):
        install_package(paths, package, tmp_path / 'archives')
    cluster_site(paths, execution=DEFAULTS)
    target = root / 'src/usr/config/experiment/pilot'
    target.mkdir(parents=True, exist_ok=True)
    for name, text in PILOT_RECIPES.items():
        (target / f'{name}.yaml').write_text(text)
    monkeypatch.setenv('THETA_WORKSPACE', str(root))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    overrides = [f'++site.execution.work_root={work}', f'++site.execution.scratch_root={scratch}',
                 f'++site.python_executable={sys.executable}']
    # What site push does: bundle the experiment, then activate it in the remote workspace.
    stage = tmp_path / 'upload'
    manifest = bundles.prepare(root, f'pilot/{recipe}', overrides, stage)
    assert {item['id'] for item in manifest['providers']} >= {'sb3-suite', 'gym_classic_control', 'slurm'}
    transfer.activate(root, stage)
    composed = compose_experiment(f'pilot/{recipe}', overrides, workspace=paths)
    assert validate_composed(composed) == []
    fake_submit(monkeypatch)
    record = execution.submit(root, composed.experiment, overrides, composed.cfg)
    code = execution.run(root, record['execution_id'])
    publication = read_json(work / 'src/usr/results/sites' / record['execution_id'] / 'publication.json')
    log = next(item for item in publication['files'] if item['path'].endswith('pipeline.log'))
    text = b''.join((work / chunk['source']).read_bytes() for chunk in log['chunks']).decode(errors='replace')
    assert code == 0, text
    assert publication['sealed']
    assert any(item['path'].endswith('metrics.csv') for item in publication['files'])
    assert any(item['category'] == 'checkpoints' for item in publication['files'])
    if recipe == 'cpu_study':
        assert publication['durable']['database']['sha256']
        assert any(item['path'].endswith('tuning_summary.json') for item in publication['files'])
        parent = read_json(execution.receipt_path(root, record['execution_id']))
        parent['scheduler_state'] = 'completed'
        write_json(execution.receipt_path(root, record['execution_id']), parent)
        export = next(item for item in publication['files'] if item['category'] == 'database')
        with __import__('omegaconf').open_dict(composed.cfg):
            composed.cfg.study_continuation = {'execution_id': record['execution_id'],
                'export_sha256': export['sha256'], 'study_name': 'cpu_pilot'}
            composed.cfg.tuning.options.optuna.n_trials = 1
        continued = execution.submit(root, composed.experiment,
            [*overrides, '++tuning.options.optuna.n_trials=1'], composed.cfg)
        code = execution.run(root, continued['execution_id'])
        manifest = read_json(work / 'src/usr/results/sites' / continued['execution_id'] / 'publication.json')
        log = next(item for item in manifest['files'] if item['path'].endswith('pipeline.log'))
        output = b''.join((work / chunk['source']).read_bytes() for chunk in log['chunks']).decode(errors='replace')
        assert code == 0, output
        database = next(item for item in manifest['files'] if item['category'] == 'database')
        copy = tmp_path / 'continued.sqlite'
        copy.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in database['chunks']))
        import optuna
        study = optuna.load_study(study_name='cpu_pilot', storage='sqlite:///' + str(copy))
        assert len(study.trials) == 4 and all(trial.state.name == 'COMPLETE' for trial in study.trials)



def test_signal_stageout_and_abrupt_unsealed_boundary(tmp_path, monkeypatch):
    import signal
    script = '''import json,os,pathlib,sys,time
root=pathlib.Path.cwd()
name=json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
p=root/'src/usr/results/logs/g'/name/'m/version_0'
p.mkdir(parents=True)
(p/'metrics.csv').write_text('step,reward\\n1,7\\n')
print('ready',flush=True)
time.sleep(30)
'''
    root, scratch, work, config = managed(tmp_path, script)
    config.site.execution.metrics_seconds = 1
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    # Use a real separate allocation owner process, not a mocked training Popen.
    owner = subprocess.Popen([sys.executable, '-m', 'src.app.sites', '_run', '--root', str(root), record['execution_id']],
                             env={**os.environ, 'PYTHONPATH': str(IDE)},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    manifest = work / 'src/usr/results/sites' / record['execution_id'] / 'publication.json'
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if any(item['path'].endswith('metrics.csv') for item in read_json(manifest).get('files', [])): break
        time.sleep(.05)
    else:
        owner.kill(); pytest.fail('Allocation did not publish before signal')
    assert not read_json(manifest)['sealed']
    owner.send_signal(signal.SIGTERM)
    out, error = owner.communicate(timeout=10)
    final = read_json(execution.receipt_path(root, record['execution_id']))
    assert final['state'] == 'cancelled', out + error
    assert final['publication_state'] == 'sealed'
    assert final['artifacts_finalized'] is False  # never invent scheduler termination
    # A payload stopped without finish remains unsealed and cannot be finalized.
    final['scheduler_state'] = 'cancelled'
    write_json(execution.receipt_path(root, record['execution_id']), final)
    value = read_json(manifest); value['sealed'] = False; write_json(manifest, value)
    with pytest.raises(ValueError, match='unsealed'): execution.finalize(root, record['execution_id'])



def test_managed_cleanup_requires_full_copy_and_transfer_release(tmp_path, monkeypatch):
    script = '''import json,pathlib,sys
name=json.loads(next(a.split('=',1)[1] for a in sys.argv if a.startswith('++experiment_id=')))
root=pathlib.Path('src/usr/results/checkpoints/g')/name
root.mkdir(parents=True)
(root/'model.ckpt').write_bytes(b'committed checkpoint')
'''
    root, scratch, work, config = managed(tmp_path, script)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], config)
    assert execution.run(root, record['execution_id']) == 0
    path = execution.receipt_path(root, record['execution_id'])
    value = read_json(path); value['scheduler_state'] = 'completed'; write_json(path, value)
    local = tmp_path / 'local'; local.mkdir()
    write_json(local / '.thetaide/workspace.json', {'id': record['project_id']})
    transfer.configure(local, 'cluster', 'fake', str(root))
    def remote(_root, site, command, *args, **kwargs):
        if command == '_index': return transfer.index(root, args[1])
        if command == '_release': return transfer.release(root, args[0])
        if command == '_renew': return transfer.renew(root, args[0])
        if command == '_clean': return transfer.clean_remote(root, args[0], json.loads(kwargs['input']), preview='--preview' in args)
        pytest.fail(command)
    monkeypatch.setattr(transfer, 'remote', remote)
    def pull(args, **kwargs):
        names = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--files-from=')))
        origin = Path(args[-2].split(':', 1)[1])
        for name in names.read_bytes().decode().strip('\0').split('\0'):
            target = Path(args[-1]) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / name, target)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(transfer, 'run_transfer', pull)
    transfer.sync(local, 'cluster', 'metrics')
    with pytest.raises(FileNotFoundError): transfer.clean(local, 'cluster', record['execution_id'])
    transfer.sync(local, 'cluster', 'results')
    preview = transfer.clean(local, 'cluster', record['execution_id'], preview=True)
    assert preview['payload_bytes'] > 0 and (work / 'src/usr/results/sites' / record['execution_id']).exists()
    lease = transfer.index(root, 'results')
    with pytest.raises(ValueError, match='lease'): transfer.clean(local, 'cluster', record['execution_id'])
    transfer.release(root, lease['generation'])
    assert transfer.clean(local, 'cluster', record['execution_id'])['removed'] > 0
    assert not (work / 'src/usr/results/sites' / record['execution_id']).exists()
    assert not (work / 'inputs' / record['execution_id']).exists()
    assert path.exists() and (path.parent / 'artifacts.json').exists()
    assert (local / 'src/usr/results/checkpoints/g' / record['result_experiment_id'] / 'model.ckpt').read_bytes() == b'committed checkpoint'



def test_site_gres_template_cannot_bypass_normalized_count():
    config = cfg(resources={'partition': 'gpu', 'gpus': 1, 'gpu_type': 'h200'})
    config.site.gres_format = 'gpu:{gpu_type}:8'
    with pytest.raises(ConfigurationError, match='match'): validate_allocation(config)
