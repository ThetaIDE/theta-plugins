"""Portable tuner publication and continuation through Slurm, with no Optuna or SQLite state."""
import builtins
import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from package_fixtures import activate_fake_bundle, fake_bundles, install_package
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.tuner_execution import read_remote_snapshots, record_plan
from src.app.sites import execution
from src.app.sites.continuation import restore
from src.app.sites.storage import read_json, write_json
from src.app.workspace import WorkspacePaths
from test_slurm_execution import fake_submit, managed

FIXTURE = Path(__file__).parent / 'fixtures/portable_tuner'


def install(root):
    """Slurm from this checkout plus an independent JSON tuner authored in the workspace."""
    install_package(root, 'slurm', root.parent / (root.name + '-archives'))
    shutil.copytree(FIXTURE, root / 'src/usr/tuners/fixture-grid')
    return WorkspacePaths(root)


def refresh_bundle(root, cfg):
    activate_fake_bundle(root, 'g/e', cfg)



def test_independent_worker_publication_monitor_and_continuation(tmp_path, monkeypatch):
    script = '''import json, os
from pathlib import Path
p = Path(os.environ['GRID_STATE_PATH'])
p.parent.mkdir(parents=True)
p.write_text(json.dumps({'trials': [{'id': 0, 'state': 'complete', 'value': 7}, {'id': 1, 'state': 'running'}]}))
'''
    root, scratch, work, cfg = managed(tmp_path, script)
    paths = install(root)
    refresh_bundle(root, cfg)
    cfg.tuning = {'component': 'fixture-grid', 'options': {'fixture-grid': {'attempts': 2}}}
    fake_submit(monkeypatch)
    original_import = builtins.__import__
    def block(name, *args, **kwargs):
        if name.split('.')[0] in ('optuna', 'sqlalchemy') or 'bundled.optuna' in name:
            raise AssertionError('Independent tuner imported Optuna')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', block)
    parent = execution.submit(root, 'g/e', [], cfg)
    assert parent['tuner_execution']['component'] == 'fixture-grid'
    assert 'optuna' not in json.dumps(parent['overrides'])
    assert execution.run(root, parent['execution_id']) == 0
    parent = read_json(execution.receipt_path(root, parent['execution_id']))
    manifest = read_json(work / 'src/usr/results/sites' / parent['execution_id'] / 'publication.json')
    export = next(item for item in manifest['files'] if item['category'] == 'database')
    assert export['path'].endswith('/search.json')
    assert 'optuna' not in manifest['dependencies']
    # Materialize the verified closed export exactly as the generic downloader does.
    target = root / export['path']; target.parent.mkdir(parents=True)
    target.write_bytes(b''.join((work / chunk['source']).read_bytes() for chunk in export['chunks']))
    assert read_remote_snapshots(root, parent)[0][1]['trials'][0]['value'] == 7
    parent.update(state='timeout', scheduler_state='timeout')
    write_json(execution.receipt_path(root, parent['execution_id']), parent)
    cfg.tuning.options['fixture-grid'].attempts = 5
    cfg.study_continuation = {'execution_id': parent['execution_id'], 'export_sha256': export['sha256'], 'study_name': 'grid'}
    child = execution.submit(root, 'g/e', [], cfg)
    fresh = scratch / 'continued'; fresh.mkdir()
    install(fresh)
    restore(fresh, child)
    plan = record_plan(child)
    continued = read_json(WorkspacePaths(fresh).results / plan.artifact('search').path)
    assert [item['state'] for item in continued['trials']] == ['complete', 'failed']
    assert continued['parent'] == parent['execution_id']
    assert not list(root.rglob('*.sqlite'))



def test_ordinary_remote_execution_has_no_tuner_or_storage_defaults(tmp_path, monkeypatch):
    script = 'import os\nassert "THETA_SQLITE_ROOT" not in os.environ\n'
    root, scratch, work, cfg = managed(tmp_path, script)
    fake_submit(monkeypatch)
    monkeypatch.setattr('src.app.pipeline.tuners.get_tuner', lambda *a, **kw: pytest.fail('Ordinary execution resolved a tuner'))
    record = execution.submit(root, 'g/e', [], cfg)
    assert record['tuner_execution'] is None
    assert record_plan(record) is None
    assert execution.run(root, record['execution_id']) == 0



def test_shared_workspace_allocation_applies_portable_tuner_environment(tmp_path, monkeypatch):
    script = '''import os
from pathlib import Path



FIXTURE = Path(__file__).parent / 'fixtures/portable_tuner'


def install(root):
    """Slurm from this checkout plus an independent JSON tuner authored in the workspace."""
    install_package(root, 'slurm', root.parent / (root.name + '-archives'))
    target = root / 'src/usr/tuners/fixture-grid'
    shutil.copytree(FIXTURE, target)
    return WorkspacePaths(root)
p=Path(os.environ['GRID_STATE_PATH']); p.parent.mkdir(parents=True)
p.write_text('{"trials": []}')
'''
    root, scratch, work, cfg = managed(tmp_path, script)
    install(root)
    cfg.site.execution = None
    cfg.tuning = {'component': 'fixture-grid'}
    fake_bundles(monkeypatch, cfg)
    fake_submit(monkeypatch)
    record = execution.submit(root, 'g/e', [], cfg)
    assert execution.run(root, record['execution_id']) == 0
    assert (WorkspacePaths(root).results / record_plan(record).artifact('search').path).is_file()


def test_slurm_rejects_parallel_tuner_trials(tmp_path):
    paths = install(tmp_path)
    cfg = OmegaConf.create({'tuning': {'component': 'fixture-grid', 'options': {'fixture-grid': {'workers': 2}}}})
    cfg.site = {'scheduler': 'slurm'}
    with pytest.raises(ConfigurationError, match='concurrency=1'):
        execution.validate_allocation(cfg, paths=paths)
