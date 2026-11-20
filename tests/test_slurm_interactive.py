"""Interactive Slurm launch contracts with synthetic profiles and no cluster connection."""
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from package_fixtures import package_module
from src.app.api import config as config_api
from src.app.api import sites
from test_slurm_execution import managed

commands = package_module('slurm', 'commands.py')
execution = package_module('slurm', 'execution.py')
BUNDLE = 'a' * 64


def test_interactive_launcher_prepares_uploads_and_forwards_quoted_compute_command(tmp_path, monkeypatch):
    events = []
    site = {'resources': {'cores': 4}, 'execution': {'runtime': 'scratch'}, 'interactive_allocation': {
        'prepare': ['network-setup'], 'command': ['allocation-helper', '--']}}
    connection = {'remote_root': '/home/test/theta space', 'python': '/home/test/theta space/venv/bin/python'}
    monkeypatch.setattr(config_api, 'compose_experiment', lambda *_a, **_k: SimpleNamespace(cfg=SimpleNamespace(site=site)))
    monkeypatch.setattr(execution, 'validate_allocation', lambda *_a, **_k: {'cores': 4, 'ntasks': 1, 'time': '0:15:00'})
    monkeypatch.setattr(sites, 'connection', lambda *_: connection)
    monkeypatch.setattr(sites, 'load_site', lambda *_: site)
    monkeypatch.setattr(subprocess, 'run', lambda argv, **_: events.append(('prepare', argv)))
    monkeypatch.setattr(sites, 'push', lambda *_, **_k: events.append(('push',)) or {'bundle_sha256': BUNDLE})
    monkeypatch.setattr(os, 'execvp', lambda program, argv: events.append(('exec', program, argv)))

    assert commands.launch_in_terminal(tmp_path, 'cluster/default', 'group/Smoke Test') == 0
    assert events[:2] == [('prepare', ['network-setup']), ('push',)]
    _, program, argv = events[2]
    assert program == 'allocation-helper'
    assert argv[:2] == ['allocation-helper', '--']
    assert shlex.split(argv[2]) == [
        '--ntasks=1', '--cpus-per-task=4', '--time=0:15:00', '--chdir=/home/test/theta space',
        'srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--chdir=/home/test/theta space',
        'env', f'{commands.INTERACTIVE_BUNDLE_ENV}={BUNDLE}', 'venv/bin/python', 'run_pipeline.py', 'group/Smoke Test',
    ]


def test_interactive_submit_uses_current_step_and_skips_sbatch(tmp_path, monkeypatch):
    root, _, _, cfg = managed(tmp_path)
    cfg.site.interactive_allocation = {'command': ['allocation-helper']}
    manifest = sites.current_bundle(root)
    monkeypatch.setenv('SLURM_JOB_ID', '721')
    monkeypatch.setenv(commands.INTERACTIVE_BUNDLE_ENV, manifest['bundle_sha256'])

    with pytest.raises(ValueError, match='compute step'):
        execution.submit(root, 'g/e', [], cfg)

    monkeypatch.setenv('SLURM_STEP_ID', '0')
    monkeypatch.setenv(commands.INTERACTIVE_BUNDLE_ENV, BUNDLE)
    with pytest.raises(ValueError, match='bundle changed'):
        execution.submit(root, 'g/e', [], cfg)

    monkeypatch.setenv(commands.INTERACTIVE_BUNDLE_ENV, manifest['bundle_sha256'])
    def run(argv, *, cwd):
        assert cwd == root
        assert argv[0] == '/bin/bash'
        assert argv[1].endswith('/job.sbatch')
        assert 'unset THETA_CONFIG_SNAPSHOT THETA_INPUTS_SNAPSHOT THETA_PIPELINE_PID' in Path(argv[1]).read_text()
        receipt, = (root / 'src/usr/results/sites').glob('*/submission.json')
        record = json.loads(receipt.read_text())
        assert record['job_id'] == '721'
        record['state'] = 'completed'
        sites.write_json(receipt, record)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, 'run', run)
    assert execution.submit(root, 'g/e', [], cfg)['state'] == 'completed'
