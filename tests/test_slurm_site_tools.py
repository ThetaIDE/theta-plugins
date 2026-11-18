"""Slurm's site commands: first-run setup, shortcuts and the copied submission command."""
import os
import sys
from pathlib import Path

import pytest
import PyQt6.QtWebEngineWidgets  # noqa: F401  (must load before the QApplication exists)
from PyQt6.QtWidgets import QApplication

from package_fixtures import IDE, package_module, package_workspace
from src.app.sites import transfer
from src.app.sites.catalog import defaults_path
from src.app.sites.transport import ssh_arguments

_slurm_commands = package_module('slurm', 'commands.py')
bootstrap_command, submission_command, shortcuts, launch_in_terminal = (
    _slurm_commands.bootstrap_command, _slurm_commands.submission_command, _slurm_commands.shortcuts,
    _slurm_commands.launch_in_terminal)
app = QApplication.instance() or QApplication(['test-slurm-site-tools'])


def component(root):
    """An authored two-variant Slurm site, ``example``."""
    from src.app.sites.catalog import configure, create
    create(root, 'example', {'scheduler': 'slurm', 'resources': {'cores': 4}},
           {'cpu': {'resources': {'gpus': 0}}, 'gpu': {'resources': {'gpus': 1}}})
    configure(root, 'example/gpu', {'host': 'shared-alias', 'remote_root': '/scratch/theta'})


def test_user_configuration_and_setup_are_offline(tmp_path, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network attempted')))
    transfer.configure(tmp_path, 'cluster', 'alias', '/scratch/project')
    profile = tmp_path / 'src/usr/sites/cluster/configs/site/cluster/_defaults.yaml'
    from src.app.sites.catalog import configure
    configure(tmp_path, 'cluster', {'rsync_executable': '/custom/rsync'})
    assert transfer.connection(tmp_path, 'cluster')['rsync_executable'] == '/custom/rsync'
    script = bootstrap_command(tmp_path, 'cluster', 'g/test')
    assert 'site push cluster g/test' in script
    # Setup installs the requirements the pushed bundle lists, not those of every installed package.
    assert 'pip install -r .thetaide/requirements.txt' in script
    assert 'python3 -m venv' in script
    paths = shortcuts(tmp_path, 'cluster')
    assert len(paths) == 3
    push = next(Path(path) for path in paths if '/theta-push-' in path).read_text()
    assert push.rstrip().endswith('cluster "$@"')  # The shortcut takes the experiment to push.
    assert any('src/usr/sites/cluster/ssh_config' in value for value in ssh_arguments(tmp_path, transfer.connection(tmp_path, 'cluster')))
    command = submission_command(transfer.connection(tmp_path, 'cluster'), 'g/test', 'cluster')
    assert command == 'venv/bin/python3 run_pipeline.py g/test'


def test_terminal_upload_finishes_before_interactive_ssh(tmp_path, monkeypatch, capsys):
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    events = []
    def upload(root, site, experiment, **kwargs):
        events.append(('upload', root, site, experiment))
        assert kwargs == {'live_output': True}
        print('rsync progress visible')
    def replace(program, command): events.append(('ssh', program, command))
    monkeypatch.setattr(transfer, 'push', upload)
    monkeypatch.setattr(transfer, 'sync', lambda *_a, **_k: pytest.fail('Run attempted to download results'))
    monkeypatch.setattr(os, 'execvp', replace)
    assert launch_in_terminal(tmp_path, 'cluster', 'cartpole/single') == 0
    assert [item[0] for item in events] == ['upload', 'ssh']
    assert events[0][3] == 'cartpole/single'  # The upload is that experiment's bundle.
    command = events[1][2]
    assert command[0] == 'ssh' and '-t' in command and 'fake' in command
    assert not any('sbatch' in str(item) for item in command)
    output = capsys.readouterr().out
    assert output.index('rsync progress visible') < output.index('Opening the cluster SSH terminal')
    assert 'paste the copied command' in output
    assert 'venv/bin/python3 run_pipeline.py cartpole/single' in output
    assert 'dry_run=true' not in output and '--verify' not in output and 'site=' not in output



def test_failed_upload_never_opens_ssh(tmp_path, monkeypatch):
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    def fail(*_a, **_k): raise RuntimeError('Permission denied')
    monkeypatch.setattr(transfer, 'push', fail)
    monkeypatch.setattr(os, 'execvp', lambda *_: pytest.fail('SSH opened after failed upload'))
    with pytest.raises(RuntimeError, match='Permission denied'):
        launch_in_terminal(tmp_path, 'cluster', 'cartpole/single')



def test_minimal_command_uses_project_relative_interpreter_and_quotes_experiment():
    import shlex
    submission_command = _slurm_commands.submission_command
    config = {'remote_root': '/home/user/theta', 'python': '/home/user/theta/venv/bin/python3'}
    command = submission_command(config, 'cartpole/my run', 'cluster/cpu')
    assert shlex.split(command) == ['venv/bin/python3', 'run_pipeline.py', 'cartpole/my run']
    assert '&&' not in command and 'site=' not in command
    config['python'] = '/home/user/env/bin/python3'
    assert shlex.split(submission_command(config, 'g/e', 'cluster'))[0] == '/home/user/env/bin/python3'



def test_upload_output_and_manual_ssh_input_share_one_real_pty(tmp_path):
    import time
    from frontend.terminal.pty_session import PtySession
    mock_ssh = tmp_path / 'mock_ssh.py'
    mock_ssh.write_text('print("SSH READY",flush=True)\ncommand=input()\nprint("USER COMMAND: "+command,flush=True)\n')
    driver = tmp_path / 'launch.py'
    project = IDE
    workspace = package_workspace('slurm')
    driver.write_text(
        'import sys,time\n'
        f'sys.path.insert(0,{str(project)!r})\n'
        'from pathlib import Path\n'
        'from src.app.sites import transfer\n'
        'from src.app.workspace import WorkspacePaths\n'
        'from src.app.core.provider_catalog import get_catalog\n'
        'from src.app.core.provider_loader import load_module\n'
        f'paths = WorkspacePaths(Path({str(workspace.root)!r}), {str(workspace.data_dir)!r})\n'
        "slurm = load_module(get_catalog(paths).package('slurm'), 'commands.py', paths=paths)\n"
        'transfer.connection=lambda *_: {"host":"mock","remote_root":"/home/user/theta","python":"venv/bin/python3"}\n'
        'def push(*args,**kwargs):\n'
        ' assert kwargs["live_output"]\n'
        ' print("RSYNC START",flush=True)\n'
        ' time.sleep(.1)\n'
        ' print("RSYNC 100%",flush=True)\n'
        'transfer.push=push\n'
        f'slurm.ssh_arguments=lambda *_: [sys.executable,{str(mock_ssh)!r}]\n'
        f'slurm.launch_in_terminal({str(tmp_path)!r},"cluster","g/exp")\n')
    session = PtySession(command=[sys.executable, str(driver)], cwd=str(tmp_path))
    chunks, exits = [], []
    session.data_ready.connect(chunks.append)
    session.process_exited.connect(exits.append)
    try:
        assert session.start()
        deadline = time.monotonic() + 10
        while 'SSH READY' not in ''.join(chunks) and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        output = ''.join(chunks)
        assert output.index('RSYNC START') < output.index('RSYNC 100%') < output.index('SSH READY')
        assert 'USER COMMAND:' not in output
        session.write('venv/bin/python3 run_pipeline.py g/exp\n')
        deadline = time.monotonic() + 5
        while not exits and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        assert 'USER COMMAND: venv/bin/python3 run_pipeline.py g/exp' in ''.join(chunks)
        assert exits == [0]
    finally:
        session.close()


def test_shortcuts_do_not_reconfigure_site(tmp_path):
    from package_fixtures import package_module
    shortcuts = package_module('slurm', 'commands.py').shortcuts
    component(tmp_path)
    path = defaults_path(tmp_path, 'example/gpu')
    original = path.read_bytes()
    outputs = shortcuts(tmp_path, 'example/gpu')
    assert all(Path(p).is_relative_to(tmp_path / 'src/usr/sites/example') for p in outputs)
    assert path.read_bytes() == original
