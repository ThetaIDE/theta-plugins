"""Slurm project setup and login handoff commands. Network execution is explicit."""
from pathlib import Path
import os, shlex, subprocess, sys

INTERACTIVE_BUNDLE_ENV = "THETA_SLURM_INTERACTIVE_BUNDLE"
from src.app.api.sites import BUNDLE_REQUIREMENTS, rsync_shell, safe_path, ssh_arguments
from src.app.api.workspace import WorkspacePaths

def submission_command(config, experiment, site):
    """The saved experiment owns its site/options; the shell is already at project root."""
    python = Path(config['python'])
    remote_root = Path(config['remote_root'])
    if python.is_absolute() and python.is_relative_to(remote_root):
        python = python.relative_to(remote_root)
    return shlex.join([str(python), 'run_pipeline.py', experiment])


def _command_vector(settings, key, *, optional=False):
    value = settings.get(key)
    if optional and value is None:
        return []
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'interactive_allocation.{key} must be a nonempty list of command arguments')
    return [os.path.expanduser(value[0]), *value[1:]]


def _interactive_arguments(config, experiment, site, resources, extra_flags, digest):
    """Return a shell-quoted argument string for a local Slurm allocation helper."""
    options = {'nodes': 'nodes', 'ntasks': 'ntasks', 'cores': 'cpus-per-task',
               'memory': 'mem', 'time': 'time', 'partition': 'partition',
               'account': 'account', 'qos': 'qos', 'gres': 'gres'}
    args = [f'--{flag}={resources[key]}' for key, flag in options.items() if resources.get(key)]
    root = config['remote_root']
    args += [f'--chdir={root}', *extra_flags]
    args += ['srun', '--nodes=1', '--ntasks=1', f'--cpus-per-task={resources["cores"]}', f'--chdir={root}',
             'env', f'{INTERACTIVE_BUNDLE_ENV}={digest}']
    return shlex.join(args) + ' ' + submission_command(config, experiment, site)


def _launch_interactive(root, site, experiment, config, settings, push):
    from src.app.api.config import compose_experiment
    from .execution import validate_allocation

    if not isinstance(settings, dict):
        raise ValueError('interactive_allocation must be a mapping')
    launcher = _command_vector(settings, 'command')
    prepare = _command_vector(settings, 'prepare', optional=True)
    paths = WorkspacePaths(root)
    cfg = compose_experiment(experiment, workspace=paths).cfg
    resources = validate_allocation(cfg, paths=paths)
    if not cfg.site.get('execution'):
        raise ValueError('Interactive allocation requires a scratch execution policy')
    if prepare:
        subprocess.run(prepare, cwd=root, check=True)
    uploaded = push(root, site, experiment, live_output=True)
    arguments = _interactive_arguments(config, experiment, site, resources,
                                        cfg.site.get('extra_sbatch_flags') or (), uploaded['bundle_sha256'])
    print('Project upload complete. Requesting an interactive allocation and starting the compute step:', flush=True)
    print(arguments, flush=True)
    os.execvp(launcher[0], [*launcher, arguments])
    return 0


def launch_in_terminal(root, site, experiment):
    """Called inside the local terminal PTY. Upload, then replace this process with SSH."""
    import os
    from src.app.api.sites import connection, load_site, push
    root = Path(root).resolve()
    config = connection(root, site)
    print(f'Theta: preparing {experiment} for {site}', flush=True)
    interactive = load_site(root, site).get('interactive_allocation')
    if interactive is not None:
        return _launch_interactive(root, site, experiment, config, interactive, push)
    push(root, site, experiment, live_output=True)
    print('\nProject upload complete. Opening the cluster SSH terminal.', flush=True)
    print('Once connected, paste the copied command and press Enter to validate and submit.', flush=True)
    print('Run command (also copied by the desktop):\n' + submission_command(config, experiment, site), flush=True)
    command = [*ssh_arguments(root, config, True), '--', config['host'],
               'cd ' + shlex.quote(config['remote_root']) + ' && exec bash -l']
    os.execvp(command[0], command)
    return 0


def bootstrap_command(root, site, experiment):
    """First-run setup for one experiment: environment, bundle upload, then the bundle's requirements."""
    from src.app.api.sites import connection, load_site
    root = Path(root).resolve()
    if load_site(root, site).get('scheduler') == 'local':
        return shlex.join([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    config = connection(root, site)
    profile = load_site(root, site)
    runtime_python = profile.get('python_executable')
    if runtime_python in (None, 'auto'):
        runtime_python = str(Path(config['remote_root']) / profile.get('venv_dir', 'venv') / 'bin/python3')
    elif not Path(runtime_python).is_absolute():
        runtime_python = str(Path(config['remote_root']) / runtime_python)
    environment_dir = str(Path(runtime_python).parent.parent)
    if Path(runtime_python).parent.name != 'bin':
        raise ValueError('Generate setup for a venv bin/python interpreter, or provision the custom interpreter manually')
    setup = ('mkdir -p ' + shlex.quote(config['remote_root']) + ' && cd ' + shlex.quote(config['remote_root']) +
             ' && python3 -m venv ' + shlex.quote(environment_dir))
    # The pushed bundle lists the requirements of the providers this experiment selects.
    install = ('cd ' + shlex.quote(config['remote_root']) + ' && ' +
               shlex.join([config['python'], '-m', 'pip', 'install', '-r', BUNDLE_REQUIREMENTS]))
    # Only emit text: the user chooses when to execute SSH, upload and package installation.
    return '\n'.join(['set -e', shlex.join([*ssh_arguments(root, config, True), '--', config['host'], setup]),
        shlex.join([sys.executable, str(WorkspacePaths(root).application / 'run_pipeline.py'), 'site', 'push', site, experiment, '--root', str(root)]),
        shlex.join([*ssh_arguments(root, config, True), '--', config['host'], install])])


def shortcuts(root, site):
    from src.app.api.sites import connection
    config = connection(root, site)
    directory = safe_path(root, config.get('_shortcuts', {}).get('directory', 'src/usr/sites/' + site.split('/')[0] + '/bin'))
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    import hashlib
    tag = hashlib.sha256(site.encode()).hexdigest()[:12]
    # The launcher belongs to the application, which need not be the workspace.
    launcher = str(WorkspacePaths(root).application / 'run_pipeline.py')
    commands = {
        'ssh': shlex.join([sys.executable, launcher, 'site', 'ssh', site, '--root', str(root)]),
        'push': shlex.join([sys.executable, launcher, 'site', 'push', '--root', str(root), site]) + ' "$@"',
        'sync': shlex.join([sys.executable, launcher, 'site', 'sync', site, '--root', str(root)]),
    }
    for action, command in commands.items():
        path = directory / f'theta-{action}-{tag}'
        path.write_text('#!/bin/sh\ncd ' + shlex.quote(str(root)) + ' || exit 1\nexec ' + command + '\n')
        path.chmod(0o700)
        paths.append(str(path))
    return paths


def verify_environment(bundle):
    """The running interpreter satisfies the requirements of the bundle's application and providers."""
    from importlib.metadata import version, PackageNotFoundError
    from packaging.requirements import Requirement
    for value in [*bundle['requirements']['application'], *bundle['requirements']['providers']]:
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate(): continue
        try: installed = version(requirement.name)
        except PackageNotFoundError as exc:
            raise ValueError(f'Missing dependency {value}; install the bundle requirements: '
                             f'python -m pip install -r {BUNDLE_REQUIREMENTS}') from exc
        if requirement.specifier and installed not in requirement.specifier:
            raise ValueError(f'Dependency {value} conflicts with installed {installed}')
