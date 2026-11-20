"""One immutable experiment bundle per SLURM allocation, with portable receipts."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import shutil
import tempfile
import time
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.app.api.execution import saved_run_id
from src.app.api.sites import (
    BUNDLE_ARCHIVES, digest, read_json, record_lock, safe_path, site_name, transfer_lock, write_json,
)
from src.app.api import sites as host

TERMINAL = host.TERMINAL

def worker_threads(config, environment):
    """Allocation size is a ceiling, not the thread count of each serial worker."""
    site = config.get('site') or {}
    extra = site.get('extra_env_vars') or {}
    requested = int(site.get('threads_per_worker') or extra.get('THETA_MANAGED_THREADS')
                    or extra.get('OMP_NUM_THREADS') or 1)
    allocated = int(environment.get('SLURM_CPUS_PER_TASK') or
                    (config.get('resources') or {}).get('cores') or 1)
    return max(1, min(requested, allocated))


def accounting_details(fields):
    names = ('scheduler_state_raw', 'scheduler_exit_code', 'scheduler_reason',
             'scheduler_elapsed', 'scheduler_time_limit', 'scheduler_start', 'scheduler_end')
    return {key: value.strip() for key, value in zip(names, fields[1:])}


def validate_allocation(cfg, *, paths=None):
    from src.app.api.errors import ConfigurationError
    site = cfg.get('site') or {}
    from .resources import normalize
    resources = normalize(cfg)
    trainer = cfg.get('trainer') or {}
    if int(resources.get('nodes') or 1) != 1 or int(trainer.get('num_nodes') or 1) != 1:
        raise ConfigurationError('SLURM experiment allocations currently support one node')
    from omegaconf import OmegaConf
    devices = trainer.get('devices', 1)
    if devices not in (1, 'auto') and not ((isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices)) and len(devices) == 1):
        raise ConfigurationError('SLURM experiment allocations currently support one training device')
    if resources.get('gpus') is not None and (isinstance(resources['gpus'], bool) or not isinstance(resources['gpus'], int) or resources['gpus'] not in (0, 1)):
        raise ConfigurationError('Request at most one GPU per experiment allocation')
    if cfg.get('plot_only'):
        raise ConfigurationError('Slurm plot_only is not supported yet; plot synced prior results locally (wipe=false)')
    raw_config = OmegaConf.to_container(OmegaConf.masked_copy(cfg, [key for key in cfg if key != 'hydra']), resolve=False) if OmegaConf.is_config(cfg) else cfg
    from urllib.parse import urlsplit
    def check_secrets(value):
        if isinstance(value, dict):
            for child in value.values(): check_secrets(child)
        elif isinstance(value, list):
            for child in value: check_secrets(child)
        elif isinstance(value, str) and '://' in value and urlsplit(value).password:
            raise ConfigurationError('Database credentials must use an environment reference, not a captured URL')
    check_secrets(raw_config)
    from src.app.api.tuning import prepare_plan
    plan = prepare_plan(cfg, '0' * 32, paths=paths)
    if plan is not None and plan.concurrency != 1:
        raise ConfigurationError('Slurm supports tuner concurrency=1 only; allocation-scoped parallel trials are not qualified. Local parallel trials remain supported.')
    if (cfg.get('variability') or {}).get('parallel_runs', 1) != 1:
        raise ConfigurationError('Slurm supports variability.parallel_runs=1 only')
    if cfg.get('workflow'):
        raise ConfigurationError('SLURM workflow execution is not supported yet; submit individual experiments')
    if cfg.get('dash') or cfg.get('dash_only'):
        raise ConfigurationError('Remote dashboards are not supported; use Sync Metrics to inspect the study')
    if cfg.get('recover'):
        raise ConfigurationError('Remote recovery requires an explicit prior execution and is not supported yet; original checkpoints are retained')
    for key in ('cores', 'nodes'):
        if resources.get(key) is not None and (isinstance(resources[key], bool) or not isinstance(resources[key], int) or resources[key] < 1):
            raise ConfigurationError(f'resources.{key} must be positive')
    return resources


def submit(root, experiment, overrides, cfg):
    from .commands import INTERACTIVE_BUNDLE_ENV

    interactive = (cfg.get('site') or {}).get('interactive_allocation')
    allocation = None
    if interactive is not None:
        job_id = os.environ.get('SLURM_JOB_ID', '')
        step_id = os.environ.get('SLURM_STEP_ID', '')
        bundle_id = os.environ.get(INTERACTIVE_BUNDLE_ENV, '')
        if not (job_id.isdigit() and step_id.isdigit() and re.fullmatch(r'[a-f0-9]{64}', bundle_id)):
            raise ValueError('Start this site with site launch to run in its interactive Slurm compute step')
        allocation = job_id, bundle_id
    # The same lock protects source references, continuation ownership, and GC.
    with transfer_lock(root):
        record = _submit(root, experiment, overrides, cfg, allocation=allocation)
    if allocation:
        script = host.receipt_path(root, record['execution_id']).with_name('job.sbatch')
        result = subprocess.run(['/bin/bash', str(script)], cwd=root)
        if result.returncode:
            raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
        record = read_json(script.with_name('submission.json'))
    return record


def _submit(root, experiment, overrides, cfg, *, allocation=None):
    """Prepare one execution inside an existing allocation or submit it to Slurm."""
    from omegaconf import OmegaConf
    from .headers import generate_sbatch_header
    root = Path(root).resolve()
    from src.app.api.sites import resolve_policy as resolve
    policy = resolve(cfg, root)
    if allocation and not policy:
        raise ValueError('Interactive allocation requires a scratch execution policy')
    from src.app.api.workspace import WorkspacePaths
    paths = WorkspacePaths(root)
    resources = validate_allocation(cfg, paths=paths)
    from .resources import submission_environment
    submit_env = submission_environment()
    site = cfg.get('site') or {}
    python = str(site.get('python_executable') or 'auto')
    if python == 'auto':
        venv = Path(str(site.get('venv_dir') or 'venv'))
        python = str((root / venv / 'bin/python3').absolute())
    if not Path(python).is_absolute(): python = str(root / python)
    if not Path(python).is_file(): raise ValueError(f'Cluster Python does not exist: {python}')
    execution_id = uuid.uuid4().hex
    path = host.receipt_path(root, execution_id)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = directory / 'workspace'
        if policy:
            source = host.current_bundle(root)
            if allocation and source.get('bundle_sha256') != allocation[1]:
                raise ValueError('The uploaded bundle changed while waiting for the allocation. Launch again to upload it.')
            pushed = (source.get('experiment') or {}).get('selector')
            if pushed != experiment:
                raise ValueError(f'The pushed bundle is for {pushed or "no experiment"}; push {experiment} to this site first')
            bundle = root / BUNDLE_ARCHIVES / (str(source.get('bundle_sha256')) + '.zip')
            if not source.get('archive_sha256') or not bundle.is_file():
                raise ValueError('Scratch execution requires the bundle archive from site push; push this experiment first')
            if sum(item.stat().st_size for item in (root / BUNDLE_ARCHIVES).glob('*.zip')) > policy['max_source_bytes']:
                raise ValueError('Bundle archive store exceeds max_source_bytes; preview managed maintenance')
            from src.app.api.sites import own_work_root, stage_inputs
            own_work_root(policy['work_root'], source['project_id'])
            inputs = Path(policy['work_root']) / 'inputs' / execution_id
            inputs.mkdir(parents=True, exist_ok=False)
            try:
                stage_inputs(root, inputs, policy)
            except BaseException:
                shutil.rmtree(inputs)
                raise
        else:
            source = host.prepare_bundle(root, experiment, overrides, snapshot)
            control, frozen = paths, WorkspacePaths(snapshot)
            # The frozen workspace writes results and reads datasets in the control workspace.
            for link, target in ((frozen.results, control.results), (frozen.datasets, control.datasets)):
                if link == frozen.results or target.exists():
                    link.parent.mkdir(parents=True, exist_ok=True)
                    link.symlink_to(target, target_is_directory=True)
        from .commands import verify_environment
        verify_environment(source)
        result_id = saved_run_id(str(cfg.experiment_id), f'__{execution_id}')
        command_overrides = [arg for arg in overrides if not arg.lstrip('+').startswith(('experiment_id=', 'group='))]
        command_overrides += [f'++experiment_id={json.dumps(result_id)}', f'++group={json.dumps(str(cfg.group))}']
        from src.app.api.tuning import prepare_plan, launch_bindings, apply_overrides
        tuner_plan = prepare_plan(cfg, execution_id, paths=paths)
        command_overrides = apply_overrides(command_overrides,
            launch_bindings(cfg, tuner_plan, snapshot, paths=paths))
        record = {'schema_version': 1, 'scheduler_component': 'slurm', 'project_id': source['project_id'], 'execution_id': execution_id,
                  'attempt_id': uuid.uuid4().hex, 'site': site_name(cfg), 'experiment': experiment,
                  'group': str(cfg.group), 'result_experiment_id': result_id,
                  'created_at': datetime.now(timezone.utc).isoformat(), 'state': 'prepared',
                  'job_id': None, 'bundle_sha256': source['bundle_sha256'],
                  'overrides': command_overrides, 'resources': resources,
                  'config': OmegaConf.to_container(cfg, resolve=False), 'artifacts_finalized': False,
                  'execution_policy': policy, 'control_root': str(root), 'bundle_archive_sha256': source.get('archive_sha256'),
                  'publication_state': 'pending' if policy else None,
                  'tuner_execution': tuner_plan.record() if tuner_plan else None}
        request = cfg.get('study_continuation')
        if request:
            if not policy: raise ValueError('Study continuation requires scratch execution')
            if cfg.get('remake') or cfg.get('recover'): raise ValueError('Continuation cannot remake or resume a training trial')
            from src.app.api.sites import pin_continuation as pin
            record['continuation'] = pin(root, record, dict(request))
            write_json(path, record)
            command_overrides = apply_overrides(command_overrides,
                launch_bindings(cfg, tuner_plan, snapshot, continuation=record['continuation'], paths=paths))
            record['overrides'] = command_overrides
        # Hydra metadata contains job-only interpolations and isn't needed in a receipt.
        record['config'].pop('hydra', None)
        write_json(path, record)
        write_json(directory / 'config.json', record['config'])
        header = generate_sbatch_header('theta_' + execution_id[:12], directory.relative_to(root), cfg, resources=resources)
        allowance = int(policy['stage_out_seconds']) if policy else 60
        header += f'#SBATCH --no-requeue\n#SBATCH --signal=B:TERM@{allowance}\n'
        script = header + '\nset -eo pipefail\n'
        script += 'cd ' + shlex.quote(str(root)) + '\n'
        script += 'unset THETA_RESOURCE_POOL THETA_RESOURCE_QUEUE_PARENT THETAIDE_SETTINGS_FILE\n'
        # The allocation loads its own pinned inputs from the extracted bundle.
        script += 'unset THETA_CONFIG_SNAPSHOT THETA_INPUTS_SNAPSHOT THETA_PIPELINE_PID\n'
        script += 'export THETA_WORKSPACE=' + shlex.quote(str(snapshot)) + '\n'
        script += 'export THETA_DATA_DIR=' + shlex.quote(str(snapshot / '.thetaide')) + '\n'
        bootstrap_root = root if policy else snapshot
        pythonpath = [str(bootstrap_root), *(str(snapshot / str(extra)) for extra in site.get('extra_pythonpath') or [])]
        script += 'export PYTHONPATH=' + shlex.quote(':'.join(pythonpath)) + '\n'
        if site.get('module_loads'):
            script += 'if ! type module >/dev/null 2>&1; then\n'
            script += '  if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; '
            script += 'elif [ -f /usr/share/modules/init/bash ]; then source /usr/share/modules/init/bash; fi\nfi\n'
        for module in site.get('module_loads') or []:
            script += 'module load ' + shlex.quote(str(module)) + '\n'
        for key, value in (site.get('extra_env_vars') or {}).items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', str(key)): raise ValueError('Invalid environment variable')
            if str(key).startswith('THETA_') or key == 'PYTHONPATH': raise ValueError(f'{key} is managed by Theta')
            script += 'export ' + str(key) + '=' + shlex.quote(str(value)) + '\n'
        script += 'set -u\n'
        # Slurm signals the batch PID; the supervisor must own that PID to finalize.
        script += 'exec ' + shlex.join(host.site_cli_command(python, '_run', '--root', str(root), execution_id)) + '\n'
        (directory / 'job.sbatch').write_text(script)
        if allocation:
            record.update(state='pending', job_id=allocation[0])
            write_json(path, record)
            print(json.dumps({'execution_id': execution_id, 'state': record['state'],
                              'job_id': record['job_id'], 'receipt': str(path)}, indent=2), flush=True)
            return record
        record['state'] = 'submitting'
        write_json(path, record)
        try:
            result = subprocess.run(['sbatch', '--parsable', '--comment=theta:' + source['project_id'] + ':' + execution_id,
                                     str(directory / 'job.sbatch')], cwd=root, env=submit_env, capture_output=True, text=True, timeout=60)
            if result.returncode:
                record.update(state='submission_failed', error=result.stderr.strip())
            else:
                match = re.fullmatch(r'(\d+)(?:;([A-Za-z0-9_.-]+))?', result.stdout.strip())
                if not match: raise ValueError('Unrecognized sbatch response; reconcile before resubmitting')
                record.update(state='pending', job_id=match[1], cluster=match[2])
        except FileNotFoundError:
            record.update(state='submission_failed', error='sbatch is unavailable; submit from a cluster login node')
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            record.update(state='submission_unknown', error=str(exc))
        # The batch worker can start before sbatch returns. Do not overwrite its state.
        with record_lock(path):
            current = read_json(path)
            if current.get('started_at'):
                record = {**record, **current, 'job_id': record.get('job_id') or current.get('job_id')}
            write_json(path, record)
        print(json.dumps({'execution_id': execution_id, 'state': record['state'], 'job_id': record['job_id'],
                          'receipt': str(path)}, indent=2))
        if record['state'] in ('submission_failed', 'submission_unknown'):
            raise RuntimeError(record.get('error', record['state']))
        return record
    except BaseException:
        if path.exists():
            failed = read_json(path)
            if failed.get('state') == 'prepared':
                failed.update(state='submission_failed', error='Submission preparation failed; no sbatch call made')
                write_json(path, failed)
            if failed.get('state') == 'submission_failed':
                payload = Path(policy['work_root']) / 'inputs' / execution_id if policy else directory / 'workspace'
                if payload.exists() and not payload.is_symlink(): shutil.rmtree(payload)
        else:
            if policy:
                inputs = Path(policy['work_root']) / 'inputs' / execution_id
                if inputs.exists(): shutil.rmtree(inputs)
            shutil.rmtree(directory)
        raise


def run(root, execution_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('The allocation worker requires SLURM_JOB_ID; never run it on a login node')
    root = Path(root).resolve()
    path = host.receipt_path(root, execution_id)
    with record_lock(path):
        record = read_json(path)
        if record.get('started_at'): raise ValueError('Execution already started; use a new attempt')
    from src.app.api.tuning import record_overrides
    overrides = record_overrides(record)
    policy = record.get('execution_policy')
    runtime = root
    snapshot = path.parent / 'workspace'
    if policy:
        from src.app.api.sites import assert_local_scratch
        scratch = Path(policy['scratch_root'])
        assert_local_scratch(scratch)
        runtime = Path(tempfile.mkdtemp(prefix='theta-' + execution_id + '-', dir=scratch))
        snapshot = runtime
        write_json(runtime / '.theta-owner.json', {'project_id': record['project_id'],
                   'execution_id': execution_id, 'job_id': os.environ.get('SLURM_JOB_ID'),
                   'host': __import__('socket').gethostname(), 'pid': os.getpid()})
        bundle = root / BUNDLE_ARCHIVES / (record['bundle_sha256'] + '.zip')
        if digest(bundle) != record['bundle_archive_sha256']: raise ValueError('Bundle archive changed')
        import zipfile
        with zipfile.ZipFile(bundle) as archive:
            required = sum(item.file_size for item in archive.infolist())
        required += read_json(Path(policy['work_root']) / 'inputs' / execution_id / '.theta-inputs.json').get('bytes', 0)
        if shutil.disk_usage(runtime).free < required + policy['scratch_reserve_bytes']:
            raise ValueError('Insufficient job-local scratch for bundle/inputs and reserve')
        source = host.extract_bundle(bundle, runtime)
        if source['bundle_sha256'] != record['bundle_sha256']: raise ValueError('Bundle identity mismatch')
        inputs = Path(policy['work_root']) / 'inputs' / execution_id
        if inputs.is_dir(): shutil.copytree(inputs, runtime, dirs_exist_ok=True, copy_function=shutil.copyfile)
    else:
        source = host.verify_bundle(snapshot)
        if source['bundle_sha256'] != record['bundle_sha256']: raise ValueError('Bundle identity mismatch')
    with record_lock(path):
        record = read_json(path)
        if record.get('started_at'): raise ValueError('Execution already started; use a new attempt')
        record.update(state='running', started_at=datetime.now(timezone.utc).isoformat(),
                      job_id=os.environ.get('SLURM_JOB_ID', record.get('job_id')))
        write_json(path, record)
    environment = os.environ.copy()
    environment.update(THETA_IN_ALLOCATION='1', THETA_WORKSPACE=str(snapshot),
                       THETA_DATA_DIR=str(snapshot / '.thetaide'),
                       PYTHONPATH=str(snapshot) + os.pathsep + environment.get('PYTHONPATH', ''),
                       THETA_EXECUTION_ID=execution_id, THETA_OUTPUT_ROOT=str(runtime),
                       THETA_PYTHON_EXECUTABLE=sys.executable,
                       THETA_MANAGED_THREADS=str(worker_threads(record['config'], environment)),
                       THETA_MANAGED_SERIAL='1')
    for key in ('THETA_RESOURCE_POOL', 'THETA_RESOURCE_QUEUE_PARENT', 'THETAIDE_SETTINGS_FILE', 'THETA_PROVIDER_SNAPSHOT'):
        environment.pop(key, None)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = environment['THETA_MANAGED_THREADS']
    from src.app.api.tuning import record_plan, record_config, launch_bindings, apply_overrides
    from src.app.api.workspace import WorkspacePaths
    provider_paths = WorkspacePaths(snapshot)
    tuner_plan = record_plan(record, paths=provider_paths)
    worker_config = record_config(record)
    bindings = launch_bindings(worker_config, tuner_plan, runtime,
        continuation=record.get('continuation'), paths=provider_paths)
    overrides = apply_overrides(overrides, bindings)
    environment.update(bindings.environment)
    publisher = None
    if policy:
        from src.app.api.sites import Publisher
        methods = record['config'].get('methods') or {'run': {}}
        method_count = sum(name != 'params' and (not isinstance(value, dict) or value.get('enabled', True)) for name, value in methods.items())
        variability = record['config'].get('variability') or {}
        seeds = len(variability.get('training_seeds') or []) or variability.get('training_runs', 1)
        planned_runs = max(1, method_count * seeds * (tuner_plan.trials if tuner_plan and record['config'].get('tuning') is not None else 1))
        environment.update(THETA_COMMIT_DIR=str(runtime / '.thetaide/commits'),
                           THETA_DB_EXPORT_REQUEST=str(runtime / '.thetaide/db-export-request.json'),
                           THETA_PUBLICATION_STOP=str(runtime / '.thetaide/publication-stop'),
                           THETA_DIAGNOSTIC_BUDGET=json.dumps({**{key: policy[key] for key in ('diagnostic_bytes', 'diagnostic_files', 'snapshot_bytes')}, 'planned_runs': planned_runs}),
                           THETA_STREAM_FLUSH_SECONDS=str(policy['stream_flush_seconds']),
                           THETA_SYSTEM_INTERVAL=str(policy['system_interval']),
                           THETA_DIAGNOSTIC_STRICT='1' if policy['diagnostic_strict'] else '0',
                           THETA_DEFER_MONITOR_REPORTS='1')
        if record.get('continuation'):
            from src.app.api.sites import restore_continuation as restore
            restore(runtime, record)
        publisher = Publisher(runtime, policy['work_root'], record, policy)
        controls = WorkspacePaths(runtime).results / 'sites' / execution_id
        controls.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'job.sbatch'):
            shutil.copyfile(path.parent / name, controls / name)
        runtime_log = controls / 'pipeline.log'
    else:
        runtime_log = path.parent / 'pipeline.log'
    process = None
    interrupted = []
    stop_started = []
    def stop(signum, _frame):
        interrupted.append(signum)
        if not stop_started: stop_started.append(time.monotonic())
        if process and process.poll() is None:
            try: os.killpg(process.pid, signum)
            except ProcessLookupError: pass
    deadline = float(os.environ.get('SLURM_JOB_END_TIME') or 0)
    allowance = int(policy['stage_out_seconds']) if policy else 60
    termination_reason = None
    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        from src.app.api.execution import ProcessOutput
        import traceback
        error_log = runtime_log.with_suffix('.err')
        with runtime_log.open('w', encoding='utf-8', errors='replace') as output, error_log.open('w', encoding='utf-8', errors='replace') as errors:
            output.write(f"[Theta] Allocation {record.get('job_id')}: worker_threads={environment['THETA_MANAGED_THREADS']}, requested_time={record.get('resources', {}).get('time') or 'partition default'}, deadline={datetime.fromtimestamp(deadline, timezone.utc).isoformat() if deadline else 'unknown'}.\n")
            output.flush()
            process = subprocess.Popen([sys.executable, '-u', str(snapshot / 'run_pipeline.py'), record['experiment'],
                                        *overrides], cwd=runtime, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       encoding='utf-8', errors='replace', bufsize=1, start_new_session=True)
            capture = ProcessOutput(process, output, errors, stderr=sys.stderr)
            try:
                while process.poll() is None:
                    if interrupted: break
                    if publisher:
                        try:
                            publisher.tick()
                        except Exception as exc:
                            record['publication_error'] = str(exc)
                            capture.write_error('[Theta] Result publication failed:\n' + traceback.format_exc())
                            (runtime / '.thetaide/publication-stop').touch()
                            stop(signal.SIGTERM, None)
                            break
                    try: process.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
                if interrupted:
                    near_deadline = deadline and 0 <= deadline - time.time() <= allowance + 60
                    termination_reason = ('Result publication failed: ' + record['publication_error'] if record.get('publication_error') else
                                          'Likely Slurm pre-timeout warning: allocation deadline is near' if near_deadline else
                                          'External termination signal; inspect scheduler accounting for attribution')
                    capture.write_error(f'[Theta] Received {signal.Signals(interrupted[-1]).name}: {termination_reason}. Finalizing results.\n')
                if interrupted and process.poll() is None:
                    try: process.wait(timeout=int(policy['stage_out_seconds']) // 2 if policy else 30)
                    except subprocess.TimeoutExpired:
                        capture.write_error('[Theta] Training did not stop in time; sending SIGKILL before final stage-out.\n')
                        os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
            except BaseException:
                capture.write_error('[Theta] Execution supervisor failed:\n' + traceback.format_exc())
                if process.poll() is None:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
                raise
            finally:
                capture.finish()
    finally:
        if interrupted or code != 0:
            try:
                host.close_unfinished_runs(runtime, record, 'cancelled' if interrupted else 'failed', paths=provider_paths)
            except Exception:
                import traceback
                print('[Theta] Could not finalize interrupted trial statuses:\n' + traceback.format_exc(), file=sys.stderr, flush=True)
        if publisher:
            try:
                allowance = policy['stage_out_seconds']
                if stop_started: allowance = max(1, allowance - (time.monotonic() - stop_started[0]))
                publisher.finish(seconds=allowance)
                record['publication_state'] = 'sealed'
            except Exception as exc:
                record.update(publication_state='incomplete', publication_error=str(exc))
                import traceback
                diagnostic = '[Theta] Final result publication failed:\n' + traceback.format_exc()
                print(diagnostic, file=sys.stderr, flush=True)
                for diagnostic_log in (runtime_log, runtime_log.with_suffix('.err')):
                    try:
                        with diagnostic_log.open('a', encoding='utf-8', errors='replace') as stream:
                            stream.write(diagnostic)
                    except OSError as log_error:
                        print(f'[Theta] Could not save {diagnostic_log.name}: {log_error}', file=sys.stderr, flush=True)
            finally:
                record['last_durable'] = publisher.manifest['durable']
                publisher.close()
        for number, handler in previous.items(): signal.signal(number, handler)
        with record_lock(path):
            record = {**read_json(path), **{key: value for key, value in record.items() if key in ('publication_state', 'publication_error', 'last_durable')}}
            training_state = 'cancelled' if interrupted else ('completed' if code == 0 else 'failed')
            record.update(state=training_state, training_state=training_state,
                          termination_reason=termination_reason, termination_signal=interrupted[-1] if interrupted else None,
                          exit_code=code, ended_at=datetime.now(timezone.utc).isoformat())
            write_json(path, record)
    if publisher and record.get('publication_state') == 'sealed':
        shutil.rmtree(runtime)  # allocation-owned temporary tree only, after durable sealing
    return (code if code >= 0 else 128 - code) or (1 if record.get('publication_state') == 'incomplete' else 0)


def refresh_status(root, record):
    if record.get('scheduler_state') in TERMINAL or record.get('state') == 'submission_failed': return record
    job_id = record.get('job_id')
    try:
        if not job_id:
            # Reconcile the case where sbatch accepted the job but its response was lost.
            result = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i|%k'],
                                    capture_output=True, text=True, timeout=20, check=True)
            token = 'theta:' + record['project_id'] + ':' + record['execution_id']
            matches = [line.split('|')[0].strip() for line in result.stdout.splitlines() if line.partition('|')[2].strip() == token]
            if len(matches) == 1: job_id = record['job_id'] = matches[0]
        if not job_id: return record
        result = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%T'],
                                capture_output=True, text=True, timeout=20, check=True)
        value = result.stdout.strip().splitlines()
        if value:
            record['scheduler_state'] = value[0].lower()
        else:
            record['scheduler_state'] = 'unknown'
            result = subprocess.run(['sacct', '--noheader', '--parsable2', '--allocations', '--jobs', job_id,
                                     '--format=JobIDRaw,State%80,ExitCode,Reason%120,Elapsed,Timelimit,Start,End'], capture_output=True, text=True, timeout=20, check=True)
            for line in result.stdout.splitlines():
                parts = line.split('|')
                if len(parts) >= 3 and parts[0] == job_id:
                    state = parts[1].split()[0].rstrip('+').lower()
                    record['scheduler_state'] = state
                    record.update(accounting_details(parts))
                    if state in TERMINAL:
                        training = record.get('training_state') or (record.get('state') if record.get('state') in TERMINAL else 'unknown')
                        record['training_state'] = training
                        record['state'] = training if state == 'completed' and training in ('failed', 'cancelled') else state
                        record['scheduler_exit_code'] = parts[2]
                        record.setdefault('exit_code', parts[2])
                    break
        if record.get('scheduler_state') == 'unknown':
            record['status_error'] = 'No scheduler accounting record is available yet'
        else:
            record['scheduler_checked_at'] = datetime.now(timezone.utc).isoformat()
            record.pop('status_error', None)
    except (OSError, subprocess.SubprocessError) as exc:
        record['status_error'] = str(exc)
    # Re-read so a worker finishing during the scheduler query wins.
    with record_lock(host.receipt_path(root, record['execution_id'])):
        current = read_json(host.receipt_path(root, record['execution_id']))
        if current.get('state') in TERMINAL and record.get('scheduler_state') not in TERMINAL:
            record = {**record, **current, 'scheduler_state': record.get('scheduler_state', current.get('scheduler_state'))}
        write_json(host.receipt_path(root, record['execution_id']), record)
    return record


def refresh_many(root, records, *, minimum_seconds=10):
    """Two batched scheduler RPCs per 200 unresolved jobs; terminal state is cached."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    pending = []
    for record in records:
        if record.get('scheduler_state') in TERMINAL or record.get('state') == 'submission_failed': continue
        checked = record.get('scheduler_checked_at')
        if checked and now - datetime.fromisoformat(checked) < timedelta(seconds=minimum_seconds): continue
        pending.append(record)
    for offset in range(0, len(pending), 200):
        batch = pending[offset:offset + 200]
        try:
            known = [str(item['job_id']) for item in batch if item.get('job_id')]
            command = ['squeue', '--noheader', '--format=%i|%T|%k']
            command += ['--jobs', ','.join(known)] if len(known) == len(batch) else ['--user', str(os.getuid())]
            queue = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
            states = {}
            for line in queue.stdout.splitlines():
                fields = line.split('|')
                if len(fields) >= 2:
                    states[fields[0].strip()] = fields[1].strip().lower()
                    if len(fields) >= 3:
                        for record in batch:
                            if fields[2].strip() == f'theta:{record["project_id"]}:{record["execution_id"]}': record['job_id'] = fields[0].strip()
            missing = [str(item['job_id']) for item in batch if item.get('job_id') and str(item['job_id']) not in states]
            exits = {}
            details = {}
            if missing:
                accounting = subprocess.run(['sacct', '--noheader', '--parsable2', '--allocations', '--jobs', ','.join(missing),
                                             '--format=JobIDRaw,State%80,ExitCode,Reason%120,Elapsed,Timelimit,Start,End'], capture_output=True, text=True, timeout=20, check=True)
                for line in accounting.stdout.splitlines():
                    fields = line.split('|')
                    if len(fields) >= 3 and fields[0] in missing:
                        states[fields[0]] = fields[1].split()[0].rstrip('+').lower()
                        exits[fields[0]] = fields[2]
                        details[fields[0]] = accounting_details(fields)
            for record in batch:
                path = host.receipt_path(root, record['execution_id'])
                with record_lock(path):
                    current = read_json(path)
                    state = states.get(str(record.get('job_id')), 'unknown')
                    current.update(job_id=record.get('job_id'), scheduler_state=state, scheduler_checked_at=now.isoformat())
                    current.update(details.get(str(record.get('job_id')), {}))
                    if state in TERMINAL:
                        training = current.get('training_state') or (current.get('state') if current.get('state') in TERMINAL else 'unknown')
                        current.update(training_state=training,
                                       state=training if state == 'completed' and training in ('failed', 'cancelled') else state,
                                       scheduler_exit_code=exits.get(str(record.get('job_id'))))
                    if state == 'unknown': current['status_error'] = 'Scheduler/accounting does not confirm termination'
                    else: current.pop('status_error', None)
                    write_json(path, current)
                    record.update(current)
        except (OSError, subprocess.SubprocessError) as exc:
            for record in batch: record['status_error'] = str(exc)
    return records


class Scheduler:
    api_version = 2

    @staticmethod
    def launch_in_terminal(root, site, experiment):
        from .commands import launch_in_terminal
        return launch_in_terminal(root, site, experiment)

    @staticmethod
    def remote_requirements(site_config):
        """Commands the generic site check verifies on the login node."""
        return [('sbatch', None), ('squeue', None), ('sacct', 'sacct unavailable: accounting will be unknown')]

    @staticmethod
    def submission_command(config, experiment, site):
        from .commands import submission_command
        return submission_command(config, experiment, site)
    validate_allocation = staticmethod(validate_allocation)
    submit = staticmethod(submit)
    run = staticmethod(run)
    refresh_status = staticmethod(refresh_status)
    refresh_many = staticmethod(refresh_many)
    worker_threads = staticmethod(worker_threads)
    accounting_details = staticmethod(accounting_details)

    @staticmethod
    def cancel(root, record):
        job = str(record.get('job_id') or '')
        if not job.isdigit(): raise ValueError('Cannot cancel an unresolved scheduler job; reconcile its submission first')
        subprocess.run(['scancel', job], check=True, capture_output=True, text=True, timeout=20)
        # Cancellation is a request. Only scheduler reconciliation proves termination.
        return refresh_status(root, record)
