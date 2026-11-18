"""Adaptive fresh-process execution for local Optuna trial batches."""
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from src.app.api.execution import ResourcePool, uses_accelerator, workload_estimate, host_capacity
from src.app.api.execution import kill_process_tree, task_worker_command


class TrialResult:
    def __init__(self, result):
        self.result = result

    @property
    def return_value(self):
        if 'error' in self.result:
            raise RuntimeError(self.result['error'])
        return self.result['value']


def launch_trials(impl, jobs=(), on_result=None, *, next_job=None, on_start=None, on_wait=None, can_start=None, total_jobs=None):
    """Refill free capacity continuously, sampling future jobs only as needed."""
    pool = ResourcePool(os.environ['THETA_RESOURCE_POOL'], os.environ.get('THETAIDE_SETTINGS_FILE'))
    script = str(Path(inspect.getsourcefile(impl.task_function) or sys.argv[0]).resolve())
    pending, active = [], {}
    targets = {}
    coordinator_tickets = {}
    worker_tickets = {}
    queue_parent = os.environ.get('THETA_RESOURCE_QUEUE_PARENT')
    results = []
    iterator = iter(jobs)
    exhausted = False
    probe = None
    if total_jobs is None and next_job is None and hasattr(jobs, '__len__'):
        total_jobs = len(jobs)
    last_reason = None
    coordinator_limit = max(1, host_capacity()[0])
    with tempfile.TemporaryDirectory(prefix='trials-', dir=pool.path.parent) as directory:
        directory = Path(directory)
        def prepare(index, overrides):
            nonlocal probe, queue_parent
            config = impl.hydra_context.config_loader.load_sweep_config(impl.config, list(overrides))
            with open_dict(config):
                config.hydra.job.id = impl.job_idx + index
                config.hydra.job.num = impl.job_idx + index
                # Freeze the coordinator's timestamp-based sweep root, so
                # workers starting later cannot drift into a different folder.
                config.hydra.sweep.dir = str(impl.config.hydra.sweep.dir)
            targets[index] = "trial:" + str(OmegaConf.select(config, "theta_tuning.log_version", default=""))
            config_path = directory / f'{index}.yaml'
            result_path = directory / f'{index}.json'
            OmegaConf.save(config, config_path)
            # Resolve only task settings: Hydra sweep nodes contain placeholders
            # which are normally filled by the launcher.
            task = OmegaConf.to_container(OmegaConf.masked_copy(config, [k for k in config if k != 'hydra']), resolve=True)
            spec = workload_estimate(task)
            spec['coordinator'] = bool(task.get('theta_execution') or task.get('theta_objective_runner'))
            probe = (spec, uses_accelerator(task))
            pending.append((index, config_path, result_path, *probe))
            if not spec.get('coordinator'):
                worker_tickets[index] = pool.enqueue(parent=queue_parent)[0]
                queue_parent = None
        try:
            while not exhausted or pending or active:
                from src.app.api.execution import cancellation_requested as requested
                for item in list(pending):
                    index = item[0]
                    if requested([targets[index]]):
                        pending.remove(item)
                        if index in worker_tickets:
                            pool.dequeue(worker_tickets.pop(index))
                        results[index] = TrialResult({'value': {'theta_trial_status': 'pruned'}})
                        if on_result:
                            on_result(index, results[index])
                pool.sample()
                # Tell Optuna about completed work before sampling a replacement.
                for index, (process, token, result_path) in list(active.items()):
                    cancelled = requested([targets[index]])
                    if cancelled and process.poll() is None:
                        kill_process_tree(process.pid)
                        process.wait()
                    if process.poll() is None:
                        continue
                    if index in coordinator_tickets:
                        pool.dequeue(coordinator_tickets.pop(index))
                    try:
                        result = json.loads(result_path.read_text())
                    except (OSError, ValueError):
                        result = {'error': f'Trial process exited with code {process.returncode} without an objective.'}
                    if cancelled:
                        result = {'value': {'theta_trial_status': 'pruned'}}
                    results[index] = TrialResult(result)
                    if token:
                        pool.release(token)
                    del active[index]
                    if on_result:
                        on_result(index, results[index])
                while pending or not exhausted:
                    if len(active) >= impl.n_jobs:
                        break
                    if probe and probe[0].get('coordinator') and len(active) >= coordinator_limit:
                        break
                    if can_start and not can_start():
                        exhausted = True
                        for index, *_rest in pending:
                            if index in worker_tickets:
                                pool.dequeue(worker_tickets.pop(index))
                            results[index] = TrialResult({'value': {'theta_trial_status': 'pruned'}})
                            if on_result:
                                on_result(index, results[index])
                        pending.clear()
                        break
                    if not pending:
                        if total_jobs is not None and len(results) >= total_jobs:
                            exhausted = True
                            break
                        # Avoid drawing suggestions while another resource
                        # limit, warmup or pause prevents using the free slot.
                        if probe is not None and not probe[0].get('coordinator'):
                            _token, reason = pool.acquire(probe[0], accelerator=probe[1], reserve=False)
                            if reason:
                                if reason != last_reason:
                                    print(f'[System usage] {reason}', flush=True)
                                    last_reason = reason
                                break
                        overrides = next_job() if next_job else next(iterator, None)
                        if overrides is None:
                            exhausted = True
                            break
                        index = len(results)
                        results.append(None)
                        prepare(index, overrides)
                    index, config_path, result_path, spec, accelerator = pending[0]
                    if requested([targets[index]]):
                        pending.pop(0)
                        if index in worker_tickets:
                            pool.dequeue(worker_tickets.pop(index))
                        results[index] = TrialResult({'value': {'theta_trial_status': 'pruned'}})
                        if on_result:
                            on_result(index, results[index])
                        continue
                    token, reason = ((None, '') if spec.get('coordinator') else pool.acquire(spec, accelerator=accelerator, ticket=worker_tickets.get(index)))
                    if reason:
                        if reason != last_reason:
                            print(f'[System usage] {reason}', flush=True)
                            last_reason = reason
                        break
                    try:
                        options = {}
                        if spec.get('coordinator'):
                            ticket = pool.enqueue(parent=queue_parent)[0]
                            queue_parent = None
                            coordinator_tickets[index] = ticket
                            options['env'] = dict(os.environ, THETA_RESOURCE_QUEUE_PARENT=ticket)
                        process = subprocess.Popen(task_worker_command(sys.executable, script, config_path, result_path), **options)
                        if token:
                            pool.attach(token, process.pid)
                    except BaseException:
                        if token:
                            pool.release(token)
                        raise
                    worker_tickets.pop(index, None)
                    active[index] = (process, token, result_path)
                    pending.pop(0)
                    last_reason = None
                    kind = 'coordinator' if spec.get('coordinator') else 'worker'
                    print(f'[System usage] Started trial {kind} {process.pid}; {len(active)} active.', flush=True)
                    if on_start:
                        on_start(index, process.pid)
                if not exhausted or pending or active:
                    time.sleep(.2)
        finally:
            for ticket in worker_tickets.values():
                pool.dequeue(ticket)
            for ticket in coordinator_tickets.values():
                pool.dequeue(ticket)
            for process, token, _path in active.values():
                if process.poll() is None:
                    kill_process_tree(process.pid)
                    process.wait()
                if token:
                    pool.release(token)
    return results
