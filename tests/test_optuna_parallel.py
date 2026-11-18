import pytest

def test_trial_runs_once_without_capacity_restarts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from omegaconf import OmegaConf
    from package_fixtures import package_module
    parallel = package_module('optuna', 'parallel.py')

    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    processes, waiting, completed = [], [], []

    class Pool:
        path = tmp_path / 'pool.db'
        def __init__(self, *args): pass
        def sample(self): pass
        def enqueue(self, count=1, **kwargs): return ['ticket'] * count
        def dequeue(self, ticket): pass
        def acquire(self, *args, **kwargs): return 'token', ''
        def attach(self, *args): pass
        def release(self, *args): pass

    class Process:
        pid = 123
        returncode = None
        def __init__(self, args):
            processes.append(self)
            Path(args[-1]).write_text('{"value": 42}')
            self.polls = 0
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0
        def wait(self): self.returncode = -15

    monkeypatch.setattr(parallel, 'ResourcePool', Pool)
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    monkeypatch.setattr(parallel, 'kill_process_tree', lambda pid: None)
    monkeypatch.setattr(parallel.time, 'sleep', lambda _: None)
    def load(config, overrides):
        return OmegaConf.create({'hydra': {'job': {}, 'sweep': {'dir': str(tmp_path)}},
                                 'theta_tuning': {'log_version': 'same-trial'}})
    impl = SimpleNamespace(task_function=load, config=OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}}),
        job_idx=0, n_jobs=2, hydra_context=SimpleNamespace(config_loader=SimpleNamespace(load_sweep_config=load)))
    results = parallel.launch_trials(impl, [('x=1',)], on_wait=waiting.append,
                                    on_result=lambda i, result: completed.append(i))
    assert len(processes) == 1 and waiting == [] and completed == [0]
    assert len(results) == 1 and results[0].return_value == 42


def test_cancel_pending_trial_keeps_next_trial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from src.app.services.cancellation import request_path
    from package_fixtures import package_module
    _optuna_parallel = package_module('optuna', 'parallel.py')
    launch_trials = _optuna_parallel.launch_trials
    from package_fixtures import package_module
    parallel = package_module('optuna', 'parallel.py')

    monkeypatch.setenv('THETA_CANCEL_DIR', str(tmp_path))
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    request_path(tmp_path, 'trial:selected').touch()
    started = []

    class Pool:
        path = tmp_path / 'pool.db'
        def __init__(self, *args): pass
        def sample(self): pass
        def enqueue(self, count=1, **kwargs): return ['ticket'] * count
        def dequeue(self, ticket): pass
        def acquire(self, *args, **kwargs): return 'token', None
        def attach(self, *args): pass
        def release(self, *args): pass

    class Process:
        pid = 123
        returncode = 0
        def __init__(self, args):
            started.append(args[-2])
            from pathlib import Path
            Path(args[-1]).write_text('{"value": 42}')
        def poll(self): return 0

    monkeypatch.setattr(parallel, 'ResourcePool', Pool)
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    def load(config, overrides):
        return OmegaConf.create({'hydra': {'job': {}, 'sweep': {'dir': str(tmp_path)}}, 'theta_tuning': {'log_version': overrides[0]}})
    impl = SimpleNamespace(task_function=load, config=OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}}), job_idx=0, n_jobs=2,
                           hydra_context=SimpleNamespace(config_loader=SimpleNamespace(load_sweep_config=load)))
    results = launch_trials(impl, [('selected',), ('sibling',)])
    assert results[0].return_value == {'theta_trial_status': 'pruned'}
    assert results[1].return_value == 42
    assert len(started) == 1


def test_managed_optuna_batches_are_single_trials(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    optuna = pytest.importorskip('optuna')
    from package_fixtures import package_module
    _optuna_sweeper = package_module('optuna', 'sweeper.py')
    ThetaOptunaSweeper = _optuna_sweeper.ThetaOptunaSweeper
    batches = []
    def launch(jobs, initial_job_idx):
        batches.append(len(jobs))
        return [SimpleNamespace(return_value=.5) for job in jobs]
    sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(), 'minimize',
                                 f'sqlite:///{tmp_path}/study.db', 'managed', 3, 8, None, 'none', 5, 0, 0, 0)
    sweeper.impl.config = OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}})
    sweeper.impl.launcher = SimpleNamespace(launch=launch)
    monkeypatch.setenv('THETA_MANAGED_SERIAL', '1')
    sweeper.sweep(['+x=interval(1.0,2.0)'])
    assert batches == [1, 1, 1]
