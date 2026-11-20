"""Hydra Optuna sweeper that passes Optuna trial identity to each job.

The upstream Hydra adapter uses an independent job counter, which restarts at
zero on resume. Explicit trial identity prevents checkpoint collisions.
"""

import json
import math
import logging
import time
import uuid
import os
from datetime import datetime
from pathlib import Path

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.plugins.sweeper import Sweeper
from omegaconf import OmegaConf
from optuna.distributions import BaseDistribution, FloatDistribution
from src.app.api.tuning import trial_log_version

logger = logging.getLogger(__name__)


def _distribution_from_config(config):
    from optuna.distributions import CategoricalDistribution, IntDistribution
    kind = config["type"]
    if kind == "categorical":
        return CategoricalDistribution(list(config["choices"]))
    if kind == "int":
        return IntDistribution(int(config["low"]), int(config["high"]),
                               log=bool(config.get("log", False)), step=int(config.get("step") or 1))
    if kind == "float":
        return FloatDistribution(float(config["low"]), float(config["high"]),
                                 log=bool(config.get("log", False)), step=config.get("step"))
    raise ValueError(f"Unsupported Optuna distribution type: {kind!r}")


def _distribution_from_override(override):
    """Translate the public Hydra override types into current Optuna distributions."""
    from optuna.distributions import CategoricalDistribution, IntDistribution
    if not override.is_sweep_override():
        return override.get_value_element_as_str()
    value = override.value()
    if override.is_choice_sweep() or (override.is_range_sweep() and value.shuffle):
        choices = list(override.sweep_iterator())
        if any(not isinstance(item, (str, int, float, bool)) for item in choices):
            raise ValueError('Optuna choices must be scalar values')
        return CategoricalDistribution(choices)
    if override.is_range_sweep():
        if any(isinstance(part, float) for part in (value.start, value.stop, value.step)):
            return FloatDistribution(float(value.start), float(value.stop), step=float(value.step))
        return IntDistribution(int(value.start), int(value.stop), step=int(value.step))
    if override.is_interval_sweep():
        cls = IntDistribution if isinstance(value.start, int) and isinstance(value.end, int) else FloatDistribution
        return cls(value.start, value.end, log='log' in value.tags)
    raise ValueError(f'Unsupported Optuna sweep override: {override}')


def _export_summary(config, study):
    from .utils import results_root, write_study_summary

    agent = config.get("agent")
    model = config.get("model")
    name = agent.get("name") if hasattr(agent, "get") else None
    name = name or (model.get("name") if hasattr(model, "get") else None)
    group, experiment = config.get("group"), config.get("experiment_id")
    if not name or not group or not experiment:
        return
    root = results_root() / "checkpoints" / str(group) / str(experiment) / str(name).replace("/", "_")
    try:
        if study.user_attrs.get("theta_log_method") != str(name):
            study.set_user_attr("theta_log_method", str(name))
        write_study_summary(study, root)
    except Exception:
        logger.exception("Could not write tuning summary for study %s", study.study_name)


def _resumed_sampler(config, existing):
    from hydra.utils import instantiate
    sampler = OmegaConf.select(config, "hydra.sweeper.sampler")
    seed = sampler.get("seed")
    return instantiate(sampler, seed=None if seed is None else (int(seed) + existing) % 2**32)


class ThetaOptunaSweeper(Sweeper):
    def __init__(self, sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                 pruner, pruner_startup_trials, pruner_warmup_steps, timeout_seconds, max_failed_trials):
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("Optuna n_trials must be a positive integer")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs <= 0:
            raise ValueError("Optuna n_jobs must be a positive integer")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds < 0:
            raise ValueError("Optuna timeout_seconds must be a nonnegative integer")
        if isinstance(max_failed_trials, bool) or not isinstance(max_failed_trials, int) or max_failed_trials < 0:
            raise ValueError("Optuna max_failed_trials must be a nonnegative integer")
        from types import SimpleNamespace
        self.impl = SimpleNamespace(sampler=sampler, direction=direction, storage=storage, study_name=study_name,
            n_trials=n_trials, n_jobs=n_jobs, job_idx=0,
            search_space={str(key): _distribution_from_config(value) for key, value in (search_space or {}).items()})
        from .policy import make_pruner
        self.pruner_name = pruner
        self.pruner_startup_trials = pruner_startup_trials
        self.pruner_warmup_steps = pruner_warmup_steps
        self.pruner = make_pruner(pruner, pruner_startup_trials, pruner_warmup_steps)
        self.timeout_seconds = timeout_seconds
        self.max_failed_trials = max_failed_trials

    def setup(self, *, hydra_context, task_function, config):
        os.environ.setdefault("THETA_PIPELINE_PID", str(os.getpid()))
        if os.environ.get('THETA_MANAGED_SERIAL') == '1':
            target = config.hydra.launcher.get('_target_', '')
            if target != 'hydra._internal.core_plugins.basic_launcher.BasicLauncher':
                raise ValueError('Resource-managed local sweeps require the basic Hydra launcher; '
                                 'parallelism is controlled by n_jobs and Settings → Training Pipeline.')
        from hydra.core.plugins import Plugins
        self.impl.job_idx = 0
        self.impl.config, self.impl.hydra_context = config, hydra_context
        self.impl.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function)
        self.impl.task_function = task_function

    def sweep(self, arguments):
        impl = self.impl
        parsed = OverridesParser.create().parse_overrides(arguments)
        search_space = dict(impl.search_space)
        fixed = {}
        override_keys = {}
        for override in parsed:
            value = _distribution_from_override(override)
            name = override.key_or_group
            override_keys[name] = override.get_key_element()
            (search_space if isinstance(value, BaseDistribution) else fixed)[name] = value
        for name in fixed:
            search_space.pop(name, None)

        if isinstance(impl.direction, (list, tuple)):
            raise ValueError("Theta's Optuna sweeper currently supports one objective")
        direction = impl.direction if isinstance(impl.direction, str) else impl.direction.name
        from .utils import get_optuna_storage
        storage = get_optuna_storage(impl.storage)
        try:
            # The pipeline prepares a new study before launching Hydra. Attach
            # explicitly rather than logging that we are reusing an old study.
            study = optuna.load_study(study_name=impl.study_name, storage=storage,
                                      sampler=impl.sampler, pruner=self.pruner)
            if study.direction.name.lower() != direction:
                raise ValueError(f"Study {study.study_name!r} has a different objective direction")
        except KeyError:
            study = optuna.create_study(study_name=impl.study_name, storage=storage,
                                        sampler=impl.sampler, pruner=self.pruner,
                                        direction=direction, load_if_exists=True)
        if study.trials and not isinstance(impl.sampler, optuna.samplers.QMCSampler):
            # A seeded sampler rebuilt on resume restarts its random stream and
            # replays the study's first suggestions; offset it by the trials so far.
            # QMC already continues its sequence from the stored trial count.
            study.sampler = impl.sampler = _resumed_sampler(impl.config, len(study.trials))
        tuning = impl.config.get("tuning")
        from src.app.api.tuning import objective_metric
        metric = objective_metric(impl.config, "") or None
        if self.pruner_name != "none" and (not impl.storage or not metric):
            raise ValueError("Optuna pruning requires persistent storage and an objective metric.")
        if metric:
            recorded_metric = study.user_attrs.get("theta_objective_metric")
            if recorded_metric and recorded_metric != metric:
                raise ValueError(f"Study {study.study_name!r} already optimizes {recorded_metric!r}, not {metric!r}")
            study.set_user_attr("theta_objective_metric", str(metric))
        study.set_user_attr("theta_last_stop_reason", "running")
        study.set_user_attr("theta_requested_trials", impl.n_trials)
        study.set_user_attr("theta_initial_trials", len(study.trials))
        remaining = impl.n_trials
        failed_count = 0
        start = time.monotonic()
        while remaining > 0 and (not self.timeout_seconds or time.monotonic() - start < self.timeout_seconds):
            from src.app.api.tuning import check_artifact_admission as admission
            admission()
            managed = bool(os.environ.get('THETA_RESOURCE_POOL'))
            batch = 1 if os.environ.get('THETA_MANAGED_SERIAL') == '1' else impl.n_jobs
            trials = [] if managed else [study.ask() for _ in range(min(remaining, batch))]
            jobs = []

            def make_job(trial):
                version = trial_log_version({"theta_tuning": {"study_name": study.study_name,
                                                                  "trial_number": trial.number}}) + "_" + uuid.uuid4().hex[:12]
                trial.set_user_attr("theta_log_version", version)
                if managed:
                    trial.set_user_attr('theta_execution_state', 'pending')
                for name, distribution in search_space.items():
                    trial._suggest(name, distribution)
                # Apply fixed parent containers before sampled nested values.
                params = dict(fixed)
                params.update(trial.params)
                args = [f"{override_keys.get(name, name)}={value}" for name, value in params.items()]
                args.extend([
                        "++theta_tuning.component=optuna",
                        f"++theta_tuning.study_name={json.dumps(study.study_name)}",
                        f"++theta_tuning.trial_number={trial.number}",
                        f"++theta_tuning.log_version={json.dumps(version)}",
                    ])
                if impl.storage and metric:
                    args.extend([
                            f"++theta_tuning.optuna_trial_id={trial._trial_id}",
                            f"++theta_tuning.storage={json.dumps(str(impl.storage))}",
                            f"++theta_tuning.objective_metric={json.dumps(str(metric))}",
                            f"++theta_tuning.pruner={self.pruner_name}",
                            f"++theta_tuning.pruner_startup_trials={self.pruner_startup_trials}",
                            f"++theta_tuning.pruner_warmup_steps={self.pruner_warmup_steps}",
                        ])
                return tuple(args)

            try:
                for trial in trials:
                    jobs.append(make_job(trial))
                _export_summary(impl.config, study)
            except BaseException as exc:
                for trial in trials:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "launch_error")
                _export_summary(impl.config, study)
                raise
            failure = None
            consumed = set()

            def consume(index, result):
                nonlocal failed_count, failure
                trial = trials[index]
                try:
                    returned = result.return_value
                    if isinstance(returned, dict) and returned.get('theta_trial_status') == 'pruned':
                        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    else:
                        value = float(returned)
                        if not math.isfinite(value):
                            raise ValueError('Tuning objective must be finite')
                        study.tell(trial, value)
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception as exc:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    failed_count += 1
                    failure = failure or exc
                state = study._storage.get_trial(trial._trial_id).state
                if state in (optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL):
                    from src.app.api.tuning import finish_trial_runs
                    finish_trial_runs(impl.config.get("group"), impl.config.get("experiment_id"),
                                      trial.user_attrs.get("theta_log_version"),
                                      "cancelled" if state == optuna.trial.TrialState.PRUNED else "failed")
                from src.app.api.tuning import request_artifact_snapshot as request_database_export
                request_database_export({'study': study.study_name, 'trial': trial.number, 'state': state.name})
                consumed.add(index)
                _export_summary(impl.config, study)

            def can_start():
                from src.app.api.tuning import check_artifact_admission as admission
                admission()
                return (not self.timeout_seconds or time.monotonic() - start < self.timeout_seconds) and failed_count <= self.max_failed_trials

            def next_job():
                if len(trials) >= remaining or not can_start():
                    return None
                trial = study.ask()
                trials.append(trial)
                args = make_job(trial)
                _export_summary(impl.config, study)
                return args

            def started(index, pid):
                trial = trials[index]
                trial.set_user_attr('theta_execution_state', 'running')
                trial.set_user_attr('theta_worker_started_at', datetime.now().isoformat())
                trial.set_user_attr('theta_worker_pid', pid)
                _export_summary(impl.config, study)

            def waiting(index):
                trial = trials[index]
                trial.set_user_attr('theta_execution_state', 'pending')
                trial.set_user_attr('theta_worker_started_at', None)
                trial.set_user_attr('theta_worker_pid', None)
                _export_summary(impl.config, study)

            try:
                if os.environ.get('THETA_RESOURCE_POOL'):
                    from .parallel import launch_trials
                    results = launch_trials(impl, on_result=consume, next_job=next_job, on_start=started, on_wait=waiting,
                                            can_start=can_start, total_jobs=remaining)
                else:
                    results = impl.launcher.launch(jobs, initial_job_idx=impl.job_idx)
            except BaseException as exc:
                for trial in trials:
                    if study._storage.get_trial(trial._trial_id).state == optuna.trial.TrialState.RUNNING:
                        study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "launch_error")
                _export_summary(impl.config, study)
                raise
            impl.job_idx += len(results)
            for index, result in enumerate(results):
                if index < len(trials) and index not in consumed:
                    consume(index, result)
            if len(results) != len(trials):
                for trial in trials[len(results):]:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                study.set_user_attr("theta_last_stop_reason", "launch_error")
                _export_summary(impl.config, study)
                raise RuntimeError("Launcher did not return a result for every tuning trial")
            if failure and failed_count > self.max_failed_trials:
                study.set_user_attr("theta_last_stop_reason", "failure_limit")
                _export_summary(impl.config, study)
                raise failure
            _export_summary(impl.config, study)
            remaining -= len(trials)
        study.set_user_attr("theta_last_stop_reason", "time_budget" if remaining else "trial_limit")
        _export_summary(impl.config, study)
        try:
            best = study.best_trial
        except ValueError:
            _export_summary(impl.config, study)
            raise ValueError(f"Optuna study {study.study_name!r} has no completed trials") from None
        OmegaConf.save(
            OmegaConf.create({"name": "optuna", "best_params": best.params, "best_value": best.value}),
            f"{impl.config.hydra.sweep.dir}/optimization_results.yaml",
        )
