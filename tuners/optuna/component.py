"""Optuna-specific compilation and lifecycle implementation."""

from __future__ import annotations

from typing import Any
import json
import math


def _sweep_expression(target, dimension):
    """Compile a typed dimension into the Hydra Optuna sweeper's expression."""
    if not hasattr(dimension, "get"):
        raise ValueError(f"Sweep dimension {target!r} must be a mapping")
    kind = dimension.get("type")
    if kind == "choice":
        values = dimension.get("values")
        if not isinstance(values, (list, tuple)) or not values or any(v is None or v == "" for v in values):
            raise ValueError(f"Sweep dimension {target!r} requires non-empty choices")
        if any(not isinstance(value, (str, bool, int, float)) or
               (isinstance(value, float) and not math.isfinite(value)) for value in values):
            raise ValueError(f"Sweep dimension {target!r} choices must be finite scalar values")
        return "choice(" + ",".join(json.dumps(v) for v in values) + ")"
    if kind not in ("int", "float", "log_float"):
        raise ValueError(f"Sweep dimension {target!r} has unsupported type {kind!r}")
    low, high, step = dimension.get("low"), dimension.get("high"), dimension.get("step")
    def valid_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not valid_number(low) or not valid_number(high) or low > high:
        raise ValueError(f"Sweep dimension {target!r} requires ordered finite bounds")
    if kind == "int" and (not isinstance(low, int) or not isinstance(high, int)):
        raise ValueError(f"Sweep dimension {target!r} requires integer bounds")
    if step is not None and (not valid_number(step) or step <= 0 or (kind == "int" and not isinstance(step, int))):
        raise ValueError(f"Sweep dimension {target!r} requires a positive compatible step")
    if kind == "log_float":
        if low <= 0 or step is not None:
            raise ValueError(f"Sweep dimension {target!r} requires positive log bounds and no step")
        return f"tag(log, interval({low}, {high}))"
    if kind == "int":
        # The installed Hydra Optuna plugin treats range.stop as an inclusive
        # Optuna bound (unlike Python's range).
        return f"range({low}, {high}, {step or 1})"
    if step is not None:
        return f"range({float(low)}, {float(high)}, {float(step)})"
    return f"interval({low}, {high})"




class OptunaTuner:
    remote_api_version = 1

    def execution_plan(self, cfg, *, execution_id):
        from .remote import execution_plan
        return execution_plan(self, cfg, execution_id=execution_id)

    def launch_bindings(self, cfg, **context):
        from .remote import launch_bindings
        return launch_bindings(cfg, **context)

    def continuation_config(self, cfg):
        from .remote import continuation_config
        return continuation_config(cfg)

    def snapshot_artifact(self, source, target, *, artifact):
        from .remote import snapshot_artifact
        return snapshot_artifact(source, target, artifact=artifact)

    def restore_continuation(self, cfg, **context):
        from .remote import restore_continuation
        return restore_continuation(cfg, **context)

    def preview_override(self, override):
        from optuna.distributions import BaseDistribution, CategoricalDistribution
        from .sweeper import _distribution_from_override
        value = _distribution_from_override(override)
        if isinstance(value, BaseDistribution):
            return value.choices[0] if isinstance(value, CategoricalDistribution) else value.low
        return None

    api_version = 1

    def job_entrypoint(self, cfg, method_cfg, script, python):
        return script, python

    def settings(self):
        fields = [
            {"key": "n_trials", "label": "Trials", "type": "int", "default": 40, "min": 1, "max": 100000},
            {"key": "n_jobs", "label": "Maximum concurrent trials", "type": "int", "default": 1, "min": 1, "max": 100000,
             "help": "Maximum trials running at once. Training Pipeline resource limits may reduce concurrency further."},
            {"key": "sampler", "label": "Sampler", "type": "choice", "default": "tpe", "choices": ["tpe", "random", "qmc"],
             "help": "QMC uses scrambled Sobol samples; categorical dimensions fall back to independent sampling."},
            {"key": "seed", "label": "Sampler seed", "type": "int", "default": 42, "min": 0, "max": 2147483647},
            {"key": "tpe_startup_trials", "label": "TPE random startup trials", "type": "int", "default": 10, "min": 0, "max": 100000,
             "when": {"sampler": "tpe"}},
            {"key": "tpe_candidates", "label": "TPE candidates", "type": "int", "default": 24, "min": 1, "max": 100000,
             "when": {"sampler": "tpe"}},
            {"key": "tpe_multivariate", "label": "TPE multivariate sampling", "type": "bool", "default": False,
             "when": {"sampler": "tpe"}},
            {"key": "storage", "label": "Storage URL", "type": "text", "default": ""},
            {"key": "study_mode", "label": "Study", "type": "choice", "default": "new", "choices": ["new", "resume"]},
            {"key": "study_name", "label": "Study name", "type": "text", "default": ""},
            {"key": "pruner", "label": "Early stopping", "type": "choice", "default": "none",
             "choices": ["none", "median"],
             "help": "Stop trials whose progress falls below the median of earlier trials at the same evaluation "
                     "checkpoint: the objective component's intermediate score, or the reported metric."},
            {"key": "pruner_startup_trials", "label": "Completed trials before pruning", "type": "int", "default": 5,
             "min": 0, "max": 100000, "when": {"pruner": "median"}},
            {"key": "pruner_warmup_steps", "label": "Warmup evaluation checkpoints", "type": "int", "default": 0,
             "min": 0, "max": 100000, "when": {"pruner": "median"},
             "help": "Never prune before this evaluation checkpoint (0 is the evaluation before training)."},
            {"key": "timeout_seconds", "label": "Time budget (seconds)", "type": "int", "default": 0, "min": 0, "max": 2147483647,
             "help": "Zero means no limit. Running trials finish; no new trials start after the budget expires."},
            {"key": "max_failed_trials", "label": "Allowed failed trials", "type": "int", "default": 0, "min": 0, "max": 100000,
             "help": "Maximum failed trials in this run before the sweep stops. Pruned trials do not count."},
        ]
        for field in fields:
            field["optional"] = field["key"] not in ("n_trials", "sampler")
            if field["key"] == "study_name":
                field["required_when"] = {"study_mode": "resume"}
            if field["key"] == "seed" or field["key"].startswith("tpe_"):
                field["parent"] = "sampler"
            elif field["key"].startswith("pruner_"):
                field["parent"] = "pruner"
        return fields

    @staticmethod
    def _options(cfg):
        tuning = cfg.get("tuning", {}) if cfg is not None and hasattr(cfg, "get") else {}
        if not hasattr(tuning, "get"):
            return {}
        all_options = tuning.get("options", {})
        if not hasattr(all_options, "get"):
            raise ValueError("tuning.options must be a mapping")
        options = all_options.get("optuna", {})
        if not hasattr(options, "get"):
            raise ValueError("tuning.options.optuna must be a mapping")
        return {field["key"]: options.get(field["key"], field["default"])
                for field in OptunaTuner().settings()}

    @staticmethod
    def _require_dependencies():
        from importlib.util import find_spec
        from src.app.api.errors import ConfigurationError

        if find_spec("optuna") is None:
            raise ConfigurationError("The Optuna tuner component needs its declared Python dependencies (optuna, sqlalchemy); install them, then retry.")

    def validate_config(self, cfg):
        self._require_dependencies()
        tuning = self._options(cfg)
        for key, default in (("n_trials", 40), ("n_jobs", 1)):
            value = tuning.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Sweep {key} must be a positive integer.")
        if tuning.get("sampler") not in ("tpe", "random", "qmc"):
            raise ValueError("Optuna sampler must be 'tpe', 'random', or 'qmc'.")
        startup, candidates, multivariate = (tuning.get("tpe_startup_trials"), tuning.get("tpe_candidates"),
                                            tuning.get("tpe_multivariate"))
        if tuning.get("sampler") == "tpe":
            if isinstance(startup, bool) or not isinstance(startup, int) or startup < 0:
                raise ValueError("TPE startup trials must be a nonnegative integer.")
            if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates < 1:
                raise ValueError("TPE candidates must be a positive integer.")
            if not isinstance(multivariate, bool):
                raise ValueError("TPE multivariate sampling must be a boolean.")
        if tuning.get("study_mode") not in ("new", "resume"):
            raise ValueError("Optuna study mode must be 'new' or 'resume'.")
        if tuning.get("pruner") not in ("none", "median"):
            raise ValueError("Optuna early stopping must be 'none' or 'median'.")
        counts = ["timeout_seconds", "max_failed_trials"]
        if tuning.get("pruner") == "median":
            counts += ["pruner_startup_trials", "pruner_warmup_steps"]
        for key in counts:
            value = tuning.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Optuna {key} must be a nonnegative integer.")
        if tuning.get("study_mode") == "resume" and not str(tuning.get("study_name") or "").strip():
            raise ValueError("Resuming an Optuna study requires a study name.")
        explicit_name = str(tuning.get("study_name") or "").strip()
        if explicit_name:
            from src.app.api.config import parse_methods_dict
            if len(parse_methods_dict(cfg)) > 1 and "{method}" not in explicit_name:
                raise ValueError("A study name shared by multiple methods must include {method}.")
            try:
                explicit_name.format(method="method", experiment="experiment", group="group")
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError("Study name supports only {method}, {experiment}, and {group} placeholders.") from exc
        seed = tuning.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2147483647:
            raise ValueError("Optuna sampler seed must be a nonnegative integer.")

    def pruning_enabled(self, cfg):
        runtime = cfg.get("theta_tuning") or {}
        return runtime.get("pruner", self._options(cfg)["pruner"]) != "none"

    def validate_dimension(self, target, dimension):
        if not str(target).startswith(("agent.", "model.", "config.")):
            raise ValueError(f"Sweep dimension {target!r} requires an explicit agent, model, or config path")
        _sweep_expression(target, dimension)

    def launch_dashboard(self, storage):
        from .utils import launch_optuna_dashboard
        return launch_optuna_dashboard(storage)

    def training_callbacks(self, cfg):
        runtime = cfg.get("theta_tuning")
        if not hasattr(runtime, "get") or runtime.get("optuna_trial_id") is None:
            return []
        from .pruning import OptunaPruningCallback
        return [OptunaPruningCallback(runtime)]

    def trial_monitor(self, cfg):
        runtime = cfg.get("theta_tuning")
        if not hasattr(runtime, "get") or runtime.get("optuna_trial_id") is None:
            return None
        from .monitor import OptunaTrialMonitor
        return OptunaTrialMonitor(runtime)

    def compile(self, overrides: list[str], **context: Any) -> list[str]:
        cfg = context["cfg"]
        extra_args = context["extra_args"]
        study_name = context["study_name"]
        method_cfg = context["method_cfg"]
        uses_agents = context["uses_agents"]
        model_keys = context["model_keys"]
        for arg in extra_args or []:
            key = arg.lstrip("+").split("=", 1)[0]
            if key == "hydra/sweeper" or key.startswith("hydra.sweeper."):
                raise ValueError(f"{key} is compiled from the tuning block; set tuning.direction or "
                                 "tuning.options.optuna.<setting> instead.")
        overrides.append("hydra/sweeper=theta_optuna")
        if study_name:
            overrides.append(f"++hydra.sweeper.study_name={json.dumps(study_name)}")
        tuning = cfg["tuning"]
        options = self._options(cfg)
        overrides.append(f"++hydra.sweeper.n_trials={options['n_trials']}")
        overrides.append(f"++hydra.sweeper.n_jobs={options['n_jobs']}")
        overrides.append(f"++hydra.sweeper.storage={json.dumps(self.storage(cfg))}")
        samplers = {"tpe": "TPESampler", "random": "RandomSampler", "qmc": "QMCSampler"}
        overrides.append(f"++hydra.sweeper.sampler._target_=optuna.samplers.{samplers[options['sampler']]}")
        overrides.append(f"++hydra.sweeper.sampler.seed={options['seed']}")
        if options["sampler"] == "tpe":
            overrides.extend([
                f"++hydra.sweeper.sampler.n_startup_trials={options['tpe_startup_trials']}",
                f"++hydra.sweeper.sampler.n_ei_candidates={options['tpe_candidates']}",
                f"++hydra.sweeper.sampler.multivariate={str(options['tpe_multivariate']).lower()}",
            ])
        elif options["sampler"] == "qmc":
            overrides.append("++hydra.sweeper.sampler.scramble=true")
        for key in ("pruner", "pruner_startup_trials", "pruner_warmup_steps", "timeout_seconds", "max_failed_trials"):
            overrides.append(f"++hydra.sweeper.{key}={options[key]}")
        overrides.append(f"++hydra.sweeper.direction={tuning['direction']}")
        for key, value in (method_cfg.get("tune") or {}).items():
            if not isinstance(value, str) and (not isinstance(key, str) or not key.startswith(("agent.", "model.", "config."))):
                raise ValueError(f"Typed sweep dimension {key!r} requires an explicit agent, model, or config path")
            if key.startswith("config."):
                full_key = key[len("config."):]
            elif "." in key:
                full_key = key
            elif not uses_agents:
                full_key = f"model.{key}"
            elif key in model_keys:
                full_key = f"model.{key}"
            else:
                full_key = f"agent.{key}"
            replaced = [arg for arg in overrides if arg.split("=", 1)[0].lstrip("+") == full_key]
            prefix = "++" if any(arg.startswith("++") for arg in replaced) else ""
            overrides = [arg for arg in overrides if arg.split("=", 1)[0].lstrip("+") != full_key]
            overrides.append(f"{prefix}{full_key}={_sweep_expression(key, value)}")
        if "--multirun" not in overrides and "-m" not in overrides:
            overrides.insert(0, "--multirun")
        return overrides

    def study_name(self, cfg, method_name):
        from .utils import get_next_study_name
        explicit = str(self._options(cfg).get("study_name") or "").strip()
        return (explicit.format(method=method_name, experiment=cfg.experiment_id, group=cfg.group)
                if explicit else get_next_study_name(cfg.group, cfg.experiment_id, method_name, self.storage(cfg)))

    def storage(self, cfg):
        from .utils import default_storage_url
        value = str(self._options(cfg)["storage"] or default_storage_url())
        return value.replace("${experiment_id}", str(cfg["experiment_id"])) if "${experiment_id}" in value else value

    def prepare(self, cfg, storage, study_name):
        self._require_dependencies()
        import optuna
        from .utils import delete_optuna_study, get_optuna_storage
        from src.app.api.errors import ConfigurationError

        mode = self._options(cfg)["study_mode"]
        if mode == "resume" and cfg.get("remake", False):
            raise ConfigurationError("Cannot resume an Optuna study with remake enabled.")
        if cfg.get("remake", False):
            delete_optuna_study(storage, study_name)
        tuning = cfg.get("tuning", {})
        direction = tuning["direction"]
        if not storage:
            raise ConfigurationError("Optuna tuning requires persistent storage for trial artifacts.")
        target = get_optuna_storage(storage)
        try:
            if mode == "resume":
                study = optuna.load_study(study_name=study_name, storage=target)
                if study.direction.name.lower() != direction:
                    raise ConfigurationError(f"Study {study_name!r} has a different objective direction.")
                from src.app.api.tuning import objective_metric
                metric = objective_metric(cfg, "") or None
                recorded = study.user_attrs.get("theta_objective_metric")
                if recorded and metric and recorded != metric:
                    raise ConfigurationError(f"Study {study_name!r} optimizes {recorded!r}, not {metric!r}.")
            else:
                optuna.create_study(study_name=study_name, storage=target, direction=direction, load_if_exists=False)
        except optuna.exceptions.DuplicatedStudyError as exc:
            raise ConfigurationError(f"Study {study_name!r} already exists; choose another name or resume it.") from exc
        except KeyError as exc:
            raise ConfigurationError(f"Study {study_name!r} does not exist and cannot be resumed.") from exc

    def read_artifact_snapshot(self, artifact, path):
        return self.read_study_snapshot(path)

    def read_study_snapshot(self, database):
        from .utils import read_study_snapshot
        info = database.stat()
        return read_study_snapshot(str(database), info.st_mtime_ns, info.st_size)

    def finalize_interrupted(self, runtime, record):
        """Close interrupted trials before the allocation's final database export."""
        import optuna
        from .utils import write_study_summary
        from src.app.api.workspace import WorkspacePaths
        results = WorkspacePaths(runtime).results
        database = results / 'optuna' / record['execution_id'] / 'study.sqlite'
        if not database.is_file():
            return
        storage = 'sqlite:///' + str(database.resolve())
        checkpoint = results / 'checkpoints' / record['group'] / record['result_experiment_id']
        import json
        for summary in checkpoint.glob('*/tuning_summary.json'):
            name = json.loads(summary.read_text())['study']
            study = optuna.load_study(study_name=name, storage=storage)
            for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            study.set_user_attr('theta_last_stop_reason', 'interrupted')
            write_study_summary(study, summary.parent)

    def finalize(self, group, experiment, method, storage, study_name):
        from .utils import promote_best_trial_checkpoint
        return promote_best_trial_checkpoint(group, experiment, method, storage, study_name)

    def cancel(self, cfg, method, storage, study_name):
        """The selected method stopped: close only this launch's outstanding trials."""
        import optuna
        from .utils import results_root, write_study_summary
        study = optuna.load_study(study_name=study_name, storage=storage)
        initial = int(study.user_attrs.get("theta_initial_trials", 0))
        for trial in study.trials[initial:]:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
        study.set_user_attr("theta_last_stop_reason", "cancelled")
        write_study_summary(study, results_root() / "checkpoints" / str(cfg.group) / str(cfg.experiment_id) / method)
