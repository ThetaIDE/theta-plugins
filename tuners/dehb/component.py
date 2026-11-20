"""Local, sequential DEHB with a transition budget."""
import json
from decimal import Decimal
import math
import time

from src.app.api.tuning import LocalBudgetSweeper, LocalBudgetTuner, hydra_value, progress, trial_log_version


def parameter(target, dimension):
    import ConfigSpace as cs

    if not target.startswith(("agent.", "model.")) or target in {
        "agent.name", "agent.backend", "agent.algorithm", "model.name", "model.architecture",
    }:
        raise ValueError(f"DEHB requires a fixed provider identity, got {target}")
    if not hasattr(dimension, "get"):
        raise ValueError(f"DEHB dimension {target!r} requires a mapping")
    kind = dimension.get("type")
    if kind == "choice":
        values = list(dimension.get("values") or [])
        if not values or any(type(v) not in {str, bool, int, float} or
                             (type(v) is float and not math.isfinite(v)) for v in values):
            raise ValueError(f"DEHB dimension {target!r} requires finite scalar choices")
        return cs.Constant(target, values[0]) if len(values) == 1 else cs.Categorical(target, values)
    if kind not in {"int", "float", "log_float"}:
        raise ValueError(f"Unsupported DEHB dimension {kind!r}")
    low, high = dimension.get("low"), dimension.get("high")
    numeric_types = {int} if kind == "int" else {int, float}
    if (not all(type(v) in numeric_types and math.isfinite(v) for v in (low, high))
            or low > high or (kind == "log_float" and low <= 0)):
        raise ValueError(f"Invalid bounds for DEHB dimension {target!r}")
    step = dimension.get("step")
    if step is not None:
        if kind == "log_float" or type(step) not in numeric_types or not math.isfinite(step) or step <= 0:
            raise ValueError(f"Invalid step for DEHB dimension {target!r}")
        first, last, increment = map(lambda value: Decimal(str(value)), (low, high, step))
        count = (last - first) // increment + 1
        if count > 10001:
            raise ValueError(f"DEHB dimension {target!r} has too many discrete values")
        cast = int if kind == "int" else float
        values = [cast(first + i * increment) for i in range(int(count))]
        return cs.Constant(target, values[0]) if len(values) == 1 else cs.OrdinalHyperparameter(target, values)
    if low == high:
        return cs.Constant(target, low)
    if kind == "int":
        return cs.Integer(target, (low, high))
    return cs.Float(target, (low, high), log=kind == "log_float")


class Tuner(LocalBudgetTuner):
    component = "dehb"
    budget_option = "min_transitions"

    def settings(self):
        return [
            dict(key=key, label=label, type="int", default=default, min=minimum)
            for key, label, default, minimum in [
                ("min_transitions", "Minimum transitions per seed", 128, 1),
                ("max_transitions", "Maximum transitions per seed", 512, 1),
                ("reduction_factor", "Reduction factor", 2, 2),
                ("search_cap", "Total search transitions", 4096, 1),
                ("seed", "Optimizer seed", 42, 0),
            ]
        ]

    def validate_config(self, cfg):
        super().validate_config(cfg)
        options = self.options(cfg)
        budget = options["min_transitions"]
        limit = options["max_transitions"]
        if budget >= limit:
            raise ValueError("DEHB requires multiple budget levels")
        while budget < limit:
            budget *= options["reduction_factor"]
        if budget != limit:
            raise ValueError("Maximum transitions must equal minimum times a power of the reduction factor")

    def validate_dimension(self, target, dimension):
        parameter(target, dimension)


class Sweeper(LocalBudgetSweeper):
    def sweep(self, arguments):
        import ConfigSpace as cs
        from dehb import DEHB

        cfg = self.loader.load_sweep_config(self.config, arguments)
        tuner = Tuner()
        tuner.validate_config(cfg)
        options = tuner.options(cfg)
        profile = tuner.execution_profile(cfg, dimensions=cfg.theta_hpo_space)
        rollout = profile["quantum"]
        minimum, maximum = options["min_transitions"], options["max_transitions"]
        seeds = list((cfg.get("variability") or {}).get("training_seeds") or [cfg.seed])
        archive = self.study_archive(cfg)
        space = cs.ConfigurationSpace(seed=options["seed"])
        space.add([parameter(key, dim) for key, dim in cfg.theta_hpo_space.items()])
        optimizer = DEHB(cs=space, dimensions=len(space), min_fidelity=minimum,
                         max_fidelity=maximum, eta=options["reduction_factor"], n_workers=1,
                         seed=options["seed"], save_freq="step", output_path=str(archive / "optimizer"))
        tuning = cfg.tuning
        direction = str(tuning.get("direction") or "maximize")
        trials = []
        summary = dict(schema_version=1, component="dehb", study=self.study_name,
                       direction=direction, objective_metric=tuning.metric, trials=trials, training_seeds=seeds,
                       search_cap=options["search_cap"], best_trial_id=None)

        spent = 0
        attempt_limit = options["search_cap"] // (minimum * len(seeds))
        while len(trials) < attempt_limit and spent + minimum * len(seeds) <= options["search_cap"]:
            job = optimizer.ask()
            budget = round(job["fidelity"])
            reservation = budget * len(seeds)
            if budget % rollout or not minimum <= budget <= maximum:
                raise RuntimeError(f"Unexpected DEHB fidelity {job['fidelity']}")
            if spent + reservation > options["search_cap"]:
                break
            index = len(trials)
            params = {key: value.item() if hasattr(value, "item") else value for key, value in dict(job["config"]).items()}
            runtime = dict(component="dehb", study_name=self.study_name, trial_number=index)
            trial = dict(id=index, state="running", parameters=params, budget=budget,
                         reserved_transitions=reservation,
                         log_version=trial_log_version({"theta_tuning": runtime}))
            trials.append(trial)
            self.publish(archive, summary)
            replacements = [f"++{key}={hydra_value(value)}" for key, value in params.items()]
            replacements += [f"++{profile['budget_path']}={budget}",
                             f"++theta_tuning={hydra_value(runtime)}", "++theta_final_evaluation=true"]
            replaced = {arg.split("=", 1)[0].lstrip("+") for arg in replacements}
            overrides = [arg for arg in arguments if arg.split("=", 1)[0].lstrip("+") not in replaced]
            started = time.monotonic()
            value = None
            interrupted = None
            try:
                result, = self.launcher.launch([overrides + replacements], initial_job_idx=index)
                value = float(result.return_value)
            except KeyboardInterrupt as exc:
                interrupted = exc
                trial["error"] = "Study interrupted"
            except Exception as exc:
                trial["error"] = str(exc)
            trial["duration_seconds"] = time.monotonic() - started
            records = self.worker_results(cfg, trial)
            observed = {seed: progress(record, profile["unit"]) for seed, record in records.items()}
            observed = {seed: count for seed, count in observed.items() if count is not None}
            charged = sum(observed.get(seed, budget) for seed in seeds)
            spent += charged
            trial.update(actual_transitions=sum(observed.values()), charged_transitions=charged,
                         reported_seeds=len(observed))
            complete = value is not None and math.isfinite(value) and all(
                observed.get(seed) == budget and records[seed]["status"] == "completed" for seed in seeds)
            trial["state"] = "complete" if complete else "fail"
            if complete:
                trial["value"] = value
            fitness = (value if direction == "minimize" else -value) if complete else math.inf
            optimizer.tell(job, {"fitness": fitness, "cost": charged, "info": {"trial": index}})
            winning = summary["best_trial_id"]
            if complete and budget == maximum and (winning is None or (
                value < trials[winning]["value"] if direction == "minimize" else value > trials[winning]["value"]
            )):
                summary.update(best_trial_id=index, best_params=params, best_value=value)
            summary["charged_transitions"] = spent
            if interrupted is not None:
                summary["outcome"] = "interrupted"
            self.publish(archive, summary)
            if interrupted is not None:
                raise interrupted
        if summary["best_trial_id"] is not None:
            (archive / "winner.json").write_text(json.dumps({
                "parameters": summary["best_params"], profile["budget_path"]: maximum,
                "tuning_seeds": seeds, "trial": summary["best_trial_id"],
            }, indent=2))
        summary["outcome"] = "no_full_budget_configuration_completed" if summary["best_trial_id"] is None else "completed"
        self.publish(archive, summary)
        return []
