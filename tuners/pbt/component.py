"""Synchronous population-based training over Theta worker records."""
import random
import math
import time

from src.app.api.tuning import LocalBudgetSweeper, LocalBudgetTuner, hydra_value, progress, trial_log_version


class Tuner(LocalBudgetTuner):
    component = "pbt"
    budget_option = "segment_transitions"

    def settings(self):
        return [dict(key=key, label=label, type="int", default=default, min=minimum)
                for key, label, default, minimum in [
                    ("population", "Population size", 3, 2),
                    ("rounds", "Training rounds", 3, 2),
                    ("segment_transitions", "Transitions per round", 128, 1),
                    ("seed", "Controller seed", 42, 0),
                ]]

    def validate_config(self, cfg):
        super().validate_config(cfg)
        if cfg.get("variability") is not None:
            raise ValueError("PBT uses one training stream per population member")
        if len(cfg.get("methods") or {}) != 1:
            raise ValueError("PBT requires exactly one training method")

    def execution_profile(self, cfg, *, dimensions, paths=None):
        profile = super().execution_profile(cfg, dimensions=dimensions, paths=paths)
        if not profile.get("transfer_format"):
            raise ValueError("PBT requires a runner with compatible dynamic checkpoint transfer")
        mutable = profile["mutable_parameters"]
        for key, dimension in dimensions.items():
            if key not in mutable:
                raise ValueError(f"Selected runner cannot mutate {key}")
            self.validate_dimension(key, dimension)
            mutable[key].validate(dimension["low"])
            mutable[key].validate(dimension["high"])
        return profile

    def validate_dimension(self, target, dimension):
        low, high = dimension.get("low"), dimension.get("high")
        if (dimension.get("type") not in {"float", "log_float"} or dimension.get("step") is not None
                or not all(type(v) in {int, float} and math.isfinite(v) for v in (low, high))
                or not low < high or (low <= 0 and dimension["type"] == "log_float")):
            raise ValueError(f"PBT requires valid continuous bounds for {target}")


class Sweeper(LocalBudgetSweeper):
    def sweep(self, arguments):
        cfg = self.loader.load_sweep_config(self.config, arguments)
        tuner = Tuner()
        tuner.validate_config(cfg)
        options = tuner.options(cfg)
        dimensions = cfg.theta_hpo_space
        profile = tuner.execution_profile(cfg, dimensions=dimensions)
        segment = options["segment_transitions"]
        rng = random.Random(options["seed"])
        seeds = random.Random(int(cfg.seed)).sample(range(2**32), options["population"])
        eval_seed = (int(cfg.seed) + 2**15) % 2**32
        archive = self.study_archive(cfg)
        trials = []
        summary = dict(schema_version=1, component="pbt", study=self.study_name,
                       direction=cfg.tuning.direction, objective_metric=cfg.tuning.metric,
                       best_trial_id=None, trials=trials, evaluation_seed=eval_seed,
                       training_seeds=seeds, actual_transitions=0)

        params = [
            {key: math.exp(rng.uniform(math.log(dim.low), math.log(dim.high)))
             if dim.type == "log_float" else rng.uniform(dim.low, dim.high)
             for key, dim in dimensions.items()}
            for _ in seeds
        ]
        parents = [None] * len(seeds)
        for round_index in range(options["rounds"]):
            current = []
            for member, seed in enumerate(seeds):
                index = len(trials)
                start = round_index * segment
                parent = parents[member]
                schedule = [{"step": start, "parameters": params[member]}]
                dynamic = dict(api_version=1, start=start, schedule=schedule)
                if parent:
                    dynamic["source"] = parent["checkpoint"]
                runtime = dict(component="pbt", study_name=self.study_name, trial_number=index)
                trial = dict(id=index, member=member, round=round_index + 1, state="running",
                             log_version=trial_log_version({"theta_tuning": runtime}),
                             budget=start + segment, parameters=params[member],
                             parent=parent["id"] if parent else None,
                             schedule=(parent["schedule"] if parent else []) + schedule)
                trials.append(trial)
                self.publish(archive, summary)
                replacements = dict(seed=seed, variability_evaluation_seed=eval_seed,
                                    theta_dynamic=dynamic, theta_final_evaluation=True, theta_tuning=runtime)
                replacements[profile["budget_path"]] = segment
                replaced = replacements.keys()
                overrides = [arg for arg in arguments if arg.split("=", 1)[0].lstrip("+") not in replaced]
                started = time.monotonic()
                interrupted = None
                try:
                    result, = self.launcher.launch(
                        [overrides + [f"++{key}={hydra_value(value)}" for key, value in replacements.items()]],
                        initial_job_idx=index)
                    score = float(result.return_value)
                    if math.isfinite(score):
                        trial["value"] = score
                except KeyboardInterrupt as exc:
                    interrupted = exc
                    trial["error"] = "Study interrupted"
                except Exception as exc:
                    trial["error"] = str(exc)
                record = self.worker_results(cfg, trial).get(seed, {})
                cost = progress(record, profile["unit"])
                artifacts = (record.get("result") or {}).get("artifacts") or []
                checkpoint = next((artifact["path"] for artifact in artifacts
                                   if artifact.get("format") == profile["transfer_format"]
                                   and artifact.get("role") == "training_state"), None)
                trial["checkpoint"] = checkpoint
                trial["inference_checkpoint"] = next((artifact["path"] for artifact in artifacts
                                                       if artifact.get("role") == "final"), None)
                trial.update(actual_transitions=cost, duration_seconds=time.monotonic() - started)
                summary["actual_transitions"] += cost or 0
                trial["state"] = "complete" if cost == segment and checkpoint and record.get("status") == "completed" and "value" in trial else "fail"
                current.append(trial)
                if interrupted is not None:
                    summary["outcome"] = "interrupted"
                self.publish(archive, summary)
                if interrupted is not None:
                    raise interrupted
            if any(trial["state"] != "complete" for trial in current):
                summary["outcome"] = "incomplete_round"
                break
            ranked = sorted(current, key=lambda row: row["value"], reverse=cfg.tuning.direction == "maximize")
            winner = ranked[0]
            summary.update(best_trial_id=winner["id"], best_value=winner["value"],
                           winner_checkpoint=winner["checkpoint"],
                           winner_inference_checkpoint=winner["inference_checkpoint"],
                           winner_schedule=winner["schedule"], completed_rounds=round_index + 1,
                           outcome="completed")
            parents = current.copy()
            recipient = ranked[-1]["member"]
            parents[recipient] = winner
            mutation = dict(winner["parameters"])
            key = rng.choice(list(mutation))
            dim = dimensions[key]
            value = mutation[key]
            for factor in rng.sample([0.8, 1.2], 2):
                changed = max(dim.low, min(dim.high, value * factor))
                if changed != value:
                    break
            if changed == value:
                changed = (dim.low + dim.high) / 2
            mutation[key] = changed
            params[recipient] = mutation
            self.publish(archive, summary)
        self.publish(archive, summary)
        return []
