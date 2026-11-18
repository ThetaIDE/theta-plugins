"""Optuna pruning policies, independent of any training engine."""

import optuna


def make_pruner(name, startup_trials=5, warmup_steps=0):
    if name == "none":
        return optuna.pruners.NopPruner()
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (startup_trials, warmup_steps)):
        raise ValueError("Pruner startup and warmup counts must be nonnegative integers")
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=startup_trials, n_warmup_steps=warmup_steps)
    raise ValueError(f"Unsupported Optuna pruner {name!r}")
