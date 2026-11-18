"""Optuna's Lightning trial monitor and pruning policy."""

from lightning.pytorch.callbacks import Callback

from src.app.api.tuning import objective_value
from .policy import make_pruner
from .monitor import OptunaTrialMonitor, TrialPruned


class OptunaPruningCallback(Callback):
    def __init__(self, runtime):
        self.runtime = runtime
        self.monitor = OptunaTrialMonitor(runtime)
        self._reported = set()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.world_size != 1:
            if self.runtime.get("pruner", "median") != "none":
                raise ValueError("Optuna pruning currently requires a single Lightning process")
            return
        metric = str(self.runtime["objective_metric"])
        if metric not in trainer.callback_metrics:
            return
        step = int(trainer.current_epoch)
        if step in self._reported:
            return
        value = objective_value(trainer.callback_metrics, metric)
        self.monitor.report(value, step)
        self._reported.add(step)

    def on_fit_end(self, trainer, pl_module):
        if not self._reported and self.runtime.get("pruner", "median") != "none":
            raise ValueError(
                f"Optuna pruning requires the objective {self.runtime['objective_metric']!r} "
                "to be reported during validation."
            )
