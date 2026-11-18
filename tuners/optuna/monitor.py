"""Engine-independent reporting of intermediate Optuna trial values."""

import optuna

from .policy import make_pruner


class TrialPruned(optuna.TrialPruned):
    """A clean tuner stop that telemetry can distinguish from a failure."""

    theta_trial_status = "pruned"


class OptunaTrialMonitor:
    def __init__(self, runtime):
        self.runtime = runtime
        self._trial = None

    def report(self, value, step):
        if self._trial is None:
            runtime = self.runtime
            pruner = make_pruner(runtime["pruner"], runtime["pruner_startup_trials"], runtime["pruner_warmup_steps"])
            study = optuna.load_study(study_name=runtime["study_name"], storage=runtime["storage"], pruner=pruner)
            trial = optuna.trial.Trial(study, int(runtime["optuna_trial_id"]))
            if trial.number != int(runtime["trial_number"]):
                raise ValueError("Optuna trial identity does not match the training job")
            self._trial = trial
        self._trial.report(float(value), int(step))
        if self._trial.should_prune():
            raise TrialPruned(f"Trial {self._trial.number} pruned at evaluation {step}")
