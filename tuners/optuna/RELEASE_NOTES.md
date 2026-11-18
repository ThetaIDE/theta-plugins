# 1.3.0

Pruning works with objective components and repeated training runs. Such a trial
trains its seeds in separate workers; the IDE scores the running seed panel with the
objective's `intermediate_function` at each evaluation checkpoint every seed has
reached, reports it to the trial, and stops every seed when the median pruner prunes
it. Requires an IDE with Theta SDK 3.4 (`ObjectiveProgress`, coordinator-side
progress scoring). Objectives without an intermediate function still reject pruning.

# 1.2.1

Source moved to theta-plugins `tuners/optuna`; behavior is unchanged from 1.2.0.

# 1.2.0

The `theta_optuna` sweeper profile targets the IDE's generic tuner adapter,
`hydra_plugins.theta_sweeper.ThetaSweeper` with `tuner: optuna`, which forwards to this
package's `hydra_sweeper` export. Requires an IDE with Theta SDK 3.1; the Optuna-specific
adapter `theta_optuna_sweeper` no longer exists in the IDE.

# 1.1.1

Native storage-v3 package with public SDK imports and workspace-owned results paths.
