"""Optuna-owned portable storage, launch and continuation policy."""
from __future__ import annotations

import json

from src.app.api.tuning import TunerArtifact, TunerExecutionPlan, TunerLaunchBindings
from src.app.api.workspace import WorkspacePaths


def execution_plan(tuner, cfg, *, execution_id):
    options = tuner._options(cfg)
    storage = options['storage']
    managed = not storage or str(storage).startswith('sqlite:')
    return TunerExecutionPlan('optuna', execution_id,
        artifacts=(TunerArtifact('study', f'optuna/{execution_id}/study.sqlite', True),) if managed else (),
        concurrency=options['n_jobs'], trials=options['n_trials'],
        storage_kind='managed' if managed else 'external', distributions=('optuna', 'sqlalchemy'),
        continuation_notes='reinitialized from configured seed; suggestion sequence can differ')


def launch_bindings(cfg, *, plan, runtime, continuation=None):
    overrides, environment = [], {}
    if plan.storage_kind == 'managed':
        database = WorkspacePaths(runtime).results / plan.artifact('study').path
        if cfg.get('tuning') is not None:
            overrides.append('++tuning.options.optuna.storage=' + json.dumps('sqlite:///' + str(database)))
        environment['THETA_SQLITE_ROOT'] = str(database.parent)
    if continuation:
        overrides += ['++tuning.options.optuna.study_mode=resume',
                      '++tuning.options.optuna.study_name=' + json.dumps(continuation['study_name'])]
    return TunerLaunchBindings(tuple(overrides), environment)


def continuation_config(tuning):
    tuning = json.loads(json.dumps(tuning))
    options = (tuning.get('options') or {}).get('optuna') or {}
    for values in (tuning, options):
        for key in ('n_trials', 'n_jobs', 'storage', 'study_name', 'study_mode', 'timeout_seconds'):
            values.pop(key, None)
    return tuning


def snapshot_artifact(source, target, *, artifact):
    from src.app.api.tuning import sqlite_snapshot
    sqlite_snapshot(source, target)


def restore_continuation(cfg, *, plan, runtime, lineage):
    import sqlite3
    from contextlib import closing
    target = WorkspacePaths(runtime).results / plan.artifact(lineage.get('artifact_id', 'study')).path
    with closing(sqlite3.connect(target)) as db:
        if db.execute('PRAGMA quick_check').fetchone() != ('ok',):
            raise ValueError('Invalid continuation database')
    import optuna
    storage = optuna.storages.RDBStorage('sqlite:///' + str(target))
    try:
        study = optuna.load_study(storage=storage, study_name=lineage['study_name'])
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study._storage.set_trial_user_attr(trial._trial_id, 'theta_interruption',
                    {'parent_execution_id': lineage['parent_execution_id'], 'policy': 'fail; retry uses new trial number'})
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        study.set_user_attr('theta_continuation', {key: lineage[key] for key in ('parent_execution_id', 'sampler_restart', 'training_resume')})
    finally:
        storage.engine.dispose()
