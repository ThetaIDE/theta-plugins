"""New sweeps must not reuse cancelled or archived study identities."""
import json

import optuna
from omegaconf import OmegaConf

from package_fixtures import package_module
_optuna_component = package_module('optuna', 'component.py')
OptunaTuner = _optuna_component.OptunaTuner
from package_fixtures import package_module
_optuna_utils = package_module('optuna', 'utils.py')
get_next_study_name = _optuna_utils.get_next_study_name


def test_new_sweep_skips_studies_without_training_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = f'sqlite:///{tmp_path}/studies.db'
    for name in ('e_m_v0', 'e_m_v6', 'other_m_v99', 'e_m_vnot-a-version'):
        optuna.create_study(study_name=name, storage=storage)
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'e',
                            'tuning': {'options': {'optuna': {'storage': storage, 'study_mode': 'new'}}}})
    assert OptunaTuner().study_name(cfg, 'm') == 'e_m_v7'
    # Allocation inspects identities; it does not resume, delete or create studies.
    assert len(optuna.study.get_all_study_names(storage=storage)) == 4
    assert not optuna.load_study(study_name='e_m_v6', storage=storage).trials


def test_sweep_archives_reserve_names(workspace_results, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Training output folders do not name studies.
    (tmp_path / 'src/usr/results/logs/g/e/m/version_2').mkdir(parents=True)
    assert get_next_study_name('g', 'e', 'm') == 'e_m_v0'
    checkpoints = tmp_path / 'src/usr/results/checkpoints/g/e/m'
    (checkpoints / 'sweeps/e_m_v8').mkdir(parents=True)
    assert get_next_study_name('g', 'e', 'm') == 'e_m_v9'
    (checkpoints / 'tuning_summary.json').write_text(json.dumps({'study': 'e_m_v10'}))
    assert get_next_study_name('g', 'e', 'm') == 'e_m_v11'


def test_explicit_resume_study_name_is_preserved():
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'e',
                            'tuning': {'options': {'optuna': {'study_name': 'cancelled_{method}', 'study_mode': 'resume'}}}})
    assert OptunaTuner().study_name(cfg, 'm') == 'cancelled_m'
