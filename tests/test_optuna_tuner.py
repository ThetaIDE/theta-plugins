"""The Optuna tuner's settings, compilation and study lifecycle."""
from package_fixtures import install_package
import importlib.util
import os
import subprocess
import sys
from omegaconf import OmegaConf
from package_fixtures import IDE
from src.app.core.contributions import ContributionRegistry
from src.app.core.provider_catalog import get_catalog
from src.app.pipeline.exceptions import ConfigurationError
from src.app.workspace import WorkspacePaths
import pytest

from package_fixtures import package_workspace
from src.app.pipeline.commands import build_method_overrides


@pytest.fixture(autouse=True)
def optuna_workspace(monkeypatch):
    """Sweeps compile through the selected tuner, so Optuna is installed in the workspace."""
    monkeypatch.setenv("THETA_WORKSPACE", str(package_workspace("optuna").root))


@pytest.mark.parametrize("arg", ["hydra.sweeper.n_trials=5", "++hydra.sweeper.direction=minimize",
                                 "hydra/sweeper=other"])
def test_sweeper_settings_come_only_from_the_tuning_block(arg):
    method = {"agent": "cql", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    with pytest.raises(ValueError, match="compiled from the tuning block"):
        build_method_overrides(
            "cql", method, cfg={"paradigm": "offline_rl", "tuning": {"component": "optuna", "direction": "maximize"}},
            extra_args=[arg])


def test_missing_optuna_dependencies_have_an_actionable_error(monkeypatch):
    from package_fixtures import package_module
    _optuna_component = package_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ConfigurationError, match="declared Python dependencies"):
        OptunaTuner().validate_config({"tuning": {"component": "optuna"}})



def test_optuna_component_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from package_fixtures import package_workspace
    workspace = package_workspace("optuna")
    cfg = {"paradigm": "online_rl", "tuning": {"component": "optuna",
        "metric": "eval/reward", "direction": "maximize",
        "options": {"optuna": {"n_trials": 7, "sampler": "random", "seed": 23}},
    }}
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("method", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.n_trials=7" in overrides
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.RandomSampler" in overrides
    assert "++hydra.sweeper.sampler.seed=23" in overrides



def test_qmc_and_tpe_specific_sampler_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from package_fixtures import package_workspace
    workspace = package_workspace("optuna")
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    cfg = {"paradigm": "online_rl", "tuning": {"component": "optuna", "direction": "maximize", "options": {"optuna": {
        "sampler": "qmc", "seed": 8,
    }}}}
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.QMCSampler" in overrides
    assert "++hydra.sweeper.sampler.scramble=true" in overrides
    cfg["tuning"]["options"]["optuna"].update({"sampler": "tpe", "tpe_startup_trials": 3,
                                                "tpe_candidates": 12, "tpe_multivariate": True})
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.sampler.n_startup_trials=3" in overrides
    assert "++hydra.sweeper.sampler.n_ei_candidates=12" in overrides
    assert "++hydra.sweeper.sampler.multivariate=true" in overrides



def test_pruning_time_and_failure_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from package_fixtures import package_workspace
    workspace = package_workspace("optuna")
    cfg = {"paradigm": "supervised", "tuning": {"component": "optuna", "direction": "minimize", "options": {"optuna": {
        "pruner": "median", "pruner_startup_trials": 2, "pruner_warmup_steps": 3,
        "timeout_seconds": 60, "max_failed_trials": 1,
    }}}}
    method = {"model": "dnn", "tune": {"model.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.pruner=median" in overrides
    assert "++hydra.sweeper.pruner_startup_trials=2" in overrides
    assert "++hydra.sweeper.pruner_warmup_steps=3" in overrides
    assert "++hydra.sweeper.timeout_seconds=60" in overrides
    assert "++hydra.sweeper.max_failed_trials=1" in overrides



def test_optuna_storage_override_is_identical_for_job_and_study():
    from src.app.pipeline.commands import build_method_overrides
    from package_fixtures import package_workspace
    workspace = package_workspace("optuna")
    from package_fixtures import package_module
    _optuna_component = package_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner

    cfg = OmegaConf.create({"experiment_id": "e", "paradigm": "online_rl",
                            "tuning": {"component": "optuna", "direction": "maximize", "options": {"optuna": {
                                "storage": "sqlite:///folder with spaces/new.db"}}}})
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("method", method, cfg=cfg, study_name="study one", workspace=workspace)
    assert OptunaTuner().storage(cfg) == "sqlite:///folder with spaces/new.db"
    assert '++hydra.sweeper.storage="sqlite:///folder with spaces/new.db"' in overrides
    assert '++hydra.sweeper.study_name="study one"' in overrides
    with pytest.raises(ValueError, match="compiled from the tuning block"):
        build_method_overrides("method", method, cfg=cfg,
                               extra_args=["hydra.sweeper.storage=sqlite:///different.db"], workspace=workspace)



def test_optuna_new_and_resume_studies_are_explicit(tmp_path):
    from package_fixtures import package_module
    _optuna_component = package_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner

    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "paradigm": "online_rl",
        "tuning": {"component": "optuna", "metric": "eval/reward", "direction": "maximize", "options": {"optuna": {
            "study_name": "study", "study_mode": "new"}}},
    })
    tuner = OptunaTuner()
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    tuner.prepare(cfg, storage, "study")
    with pytest.raises(ConfigurationError, match="already exists"):
        tuner.prepare(cfg, storage, "study")
    cfg.tuning.options.optuna.study_mode = "resume"
    tuner.prepare(cfg, storage, "study")
    with pytest.raises(ConfigurationError, match="does not exist"):
        tuner.prepare(cfg, storage, "missing")
    cfg.tuning.direction = "minimize"
    with pytest.raises(ConfigurationError, match="different objective direction"):
        tuner.prepare(cfg, storage, "study")



def test_optuna_sweeper_does_not_import_lightning_when_pruning_is_off(tmp_path):
    project = IDE
    script = tmp_path / "without_lightning.py"
    script.write_text(
        "import sys\n"
        "class BlockLightning:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'lightning' or fullname.startswith('lightning.'):\n"
        "            raise ModuleNotFoundError('Lightning intentionally unavailable')\n"
        "sys.meta_path.insert(0, BlockLightning())\n"
        "import optuna\n"
        "from hydra_plugins.theta_sweeper import ThetaSweeper\n"
        "ThetaSweeper('optuna', sampler=optuna.samplers.RandomSampler(), direction='minimize', storage=None,"
        " study_name='test', n_trials=1, n_jobs=1, search_space=None, pruner='none', pruner_startup_trials=5,"
        " pruner_warmup_steps=0, timeout_seconds=0, max_failed_trials=0)\n",
        encoding="utf-8",
    )
    from package_fixtures import package_workspace
    workspace = package_workspace("optuna")
    # The saved sweeper target resolves from the selected workspace's installed Optuna component.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "THETA_WORKSPACE": str(workspace.root), "THETA_DATA_DIR": str(workspace.data_dir)}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('cfg,active', [
    ({'tuning': {'options': {'optuna': {'pruner': 'median'}}}}, True),
    ({'theta_tuning': {'pruner': 'none'}, 'tuning': {'options': {'optuna': {'pruner': 'median'}}}}, False),
    ({'theta_tuning': {'pruner': 'median'}}, True),
    ({}, False),
])
def test_optuna_reads_pruning_from_its_options_or_the_trial_runtime(cfg, active, tmp_path):
    from package_fixtures import install_package
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_package(paths, 'optuna', tmp_path / 'archives')
    component = load_module(get_catalog(paths).package('optuna'), 'component.py', paths=paths)
    assert component.OptunaTuner().pruning_enabled(cfg) is active


def test_optional_sweep_controls_keep_hydra_prefix_without_polluting_trial_names(tmp_path):
    optuna = pytest.importorskip("optuna")
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from package_fixtures import package_module
    _optuna_sweeper = package_module('optuna', 'sweeper.py')
    ThetaOptunaSweeper = _optuna_sweeper.ThetaOptunaSweeper

    method = {
        "agent": {"name": "sb3/ppo", "learning_rate_final": 0.0001}, "model": "blendrl",
        "tune": {"agent.learning_rate_final": {"type": "choice", "values": [1e-6, 1e-4]}},
    }
    args = build_method_overrides("m", method, cfg={"paradigm": "online_rl", "tuning": {"component": "optuna", "direction": "minimize"}})
    dimension = next(arg for arg in args if "agent.learning_rate_final=" in arg)
    assert dimension.startswith("++agent.learning_rate_final=choice(")
    jobs_seen = []

    class Launcher:
        def launch(self, jobs, initial_job_idx):
            jobs_seen.extend(jobs)
            return [SimpleNamespace(return_value=0.5) for _ in jobs]

    storage = f"sqlite:///{tmp_path / 'optional.db'}"
    sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(seed=1), "minimize", storage,
                                 "optional", 1, 1, None, "none", 5, 0, 0, 0)
    sweeper.impl.config = OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}})
    sweeper.impl.launcher = Launcher()
    sweeper.sweep([dimension])
    params = optuna.load_study(study_name="optional", storage=storage).trials[0].params
    assert set(params) == {"agent.learning_rate_final"}
    assert any(arg.startswith("++agent.learning_rate_final=") for arg in jobs_seen[0])


def test_optuna_archive_has_setup_and_hydra_sweeper_export(tmp_path, monkeypatch):
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_package(paths, 'optuna', tmp_path / 'archives')
    monkeypatch.setenv('THETA_WORKSPACE', str(paths.root))
    registry = ContributionRegistry(paths)
    assert registry.contributions('configuration.tuning')
    package = get_catalog(paths).package('optuna')
    current = load_module(package, 'component.py', paths=paths)
    assert current.OptunaTuner().api_version == 1
    from hydra_plugins.theta_sweeper import _implementation
    assert _implementation('optuna') is load_module(package, 'sweeper.py', paths=paths).ThetaOptunaSweeper
    for removed in ('src.app.tuners.optuna.component', 'src.app.pipeline.tuners_optuna'):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(removed)



def test_installed_optuna_runs_a_real_study_without_upstream_sweeper(tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_package(paths, 'optuna', tmp_path / 'archives')
    package = get_catalog(paths).package('optuna')
    sweeper = load_module(package, 'sweeper.py', paths=paths)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parser = OverridesParser.create()
    distribution = sweeper._distribution_from_override(parser.parse_override('rate=range(0.1,0.5,0.1)'))
    assert distribution.step == 0.1 and distribution.low == 0.1
    import optuna
    storage = load_module(package, 'utils.py', paths=paths).get_optuna_storage('sqlite:///' + str(tmp_path / 'study.sqlite'))
    study = optuna.create_study(storage=storage, study_name='standalone', direction='minimize')
    study.optimize(lambda trial: trial.suggest_float('rate', 0.1, 0.5), n_trials=2)
    assert len(study.trials) == 2
    assert study.best_value is not None
