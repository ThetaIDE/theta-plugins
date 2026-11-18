"""Install this repository's packages from source into disposable workspaces.

Tests of the Optuna tuner and the Slurm scheduler use these helpers; they never read the
IDE's own test fixtures or any user workspace.
"""
import json
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
IDE = Path(os.environ.get('THETA_IDE_ROOT', ROOT.parent / 'Theta-IDE')).resolve()


def source(package_id):
    for path in sorted(ROOT.glob('*/*/.theta_component.json')):
        if json.loads(path.read_text())['id'] == package_id:
            return path.parent
    raise KeyError(f'No package {package_id!r} in {ROOT}')


def build_package(package_id, output):
    """The package's archive, built exactly as ``scripts/package_plugins.py`` builds it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('package_plugins', ROOT / 'scripts/package_plugins.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    folder = source(package_id)
    meta = json.loads((folder / '.theta_component.json').read_text())
    target = Path(output) / f"{meta['id']}-{meta['version']}.zip"
    builder.package_component(folder, target)
    return target


def install_package(paths, package_id, output):
    """Install a package and its component dependencies from source, without Python dependencies."""
    from src.app.components.installer import HubInstaller
    from src.app.components.lifecycle import dependencies
    from src.app.components.models import HubComponent, ReleaseInfo
    from src.app.core.provider_catalog import get_catalog
    from src.app.workspace import WorkspacePaths
    paths = paths if isinstance(paths, WorkspacePaths) else WorkspacePaths(paths)
    if get_catalog(paths).package(package_id) is not None: return
    archive = build_package(package_id, output)
    with ZipFile(archive) as bundle: meta = json.loads(bundle.read('.theta_component.json'))
    for dependency, _ in dependencies(meta):
        if get_catalog(paths).package(dependency) is None:
            install_package(paths, dependency, output)
    component = HubComponent.from_dict(meta)
    component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), '')}
    HubInstaller(paths.root, paths.data_dir).install(component, install_dependencies=False)


_SESSION_WORKSPACES = {}


def package_workspace(package_id):
    """A private workspace with one package installed once per test session. Do not change it."""
    from src.app.workspace import WorkspacePaths
    if package_id not in _SESSION_WORKSPACES:
        root = Path(tempfile.mkdtemp(prefix=f'theta-{package_id}-'))
        paths = WorkspacePaths(root / 'workspace')
        install_package(paths, package_id, root / 'archives')
        _SESSION_WORKSPACES[package_id] = paths
    return _SESSION_WORKSPACES[package_id]


def package_module(package_id, relative):
    """Load ``relative`` from an installed copy of the package, never from this checkout."""
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_module
    paths = package_workspace(package_id)
    return load_module(get_catalog(paths).package(package_id), relative, paths=paths)


def optuna_plan(execution_id):
    """The serialized managed-SQLite plan the Optuna tuner records for a study allocation."""
    from src.app.pipeline.tuner_execution import TunerArtifact, TunerExecutionPlan
    return TunerExecutionPlan('optuna', execution_id,
        artifacts=(TunerArtifact('study', f'optuna/{execution_id}/study.sqlite', True),),
        trials=40, storage_kind='managed', distributions=('optuna', 'sqlalchemy'),
        continuation_notes='reinitialized from configured seed; suggestion sequence can differ').record()


def cluster_site(root, *, execution=None):
    """Author a two-partition Slurm site, ``cluster``, in the workspace at ``root``.

    ``cluster/cpu`` is a CPU partition without --time or --gres, ``cluster/gpu`` a consolidated
    GPU partition, and ``cluster/default`` inherits the shared defaults.
    """
    from src.app.sites.catalog import create
    from src.app.workspace import WorkspacePaths
    root = root.root if isinstance(root, WorkspacePaths) else Path(root)
    defaults = {'scheduler': 'slurm', 'python_executable': 'auto', 'venv_dir': 'venv',
                'mail_user': None, 'mail_type': 'END,FAIL', 'consolidate': False,
                'resources': {'time': None, 'nodes': 1, 'gpu_type': None, 'gres': None},
                'module_loads': [], 'extra_pythonpath': [],
                'extra_env_vars': {'OMP_NUM_THREADS': '1'}, 'extra_sbatch_flags': [],
                'dataset_search_dirs': []}
    variants = {
        'cpu': {'scheduler': 'slurm', 'resources': {'partition': 'cpu', 'cores': 4,
                   'memory': '8G', 'gpus': 0, 'no_gres': True}},
        'gpu': {'scheduler': 'slurm', 'consolidate': True, 'gres_format': 'gpu:{gpu_type}:{gpus}',
                'resources': {'partition': 'gpu', 'gpus': 1, 'cores': 16, 'memory': '32G',
                              'no_gres': False}, 'module_loads': ['cuda']},
    }
    if execution is not None:
        variants['cpu']['execution'] = dict(execution)
    return create(root, 'cluster', defaults, variants)


def sweep_workspace(root):
    """A workspace at ``root`` with Optuna installed, for scripts that launch Hydra sweeps directly.

    Returns the workspace and the ``hydra.searchpath=[...]`` override such a script needs to see
    package and framework profiles. Run the script with ``THETA_WORKSPACE`` set to ``paths.root``.
    """
    from src.app.workspace import WorkspacePaths
    root = Path(root)
    paths = WorkspacePaths(root)
    paths.config.mkdir(parents=True, exist_ok=True)
    (paths.config / 'config.yaml').write_text('defaults:\n  - site: local/default\n  - _self_\n')
    install_package(paths, 'optuna', root.parent / (root.name + '-archives'))
    roots = [path for path in paths.config_roots() if path != paths.config]
    return paths, 'hydra.searchpath=[' + ','.join('file://' + str(path) for path in roots) + ']'


def sweeper_args(storage, study, n_trials, sampler='TPESampler', **settings):
    """Every Hydra sweeper value, as the Optuna tuner compiles them from a tuning block."""
    values = dict(storage=storage, study_name=study, n_trials=n_trials, n_jobs=1, direction='minimize',
                  pruner='none', pruner_startup_trials=5, pruner_warmup_steps=0, timeout_seconds=0,
                  max_failed_trials=0)
    values.update(settings)
    return ['hydra/sweeper=theta_optuna', f'++hydra.sweeper.sampler._target_=optuna.samplers.{sampler}',
            *(f'hydra.sweeper.{key}={value}' for key, value in values.items())]


# Bundles of small fake projects whose own root is the application, for scheduler tests that
# exercise submission and allocation workers with a fake ``run_pipeline.py``.
PYPROJECT = '[project]\nname = "fake-application"\nversion = "0.0.0"\ndependencies = []\n'


def fake_selection(root, experiment, cfg, overrides=()):
    from src.app.core.provider_catalog import get_catalog
    from src.app.sites.bundle import Selection
    from src.app.workspace import WorkspacePaths
    if not (Path(root) / 'pyproject.toml').is_file():
        (Path(root) / 'pyproject.toml').write_text(PYPROJECT)
    paths = WorkspacePaths(root, application=root)
    packages = [package for package in get_catalog(paths).packages
                if package.origin not in ('bundled', 'python_distribution')
                and package.root.resolve().is_relative_to(paths.root)]
    return Selection(paths, experiment, list(overrides), cfg, packages, [])


def fake_bundles(monkeypatch, cfg):
    """Push and shared-workspace submission bundle the fake project instead of composing it."""
    from src.app.sites import bundle
    monkeypatch.setattr(bundle, 'select', lambda root, experiment, overrides=(), *, paths=None:
                        fake_selection(root, experiment, cfg, overrides))
    monkeypatch.setattr(bundle, 'check', lambda directory, manifest: None)


def activate_fake_bundle(root, experiment, cfg):
    """What a site push leaves in a remote workspace: the activated bundle and its archive."""
    from src.app.sites import bundle, transfer
    with tempfile.TemporaryDirectory(prefix='theta-test-bundle-') as stage:
        bundle.build(fake_selection(root, experiment, cfg), stage)
        transfer.activate(root, stage)
    return bundle.current(root)
