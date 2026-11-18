"""Maintained archive integration against the sibling IDE; no Hub/network or live-workspace writes.

Run with: ../Theta-IDE/venv/bin/python -m pytest tests/test_native_integration.py -q
"""
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
IDE = Path(os.environ.get('THETA_IDE_ROOT', ROOT.parent / 'Theta-IDE')).resolve()
sys.path.insert(0, str(IDE))
from src.app.components.installer import HubInstaller
from src.app.components.lifecycle import dependencies
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_catalog import ProviderCatalog, content_hash, file_hash
from src.app.core.provider_loader import load_capability, load_module, provider_export, resolve_provider
from src.app.workspace import WorkspacePaths

IDS = [meta['id'] for meta in (json.loads(p.read_text()) for p in sorted(ROOT.glob('*/*/.theta_component.json')))
       if meta.get('storage_version') == 3]


@pytest.fixture(scope='session')
def native(tmp_path_factory):
    base = tmp_path_factory.mktemp('native-integration')
    output = base / 'archives'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/package_plugins.py'), '--ide-root', str(IDE), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = json.loads((output/'index.json').read_text())['packages']
    paths = WorkspacePaths(base/'workspace', base/'custom-state')
    installer = HubInstaller(paths.root, paths.data_dir)
    pending = {m['id']:m for m in records}
    while pending:
        progress = False
        catalog = ProviderCatalog(paths)
        for key, meta in list(pending.items()):
            if any(catalog.package(dep) is None for dep, _ in dependencies(meta)): continue
            component = HubComponent.from_dict(meta)
            component.releases = {component.version:ReleaseInfo(component.version,(output/meta['archive']).as_uri(),meta['sha256'])}
            installer.install(component, install_dependencies=False)
            pending.pop(key); progress = True
        assert progress, f'Unresolved local dependencies: {sorted(pending)}'
    catalog = ProviderCatalog(paths)
    assert not catalog.diagnostics, catalog.diagnostics
    assert {p.id for p in catalog.packages if p.managed} == set(IDS)
    assert not paths.research.exists()
    assert not (paths.data_dir/'plugins').exists()
    return SimpleNamespace(base=base, output=output, paths=paths, installer=installer, records=records)


@pytest.fixture(autouse=True)
def environment(native, monkeypatch):
    for key,value in native.paths.process_environment().items():monkeypatch.setenv(key,value)
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('MPLCONFIGDIR',str(native.base/'matplotlib'))
    monkeypatch.setenv('THETA_MANAGED_THREADS','1')
    monkeypatch.setenv('PYTHONPATH',str(IDE))


def execute(native, args, timeout=90):
    result = subprocess.run([sys.executable,*args], cwd=native.paths.root,
                            capture_output=True,text=True,timeout=timeout)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    return result


@pytest.mark.parametrize('package_id', IDS)
def test_every_archive_imports_explicit_capabilities(native, package_id):
    package = ProviderCatalog(native.paths).package(package_id)
    assert package and package.managed and package.root.parent in native.paths.managed_roots
    assert not {'legacy_paths','module_aliases','target_path','legacy_archive'} & package.manifest.keys()
    for cap in package.capabilities:
        assert load_capability(package,cap,paths=native.paths) is not None


def test_metadata_listing_is_headless_and_has_no_legacy_alias_hooks(native):
    execute(native,['-c', '''import sys, importlib.abc
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','PyQt6','optuna','stable_baselines3','d3rlpy','captum'}:
            raise AssertionError('metadata imported optional code: '+fullname)
sys.meta_path.insert(0, Absent())
from src.app.core.provider_catalog import ProviderCatalog
c=ProviderCatalog()
assert not c.diagnostics, c.diagnostics
assert len([p for p in c.packages if p.managed]) == %d
assert not any(type(f).__name__ == '_LegacyFinder' for f in sys.meta_path)
''' % len(IDS)])


def test_framework_paradigms_remain_without_algorithm_plugins(native, tmp_path):
    env={**os.environ,'THETA_WORKSPACE':str(tmp_path),'THETA_DATA_DIR':str(tmp_path/'state')}
    code='''import sys,importlib.abc
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'PyQt6','optuna','stable_baselines3','d3rlpy'}:raise ImportError('absent optional provider')
sys.meta_path.insert(0,Absent())
from src.app.core.paradigm_loader import load_paradigm_definition
from src.app.pipeline.validation import list_paradigms
for name in ('online_rl','offline_rl','supervised','unsupervised'):
    assert name in list_paradigms()
    d=load_paradigm_definition(name)
    assert d.runner_cls and d.data_module_cls and d.eval_protocol_cls
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_environment_rules_helpers_and_models_use_package_resources(native):
    import torch
    from src.app.core.env_vectorized import VectorizedBaseEnv
    from src.app.core.model_registry import build_model
    from src.app.core.workspace_resources import entries
    resolve_ruleset=provider_export('nsfr','resolve_ruleset',paths=native.paths)
    rules=resolve_ruleset('cartpole','anticipatory_v1')
    assert rules.directory.is_relative_to(native.paths.installed_components/'nsfr/resources/rules')
    assert rules.directory==resolve_ruleset(None,'cartpole/anticipatory_v1').directory
    assert 'cartpole/anticipatory_v1' in entries('rules',native.paths)
    env=VectorizedBaseEnv.from_name('gym_classic_control/cartpole',n_envs=2,seed=3)
    try:
        obs=env.reset();assert obs.shape==(2,4)
        next_obs,*_=env.step(torch.zeros(2,dtype=torch.int64));assert next_obs.shape==(2,4)
        for name in ('dnn','mlp','mlp_classifier','dense_autoencoder','transformer','lstm'):
            model=build_model(name,obs_dim=4,n_actions=2,device='cpu')
            assert isinstance(model,torch.nn.Module)
    finally:env.close()
    catalog=ProviderCatalog(native.paths)
    make=provider_export('nsfr','get_nsfr_model',paths=native.paths)
    reasoner=make('cartpole','anticipatory_v1','cpu',train=True)
    observation=load_module(catalog.package('nsfr'),'resources/rules/cartpole/anticipatory_v1/valuation.py',
                            paths=native.paths).build_symbolic_observation
    values=reasoner(observation(torch.zeros(2,4)))
    assert values.shape[0]==2 and torch.isfinite(values).all()
    reasoner=provider_export('neumann','get_neumann_model',paths=native.paths)('cartpole','anticipatory_v1','cpu',train=True)
    values=reasoner(observation(torch.zeros(2,4)))
    assert values.shape==(2,2) and torch.isfinite(values).all()
    # The CartPole profile names its symbolic builder as a declared export.
    from src.app.core.provider_loader import configured_export
    builder=configured_export('export://gym_classic_control/cartpole_symbolic_observation',paths=native.paths)
    assert builder(torch.zeros(3,4)).shape==(3,2,4)
    multiply=load_module(catalog.package('neumann'),'scatter.py',paths=native.paths).scatter_mul
    x=torch.tensor([2.,3.,5.,7.],requires_grad=True)
    result=multiply(x,torch.tensor([0,1,0,1]),dim=0,dim_size=2)
    assert torch.equal(result,torch.tensor([10.,21.]))
    result.sum().backward()
    assert torch.equal(x.grad,torch.tensor([5.,7.,2.,3.]))



def test_editable_copy_edit_export_install_update_failure_uninstall(native, tmp_path):
    from src.app.components.development import ComponentDevelopment
    from src.app.services.component_exporter import ComponentExporter,ComponentExportOptions
    package=ProviderCatalog(native.paths).package('dnn');before=content_hash(package.root)
    profile=ComponentDevelopment(native.paths).copy('dnn','native_editable_dnn')
    fork=ProviderCatalog(native.paths).package('native_editable_dnn')
    source=fork.root/'dnn_model.py';source.write_text(source.read_text()+'\nNATIVE_EDIT_MARKER=17\n')
    archive=tmp_path/'edited.zip'
    exported=ComponentExporter(native.paths.root).export(profile,ComponentExportOptions(output_path=archive,version='0.1.0'))
    assert exported.success,exported.error
    with ZipFile(archive) as z:
        meta=json.loads(z.read('.theta_component.json'))
        assert not {'legacy_paths','module_aliases','target_path'} & meta.keys()
        assert 'dnn_model.py' in z.namelist()
    other=HubInstaller(tmp_path/'other',tmp_path/'other-state')
    c=HubComponent.from_dict(meta);c.releases={c.version:ReleaseInfo(c.version,archive.as_uri(),file_hash(archive))}
    other.install(c,install_dependencies=False)
    cls=resolve_provider('model','native_editable_dnn',other.paths)
    assert sys.modules[cls.__module__].NATIVE_EDIT_MARKER==17
    installed=ProviderCatalog(other.paths).package(c.id);original=content_hash(installed.root)
    # Package/record rollback must preserve a working installed native archive (installs write no config).
    for boundary in ('package','record'):
        def fail(current,identity):
            if current==boundary:raise RuntimeError('injected native lifecycle failure')
        other._failure_point=fail
        with pytest.raises(RuntimeError,match='injected'):other.install(c,install_dependencies=False)
        assert content_hash(installed.root)==original
    other._failure_point=lambda *args:None
    other.install(c,install_dependencies=False)
    assert other.uninstall(c)
    assert content_hash(package.root)==before and fork.root.is_dir()


@pytest.mark.parametrize('model', ['basics/dnn','blendrl'])
def test_real_sb3_worker_training_and_fresh_checkpoint(native, model):
    experiment=model.replace('/','_')
    overrides=['paradigm=online_rl','agent=sb3/ppo','++agent.name=sb3_agent','env=gym_classic_control/cartpole',
               'model='+model,'total_timesteps=64','agent.n_steps=32','agent.batch_size=32','agent.n_epochs=1',
               'env.num_envs=1','group=native','experiment_id='+experiment]
    if model=='blendrl':overrides+=['model.symbolic.rules=anticipatory_v1','++model.symbolic_state=blendrl/cartpole_state']
    result=execute(native,['-m','src.app.pipeline.provider_worker','src/app/train.py',*overrides])
    assert '[SB3 Engine Complete]' in result.stdout
    run=native.paths.results/'checkpoints/native'/experiment/'sb3_agent'
    record=json.loads(next(run.glob('*/execution.json')).read_text())
    assert record['status']=='completed' and record['metadata']['algorithm']['id']=='sb3/ppo'
    assert record['metadata']['backend']=='sb3'
    checkpoint=next(run.glob('*/latest_model.zip'))
    code=f'''from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_module
ThetaPPO=load_module(get_catalog().package('sb3-suite'),'theta_sb3_policy.py').ThetaPPO
model=ThetaPPO.load({str(checkpoint)!r},device='cpu')
assert model.policy.theta_model_module is not None
'''
    execute(native,['-c',code])


def test_native_scaffold_export_and_install_all_authoring_kinds(native,tmp_path):
    from src.app.services.plugin_scaffolder import PluginScaffolder,ScaffoldOptions
    from src.app.services.component_exporter import ComponentExporter,ComponentExportOptions
    from src.app.components.archive import validate_native_package
    from src.app.core.provider_catalog import kind_name
    authored=tmp_path/'authored';other=HubInstaller(tmp_path/'installed',tmp_path/'state')
    for kind in ('model','agent','env','datamodule','evaluator','plotter'):
        name='native_'+kind
        result=PluginScaffolder(authored).scaffold(ScaffoldOptions(name=name,kind=kind))
        assert result.success,result.error
        meta=json.loads((result.target_dir/'.theta_component.json').read_text())
        validate_native_package(meta,result.target_dir)
        archive=tmp_path/(name+'.zip')
        exported=ComponentExporter(authored).export(name,ComponentExportOptions(output_path=archive))
        assert exported.success,exported.error
        c=HubComponent.from_dict(exported.manifest);c.releases={c.version:ReleaseInfo(c.version,archive.as_uri(),file_hash(archive))}
        other.install(c,install_dependencies=False)
        assert resolve_provider(kind_name(kind),name,other.paths) is not None


def test_desktop_app_plugins_preserve_identity_settings_and_lease(native):
    from PyQt6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication(['--platform','offscreen'])
    from frontend.plugins.manager import PluginManager
    from frontend.plugins.commands import CommandRegistry
    from frontend.plugins.slots import ExtensionSlotRegistry
    from frontend.settings import SettingsManager
    logs=[]
    window=SimpleNamespace(log=logs.append,store=SimpleNamespace(root=native.paths.data_dir),command_registry=CommandRegistry(),extension_slots=ExtensionSlotRegistry())
    manager=PluginManager(window,paths=native.paths,settings_manager=SettingsManager(native.paths.data_dir))
    manager.discover()
    for plugin_id,package_id in [('obsidian','obsidian-notes'),('doom','doom-sim'),('antigravity','antigravity')]:
        manifest=manager.manifests[plugin_id]
        assert manifest.extra['package_id']==package_id
        # Load the real class and settings without launching a browser/terminal or external process.
        package,cap=ProviderCatalog(native.paths).resolve('plugin',plugin_id)
        cls=load_capability(package,cap,paths=native.paths)
        instance=cls(manifest)
        assert instance.manifest.id==plugin_id


def test_workspace_override_and_packaged_default_precedence(native):
    paths=native.paths
    packaged=paths.installed_components/'dnn/configs/model/basics/dnn.yaml'
    assert paths.resolve_config('model/basics/dnn.yaml')==packaged and packaged.is_file()
    override=paths.config/'model/basics/dnn.yaml'
    override.parent.mkdir(parents=True,exist_ok=True)
    try:
        override.write_text('architecture: dnn\nhidden_sizes: [7, 7]\n')
        assert paths.resolve_config('model/basics/dnn.yaml')==override
    finally:
        override.unlink()
    assert paths.resolve_config('model/basics/dnn.yaml')==packaged


@pytest.mark.parametrize('paradigm,model', [('supervised','basics/mlp_classifier'),('unsupervised','basics/dense_autoencoder')])
def test_real_static_dataset_paradigm_workers(native,paradigm,model):
    import numpy as np
    dataset=native.paths.datasets/(paradigm+'.npz');dataset.parent.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(5)
    features=rng.normal(size=(32,4)).astype('float32')
    np.savez(dataset,features=features,labels=(features[:,0]>0).astype('int64'))
    result=execute(native,['-m','src.app.pipeline.provider_worker','src/app/train.py',
        'paradigm='+paradigm,'model='+model,'dataset_path='+str(dataset),'group=native',
        'experiment_id='+paradigm,'++epochs=1','++batch_size=8','++trainer.accelerator=cpu',
        '++trainer.devices=1','++trainer.enable_progress_bar=false'])
    assert '[Training Complete]' in result.stdout
    assert list((native.paths.results/'checkpoints/native'/paradigm).rglob('*.ckpt'))


def test_real_offline_runner_uses_explicit_transition_dataset(native):
    import numpy as np
    from src.app.dataset_utils import DatasetWriter
    dataset=native.paths.datasets/'transitions'
    writer=DatasetWriter(dataset,env_name='cartpole')
    rng=np.random.default_rng(4)
    for step in range(64):
        obs=rng.normal(size=4).astype('float32')
        writer.add(obs,step%2,1.0,obs+0.01,step%8==7)
    writer.flush()
    execute(native,['-m','src.app.pipeline.provider_worker','src/app/train.py',
        'paradigm=offline_rl','agent=d3rlpy/bc','++agent.name=bc','model=d3rlpy/vector','env=gym_classic_control/cartpole',
        'dataset_path='+str(dataset),'agent.n_steps=2','agent.batch_size=8','group=native','experiment_id=offline'])
    run=native.paths.results/'checkpoints/native/offline/bc'
    record=json.loads(next(run.glob('*/execution.json')).read_text())
    assert record['status']=='completed' and record['metadata']['backend']=='d3rlpy'
    assert list(run.glob('*/latest_model.d3'))


def test_selected_objective_and_backend(native):
    from src.app.core.execution import execution_backend
    from src.app.objectives.api import create_objective,TrialResults
    from src.app.pipeline.engine import DEFAULT_ENGINE_SCRIPT,resolve_engine
    objective=create_objective('native_score',workspace=native.paths.root)
    objective.source.write_text('def evaluate(results, params):\n    return 4.5\n')
    assert objective.evaluate(TrialResults([],{}),{}).value==4.5
    # Packages select backends through capability metadata; every method runs the Theta bootstrap.
    for agent,backend in (('sb3/ppo','sb3'),('d3rlpy/cql','d3rlpy')):
        assert resolve_engine({'agent':agent})[0]==DEFAULT_ENGINE_SCRIPT
        assert execution_backend({'agent':agent},paths=native.paths)[0]==backend


def test_obsidian_activation_settings_and_loaded_provider_lease(native):
    from PyQt6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication(['--platform','offscreen'])
    package,cap=ProviderCatalog(native.paths).resolve('plugin','obsidian')
    from frontend.plugins.base import PluginManifest
    cls=load_capability(package,cap,paths=native.paths)
    plugin=cls(PluginManifest(id='obsidian',name='Obsidian',version=package.version,plugin_dir=package.root))
    settings={};listeners=[];tabs=[]
    context=SimpleNamespace(get_setting=lambda key,default=None:settings.get(key,default),
        set_setting=lambda key,value:settings.__setitem__(key,value),get_workspace_dir=lambda:native.paths.root,
        add_experiment_change_listener=listeners.append,remove_experiment_change_listener=listeners.remove,
        add_sidebar_tab=lambda **data:tabs.append(data),remove_sidebar_tab=lambda tab:tabs.clear(),log=lambda text:None)
    plugin.activate(context)
    assert tabs and listeners and plugin.panel.vault_dir==native.paths.root/'notes'
    plugin.deactivate()
    assert not tabs and not listeners


def test_train_py_resolves_the_autoencoder_from_a_composed_config(native):
    """The link a unit test cannot see: model YAML -> component registry -> module.

    Replays train.py's own resolution steps against the real Hydra composition,
    so a typo in `lightning_module` or a missing registration fails here rather
    than silently falling back to the supervised classifier at run time.
    """
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from src.app.config_resolution import configuration_context
    from src.app.core.paradigm_loader import get_component
    from src.app.pipeline.validation import paradigm_uses_agents
    paths=native.paths
    GlobalHydra.instance().clear()
    try:
        # The same search path and resolution context train.py's launch() establishes.
        with configuration_context(paths),initialize_config_dir(config_dir=str(paths.primary_config),version_base=None):
            cfg=compose(config_name='config',overrides=['paradigm=unsupervised','model=basics/autoencoder',
                'hydra.searchpath=['+','.join(paths.config_searchpath())+']'])
    finally:
        GlobalHydra.instance().clear()

    assert paradigm_uses_agents(cfg.get("paradigm")) is False

    model_cfg = cfg.model
    cls = get_component(model_cfg.get("lightning_module") or model_cfg.get("module"))
    assert cls.__name__ == "AutoencoderLightningModule"

    reserved = {
        "architecture",
        "name",
        "lightning_module",
        "module",
        "lr",
        "type",
        "epochs",
        "eval_every_n_epochs",
    }
    kwargs = {k: v for k, v in model_cfg.items() if k not in reserved and v is not None}
    module = cls(
        architecture_name=str(model_cfg.get("architecture") or model_cfg.get("type")).lower(),
        input_dim=64,
        lr=float(model_cfg.get("lr", cfg.get("lr", 1e-3))),
        **kwargs,
    )
    assert type(module.model).__name__ == "SequenceAutoencoder"
    assert module.model.latent_dim == model_cfg.get("latent_dim")




def test_native_blendrl_editable_copy_owns_its_auxiliary_plot(native):
    from src.app.services.local_components import LocalComponents
    from src.app.core.provider_loader import resolve_provider
    original=ProviderCatalog(native.paths).package('blendrl')
    before=content_hash(original.root)
    packaged=native.paths.config_resolver().resolve('model/blendrl.yaml').file
    profile=LocalComponents(native.paths).deep_copy(packaged,'native_blendrl_copy')
    copied=ProviderCatalog(native.paths).package('native_blendrl_copy')
    assert copied.manifest['parent']['id']=='blendrl'
    assert copied.manifest['dependencies']==original.manifest['dependencies']
    assert profile.is_file() and profile.is_relative_to(copied.root)
    assert (copied.root/'plot/blendrl/routing.py').is_file()
    plot=native.paths.config_resolver().resolve('plot/native_blendrl_copy/blend_routing.yaml')
    assert plot is not None and plot.package.id=='native_blendrl_copy'
    assert resolve_provider('model','native_blendrl_copy',native.paths) is not resolve_provider('model','blendrl',native.paths)
    assert resolve_provider('plotter','native_blendrl_copy/blend_routing',native.paths) is not None
    assert content_hash(original.root)==before
    local=LocalComponents(native.paths)
    component=next(c for c in local.entries() if c.id=='native_blendrl_copy')
    local.archive_authored(component,native.installer)
    assert not copied.root.exists() and original.root.is_dir()
    local.restore_authored(component)
    assert profile.is_file() and (copied.root/'plot/blendrl/routing.py').is_file()


def test_native_sb3_real_optuna_trial_and_checkpoint(native, monkeypatch):
    """A tuned recipe runs a real Optuna study through the pipeline and promotes the best checkpoint."""
    import optuna
    # Tuning is a component; install Optuna from this checkout into the workspace.
    from package_fixtures import install_package
    install_package(native.paths,'optuna',native.base/'optuna-archives')
    recipe=native.paths.config/'experiment/native/tuned.yaml'
    recipe.parent.mkdir(parents=True,exist_ok=True)
    recipe.write_text("""# @package _global_
defaults:
  - override /env: gym_classic_control/cartpole
  - _self_
paradigm: online_rl
group: native
experiment_id: tuned
total_timesteps: 8
eval_episodes: 1
eval_freq_steps: 8
no_plot: true
monitors: {}
tuning:
  component: optuna
  metric: eval/reward
  direction: maximize
  options:
    optuna: {n_trials: 1, n_jobs: 1, sampler: random, seed: 1, study_name: native-study, study_mode: new, pruner: none}
methods:
  sb3_agent:
    agent: {name: sb3/ppo, n_steps: 8, batch_size: 4, n_epochs: 1}
    model: {name: basics/dnn}
    tune:
      agent.learning_rate: {type: choice, values: [0.001, 0.002]}
""")
    for key,value in native.paths.process_environment().items(): monkeypatch.setenv(key,value)
    execute(native,[str(IDE/'run_pipeline.py'),'native/tuned'],timeout=600)
    # Promotion puts the best trial's checkpoint beside the method's checkpoints.
    root=native.paths.results/'checkpoints/native/tuned/sb3_agent'
    assert (root/'best_model.zip').is_file()
    summary=json.loads((root/'tuning_summary.json').read_text())
    assert summary['objective_metric']=='eval/reward' and len(summary['trials'])==1
    import math
    assert math.isfinite(summary['trials'][0]['value'])


def test_native_cartpole_sb3_objective_worker(native,tmp_path):
    """The real two-seed objective regression uses installed maintained archives."""
    import math
    from omegaconf import OmegaConf
    cfg=OmegaConf.create({'group':'g','experiment_id':'e','paradigm':'online_rl','seed':1,
        'total_timesteps':8,'eval_episodes':1,'eval_freq_steps':8,
        'agent':{'name':'m','algorithm':'sb3/ppo','backend':'sb3','n_steps':8,'batch_size':4,'n_epochs':1},
        'model':{'sb3_native_policy':True},'env':{'name':'cartpole','env_id':'CartPole-v1'},
        'objective':{'component':'learning_curve'},
        'variability':{'training_seeds':[1,2],'parallel_runs':2,'evaluation_seed':10}})
    OmegaConf.save(cfg,tmp_path/'test.yaml')
    execute(native,['-m','src.app.pipeline.repeated_task','--config-path',str(tmp_path),'--config-name','test'])
    path=next(native.paths.results.glob('logs/g/e/m/version_execution_*/objective.json'))
    manifest=json.loads(path.read_text())
    assert len(manifest['runs'])==2
    assert all(run['status']=='completed' for run in manifest['runs'])
    assert all([point['training_step'] for point in run['events'] if point.get('metric')=='eval/reward']==[0,8]
               for run in manifest['runs'])
    assert all((native.paths.root/run['artifacts']['checkpoints']/'latest_model.zip').is_file()
               for run in manifest['runs'])
    assert math.isfinite(manifest['result']['value'])


def test_native_sb3_replicates_record_episodes(native,monkeypatch):
    from omegaconf import OmegaConf
    from src.app.pipeline.variability import run_execution,make_plan
    from src.app.pipeline.execution_data import ExecutionData
    experiment=native.paths.config/'experiment/native_replicates.yaml'
    experiment.parent.mkdir(parents=True,exist_ok=True)
    experiment.write_text('# @package _global_\ngroup: replicates\nexperiment_id: e\nparadigm: online_rl\n'
        'total_timesteps: 8\neval_episodes: 2\neval_freq_steps: 8\n'
        'env: {name: cartpole, env_id: CartPole-v1}\n')
    cfg=OmegaConf.create({'group':'replicates','experiment_id':'e','experiment_name':'native_replicates',
        'paradigm':'online_rl','eval_episodes':2,
        'variability':{'training_seeds':[1,2],'evaluation_seed':100,'parallel_runs':1}})
    context={'methods':{'m':{'agent':'sb3/ppo','model':'basics/dnn','python':sys.executable}},
        'sanitized_extra_args':['agent.n_steps=8','agent.batch_size=4','agent.n_epochs=1'],
        'storage_url':None,'is_sweep':False,'is_interactive':True}
    monkeypatch.chdir(native.paths.root)
    run_execution(cfg,context,make_plan(cfg,context))
    execution=ExecutionData(next(native.paths.results.glob('executions/replicates/e/*/manifest.json')))
    assert len(execution.runs())==2
    for run in execution.runs():
        assert run['status']=='completed'
        assert (native.paths.root/run['checkpoints']/'latest_model.zip').is_file()
        # The runner records matched greedy and sampled evaluations; greedy keeps eval/reward.
        events=[event for event in execution.events(run) if event.get('metric')=='eval/reward']
        assert [event['training_step'] for event in events]==[0,8]
        assert all(len(event['episode_outcomes'])==2 and event['deterministic_actions'] for event in events)
        assert [event['evaluation_seed'] for event in events]==[100,101]
    assert all(point['n']==2 for point in execution.aggregate('m'))


# BlendRL and the basic models inside the sb3-suite policy adapters (moved from the IDE suite,
# which no longer depends on installed research packages).
BLEND_CFG = {'architecture': 'blendrl', 'actor_mode': 'hybrid', 'blend_function': 'softmax', 'logic_confidence_floor': .05,
             'blender': 'neural', 'blender_model': {'name': 'basics/dnn', 'hidden_sizes': [16, 16]},
             'neural': {'type': 'dnn', 'hidden_sizes': [32, 32]},
             'symbolic': {'type': 'nsfr', 'rules': 'variant2'},
             'symbolic_state': 'blendrl/cartpole_state'}
CARTPOLE = {'env_name': 'cartpole', 'action_meanings': ['left', 'right']}


def package_module(native, package_id, relative):
    return load_module(ProviderCatalog(native.paths).package(package_id), relative, paths=native.paths)


def sb3_policies(native):
    pytest.importorskip('stable_baselines3')
    return package_module(native, 'sb3-suite', 'theta_sb3_policy.py')


def bare_blender(native):
    BlenderActorCritic = package_module(native, 'blendrl', 'agents/blender_agent.py').BlenderActorCritic
    import torch
    model = BlenderActorCritic.__new__(BlenderActorCritic)
    torch.nn.Module.__init__(model)
    return model


def test_blendrl_builds_symbolic_states_with_its_selected_builder(native):
    import torch
    model = bare_blender(native)
    model.actor_mode, model.blender_mode = 'hybrid', 'neural'
    obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    model.cfg = {'model': {'symbolic_state': 'blendrl/cartpole_state'}}
    torch.testing.assert_close(model.build_logic_observation(obs), obs.unsqueeze(1).expand(-1, 2, -1))

    class Actor(torch.nn.Module):
        def get_q_values(self, neural_state, logic_state):
            return logic_state[:, 0]

    model.actor = Actor()
    torch.testing.assert_close(model.get_q_values(obs), obs)
    model = bare_blender(native)
    model.actor_mode, model.blender_mode = 'hybrid', 'neural'
    model.cfg = {'model': {}}
    with pytest.raises(ValueError, match='symbolic state builder'):
        model.build_logic_observation(obs)


def test_blendrl_rejects_an_engine_specific_configuration(native):
    model = bare_blender(native)
    model.cfg = {'model': {}}
    model.blend_function = 'gumbel_softmax'
    with pytest.raises(ValueError, match='requires blend_function=softmax'):
        model.validate_engine_compatibility('sb3', 'on_policy')
    model.validate_engine_compatibility('another_engine', 'on_policy')


def test_blendrl_null_optimizer_rates_use_the_agent_learning_rate(native):
    import torch
    model = bare_blender(native)
    model.cfg = {'model': {'logic_lr': None, 'blender_lr': .003}, 'agent': {'logic_lr': .002}}  # agent keys are not read
    model.module_types = ['logic', 'neural']
    model.actor = torch.nn.Module()
    model.actor.policy_modules = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    model.actor.blender = torch.nn.Linear(2, 2)
    assert [g['lr'] for g in model.configure_optimizer_groups(.001)] == pytest.approx([.001, .001, .003])


@pytest.mark.parametrize('actor_mode', ['hybrid', 'logic', 'neural'])
def test_sb3_ppo_trains_and_reloads_blendrl(native, tmp_path, actor_mode):
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    policy = sb3_policies(native).ThetaActorCriticPolicy
    DNNModel = package_module(native, 'dnn', 'dnn_model.py').DNNModel
    cfg = {'model': {**BLEND_CFG, 'actor_mode': actor_mode}, 'agent': {}}
    env = gym.make('CartPole-v1')
    model = PPO(policy, env, n_steps=8, batch_size=4, n_epochs=1, seed=4,
                policy_kwargs={'theta_config': cfg, 'theta_model': 'blendrl', **CARTPOLE})
    selected = model.policy.theta_model_module
    assert type(selected).__name__ == 'BlenderActorCritic'
    assert selected.module_types == ['logic', 'neural']
    neural = selected.policy_modules[1]
    assert isinstance(neural, DNNModel) and isinstance(selected.blender, DNNModel)
    assert neural.network[0].out_features == 32 and selected.blender.network[0].out_features == 16
    before = {name: value.detach().clone() for name, value in selected.named_parameters()}
    model.learn(total_timesteps=16)
    after = dict(selected.named_parameters())
    if actor_mode == 'hybrid':
        assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith('policy_modules.1.'))
        assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith('blender.'))
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    probs_before = selected.get_action_probs(obs).detach()
    model.save(tmp_path / 'blendrl.zip')
    restored = PPO.load(tmp_path / 'blendrl.zip', env=env)
    assert type(restored.policy.theta_model_module) is type(selected)
    torch.testing.assert_close(restored.policy.theta_model_module.get_action_probs(obs), probs_before)
    env.close()


def test_sb3_dqn_uses_selected_model_q_values_and_reloads(native, tmp_path):
    import gymnasium as gym
    import torch
    from stable_baselines3 import DQN
    policy = sb3_policies(native).ThetaDQNPolicy
    MLPQNetwork = package_module(native, 'mlp', 'mlp_model.py').MLPQNetwork
    cfg = {'model': {'architecture': 'basics/mlp', 'hidden_dims': [11, 7], 'activation': 'relu'}}
    env = gym.make('CartPole-v1')
    model = DQN(policy, env, buffer_size=40, learning_starts=0, batch_size=4, train_freq=1, seed=3,
                policy_kwargs={'theta_config': cfg, 'theta_model': 'basics/mlp', **CARTPOLE})
    selected = model.policy.q_net.theta_model_module
    assert isinstance(selected, MLPQNetwork)
    assert [selected.network[0].out_features, selected.network[2].out_features] == [11, 7]
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    torch.testing.assert_close(model.policy.q_net(obs), selected.get_q_values(obs))
    before = {name: value.detach().clone() for name, value in selected.named_parameters()}
    model.learn(total_timesteps=12)
    assert any(not torch.equal(before[name], value) for name, value in selected.named_parameters())
    values = selected.get_q_values(obs).detach()
    model.save(tmp_path / 'dqn.zip')
    restored = DQN.load(tmp_path / 'dqn.zip', env=env)
    assert isinstance(restored.policy.q_net.theta_model_module, MLPQNetwork)
    torch.testing.assert_close(restored.policy.q_net(obs), values)
    env.close()


def test_sb3_dqn_uses_blendrl_to_build_symbolic_observations(native):
    import gymnasium as gym
    import torch
    from stable_baselines3 import DQN
    adapter = sb3_policies(native).ThetaDQNPolicy
    env = gym.make('CartPole-v1')
    model = DQN(adapter, env, buffer_size=40, learning_starts=0, batch_size=4, train_freq=1,
                policy_kwargs={'theta_config': {'model': dict(BLEND_CFG)}, 'theta_model': 'blendrl', **CARTPOLE})
    selected = model.policy.q_net.theta_model_module
    assert type(selected).__name__ == 'BlenderActorCritic'
    obs = torch.tensor([[0., 0., -0.1, 0.]], dtype=torch.float32)
    torch.testing.assert_close(model.policy.q_net(obs), selected.get_q_values(obs))
    model.learn(total_timesteps=8)
    env.close()


def test_blendrl_routing_monitor_reports_experts_and_keeps_gradients_usable(native):
    import gymnasium as gym
    import torch
    module = sb3_policies(native)
    cfg = {'model': {**BLEND_CFG, 'logic_lr': .002, 'blender_lr': .003}}
    env = gym.make('CartPole-v1')
    model = module.ThetaPPO(module.ThetaActorCriticPolicy, env, learning_rate=.001, n_steps=8, batch_size=4, n_epochs=1,
                            policy_kwargs={'theta_config': cfg, 'theta_model': 'blendrl', **CARTPOLE})
    selected = model.policy.theta_model_module
    metrics, samples = selected.collect_monitoring(torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.]]))
    assert samples['weights'].shape == (2, 2)
    assert 'blend/logic/weight_mean' in metrics and 'blend/neural/weight_mean' in metrics
    assert 0 <= metrics['blend/logic/coverage'] <= 1
    assert 'blend/agreement/logic__neural/top_action' in metrics
    assert not selected.actor._capture_experts
    model.learn(total_timesteps=8)
    env.close()


def stability_policy(native):
    """A PPO BlendRL policy with partial NSFR rules and an independent DNN value trunk."""
    import gymnasium as gym
    module = sb3_policies(native)
    cfg = {'model': {**BLEND_CFG, 'logic_confidence_floor': .05,
                     'neural': {'type': 'dnn', 'hidden_sizes': [32, 32], 'activation': 'tanh',
                                'dueling': False, 'shared_features': False}}, 'agent': {}}
    env = gym.make('CartPole-v1')
    algo = module.ThetaPPO(module.ThetaActorCriticPolicy, env, seed=4, n_steps=16, batch_size=8, n_epochs=2,
                           policy_kwargs={'theta_config': cfg, 'theta_model': 'blendrl', **CARTPOLE})
    return algo, env


def test_blendrl_partial_rules_abstain_when_unmatched_and_report_coverage(native):
    import torch
    algo, env = stability_policy(native)
    try:
        model = algo.policy.theta_model_module
        obs = torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.], [0., 0., 0., 0.]])
        logic = model.policy_modules[0]
        mapped = model.actor._map_logic_output(logic(model.build_logic_observation(obs)), logic)
        assert mapped[0, 0] > .99
        torch.testing.assert_close(mapped[1:], torch.full((2, 2), .5))
        metrics, _ = model.collect_monitoring(obs)
        assert metrics['blend/logic/coverage'] == pytest.approx(1 / 3)
        # Neural expert options must not be silently discarded by the composite.
        assert isinstance(model.policy_modules[1].network[1], torch.nn.Tanh)
        algo.learn(total_timesteps=32)
    finally:
        env.close()


def test_blendrl_weak_rule_confidence_stays_differentiable_above_the_noise_floor(native):
    import torch
    algo, env = stability_policy(native)
    try:
        model = algo.policy.theta_model_module
        score = torch.tensor([[.3], [.9]], requires_grad=True)
        probs = model.actor._map_logic_output(score, model.policy_modules[0])
        assert .5 < probs[0, 0] < probs[1, 0] < 1
        probs[:, 0].sum().backward()
        assert torch.all(score.grad > 0)
        # Other expert plugins already return action probabilities.
        distribution = torch.tensor([[.7, .3]])
        torch.testing.assert_close(model.actor._map_logic_output(distribution, torch.nn.Identity()), distribution)
        model.actor.logic_confidence_floor = None
        torch.testing.assert_close(model.actor._map_logic_output(torch.tensor([[.02]]), model.policy_modules[0]),
                                   torch.tensor([[1., 0.]]))
    finally:
        env.close()


def test_blendrl_missing_symbolic_critic_does_not_reward_gate_collapse(native):
    import torch
    algo, env = stability_policy(native)
    try:
        model = algo.policy.theta_model_module
        assert model.logic_critic is None
        obs = torch.tensor([[0., 0., -.1, 0.], [0., 0., .1, 0.]])
        logits = torch.tensor([[5., -5.], [-5., 5.]], requires_grad=True)
        value = model.get_value(obs, blending_weights=logits.softmax(-1))
        torch.testing.assert_close(value, model.get_neural_value(obs))
        value.sum().backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-7, rtol=0)
    finally:
        env.close()


def test_blendrl_independent_value_features_keep_value_loss_out_of_policy_trunk(native, tmp_path):
    import torch
    algo, env = stability_policy(native)
    try:
        model = algo.policy.theta_model_module
        neural = model.policy_modules[1]
        obs = torch.tensor([[.01, .02, -.1, .01], [.02, .01, .1, -.01]])
        neural.get_value(obs).square().sum().backward()
        assert all(p.grad is None for p in neural.network.parameters())
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in neural.value_network.parameters())
        algo.learn(total_timesteps=32)
        checkpoint = tmp_path / 'independent.zip'
        algo.save(checkpoint)
        restored = type(algo).load(checkpoint)
        torch.testing.assert_close(restored.policy.theta_model_module.get_action_probs(obs), model.get_action_probs(obs))
        torch.testing.assert_close(restored.policy.theta_model_module.get_value(obs), model.get_value(obs))
    finally:
        env.close()


def test_blendrl_symbolic_gate_names_follow_expert_order(native):
    import torch
    BlenderActor = package_module(native, 'blendrl', 'agents/blender_agent.py').BlenderActor

    class Gate(torch.nn.Module):
        def get_prednames(self):
            return ['neural_agent', 'logic_agent']

    for kinds in [['logic', 'neural'], ['neural', 'logic']]:
        actor = BlenderActor(None, [torch.nn.Identity(), torch.nn.Identity()], kinds,
                             Gate(), 'hybrid', 'logic', 'softmax')
        assert actor.blender_id_to_pred_indices == {i: [0 if k == 'neural' else 1] for i, k in enumerate(kinds)}
