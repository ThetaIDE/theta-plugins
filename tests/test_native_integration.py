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
from src.app.core.provider_loader import load_capability, resolve_provider, provider_object
from src.app.workspace import WorkspacePaths

IDS = [json.loads(p.read_text())['id'] for p in sorted(ROOT.glob('*/*/.theta_component.json'))]


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
assert len([p for p in c.packages if p.managed]) == 37
assert not any(type(f).__name__ == '_LegacyFinder' for f in sys.meta_path)
'''])


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
    from src.app.services.rulesets import resolve_ruleset, available_rulesets
    from src.app.core.env_vectorized import VectorizedBaseEnv
    from src.app.core.model_registry import build_model
    rules=resolve_ruleset(native.paths.root,'cartpole','anticipatory_v1')
    assert rules.directory.is_relative_to(native.paths.installed_components/'nsfr/resources/rules')
    assert len([r for r in available_rulesets(native.paths.root,'cartpole') if r.name=='anticipatory_v1'])==1
    env=VectorizedBaseEnv.from_name('gym_classic_control/cartpole',n_envs=2,seed=3)
    try:
        obs=env.reset();assert obs.shape==(2,4)
        next_obs,*_=env.step(torch.zeros(2,dtype=torch.int64));assert next_obs.shape==(2,4)
        for name in ('dnn','mlp','mlp_classifier','dense_autoencoder','transformer','lstm'):
            model=build_model(name,obs_dim=4,n_actions=2,device='cpu')
            assert isinstance(model,torch.nn.Module)
    finally:env.close()
    make=provider_object('nsfr','common.py:get_nsfr_model',native.paths)
    reasoner=make('cartpole','anticipatory_v1','cpu',train=True)
    observation=provider_object('nsfr','resources/rules/cartpole/anticipatory_v1/valuation.py:build_symbolic_observation',native.paths)
    values=reasoner(observation(torch.zeros(2,4)))
    assert values.shape[0]==2 and torch.isfinite(values).all()
    reasoner=provider_object('neumann','common.py:get_neumann_model',native.paths)('cartpole','anticipatory_v1','cpu',train=True)
    values=reasoner(observation(torch.zeros(2,4)))
    assert values.shape==(2,2) and torch.isfinite(values).all()
    multiply=provider_object('neumann','scatter.py:scatter_mul',native.paths)
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
    # Package/config/record rollback must preserve a working installed native archive.
    for boundary in ('config','package','record'):
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
    overrides=['agent=sb3/ppo','env=gym_classic_control/cartpole','model='+model,
               'total_timesteps=64','agent.n_steps=32','agent.batch_size=32','agent.n_epochs=1',
               'env.num_envs=1','group=native','experiment_id='+model.replace('/','_')]
    if model=='blendrl':overrides+=['model.symbolic.rules=anticipatory_v1']
    result=execute(native,['-m','src.app.pipeline.provider_worker','package://sb3-suite/sb3_runner.py',*overrides])
    assert '[SB3 Engine Complete]' in result.stdout
    checkpoint=native.paths.results/'checkpoints/native'/model.replace('/','_')/'sb3_agent/latest_model.zip'
    assert checkpoint.is_file(),list(checkpoint.parent.glob('*'))
    code=f'''from src.app.core.provider_loader import provider_object
ThetaPPO=provider_object('sb3-suite','theta_sb3_policy.py:ThetaPPO')
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
    assert manager.manifests['model_builder'].is_core


def test_workspace_override_and_packaged_default_precedence(native):
    paths=native.paths
    active=paths.config/'model/basics/dnn.yaml'
    original=active.read_bytes()
    try:
        active.chmod(0o644);active.write_text('architecture: dnn\nhidden_sizes: [7, 7]\n')
        assert paths.resolve_config('model/basics/dnn.yaml')==active
        active.unlink()
        packaged=paths.resolve_config('model/basics/dnn.yaml')
        assert packaged==paths.installed_components/'dnn/configs/model/basics/dnn.yaml'
        assert packaged.is_file()
    finally:
        active.write_bytes(original)


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
    result=execute(native,['-m','src.app.pipeline.provider_worker','package://d3rlpy-suite/d3rlpy_runner.py',
        'agent=d3rlpy/bc','env=gym_classic_control/cartpole','paradigm=offline_rl',
        'dataset_path='+str(dataset),'agent.n_steps=2','agent.batch_size=8','group=native','experiment_id=offline'])
    assert 'Loading transitions' in result.stdout
    assert list((native.paths.results/'checkpoints/native/offline').rglob('latest_model.d3'))


def test_native_experiment_export_keeps_provider_dependency_ownership(native,tmp_path):
    from src.app.services.experiment_exporter import ExperimentExporter,ExportOptions
    result=ExperimentExporter(native.paths.root).export_group('robotics',ExportOptions(output_path=tmp_path/'robotics.zip'))
    assert result.success,result.error
    manifest=result.manifest
    assert not {'legacy_paths','module_aliases','target_path'} & manifest.keys()
    assert manifest['capabilities']==[]
    assert {'robotics-mujoco','sb3-suite','mlp'} <= {d['id'] for d in manifest['dependencies']['components']}
    with ZipFile(result.archive_path) as archive:
        assert not any(p.endswith(('.py','.pyc')) for p in archive.namelist())
        assert 'configs/experiment/robotics/_base.yaml' in archive.namelist()


def test_selected_tuner_objective_and_slurm_command(native):
    from omegaconf import OmegaConf
    from src.app.pipeline.tuners import get_tuner
    from src.app.objectives.api import create_objective,TrialResults
    from src.app.pipeline.engine import resolve_engine
    from src.app.pipeline.slurm import generate_sbatch_script
    assert get_tuner().api_version==1
    objective=create_objective('native_score',workspace=native.paths.root)
    objective.source.write_text('def evaluate(results, params):\n    return 4.5\n')
    assert objective.evaluate(TrialResults([],{}),{}).value==4.5
    cfg=OmegaConf.create({'group':'native','experiment_id':'remote','site':{},'agent':{'entrypoint':'package://sb3-suite/sb3_runner.py'}})
    assert resolve_engine({},cfg)[0]=='package://sb3-suite/sb3_runner.py'
    # Generic Slurm wrapping must preserve the native worker protocol; full per-method dispatch is in slurm_runner.
    text=generate_sbatch_script('native',['agent=sb3/ppo','model=basics/dnn'],str(native.paths.results/'slurm'),cfg)
    assert '-m src.app.pipeline.provider_worker package://sb3-suite/sb3_runner.py' in text


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

    from src.app.core.paradigm_loader import get_component
    from src.app.pipeline.validation import paradigm_uses_agents
    paths=native.paths
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(paths.framework_config),version_base=None):
            cfg=compose(config_name='config',overrides=['paradigm=unsupervised','model=basics/autoencoder',
                'hydra.searchpath=[file://'+str(paths.config)+']'])
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
    profile=LocalComponents(native.paths).deep_copy(native.paths.config/'model/blendrl.yaml','native_blendrl_copy')
    copied=ProviderCatalog(native.paths).package('native_blendrl_copy')
    assert copied.manifest['parent']['id']=='blendrl'
    assert copied.manifest['dependencies']==original.manifest['dependencies']
    assert profile==native.paths.config/'model/native_blendrl_copy.yaml'
    assert (copied.root/'plot/blendrl/routing.py').is_file()
    plot=native.paths.config/'plot/native_blendrl_copy/blend_routing.yaml'
    assert plot.is_file()
    assert resolve_provider('model','native_blendrl_copy',native.paths) is not resolve_provider('model','blendrl',native.paths)
    assert resolve_provider('model','blender',native.paths) is resolve_provider('model','blendrl',native.paths)
    assert resolve_provider('plotter','native_blendrl_copy/blend_routing',native.paths) is not None
    assert content_hash(original.root)==before
    local=LocalComponents(native.paths)
    component=next(c for c in local.entries() if c.id=='native_blendrl_copy')
    local.archive_authored(component,native.installer)
    assert not copied.root.exists() and original.root.is_dir()
    local.restore_authored(component)
    assert profile.is_file() and (copied.root/'plot/blendrl/routing.py').is_file()


def test_native_sb3_real_optuna_trial_and_checkpoint(native):
    import optuna
    from src.app.pipeline.optuna_utils import promote_best_trial_checkpoint
    storage='sqlite:///'+str(native.base/'native-study.db')
    execute(native,['-m','src.app.pipeline.provider_worker','package://sb3-suite/sb3_runner.py',
        '--multirun','agent=sb3/ppo','env=gym_classic_control/cartpole','model=basics/dnn',
        'total_timesteps=8','eval_episodes=1','+eval_freq_steps=8','agent.n_steps=8',
        'agent.batch_size=4','agent.n_epochs=1','env.num_envs=1','group=native',
        'experiment_id=tuned','+tuning.metric=eval/reward','hydra/sweeper=optuna_online',
        'hydra.sweeper.n_trials=1','hydra.sweeper.study_name=native-study',
        'hydra.sweeper.storage='+storage,'agent.learning_rate=choice(0.001,0.002)'])
    study=optuna.load_study(study_name='native-study',storage=storage)
    assert len(study.trials)==1 and study.trials[0].state==optuna.trial.TrialState.COMPLETE
    import math
    assert math.isfinite(study.best_value)
    # Promotion uses the recorded study/trial identity, independent of Hydra job index.
    previous=Path.cwd()
    try:
        os.chdir(native.paths.root)
        promote_best_trial_checkpoint('native','tuned','sb3_agent',storage,'native-study')
    finally:
        os.chdir(previous)
    root=native.paths.results/'checkpoints/native/tuned/sb3_agent'
    assert (root/'best_model.zip').is_file()
    summary=json.loads((root/'tuning_summary.json').read_text())
    assert summary['trials'][0]['value']==study.best_value
    assert summary['objective_metric']=='eval/reward'


def test_native_cartpole_sb3_objective_worker(native,tmp_path):
    """The real two-seed objective regression uses installed maintained archives."""
    import math
    from omegaconf import OmegaConf
    cfg=OmegaConf.create({'group':'g','experiment_id':'e','paradigm':'online_rl','seed':1,
        'total_timesteps':8,'eval_episodes':1,'eval_freq_steps':8,
        'agent':{'name':'m','algorithm':'PPO','n_steps':8,'batch_size':4,'n_epochs':1,
                 'entrypoint':'package://sb3-suite/sb3_runner.py'},
        'model':{'sb3_native_policy':True},'env':{'name':'cartpole','env_id':'CartPole-v1'},
        'objective':{'component':'learning_curve'},
        'theta_execution':{'script':'package://sb3-suite/sb3_runner.py'},
        'variability':{'training_seeds':[1,2],'parallel_runs':2,'evaluation_seed':10}})
    OmegaConf.save(cfg,tmp_path/'test.yaml')
    execute(native,['-m','src.app.pipeline.repeated_task','--config-path',str(tmp_path),'--config-name','test'])
    path=next(native.paths.root.glob('results/logs/g/e/m/version_execution_*/objective.json'))
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
    context={'methods':{'m':{'agent':'sb3/ppo','model':'basics/dnn',
        'entrypoint':'package://sb3-suite/sb3_runner.py','python':sys.executable}},
        'sanitized_extra_args':['agent.n_steps=8','agent.batch_size=4','agent.n_epochs=1'],
        'storage_url':None,'is_sweep':False,'is_interactive':True}
    monkeypatch.chdir(native.paths.root)
    run_execution(cfg,context,make_plan(cfg,context))
    execution=ExecutionData(next(native.paths.root.glob('results/executions/replicates/e/*/manifest.json')))
    assert len(execution.runs())==2
    for run in execution.runs():
        assert run['status']=='completed'
        assert (native.paths.root/run['checkpoints']/'latest_model.zip').is_file()
        events=execution.events(run)
        assert [event['training_step'] for event in events]==[0,8]
        assert all(len(event['episode_outcomes'])==2 and event['deterministic_actions'] for event in events)
        assert [event['evaluation_seed'] for event in events]==[100,101]
    assert all(point['n']==2 for point in execution.aggregate('m'))
