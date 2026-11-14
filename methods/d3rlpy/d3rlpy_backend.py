"""Editable d3rlpy method backend, execution contract v1.

Lineage: d3rlpy-suite 1.0.3; no installed package is changed or imported as a script.
"""
from pathlib import Path


def create_encoder(cfg):
    from d3rlpy.models.encoders import VectorEncoderFactory
    return VectorEncoderFactory(hidden_units=list(cfg.get("hidden_units", [256, 256])),
                                activation=str(cfg.get("activation", "relu")))


def execute(cfg, context):
    import time
    import numpy as np
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.constants import ActionSpace
    from omegaconf import OmegaConf
    from src.app.api import telemetry
    from src.app.api.data import DatasetReader
    from src.app.api.execution import RunResult, TrainingCancelled
    from src.app.api.packages import selected_capability
    from src.app.api.tuning import trial_log_version

    if cfg.get("resume") or cfg.get("recover") or context.checkpoint:
        raise ValueError("d3rlpy-suite supports inference reload of .d3 bundles; exact training resume is unsupported")
    context.check_cancelled()
    d3rlpy.seed(context.seed)
    agent = cfg.agent
    algorithm = str(agent.algorithm).rsplit('/', 1)[-1].lower()
    model = cfg.model
    selector = model.get('architecture')
    selected = selected_capability('model', selector, paths=context.workspace)
    if selected is None or selected.metadata.get('backend') != 'd3rlpy':
        raise ValueError("d3rlpy-suite requires an explicit d3rlpy encoder model, e.g. d3rlpy/vector")
    encoder = selected.load()(model)
    reader = DatasetReader(cfg.dataset_path)
    count = len(reader)
    if count < 2:
        raise ValueError('d3rlpy-suite needs at least two transitions')
    observations = reader.obs[:count].cpu().numpy().astype(np.float32)
    actions = reader.actions[:count].cpu().numpy()
    rewards = reader.rewards[:count].cpu().numpy().reshape(-1)
    terminals = reader.dones[:count].cpu().numpy().reshape(-1).astype(bool)
    discrete = np.issubdtype(actions.dtype, np.integer)
    actions = actions.reshape(-1) if discrete else actions
    dataset = MDPDataset(observations=observations, actions=actions, rewards=rewards, terminals=terminals,
                         action_space=ActionSpace.DISCRETE if discrete else ActionSpace.CONTINUOUS)
    names = {('bc', True): 'DiscreteBCConfig', ('bc', False): 'BCConfig',
             ('cql', True): 'DiscreteCQLConfig', ('cql', False): 'CQLConfig',
             ('iql', False): 'IQLConfig', ('td3_bc', False): 'TD3PlusBCConfig'}
    name = names.get((algorithm, discrete))
    if name is None:
        raise ValueError(f'd3rlpy/{algorithm} does not support this action space')
    import inspect
    config_cls = getattr(d3rlpy.algos, name)
    fields = inspect.signature(config_cls).parameters
    options = {key: value for key, value in agent.items() if key in fields}
    options.update(batch_size=int(agent.get('batch_size', 256)))
    rate = float(agent.get('learning_rate', 3e-4))
    for key in ('learning_rate', 'actor_learning_rate', 'critic_learning_rate'):
        if key in fields:
            options[key] = rate
    for key in ('encoder_factory', 'actor_encoder_factory', 'critic_encoder_factory'):
        if key in fields:
            options[key] = encoder
    learner = config_cls(**options).create(device=False)
    budget = int(cfg.get('n_steps', agent.get('n_steps', 50000)))
    interval = int(agent.get('n_steps_per_epoch', min(1000, budget)))
    if budget < 1 or interval < 1 or budget % interval:
        raise ValueError('n_steps must be positive and divisible by positive n_steps_per_epoch')
    sinks, system_interval = telemetry.settings_from_config(cfg)
    log = telemetry.init(str(cfg.group), str(cfg.experiment_id), str(agent.name), sinks=sinks,
                         config=OmegaConf.to_container(cfg, resolve=True), system_interval=system_interval,
                         version=cfg.get('variability_version') or trial_log_version(cfg))
    status = 'failed'
    start = time.time()
    metrics = {}
    def step_callback(algo, epoch, total_step):
        context.check_cancelled()
    try:
        for epoch, losses in learner.fitter(dataset, n_steps=budget, n_steps_per_epoch=interval,
                                            show_progress=False, callback=step_callback,
                                            with_timestamp=False, save_interval=budget + 1,
                                            logger_adapter=d3rlpy.logging.NoopAdapterFactory()):
            metrics = {f'train/{key}': float(value) for key, value in losses.items()}
            log.log(metrics, step=epoch * interval, transitions=epoch * interval)
        # Static-data fitting owns train losses; offline protocols do not perform simulator rollouts.
        for name in ('latest_model.d3', 'best_model.d3'):
            learner.save(str(context.artifacts / name))
            log.artifact(context.artifacts / name, kind='checkpoint')
        status = 'completed'
    except (TrainingCancelled, KeyboardInterrupt):
        status = 'cancelled'
        raise
    finally:
        log.save_runtime_metadata(config=OmegaConf.to_container(cfg, resolve=True), training_time_seconds=time.time()-start)
        log.finish(status=status)
    definitions = {key: {"meaning": "d3rlpy training epoch mean " + key,
                         "direction": "minimize"} for key in metrics}
    return RunResult(metrics=metrics, metric_definitions=definitions, artifacts=[{'path': str(context.artifacts / 'latest_model.d3'),
                                                'kind': 'checkpoint', 'format': 'd3rlpy.d3'}],
                     progress={'unit': 'updates', 'completed': budget})
