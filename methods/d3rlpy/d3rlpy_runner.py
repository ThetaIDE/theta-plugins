"""Universal d3rlpy Offline RL Runner for Theta-IDE.

Executes d3rlpy offline reinforcement learning algorithms (CQL, IQL, TD3+BC, BC)
as a decoupled black-box engine, logging training telemetry through src.app.telemetry
and saving checkpoints adhering to the Theta-IDE execution contract.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


import hydra
from omegaconf import DictConfig, OmegaConf

from src.app.workspace import WorkspacePaths
from src.app import telemetry


@hydra.main(version_base=None, config_path=str(WorkspacePaths.for_process().primary_config), config_name="config")
def main(cfg: DictConfig):
    if os.environ.get("THETA_MANAGED_THREADS"):
        import torch
        torch.set_num_threads(int(os.environ["THETA_MANAGED_THREADS"]))
    try:
        import d3rlpy
    except ImportError:
        print(
            "\n[Error] d3rlpy is not installed in the active environment.\n"
            "Please run: pip install d3rlpy\n"
            "or install the d3rlpy Offline RL Suite via the Theta Hub.\n"
        )
        sys.exit(1)

    try:
        import gymnasium as gym
    except ImportError:
        print("\n[Error] Gymnasium is required for evaluation. Run: pip install gymnasium\n")
        sys.exit(1)

    import numpy as np

    group = str(cfg.get("group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp"))
    agent_cfg = cfg.get("agent", {})
    agent_name = str(agent_cfg.get("name", "d3rlpy_agent"))
    algo_name = str(agent_cfg.get("algorithm", "CQL")).upper()
    n_steps = int(cfg.get("n_steps", agent_cfg.get("n_steps", 50000)))
    batch_size = int(agent_cfg.get("batch_size", 256))
    lr = float(agent_cfg.get("learning_rate", agent_cfg.get("lr", 3e-4)))

    # Resolve environment and discrete vs continuous
    env_cfg = cfg.get("env", {})
    env_name = env_cfg.get("name")
    env_id = env_cfg.get("env_id") or env_name or "CartPole-v1"
    if env_id == "cartpole":
        env_id = "CartPole-v1"
    elif env_id == "mountaincar":
        env_id = "MountainCar-v0"

    print(f"\n=== [d3rlpy Offline Engine] Initializing {algo_name} on {env_id} ===")
    print(f"Group: {group} | Experiment: {exp_id} | Total Steps: {n_steps}")

    # Telemetry initialization
    sinks, system_interval = telemetry.settings_from_config(cfg)
    logger = telemetry.init(
        group,
        exp_id,
        agent_name,
        sinks=sinks,
        config=OmegaConf.to_container(cfg, resolve=True),
        system_interval=system_interval,
    )

    ckpt_dir = Path("results/checkpoints") / group / exp_id / agent_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset
    dataset_path_val = cfg.get("dataset_path")
    if not dataset_path_val or not Path(dataset_path_val).is_dir():
        raise ValueError("The offline runner requires an explicit transition dataset_path directory")
    from src.app.dataset_utils import DatasetReader
    reader = DatasetReader(dataset_path_val)
    if not len(reader): raise ValueError(f"No transitions in {dataset_path_val}")
    print(f"[d3rlpy Engine] Loading transitions from {dataset_path_val} ({len(reader)} items)...")
    observations = reader.obs[:len(reader)].cpu().numpy()
    actions = reader.actions[:len(reader)].cpu().numpy()
    rewards = reader.rewards[:len(reader)].cpu().numpy()
    terminals = reader.dones[:len(reader)].cpu().numpy()

    obs_arr = np.array(observations, dtype=np.float32)
    act_arr = np.array(actions)
    rew_arr = np.array(rewards, dtype=np.float32)
    term_arr = np.array(terminals, dtype=bool)

    # Determine action type (discrete vs continuous)
    is_discrete = np.issubdtype(act_arr.dtype, np.integer)
    if is_discrete:
        act_arr = act_arr.astype(np.int64)
    else:
        act_arr = act_arr.astype(np.float32)

    from d3rlpy.dataset import MDPDataset
    from d3rlpy.constants import ActionSpace
    dataset = MDPDataset(observations=obs_arr, actions=act_arr, rewards=rew_arr,
                         terminals=term_arr, action_space=ActionSpace.DISCRETE if is_discrete else ActionSpace.CONTINUOUS)

    # Algorithm selection
    algo = None
    if algo_name in ("CQL", "CONSERVATIVE_Q_LEARNING"):
        if is_discrete and hasattr(d3rlpy.algos, "DiscreteCQLConfig"):
            algo = d3rlpy.algos.DiscreteCQLConfig(learning_rate=lr, batch_size=batch_size).create()
        elif hasattr(d3rlpy.algos, "CQLConfig"):
            algo = d3rlpy.algos.CQLConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("IQL", "IMPLICIT_Q_LEARNING"):
        if hasattr(d3rlpy.algos, "IQLConfig"):
            algo = d3rlpy.algos.IQLConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("TD3_BC", "TD3PLUSBC"):
        if hasattr(d3rlpy.algos, "TD3PlusBCConfig"):
            algo = d3rlpy.algos.TD3PlusBCConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("BC", "BEHAVIOR_CLONING"):
        if is_discrete and hasattr(d3rlpy.algos, "DiscreteBCConfig"):
            algo = d3rlpy.algos.DiscreteBCConfig(learning_rate=lr, batch_size=batch_size).create()
        elif hasattr(d3rlpy.algos, "BCConfig"):
            algo = d3rlpy.algos.BCConfig(learning_rate=lr, batch_size=batch_size).create()

    if algo is None:
        raise ValueError(f"d3rlpy does not provide the declared algorithm {algo_name!r} for this action space")

    start_time = time.time()
    status = "failed"
    logger.event("stage", "Offline training started", stage="training")

    try:
        eval_env = gym.make(env_id)
        eval_freq = int(agent_cfg.get("eval_freq", 1000))

        # Training loop logging to telemetry
        step = 0
        while step < n_steps:
            steps_to_run = min(eval_freq, n_steps - step)
            if hasattr(algo, "fit") and dataset is not None:
                algo.fit(dataset, n_steps=steps_to_run, n_steps_per_epoch=steps_to_run, show_progress=False)
            step += steps_to_run

            # Evaluate policy in environment
            eval_rewards = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                ep_r = 0.0
                done = False
                while not done:
                    act = algo.predict(np.expand_dims(obs, axis=0))[0] if hasattr(algo, "predict") else eval_env.action_space.sample()
                    obs, r, term, trunc, _ = eval_env.step(act)
                    ep_r += float(r)
                    done = term or trunc
                eval_rewards.append(ep_r)

            mean_eval_r = float(np.mean(eval_rewards))
            metrics = {
                "eval/reward": mean_eval_r,
                "train/reward": mean_eval_r,
                "step": step,
            }
            logger.log(metrics, step=step, transitions=step)

        eval_env.close()
        status = "completed"
    except BaseException as exc:
        logger.event("error", f"{type(exc).__name__}: {exc}", level="error")
        raise
    finally:
        end_time = time.time()
        # Save checkpoints
        try:
            if hasattr(algo, "save"):
                algo.save(str(ckpt_dir / "best_model.d3"))
                algo.save(str(ckpt_dir / "latest_model.d3"))
        except Exception as e:
            print(f"[Warning] Failed to save d3rlpy checkpoint: {e}")

        logger.save_runtime_metadata(
            config=OmegaConf.to_container(cfg, resolve=True),
            training_time_seconds=end_time - start_time,
        )
        for name in ("best_model.d3", "latest_model.d3"):
            p = ckpt_dir / name
            if p.exists():
                logger.artifact(p, kind="checkpoint")
        logger.finish(status=status)

    print(f"\n[d3rlpy Engine Complete] Total time: {end_time - start_time:.2f}s")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Telemetry logged to: {logger.version_dir}")


def launch():
    paths = WorkspacePaths.for_process()
    search = [f"file://{p}" for p in paths.config_roots() if p.is_dir() and p != paths.primary_config]
    if not any(arg.lstrip('+').startswith('hydra.searchpath=') for arg in sys.argv[1:]):
        sys.argv.append('hydra.searchpath=[' + ','.join(search) + ']')
    main()


if __name__ == "__main__":
    launch()
