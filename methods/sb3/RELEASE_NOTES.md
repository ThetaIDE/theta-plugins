# 2.2.0

PPO serves three opt-in model protocols, which a model lists in `sb3_protocols`:

- `rollout_context.v1`: the model may return `aux["rollout_context"]` while sampling (`phase="rollout"`). PPO stores it with each transition and passes it back when it re-evaluates that transition (`phase="update"`); next-state values use `phase="bootstrap"`.
- `training_progress.v1`: PPO calls `on_rollout_start(num_timesteps, total_timesteps)` before each rollout and `on_env_step(dones, infos)` after each environment step. An optional `training_metrics()` is read at each rollout end and logged as `diagnostics/<name>`.
- `auxiliary_losses.v1`: PPO adds the model's `compute_auxiliary_losses(batch, output)` terms (already weighted and signed) to each minibatch loss and logs them as `losses/train/aux_loss/<name>`.

A model that declares none of them trains with SB3's own PPO update. A model that declares any trains only with `sb3/ppo`; A2C rejects it. Requires stable-baselines3 2.9, whose PPO update this release follows.

# 2.1.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest. Algorithms block a configured `data_module` or `dataset_path`: they train from the live environment only.

# SB3 Suite 2.0.0

Algorithm capabilities for execution contract v1: the paradigm runner calls `sb3_runner.py:execute(cfg, context)` with `backend: sb3`. Profiles no longer declare a script `entrypoint`, and select `sb3/ppo`, `sb3/dqn`, `sb3/a2c` or `sb3/sac` (the bare aliases `ppo`, `dqn`, `a2c` and `sac` are removed). Every evaluation records matched greedy and sampled returns; `eval/reward` remains the greedy return. Mechanistic monitors attach after each optimizer update. Checkpoints are written to the run's artifact directory. Exact training resume is rejected; inference reload of `.zip` checkpoints is supported. Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages, with execution contract v1 and SDK 2.2.

# SB3 Suite 1.1.0

Final evaluation and diagnostics follow the final optimizer update. Best and latest checkpoints remain distinct. Explicit linear learning-rate decay is supported; training seeds and vector environment counts are honored. Routing diagnostics sample across the entire evaluation.
