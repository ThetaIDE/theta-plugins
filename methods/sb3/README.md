# Stable-Baselines3 Suite

PPO, DQN, A2C and SAC from Stable-Baselines3 as Theta algorithm capabilities
(`sb3/ppo`, `sb3/dqn`, `sb3/a2c`, `sb3/sac`) for the online RL paradigm.

## Evaluation

Every enabled evaluation runs both action-selection modes, with the same number
of episodes and matched environment reset seeds:

- `eval/greedy_reward`: return with deterministic actions (argmax for discrete policies).
- `eval/sampled_reward`: return with stochastic actions from the same frozen checkpoint.
- Each has a corresponding `_std` metric describing spread across episodes.
- `eval/reward` and `eval/reward_std` remain aliases for the greedy metrics, so
  existing objectives, pruning and best-checkpoint selection keep their meaning.

Both modes run at step zero, at the configured evaluation interval, and after
the final training update. `eval_episodes` is the episode count **per mode**;
there is no mode flag. Evaluation uses isolated random state, so its action
sampling does not consume the training random sequence. The base environment
and action seed is `variability_evaluation_seed`, falling back to
`variability.evaluation_seed` or 10000000; the evaluation index is added to it
modulo (2^32 minus 1). The same seeds are used for both modes at each checkpoint.

Per-episode outcomes, mode and seeds are saved in `evaluation_events.jsonl`.
Greedy events retain `metric: eval/reward` for existing scoring readers; sampled
events use `metric: eval/sampled_reward`. Runtime metadata records the protocol.
The live and Results metric catalogs identify both return curves explicitly.

Use this runner for all methods in a comparison to keep the protocol consistent.
Compare independent training seeds; episode standard deviation is not a
confidence interval across training runs. Include the untrained and rule-only
baselines when measuring what a hybrid policy gains from learning.

## Execution contract v1

The paradigm runner invokes `sb3_runner.py:execute(cfg, context)` as a backend.
Profiles do not select scripts. Capabilities declare `execution_contract: 1`
and `backend: sb3`; select `sb3/*` in a method's `agent`. Checkpoints are native
SB3 `.zip` files in the run's artifact directory. Exact training resume is
rejected; native inference reload is supported.

Mechanistic monitors attach at SB3's post-update boundary through
`sb3_monitoring.py`, which this package owns.
