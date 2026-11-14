# d3rlpy Offline RL Suite

BC, CQL, IQL and TD3+BC from d3rlpy as Theta algorithm capabilities
(`d3rlpy/bc`, `d3rlpy/cql`, `d3rlpy/iql`, `d3rlpy/td3_bc`) for the offline RL
paradigm, with the vector encoder model `d3rlpy/vector`.

The paradigm runner invokes `d3rlpy_backend.py:execute(cfg, context)` (execution
contract v1). Select `d3rlpy/vector` as the method's model and set `dataset_path`
to a Theta `DatasetWriter` transition directory. The backend loads each
transition once, performs the selected learner's updates and records training
losses. It never calls a simulator: training losses are not held-out evaluation.

Agent profile keys that the selected d3rlpy algorithm accepts (`gamma`,
`expectile`, `weight_temp`, `alpha`, ...) are passed to it; `learning_rate`
sets every learning rate of the algorithm. `n_steps` must be divisible by
`n_steps_per_epoch`. IQL and TD3+BC require continuous actions.

Native `.d3` bundles reload with `d3rlpy.load_learnable`; exact training resume
is rejected. Tuning and winner promotion of `.d3` bundles are not supported.
