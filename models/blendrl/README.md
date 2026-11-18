# BlendRL

Select `blendrl` with a runner advertising `model_observations.v1`, such as
`sb3/ppo`. The package declares its outputs for the experiment's Monitors box.

`blender` chooses neural or symbolic routing. Its required `blender_model` field
uses the standard model picker; use its **+** to add the selected model's parameters.
The default explicitly selects the prior implicit 64×64 DNN. The two YAML fields
remain siblings; no special conditional Blender form is used.

BlendRL mixes neural, symbolic, and other policy modules. With SB3 suite 1.0.6 or later, `model.logic_lr` and `model.blender_lr` set initial optimizer group rates; null uses the agent's learning rate. SB3's schedule scales every group proportionally. This applies to the Theta PPO, A2C and DQN adapters. Nonzero `blend_ent_coef` remains unsupported by SB3 and fails explicitly; it is not silently ignored. Hard Gumbel routing also remains unsupported by SB3.

SB3 records `blend/train/` metrics every `model.monitoring_interval` environment steps (default 256; zero disables). Evaluation summaries are recorded when the runner performs evaluation. Include `blend/` in `telemetry.metrics` when selecting metrics. Experts are named by type (`blend/train/logic/...`, `blend/train/neural/...`); a type that appears more than once is numbered from 1 in module order (`logic_1`, `logic_2`), and only then is a per-type total such as `logic/weight_mean` added. Every catalog output has a readable label and a description. Dominance uses argmax, exclusive means weight >= 0.9, and mixed means no expert reaches that threshold. Logic coverage reports the fraction of samples where a mapped logic action predicate is active before fallback. Agreement and Jensen–Shannon divergence compare expert action distributions; they do not establish safety. Normalized entropy is entropy divided by log(number of experts), with one expert defined as zero. Other runners must call the model's `collect_monitoring` hook and log its scalar results; this package does not automatically instrument them.

For state plots, set `model.monitoring_max_samples` to a positive maximum. The SB3 runner stores recent sampled observations and routing weights in `model_monitoring_samples.npz` under the run's log directory. This can contain sensitive observations: enable it only when appropriate. The `blend_routing` plotter reads those artifacts and logged metrics. It creates authority, weight distributions, and PCA plots; enable `tsne` in the plot configuration for manifold plots. PCA and t-SNE use up to 500 recent samples and 2048 evenly spaced features. A separately prepared artifact may include per-row `cohort` labels or `distance` values for conditional plots. Distance is a supplied score, not an assertion of out-of-distribution detection. Observations are sampled periodically and are not a representative dataset guarantee.
