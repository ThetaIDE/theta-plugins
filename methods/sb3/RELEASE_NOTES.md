# 2.1.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest. Algorithms block a configured `data_module` or `dataset_path`: they train from the live environment only.

# SB3 Suite 2.0.0

Algorithm capabilities for execution contract v1: the paradigm runner calls `sb3_runner.py:execute(cfg, context)` with `backend: sb3`. Profiles no longer declare a script `entrypoint`, and select `sb3/ppo`, `sb3/dqn`, `sb3/a2c` or `sb3/sac` (the bare aliases `ppo`, `dqn`, `a2c` and `sac` are removed). Every evaluation records matched greedy and sampled returns; `eval/reward` remains the greedy return. Mechanistic monitors attach after each optimizer update. Checkpoints are written to the run's artifact directory. Exact training resume is rejected; inference reload of `.zip` checkpoints is supported. Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages, with execution contract v1 and SDK 2.2.

# SB3 Suite 1.1.0

Final evaluation and diagnostics follow the final optimizer update. Best and latest checkpoints remain distinct. Explicit linear learning-rate decay is supported; training seeds and vector environment counts are honored. Routing diagnostics sample across the entire evaluation.
