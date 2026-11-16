# 1.1.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest.

The shaper reads its settings only from its own parameters: no `env.reward_shaping` block, `env.gamma` override or `EP_SHAPE_*` environment variables, and the unused `shape_rewards_ep` helper is removed. `ep_ckpt_dir` defaults to the workspace's `results/checkpoints/early_prediction` and relative paths resolve from the workspace, not the working directory; `cql_ckpt_dir` is a parameter.
