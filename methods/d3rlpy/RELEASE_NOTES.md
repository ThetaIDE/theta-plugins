# 2.1.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest. Algorithms block a configured `data_module`: they read the offline dataset from `dataset_path`.

# d3rlpy Offline RL Suite 2.0.0

Algorithm capabilities for execution contract v1: the paradigm runner calls `d3rlpy_backend.py:execute(cfg, context)` with `backend: d3rlpy`. Profiles no longer declare a script `entrypoint`, and select `d3rlpy/bc`, `d3rlpy/cql`, `d3rlpy/iql` or `d3rlpy/td3_bc` (the bare aliases are removed) with the new encoder model `d3rlpy/vector`. Agent keys the selected d3rlpy algorithm accepts, such as `gamma`, `expectile`, `weight_temp` and `alpha`, now reach it; `cql_alpha`, which no d3rlpy configuration accepts, is removed. Training records losses only; it never runs a simulator. Exact training resume is rejected; `.d3` bundles reload for inference. Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages, with execution contract v1 and SDK 2.2.
