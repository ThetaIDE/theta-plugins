## 1.3.1 (local preparation; unpublished)

`blender` is the neural/symbolic selector. Configure its model under `blender_model`. Component schemas match the structured neural/symbolic defaults and nullable learning rates.

# BlendRL 1.3.0

Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages. A `symbolic_observation_builder` may name a declared export (`export://<package>/<name>`); `module:function` references still resolve. The unused model-slot helper is removed. Selectors, defaults and checkpoint classes are unchanged.

# BlendRL 1.2.0

Partial rules now preserve confidence and abstain below logic_confidence_floor (default 0.05 in new profiles). Missing symbolic critics share the available state-value estimate. Neural child options are forwarded. Symbolic gate outputs follow actual expert ordering. Historical configs without a confidence floor retain legacy action normalization.
