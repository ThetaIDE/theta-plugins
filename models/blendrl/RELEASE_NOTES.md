# BlendRL 1.4.0

Compatibility is declared as capability `requires` rules (Theta SDK 3.1); earlier IDEs reject the manifest.

- `blender` chooses neural or symbolic routing; its model is the required `blender_model` selection (default: the prior implicit 64×64 DNN).
- The symbolic state builder is the model parameter `symbolic_state`, a picker of data modules providing `symbolic_state/v1`. Hybrid and logic actors and the symbolic blender block without one. The package provides `blendrl_symbolic_state` with the `data/blendrl/symbolic_state` and `data/blendrl/cartpole_state` profiles; `symbolic.observation_builder` and the environment's `symbolic_observation_builder` are no longer read.
- Model-owned `blendrl/reliance` monitor (disabled by default): expert and gate counterfactuals for hybrid softmax blending, with a monitor catalog describing BlendRL's metrics.
- Non-hybrid actor modes zero and renormalize incompatible experts; the symbolic blender receives the logic state in neural-only evaluation.
- `logic_confidence_floor` is a number; null no longer selects the earlier normalization. Null `logic_lr`/`blender_lr` use the agent's learning rate; matching `agent` keys are no longer read. Comma-separated `rules` strings are no longer split; use a list.
- The `blender` model alias is removed; select `blendrl`.
- Rulesets are entries of the `rules` workspace resource NSFR declares. Declares `lark`.

# BlendRL 1.3.0

Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages. A `symbolic_observation_builder` may name a declared export (`export://<package>/<name>`); `module:function` references still resolve. The unused model-slot helper is removed. Selectors, defaults and checkpoint classes are unchanged.

# BlendRL 1.2.0

Partial rules now preserve confidence and abstain below logic_confidence_floor (default 0.05 in new profiles). Missing symbolic critics share the available state-value estimate. Neural child options are forwarded. Symbolic gate outputs follow actual expert ordering. Historical configs without a confidence floor retain legacy action normalization.
