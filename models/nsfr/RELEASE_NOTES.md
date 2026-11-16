# 1.2.0

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest. Declares the `rules` workspace resource (`src/usr/rules` plus its own `resources/rules`, entries `environment/name`) and exports `resolve_ruleset`, the ruleset lookup NEUMANN and BlendRL share. The `rules` parameter is a picker of those entries; the package's editor action is replaced by the IDE's generic resource action.

# 1.1.0

Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages. Publishes `get_nsfr_model`, `get_blender_nsfr_model`, `load_module`, `bool_to_probs`, `FactsConverter`, `ValuationModule`, `build_infer_module`, `get_lang`, `get_blender_lang`, `subs_list` and `unify` as declared exports for dependent packages (NEUMANN, BlendRL). Rules, selectors and defaults are unchanged.

# 1.0.7

Rules resolve only from src/usr/rules. The matching IDE rejects legacy in/rules locations, aliases and package references; there is no discovery fallback or path translation. Rule folders can provide their own valuation.py through ruleset.yaml. Editor actions open the selected rules directory. Includes anticipatory_v1 for ordinary CartPole environments. Requires the matching IDE rules-resource update.

The non-training factory supports multi-clause rulesets with train=False.
