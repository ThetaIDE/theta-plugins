# 1.1.1

Compatibility is declared once, as capability `requires` rules in the manifest. The package-level `paradigms` list, profile `compatible_paradigms`, `execution_features.input_types` and class-level `get_supported_paradigms` are gone; nothing reads them. Requires an IDE with Theta SDK 3.1 (the compatibility-requirements release); earlier IDEs reject the manifest. Resolves rulesets through NSFR's `resolve_ruleset` export; the `rules` parameter is a picker of the `rules` resource NSFR declares.

Removes `neumann_utils.py`, `valuation.py` and `valuation_func.py`, an unused perception chain that imported modules the package does not provide. Valuation functions still come from the selected rules. Selectors and defaults are unchanged.

# 1.1.0

Package format v3 (`storage_version: 3`) with host imports from the Theta SDK only. Requires an IDE that reads storage-v3 packages. Loads NSFR helpers through its declared exports instead of importing NSFR files, and publishes `get_neumann_model` and `get_blender_neumann_model`. Selectors and defaults are unchanged.

# 1.0.7

Rules resolve only from src/usr/rules. The matching IDE rejects legacy in/rules locations, aliases and package references; there is no discovery fallback or path translation. Rule folders can provide their own valuation.py through ruleset.yaml. Editor actions open the selected rules directory. Includes anticipatory_v1 for ordinary CartPole environments. Requires the matching IDE rules-resource update.
