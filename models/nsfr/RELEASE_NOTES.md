# 1.0.7

Rules resolve only from src/usr/rules. The matching IDE rejects legacy in/rules locations, aliases and package references; there is no discovery fallback or path translation. Rule folders can provide their own valuation.py through ruleset.yaml. Editor actions open the selected rules directory. Includes anticipatory_v1 for ordinary CartPole environments. Requires the matching IDE rules-resource update.

The non-training factory supports multi-clause rulesets with train=False.
