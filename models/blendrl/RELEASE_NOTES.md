# BlendRL 1.2.0

Partial rules now preserve confidence and abstain below logic_confidence_floor (default 0.05 in new profiles). Missing symbolic critics share the available state-value estimate. Neural child options are forwarded. Symbolic gate outputs follow actual expert ordering. Historical configs without a confidence floor retain legacy action normalization.
