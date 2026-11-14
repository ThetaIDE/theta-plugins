# Anticipatory CartPole rules

Four candidates choose left/right from short (0.20 s) and long (0.30 s) pole-lean control scores, using pole angle and angular velocity plus small cart position/velocity feedback. The scores are heuristics, not exact forecasts.

This self-contained ruleset uses its own valuation.py. Select `anticipatory_v1` with the ordinary CartPole environment; no symbolic environment alias or custom observation builder is needed. NSFR learns candidate weights, not the horizons or predicate coefficients.
