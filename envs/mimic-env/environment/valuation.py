import os
import torch as th
from src.app.core.provider_loader import provider_object
bool_to_probs = provider_object('nsfr', 'utils/common.py:bool_to_probs')


VALUATION_MODE = "rigid"


def set_valuation_mode(mode: str):
    """Set valuation mode to 'rigid' or 'continuous'."""
    global VALUATION_MODE
    VALUATION_MODE = str(mode).lower()


def _is_continuous() -> bool:
    """Check if continuous soft valuations are enabled via flag or env var."""
    if VALUATION_MODE in ["continuous", "soft"]:
        return True
    return os.environ.get("MIMIC_VALUATION_MODE", "rigid").lower() in ["continuous", "soft"]


def _smooth_lt(x: th.Tensor, threshold: float, tau: float = 0.5) -> th.Tensor:
    """Continuous soft less-than: P(x < threshold) = sigmoid((threshold - x) / tau)."""
    if _is_continuous():
        return th.sigmoid((threshold - x) / tau).clamp(min=1e-4, max=1.0 - 1e-4)
    return bool_to_probs(x < threshold)


def _smooth_gt(x: th.Tensor, threshold: float, tau: float = 0.5) -> th.Tensor:
    """Continuous soft greater-than: P(x > threshold) = sigmoid((x - threshold) / tau)."""
    if _is_continuous():
        return th.sigmoid((x - threshold) / tau).clamp(min=1e-4, max=1.0 - 1e-4)
    return bool_to_probs(x > threshold)


def low_bp(agent: th.Tensor) -> th.Tensor:
    # MAP is index 15. Z-score < -1.0 indicates hypotension
    map_val = agent[..., 15]
    return _smooth_lt(map_val, -1.0)


def high_lactate(agent: th.Tensor) -> th.Tensor:
    # Lactate is index 10. Z-score > 1.0 indicates hyperlactatemia
    lactate = agent[..., 10]
    return _smooth_gt(lactate, 1.0)


def high_creatinine(agent: th.Tensor) -> th.Tensor:
    # Creatinine is index 12. Z-score > 1.0 indicates renal dysfunction
    creatinine = agent[..., 12]
    return _smooth_gt(creatinine, 1.0)


def high_bilirubin(agent: th.Tensor) -> th.Tensor:
    # Bilirubin is index 13. Z-score > 1.0 indicates hepatic dysfunction
    bilirubin = agent[..., 13]
    return _smooth_gt(bilirubin, 1.0)


def low_platelets(agent: th.Tensor) -> th.Tensor:
    # Platelets is index 11. Z-score < -1.0 indicates thrombocytopenia/coagulation issues
    platelets = agent[..., 11]
    return _smooth_lt(platelets, -1.0)


def normal_bp(agent: th.Tensor) -> th.Tensor:
    # Stable BP corresponds to z-score >= -1.0
    map_val = agent[..., 15]
    return _smooth_gt(map_val, -1.0)


def normal_lactate(agent: th.Tensor) -> th.Tensor:
    # Stable Lactate corresponds to z-score <= 1.0
    lactate = agent[..., 10]
    return _smooth_lt(lactate, 1.0)


def normal_organs(agent: th.Tensor) -> th.Tensor:
    # All organ markers within normal ranges (z-score <= 1.0 and platelets >= -1.0)
    creatinine = agent[..., 12]
    bilirubin = agent[..., 13]
    platelets = agent[..., 11]
    if _is_continuous():
        norm_cr = _smooth_lt(creatinine, 1.0)
        norm_bili = _smooth_lt(bilirubin, 1.0)
        norm_plat = _smooth_gt(platelets, -1.0)
        return (norm_cr * norm_bili * norm_plat).clamp(min=1e-4, max=1.0 - 1e-4)
    is_normal = (creatinine <= 1.0) & (bilirubin <= 1.0) & (platelets >= -1.0)
    return bool_to_probs(is_normal)


def true(env_obj: th.Tensor) -> th.Tensor:
    return bool_to_probs(th.ones_like(env_obj[..., 0], dtype=th.bool))
