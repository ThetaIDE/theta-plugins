from typing import Any

try:
    import torch as th
    from src.app.core.provider_loader import provider_object
    bool_to_probs = provider_object('nsfr', 'utils/common.py:bool_to_probs')
except ImportError:
    th = None
    def bool_to_probs(x):
        return x

def enemy_in_sight(enemy_obj) -> Any:
    # enemy_obj: [dist, angle, alive, hp_ratio]
    dist = enemy_obj[..., 0] if th is not None else enemy_obj[0]
    alive = enemy_obj[..., 2] if th is not None else enemy_obj[2]
    return bool_to_probs((alive > 0.5) & (dist < 0.7))

def facing_enemy(enemy_obj) -> Any:
    angle = enemy_obj[..., 1] if th is not None else enemy_obj[1]
    alive = enemy_obj[..., 2] if th is not None else enemy_obj[2]
    val = (th.abs(angle) < 0.15) if th is not None else (abs(angle) < 0.15)
    return bool_to_probs((alive > 0.5) & val)

def low_health(agent_obj) -> Any:
    # agent_obj: [x, y, angle, hp]
    hp = agent_obj[..., 3] if th is not None else agent_obj[3]
    return bool_to_probs(hp < 0.4)

def item_available(item_obj) -> Any:
    avail = item_obj[..., 1] if th is not None else item_obj[1]
    return bool_to_probs(avail > 0.5)

def near_exit(door_obj) -> Any:
    dist = door_obj[..., 0] if th is not None else door_obj[0]
    return bool_to_probs(dist < 0.15)
