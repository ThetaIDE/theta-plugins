import os
import pickle
from pathlib import Path


def get_ep_eval_data(exp_id, cfg, group, output_dir):
    cache_path = Path(output_dir) / "ep_eval_cache.pkl"
    remake = cfg.get("remake", False)

    ckpt_dir = Path("results/checkpoints") / group / exp_id
    if not ckpt_dir.exists():
        matches = list(Path("results/checkpoints").glob(f"**/{exp_id}"))
        if matches:
            ckpt_dir = matches[0]

    cache_stale = False
    if cache_path.exists() and ckpt_dir.exists():
        cache_mtime = cache_path.stat().st_mtime
        for ckpt_file in ckpt_dir.rglob("*.ckpt"):
            if ckpt_file.stat().st_mtime > cache_mtime:
                cache_stale = True
                break

    if cache_path.exists() and not remake and not cache_stale:
        print(f"Loading cached EP evaluation data from {cache_path}...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from src.app.core.provider_loader import provider_object
    compute_ep_eval_data = provider_object('mimic-early-prediction', 'eval_logic.py:compute_ep_eval_data')

    if not ckpt_dir.exists():
        print(f"Error: Could not find checkpoint directory for {exp_id}")
        return None

    print("Computing EP evaluation data (this may take a while)...")
    data = compute_ep_eval_data(
        checkpoint_root=str(ckpt_dir),
        dataset_path=cfg.get("dataset_path", None),
        ep_ckpt_root=cfg.get("ep_ckpt_root", "results/checkpoints/early_prediction"),
        n_splits=cfg.get("n_splits", 20),
        use_volatility=cfg.get("use_volatility", True),
    )

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    return data
