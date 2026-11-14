"""Early Prediction Sepsis Data Module.

Loads and prepares MIMIC clinical trajectory datasets for early prediction:
- Extracts sequence windows and clinical observation features.
- Computes volatility features and integrates optional CQL policy values V(s).
- Builds stratified cross-validation train/test cohort splits.
- Provides standard PyTorch DataLoaders with variable-length batch collation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.app.core.interfaces import BaseDataModule
from src.app.core.paradigm_loader import register_component
from .model import compute_volatility_features, normalize_features

log = logging.getLogger(__name__)


class EPSepsisDataset(Dataset):
    def __init__(self, X, y, input_dim):
        self.X = X
        self.y = y
        self.input_dim = input_dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        label = self.y[idx] if self.y is not None else 0.0
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_ep_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = max(lengths).item()
    input_dim = sequences[0].shape[-1]

    padded_seqs = torch.zeros(len(sequences), max_len, input_dim, dtype=torch.float32)
    padding_mask = torch.ones(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        padded_seqs[i, : len(seq), :] = seq
        padding_mask[i, : len(seq)] = False

    return padded_seqs, torch.stack(labels), lengths, padding_mask


@register_component("EPSepsisDataModule", "SepsisDataModule")
class EPSepsisDataModule(L.LightningDataModule, BaseDataModule):
    """DataModule managing MIMIC clinical trajectory loading and CV splits."""

    def __init__(self, cfg: Any = None):
        super().__init__()
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.patient_lengths: np.ndarray | None = None
        self.v_vals_all: np.ndarray | None = None
        self.c_indices_train: np.ndarray | None = None
        self.t_cutoffs_train: np.ndarray | None = None
        self.y_cohort_train: np.ndarray | None = None
        self.seq_data_cache: dict[str, list[np.ndarray]] = {}
        self.splits_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self.cfg = cfg
        self.window_hours = 12
        self.w_steps = 24
        self.tau_train = 12
        self.tau_max = 33
        self.use_volatility = True
        self.use_norm = True
        self.use_all_history = False
        self.use_all_trajectories = False
        self.n_splits = 20

        if cfg is not None:
            self.setup(cfg)

    def setup(self, stage: str | None = None, cfg: Any = None) -> None:
        """Initialize dataset arrays and precompute CQL V(s) if configured."""
        if cfg is not None:
            self.cfg = cfg
        cfg = self.cfg
        if cfg is None or self.X is not None:
            return
        ep_cfg = cfg.get("early_prediction", {}) or {}
        if hasattr(ep_cfg, "__iter__") and not isinstance(ep_cfg, dict):
            from omegaconf import OmegaConf

            ep_cfg = OmegaConf.to_container(ep_cfg, resolve=True)

        dataset_path = ep_cfg.get("dataset_path") or cfg.get("dataset_path")
        if not dataset_path and hasattr(cfg, "env"):
            ds_name = cfg.env.get("dataset_name", "")
            if ds_name:
                dataset_path = f"in/datasets/mimic/{ds_name}"
        if not dataset_path:
            raise ValueError(
                "Dataset path is required for supervised sepsis learning. Specify dataset_path in the config."
            )
        if not os.path.exists(dataset_path) and os.path.exists(f"{dataset_path}.npz"):
            dataset_path = f"{dataset_path}.npz"

        log.info("Loading MIMIC sequence dataset from: %s", dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        data = np.load(dataset_path, allow_pickle=True)
        self.X = data["X"]
        self.y = data["y"].squeeze()
        self.mask = data["mask"]
        self.patient_lengths = np.array([(self.mask[i].squeeze() != -1).sum() for i in range(len(self.X))])

        data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}
        if hasattr(data_cfg, "__iter__") and not isinstance(data_cfg, dict):
            from omegaconf import OmegaConf

            data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
        if not isinstance(data_cfg, dict):
            data_cfg = {}

        self.window_hours = ep_cfg.get("window_hours", data_cfg.get("window_hours", 12))
        self.w_steps = 2 * self.window_hours
        self.tau_train = ep_cfg.get("tau_train", data_cfg.get("tau_train", 12))
        self.tau_max = ep_cfg.get("tau_max", data_cfg.get("tau_max", 33))
        self.use_volatility = ep_cfg.get("use_volatility", data_cfg.get("use_volatility", True))
        self.use_norm = ep_cfg.get("use_norm", data_cfg.get("use_norm", True))
        self.use_all_history = ep_cfg.get("use_all_history", data_cfg.get("use_all_history", False))
        self.use_all_trajectories = ep_cfg.get("use_all_trajectories", data_cfg.get("use_all_trajectories", False))
        self.n_splits = ep_cfg.get("n_splits", data_cfg.get("n_splits", 20))

        # Precompute CQL policy values V(s) if checkpoint is supplied
        checkpoint = ep_cfg.get("checkpoint")
        if checkpoint:
            self.v_vals_all = self._compute_cql_v_values(checkpoint)

        self._build_training_cohort()
        _, self.input_dim = self.get_training_sequences()

    def _compute_cql_v_values(self, checkpoint: str) -> np.ndarray:
        from src.app.core.agent_registry import get_agent_class
        agent_cls = get_agent_class("cql")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_arg = Path(checkpoint)
        cql_ckpt_path = None

        if checkpoint_arg.is_dir():
            candidates = list(checkpoint_arg.glob("**/*.ckpt"))
            if candidates:
                cql_ckpt_path = str(candidates[-1])
        elif checkpoint_arg.exists():
            cql_ckpt_path = str(checkpoint_arg)

        if not cql_ckpt_path or not os.path.exists(cql_ckpt_path):
            raise FileNotFoundError(f"CQL checkpoint '{checkpoint}' does not exist.")

        log.info("Loading CQL agent from %s for V(s) precomputation...", cql_ckpt_path)
        cql_agent = agent_cls.load_from_checkpoint(cql_ckpt_path, map_location=device, weights_only=False)
        cql_agent.eval()

        v_vals_all = np.zeros((len(self.X), 240, 1), dtype=np.float32)
        batch_sz = 128
        with torch.no_grad():
            for i in range(0, len(self.X), batch_sz):
                batch_x = torch.tensor(self.X[i : i + batch_sz, :, :46], dtype=torch.float32).to(device)
                b_curr = batch_x.size(0)
                flat_x = batch_x.view(-1, 46)

                if hasattr(cql_agent, "get_q_values"):
                    flat_q = cql_agent.get_q_values(flat_x)
                elif hasattr(cql_agent, "model") and hasattr(cql_agent.model, "get_q_values"):
                    flat_q = cql_agent.model.get_q_values(flat_x)
                elif hasattr(cql_agent, "q_network"):
                    flat_q = cql_agent.q_network(flat_x)
                else:
                    raise AttributeError("CQL agent does not expose get_q_values or q_network.")

                q_vals = flat_q.view(b_curr, 240, -1)
                v_vals = torch.max(q_vals, dim=-1)[0].unsqueeze(-1).cpu().numpy()
                v_vals_all[i : i + batch_sz] = v_vals

        log.info("CQL V(s) precomputation complete.")
        return v_vals_all

    def _build_training_cohort(self) -> None:
        """Select patients with adequate trajectory length for training."""
        steps_early_train = 2 * self.tau_train
        if self.use_all_trajectories:
            self.c_indices_train = np.array(
                [i for i in range(len(self.X)) if self.patient_lengths[i] - steps_early_train >= 1]
            )
        else:
            min_stay_steps = 2 * max(self.tau_max, self.tau_train) + self.w_steps
            self.c_indices_train = np.array(
                [i for i in range(len(self.X)) if self.patient_lengths[i] >= min_stay_steps]
            )

        self.t_cutoffs_train = self.patient_lengths[self.c_indices_train] - steps_early_train
        self.y_cohort_train = self.y[self.c_indices_train]

    def get_training_sequences(self, use_v: bool = False) -> tuple[list[np.ndarray], int]:
        """Construct observation sequences at tau_train with volatility and optional V(s)."""
        cache_key = f"train_v_{use_v}"
        if cache_key in self.seq_data_cache:
            seqs = self.seq_data_cache[cache_key]
            return seqs, seqs[0].shape[-1]

        seqs = []
        for i, orig_idx in enumerate(self.c_indices_train):
            tc = self.t_cutoffs_train[i]
            st = 0 if self.use_all_history else max(0, tc - self.w_steps)
            raw_seq = self.X[orig_idx, st:tc, :49]
            feat_seq = compute_volatility_features(raw_seq) if self.use_volatility else raw_seq

            if use_v and self.v_vals_all is not None:
                v_seq = self.v_vals_all[orig_idx, st:tc]
                seqs.append(np.concatenate([feat_seq, v_seq], axis=-1))
            else:
                seqs.append(feat_seq)

        self.seq_data_cache[cache_key] = seqs
        return seqs, seqs[0].shape[-1]

    def get_split_indices(self, split_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_cohort_idxs, test_cohort_idxs) for a CV split."""
        if split_idx not in self.splits_cache:
            seed_val = 42 + split_idx
            tr_idxs, te_idxs = train_test_split(
                np.arange(len(self.c_indices_train)),
                test_size=0.2,
                random_state=seed_val,
                stratify=self.y_cohort_train,
            )
            self.splits_cache[split_idx] = (tr_idxs, te_idxs)
        return self.splits_cache[split_idx]

    def get_train_dataloader(
        self, split_idx: int, use_v: bool = False, batch_size: int = 64
    ) -> tuple[DataLoader, list[np.ndarray], np.ndarray, int]:
        """Build standard PyTorch DataLoader for training fold."""
        seqs, input_dim = self.get_training_sequences(use_v=use_v)
        tr_idxs, _ = self.get_split_indices(split_idx)

        x_train = [seqs[i] for i in tr_idxs]
        y_train = self.y_cohort_train[tr_idxs]

        dataset = EPSepsisDataset(x_train, y_train, input_dim)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ep_batch)
        return loader, x_train, y_train, input_dim

    def get_eval_dataloader(
        self,
        split_idx: int,
        tau: int,
        x_train: list[np.ndarray],
        use_v: bool = False,
        input_dim: int = 64,
        batch_size: int = 128,
    ) -> DataLoader | None:
        """Build PyTorch DataLoader for evaluation at lead-time horizon tau."""
        _, te_idxs = self.get_split_indices(split_idx)
        test_global_patient_indices = self.c_indices_train[te_idxs]

        steps_early_tau = 2 * tau
        valid_mask = [
            i
            for i, g_idx in enumerate(test_global_patient_indices)
            if self.patient_lengths[g_idx] - steps_early_tau >= 1
        ]
        if not valid_mask:
            return None

        test_eval_global_idxs = test_global_patient_indices[valid_mask]
        y_test_tau = self.y[test_eval_global_idxs]
        t_cutoffs_tau = self.patient_lengths[test_eval_global_idxs] - steps_early_tau

        x_test_tau = []
        for i, g_idx in enumerate(test_eval_global_idxs):
            tc = t_cutoffs_tau[i]
            st = 0 if self.use_all_history else max(0, tc - self.w_steps)
            raw_seq = self.X[g_idx, st:tc, :49]
            feat_seq = compute_volatility_features(raw_seq) if self.use_volatility else raw_seq

            if use_v and self.v_vals_all is not None:
                v_seq = self.v_vals_all[g_idx, st:tc]
                x_test_tau.append(np.concatenate([feat_seq, v_seq], axis=-1))
            else:
                x_test_tau.append(feat_seq)

        if self.use_norm:
            _, x_test_tau = normalize_features(x_train, x_test_tau)

        dataset = EPSepsisDataset(x_test_tau, y_test_tau, input_dim)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ep_batch)

    def train_dataloader(self) -> DataLoader:
        batch_size = 64
        if (
            self.cfg
            and hasattr(self.cfg, "agent")
            and hasattr(self.cfg.agent, "get")
            and self.cfg.agent.get("batch_size")
        ):
            batch_size = int(self.cfg.agent.get("batch_size"))
        loader, _, _, _ = self.get_train_dataloader(split_idx=0, batch_size=batch_size)
        return loader

    def val_dataloader(self) -> DataLoader | None:
        batch_size = 64
        if (
            self.cfg
            and hasattr(self.cfg, "agent")
            and hasattr(self.cfg.agent, "get")
            and self.cfg.agent.get("batch_size")
        ):
            batch_size = int(self.cfg.agent.get("batch_size"))
        tr_idxs, _ = self.get_split_indices(0)
        seqs, input_dim = self.get_training_sequences()
        x_train = [seqs[i] for i in tr_idxs]
        return self.get_eval_dataloader(
            split_idx=0, tau=self.tau_train, x_train=x_train, input_dim=input_dim, batch_size=batch_size
        )
