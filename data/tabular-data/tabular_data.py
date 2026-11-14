"""Tabular Data Module for Supervised Learning and Tabular RL."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.app.api.data import register_datamodule
from src.app.api.data import BaseDataModule

log = logging.getLogger(__name__)


class TabularDataset(Dataset):
    """PyTorch Dataset wrapping tabular feature arrays and targets."""

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.features = torch.tensor(features, dtype=dtype)
        self.targets = torch.tensor(targets, dtype=dtype) if targets is not None else None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.targets is not None:
            return {
                "obs": self.features[idx],
                "target": self.targets[idx],
                "y": self.targets[idx],
            }
        return {"obs": self.features[idx]}


@register_datamodule("tabular_data", "TabularDataModule")
class TabularDataModule(L.LightningDataModule, BaseDataModule):
    """General tabular dataset loader supporting CSV, NPZ, and NPY files."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        return ["supervised", "offline_rl", "unsupervised"]

    def __init__(
        self,
        cfg: Any = None,
        batch_size: int = 128,
        val_split: float = 0.2,
        test_split: float = 0.1,
        normalize: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        self.train_dataset: Optional[TabularDataset] = None
        self.val_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None

    def setup(self, stage: Optional[str] = None, cfg: Any = None) -> None:
        if cfg is not None:
            self.cfg = cfg

        dataset_path = None
        if self.cfg is not None:
            dataset_path = getattr(self.cfg, "dataset_path", None)
            if dataset_path is None and hasattr(self.cfg, "get"):
                dataset_path = self.cfg.get("dataset_path")

        if not dataset_path:
            # Generate synthetic dummy tabular data for pipeline dry-runs
            log.info("No dataset_path provided; generating synthetic tabular data.")
            features = np.random.randn(200, 16).astype(np.float32)
            targets = (features[:, 0] + features[:, 1] > 0).astype(np.float32)
        else:
            p = Path(dataset_path)
            if not p.exists():
                log.warning("Tabular dataset %s not found. Using fallback mock data.", p)
                features = np.random.randn(200, 16).astype(np.float32)
                targets = (features[:, 0] + features[:, 1] > 0).astype(np.float32)
            elif p.suffix == ".npz":
                data = np.load(p)
                features = data.get("obs") if "obs" in data else data.get("X")
                if features is None:
                    features = data[list(data.keys())[0]]
                targets = data.get("targets") if "targets" in data else data.get("y", None)
            elif p.suffix == ".npy":
                features = np.load(p)
                targets = None
            elif p.suffix == ".csv":
                raw = np.genfromtxt(p, delimiter=",", skip_header=1)
                features = raw[:, :-1].astype(np.float32)
                targets = raw[:, -1].astype(np.float32)
            else:
                raise ValueError(f"Unsupported file format for tabular data: {p.suffix}")

        n = len(features)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train = features[train_idx]
        y_train = targets[train_idx] if targets is not None else None

        X_val = features[val_idx] if len(val_idx) > 0 else None
        y_val = targets[val_idx] if (targets is not None and len(val_idx) > 0) else None

        X_test = features[test_idx] if len(test_idx) > 0 else None
        y_test = targets[test_idx] if (targets is not None and len(test_idx) > 0) else None

        if self.normalize:
            self.mean = np.mean(X_train, axis=0, keepdims=True)
            self.std = np.std(X_train, axis=0, keepdims=True) + 1e-8
            X_train = (X_train - self.mean) / self.std
            if X_val is not None:
                X_val = (X_val - self.mean) / self.std
            if X_test is not None:
                X_test = (X_test - self.mean) / self.std

        self.train_dataset = TabularDataset(X_train, y_train)
        if X_val is not None and len(X_val) > 0:
            self.val_dataset = TabularDataset(X_val, y_val)
        if X_test is not None and len(X_test) > 0:
            self.test_dataset = TabularDataset(X_test, y_test)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(range(min(self.batch_size, 10)), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(range(min(self.batch_size, 2)), batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(range(min(self.batch_size, 2)), batch_size=self.batch_size)
