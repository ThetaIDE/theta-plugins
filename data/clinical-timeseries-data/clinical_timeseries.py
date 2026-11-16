"""Clinical Timeseries Data Module for Healthcare & ICU Benchmarks."""
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


class ClinicalTimeseriesDataset(Dataset):
    """Dataset for windowed clinical timeseries and vital signs."""

    def __init__(
        self,
        sequences: List[np.ndarray] | np.ndarray,
        labels: Optional[List[float] | np.ndarray] = None,
        input_dim: Optional[int] = None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.input_dim = input_dim or (sequences[0].shape[-1] if len(sequences) > 0 else 0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx] if self.labels is not None else 0.0, dtype=torch.float32)
        return seq, label


def collate_clinical_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pads variable-length timeseries sequences into a dense tensor batch with attention mask."""
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


@register_datamodule("clinical_timeseries", "clinical_timeseries_data", "EPSepsisDataModule", "SepsisDataModule")
class ClinicalTimeseriesDataModule(L.LightningDataModule, BaseDataModule):
    """DataModule managing multi-variate ICU/EHR timeseries windowing, imputation, and cohort splits."""


    def __init__(
        self,
        cfg: Any = None,
        batch_size: int = 64,
        window_hours: int = 12,
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
        self.window_hours = window_hours
        self.val_split = val_split
        self.test_split = test_split
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[ClinicalTimeseriesDataset] = None
        self.val_dataset: Optional[ClinicalTimeseriesDataset] = None
        self.test_dataset: Optional[ClinicalTimeseriesDataset] = None

    def setup(self, stage: Optional[str] = None, cfg: Any = None) -> None:
        if cfg is not None:
            self.cfg = cfg

        dataset_path = None
        if self.cfg is not None:
            dataset_path = getattr(self.cfg, "dataset_path", None)
            if dataset_path is None and hasattr(self.cfg, "get"):
                dataset_path = self.cfg.get("dataset_path")

        if not dataset_path or not Path(str(dataset_path)).exists():
            log.info("No valid clinical dataset path found (%s). Initializing synthetic cohort.", dataset_path)
            # Create synthetic clinical trajectories
            n_patients = 100
            sequences = [np.random.randn(np.random.randint(12, 36), 34).astype(np.float32) for _ in range(n_patients)]
            labels = np.random.randint(0, 2, size=n_patients).astype(np.float32)
        else:
            p = Path(str(dataset_path))
            if p.suffix == ".npz":
                raw = np.load(p, allow_pickle=True)
                if "X" in raw and "y" in raw:
                    sequences = list(raw["X"])
                    labels = raw["y"]
                elif "observations" in raw:
                    sequences = list(raw["observations"])
                    labels = raw.get("labels", np.zeros(len(sequences), dtype=np.float32))
                else:
                    k0 = list(raw.keys())[0]
                    sequences = list(raw[k0])
                    labels = np.zeros(len(sequences), dtype=np.float32)
            else:
                sequences = [np.random.randn(24, 34).astype(np.float32) for _ in range(50)]
                labels = np.zeros(50, dtype=np.float32)

        n = len(sequences)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_val - n_test

        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        val_seqs = [sequences[i] for i in val_idx] if len(val_idx) > 0 else []
        val_labels = [labels[i] for i in val_idx] if len(val_idx) > 0 else []

        test_seqs = [sequences[i] for i in test_idx] if len(test_idx) > 0 else []
        test_labels = [labels[i] for i in test_idx] if len(test_idx) > 0 else []

        self.train_dataset = ClinicalTimeseriesDataset(train_seqs, train_labels)
        if val_seqs:
            self.val_dataset = ClinicalTimeseriesDataset(val_seqs, val_labels)
        if test_seqs:
            self.test_dataset = ClinicalTimeseriesDataset(test_seqs, test_labels)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_clinical_batch,
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
                collate_fn=collate_clinical_batch,
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
                collate_fn=collate_clinical_batch,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return DataLoader(range(min(self.batch_size, 2)), batch_size=self.batch_size)
