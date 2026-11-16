"""Offline Reinforcement Learning Replay Buffer Data Module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.app.api.data import register_datamodule
from src.app.api.data import BaseDataModule
from src.app.api.data import DatasetReader

log = logging.getLogger(__name__)


class OfflineReplayDataset(Dataset):
    """Wraps DatasetReader to serve transitions as PyTorch tensors."""

    def __init__(self, reader: DatasetReader):
        self.reader = reader

    def __len__(self) -> int:
        return self.reader.limit

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.reader[idx]


@register_datamodule("replay_buffer", "ReplayBufferDataModule", "RLReplayBufferDataModule")
class ReplayBufferDataModule(L.LightningDataModule, BaseDataModule):
    """General transition replay loader and chunk reader for offline reinforcement learning."""


    def __init__(
        self,
        cfg: Any = None,
        batch_size: int = 256,
        val_split: float = 0.1,
        chunk_size: int = 100000,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.val_split = val_split
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.reader: Optional[DatasetReader] = None
        self.val_reader: Optional[DatasetReader] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None, cfg: Any = None) -> None:
        if cfg is not None:
            self.cfg = cfg

        if self.cfg is None:
            return

        dataset_path = getattr(self.cfg, "dataset_path", None)
        if dataset_path is None and hasattr(self.cfg, "get"):
            dataset_path = self.cfg.get("dataset_path")

        if not dataset_path:
            return

        ds_path = Path(dataset_path)
        if not ds_path.exists():
            log.warning("Dataset path %s does not exist", ds_path)
            return

        val_pct = int(self.val_split * 100)
        train_pct = 100 - val_pct

        try:
            self.reader = DatasetReader(
                str(ds_path),
                percentage=train_pct,
                offset=0,
                memory_mapped=False,
            )
            self.train_dataset = OfflineReplayDataset(self.reader)
        except Exception as e:
            log.warning("Failed to initialize train DatasetReader: %s", e)

        if val_pct > 0:
            try:
                self.val_reader = DatasetReader(
                    str(ds_path),
                    percentage=val_pct,
                    offset=train_pct,
                    memory_mapped=False,
                )
                self.val_dataset = OfflineReplayDataset(self.val_reader)
            except Exception as e:
                log.debug("No validation split initialized: %s", e)

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
