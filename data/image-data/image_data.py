"""General-purpose Visual Observation Data Module.

Provides frame stacking, grayscale conversion, spatial resizing, channel transposition,
and tensor normalization for any image-based Gymnasium environment or offline dataset.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import gymnasium as gym
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.app.api.data import register_datamodule
from src.app.api.data import BaseDataModule

log = logging.getLogger(__name__)


class ImageObservationDataset(Dataset):
    """Dataset of stored or streamed observation frames."""

    def __init__(self, frames: np.ndarray | torch.Tensor, targets: Optional[np.ndarray | torch.Tensor] = None):
        self.frames = torch.as_tensor(frames, dtype=torch.float32)
        if self.frames.ndim == 4 and self.frames.shape[-1] in (1, 3, 4):
            # Convert (N, H, W, C) -> (N, C, H, W)
            self.frames = self.frames.permute(0, 3, 1, 2)
        # Normalize to [0.0, 1.0] if pixel values in [0, 255]
        if self.frames.max() > 1.0:
            self.frames = self.frames / 255.0
        self.targets = torch.as_tensor(targets, dtype=torch.long) if targets is not None else None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        frame = self.frames[idx]
        if self.targets is not None:
            return frame, self.targets[idx]
        return frame, 0


@register_datamodule("image_data", "ImageDataModule", "atari_image_data")
class ImageDataModule(L.LightningDataModule, BaseDataModule):
    """General-purpose DataModule for image observations (Atari, DOOM, Procgen, Crafter, Robotics)."""


    def __init__(
        self,
        cfg: Any = None,
        frame_stack: int = 4,
        grayscale: bool = True,
        image_size: Tuple[int, int] = (84, 84),
        clip_rewards: bool = False,
        batch_size: int = 64,
        normalize: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        self.image_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (84, 84)
        self.clip_rewards = clip_rewards
        self.batch_size = batch_size
        self.normalize = normalize

        if cfg is not None:
            dm_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else getattr(cfg, "data", {})
            if isinstance(dm_cfg, dict):
                self.frame_stack = dm_cfg.get("frame_stack", self.frame_stack)
                self.grayscale = dm_cfg.get("grayscale", self.grayscale)
                self.batch_size = dm_cfg.get("batch_size", self.batch_size)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def wrap_env(self, env: gym.Env) -> gym.Env:
        """Apply the generic visual observation wrapper stack to a Gymnasium environment."""
        if hasattr(gym.wrappers, "ResizeObservation"):
            env = gym.wrappers.ResizeObservation(env, self.image_size)
        if self.grayscale and hasattr(gym.wrappers, "GrayscaleObservation"):
            env = gym.wrappers.GrayscaleObservation(env)
        if self.frame_stack > 1 and hasattr(gym.wrappers, "FrameStackObservation"):
            env = gym.wrappers.FrameStackObservation(env, self.frame_stack)
        if hasattr(gym.wrappers, "RecordEpisodeStatistics"):
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        dummy = torch.zeros(10, self.frame_stack, *self.image_size)
        return DataLoader(ImageObservationDataset(dummy), batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        dummy = torch.zeros(2, self.frame_stack, *self.image_size)
        return DataLoader(ImageObservationDataset(dummy), batch_size=self.batch_size)
