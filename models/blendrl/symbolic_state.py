"""Convert observations to symbolic object tensors; independent of policy models."""
from __future__ import annotations

import importlib
import torch


class SymbolicStateLoader:
    """Preserve batch/device/gradients while preparing [batch, objects, features]."""

    def __init__(self, object_count=1, feature_count=None, builder=None):
        if object_count is not None and (not isinstance(object_count, int) or isinstance(object_count, bool) or object_count < 1):
            raise ValueError("object_count must be a positive integer.")
        if feature_count is not None and (not isinstance(feature_count, int) or isinstance(feature_count, bool) or feature_count < 1):
            raise ValueError("feature_count must be a positive integer or null.")
        self.object_count = object_count
        self.feature_count = feature_count
        if isinstance(builder, str):
            module, separator, function = builder.rpartition(":")
            if not separator or not module or not function:
                raise ValueError("A symbolic observation builder must use module:function format.")
            if module.startswith("package://"):
                from src.app.core.provider_loader import provider_object
                package, _, relative = module[10:].partition("/")
                builder = provider_object(package, relative + ":" + function)
            else:
                builder = getattr(importlib.import_module(module), function)
        if builder is not None and not callable(builder):
            raise ValueError("The symbolic observation builder must be callable.")
        self.builder = builder

    def __call__(self, observation):
        if not torch.is_tensor(observation) or observation.ndim < 2:
            raise ValueError("Symbolic state inputs must be batched tensors.")
        state = self.builder(observation) if self.builder else observation
        if not self.builder and state.ndim == 2:
            state = state.unsqueeze(1).expand(-1, self.object_count or 1, -1)
        if not torch.is_tensor(state) or state.ndim != 3 or state.shape[0] != observation.shape[0]:
            raise ValueError("A symbolic state loader must return [batch, objects, features] with the same batch size.")
        if self.object_count is not None and state.shape[1] != self.object_count:
            raise ValueError(f"Expected {self.object_count} symbolic objects, got {state.shape[1]}.")
        if self.feature_count is not None and state.shape[2] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} symbolic features, got {state.shape[2]}.")
        if state.device != observation.device:
            raise ValueError("A symbolic state loader must preserve the observation device.")
        return state
