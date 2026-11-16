"""Pure BlendRL routing summaries for telemetry and offline analysis.

The caller controls sampling and logging; all tensors are detached before use.
"""
from __future__ import annotations

import math

import torch
from .monitor_catalog import expert_names


def routing_metrics(weights: torch.Tensor, module_types: list[str], *, threshold: float = 0.9,
                    expert_probs: torch.Tensor | None = None) -> dict[str, float]:
    """Summarize a batch of actual routing weights, without assuming two experts.

    ``expert_probs`` is optional, with shape (batch, modules, actions). Agreement
    is a per-pair top-action rate and Jensen-Shannon divergence in nats.
    """
    if not 0.5 < threshold <= 1:
        raise ValueError('threshold must be in (0.5, 1]')
    w = weights.detach().float()
    if w.ndim != 2 or not w.shape[0] or w.shape[1] != len(module_types) or not module_types:
        raise ValueError('weights must have shape (nonempty batch, number of modules)')
    if not torch.isfinite(w).all() or (w < 0).any() or not torch.allclose(w.sum(-1), torch.ones_like(w[:, 0]), atol=1e-4):
        raise ValueError('weights must be finite probability distributions')
    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)
    n = w.shape[1]
    metrics = {'blend/entropy': entropy.mean().item(),
               'blend/entropy_normalized': (entropy.mean() / math.log(n)).item() if n > 1 else 0.0,
               'blend/mixed_share': (w.max(-1).values < threshold).float().mean().item()}
    winners = w.argmax(-1)
    names = expert_names(module_types)
    for i, name in enumerate(names):
        values = w[:, i]
        prefix = f'blend/{name}'
        metrics.update({f'{prefix}/weight_mean': values.mean().item(),
                        f'{prefix}/weight_std': values.std(unbiased=False).item(),
                        f'{prefix}/dominant_share': (winners == i).float().mean().item(),
                        f'{prefix}/exclusive_share': (values >= threshold).float().mean().item()})
    for kind in ('neural', 'logic'):
        indices = [i for i, t in enumerate(module_types) if t == kind]
        if indices:
            metrics[f'blend/{kind}/weight_mean'] = w[:, indices].sum(-1).mean().item()
    if expert_probs is not None:
        p = expert_probs.detach().float()
        if p.ndim != 3 or p.shape[:2] != w.shape or p.shape[2] == 0 or not torch.isfinite(p).all() or (p < 0).any() or not torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-4):
            raise ValueError('expert_probs must have shape (batch, modules, actions) and contain probabilities')
        for i in range(n):
            for j in range(i + 1, n):
                a, b = p[:, i], p[:, j]
                m = (a + b) / 2
                kl_a = (a * (a.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                kl_b = (b * (b.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                prefix = f'blend/agreement/{names[i]}__{names[j]}'
                metrics[f'{prefix}/top_action'] = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
                metrics[f'{prefix}/js_divergence'] = ((kl_a + kl_b) / 2).mean().item()
    return metrics
