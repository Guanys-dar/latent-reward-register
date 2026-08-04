from __future__ import annotations

from typing import Mapping

import torch


def thurstone_pairwise_loss(
    preferred: torch.Tensor,
    rejected: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    variance_scale: float = 2.0,
    min_variance: float = 0.05,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    item_variance = variance_scale * sigmas.float().square() + min_variance
    normalized_margin = (preferred.float() - rejected.float()) / torch.sqrt(2.0 * item_variance + epsilon)
    preference_probability = torch.distributions.Normal(0, 1).cdf(normalized_margin)
    return (1.0 - torch.sqrt(preference_probability.clamp_min(epsilon))).mean()


def multihead_pairwise_loss(
    preferred: Mapping[str, torch.Tensor],
    rejected: Mapping[str, torch.Tensor],
    sigmas: torch.Tensor,
    weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if preferred.keys() != rejected.keys():
        raise ValueError("Preferred and rejected scores must have identical reward heads")
    weights = weights or {}
    per_head = {
        name: thurstone_pairwise_loss(preferred[name], rejected[name], sigmas) for name in preferred
    }
    total_weight = sum(float(weights.get(name, 1.0)) for name in per_head)
    if total_weight <= 0:
        raise ValueError("At least one reward head must have positive weight")
    total = sum(per_head[name] * float(weights.get(name, 1.0)) for name in per_head) / total_weight
    return total, per_head
