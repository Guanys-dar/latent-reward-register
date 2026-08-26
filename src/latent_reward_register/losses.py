from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class GroupLossOutput:
    """Loss plus the diagnostics the training loop logs."""

    loss: torch.Tensor
    pair_count: torch.Tensor
    pair_accuracy: torch.Tensor


def _as_item_sigmas(sigmas: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Broadcast per-sample or per-group sigmas to one value per item."""
    if sigmas.ndim == 0:
        return sigmas.reshape(1, 1).expand_as(predictions)
    if sigmas.ndim == 1:
        return sigmas.reshape(-1, 1).expand_as(predictions)
    item_sigmas = sigmas
    while item_sigmas.ndim > 2:
        item_sigmas = item_sigmas.squeeze(-1)
    if item_sigmas.shape == predictions.shape:
        return item_sigmas
    if item_sigmas.shape[0] == predictions.shape[0] and item_sigmas.shape[1] == 1:
        return item_sigmas.expand_as(predictions)
    raise ValueError(f"Cannot broadcast sigmas {tuple(sigmas.shape)} to {tuple(predictions.shape)}")


def dina_thurstone_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    group_mask: torch.Tensor | None = None,
    min_target_gap: float = 0.0,
    variance_scale: float = 2.0,
    min_variance: float = 0.05,
    epsilon: float = 1e-6,
) -> GroupLossOutput:
    """Thurstone preference loss over every ordered pair inside each group.

    ``predictions`` and ``targets`` are ``(batch, group_size)``. Groups are the
    unit of the loss: all within-group pairs contribute, each group is
    normalized by its own pair count, and only then are groups averaged. A
    flat list of pairs is therefore not an equivalent input.

    Pair noise grows with the diffusion sigma, so a disagreement at high noise
    is penalized less than the same disagreement at low noise.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must match, got {predictions.shape} != {targets.shape}")
    if predictions.ndim != 2:
        raise ValueError(f"Expected (batch, group_size) predictions, got {tuple(predictions.shape)}")

    predictions = predictions.float()
    targets = targets.float()

    if group_mask is None:
        group_mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        group_mask = group_mask.to(dtype=torch.bool)

    group_size = predictions.shape[1]
    upper = torch.triu(
        torch.ones((group_size, group_size), dtype=torch.bool, device=predictions.device), diagonal=1
    ).unsqueeze(0)

    target_difference = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    prediction_difference = predictions.unsqueeze(-1) - predictions.unsqueeze(-2)

    # A pair counts only if both items are real and the teacher separates them.
    pair_mask = upper & group_mask.unsqueeze(-1) & group_mask.unsqueeze(-2)
    pair_mask = pair_mask & (target_difference.abs() > min_target_gap)

    pair_count = pair_mask.sum()
    if int(pair_count) == 0:
        zero = predictions.new_zeros(())
        return GroupLossOutput(loss=zero, pair_count=zero, pair_accuracy=zero)

    item_variance = variance_scale * _as_item_sigmas(
        sigmas.to(device=predictions.device, dtype=predictions.dtype), predictions
    ).square() + min_variance
    pair_variance = item_variance.unsqueeze(-1) + item_variance.unsqueeze(-2)

    direction = target_difference.sign()
    normalized_margin = direction * prediction_difference / torch.sqrt(pair_variance + epsilon)
    preference_probability = torch.distributions.Normal(0, 1).cdf(normalized_margin)
    # sqrt of the probability, matching the research implementation.
    pair_loss = 1.0 - torch.sqrt(preference_probability.clamp_min(epsilon))

    pairs_per_group = pair_mask.sum(dim=(-1, -2))
    valid = pairs_per_group > 0
    per_group_loss = (pair_loss * pair_mask).sum(dim=(-1, -2)) / pairs_per_group.clamp_min(1)
    loss = per_group_loss.masked_select(valid).mean()

    correct = ((prediction_difference * direction) > 0).to(dtype=predictions.dtype)
    per_group_accuracy = (correct * pair_mask).sum(dim=(-1, -2)) / pairs_per_group.clamp_min(1)
    accuracy = per_group_accuracy.masked_select(valid).mean()

    return GroupLossOutput(
        loss=loss,
        pair_count=pair_count.to(dtype=predictions.dtype),
        pair_accuracy=accuracy,
    )


def multihead_group_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    sigmas: torch.Tensor,
    head_weights: Mapping[str, float] | None = None,
    group_mask: torch.Tensor | None = None,
    min_target_gap: float = 0.0,
) -> tuple[torch.Tensor, dict[str, GroupLossOutput]]:
    """Weighted mean of the per-head group losses.

    Every head runs the same objective on its own teacher scores; SD3 exp11
    trains three heads at equal weight.
    """
    if predictions.keys() != targets.keys():
        raise ValueError("Predictions and targets must cover identical reward heads")
    if not predictions:
        raise ValueError("At least one reward head is required")

    head_weights = head_weights or {}
    per_head = {
        name: dina_thurstone_loss(
            predictions[name],
            targets[name],
            sigmas=sigmas,
            group_mask=group_mask,
            min_target_gap=min_target_gap,
        )
        for name in predictions
    }
    total_weight = sum(float(head_weights.get(name, 1.0)) for name in per_head)
    if total_weight <= 0:
        raise ValueError("At least one reward head must have positive weight")
    total = sum(per_head[name].loss * float(head_weights.get(name, 1.0)) for name in per_head) / total_weight
    return total, per_head
