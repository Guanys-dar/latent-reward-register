from __future__ import annotations

from collections.abc import Iterable

import torch

from .guidance import GuidanceSchedule, RewardGradientGuidance
from .register import RewardRegister
from .types import RegisterCondition


def reward_guided_sample(
    *,
    register: RewardRegister,
    latents: torch.Tensor,
    condition: RegisterCondition,
    sigmas: Iterable[float],
    heads: tuple[str, ...],
    schedule: GuidanceSchedule,
    guidance: RewardGradientGuidance,
    head_weights: dict[str, float] | None = None,
    step_kwargs: dict | None = None,
) -> torch.Tensor:
    sigma_values = tuple(float(value) for value in sigmas)
    if len(sigma_values) < 2:
        raise ValueError("Sampling requires at least two sigma values")
    current = latents
    for sigma_value, next_sigma_value in zip(sigma_values[:-1], sigma_values[1:]):
        sigma = torch.full((current.shape[0],), sigma_value, device=current.device, dtype=torch.float32)
        next_sigma = torch.full_like(sigma, next_sigma_value)
        base_next = register.adapter.reference_step(current, condition, sigma, next_sigma, **(step_kwargs or {}))
        scale = schedule.at(sigma_value)
        if scale == 0.0:
            current = base_next
            continue
        output = register.score_and_grad(current, condition, sigma, heads=heads)
        gradient = guidance.combine(output.gradients, head_weights)
        current, _ = guidance.guided_step(
            latents=current,
            base_next=base_next,
            gradient=gradient,
            scale=scale,
        )
    return current

