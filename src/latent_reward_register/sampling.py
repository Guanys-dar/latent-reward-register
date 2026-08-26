from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch

from .guidance import GuidanceSchedule, RewardGradientGuidance
from .teacher import RewardGradientTeacher
from .types import RegisterCondition


@dataclass(frozen=True)
class SamplingTrace:
    """Per-step record of what guidance did, for cost and ablation reporting."""

    steps: int
    guided_steps: int
    scales: tuple[float, ...]

    @property
    def guidance_fraction(self) -> float:
        return self.guided_steps / self.steps if self.steps else 0.0


def reward_guided_sample(
    *,
    register,
    latents: torch.Tensor,
    condition: RegisterCondition,
    sigmas: Iterable[float],
    heads: Sequence[str],
    schedule: GuidanceSchedule,
    guidance: RewardGradientGuidance | None = None,
    head_weights: Mapping[str, float] | None = None,
    step_kwargs: dict | None = None,
    reference_step=None,
    return_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, SamplingTrace]:
    """Sample with reward-gradient guidance applied on scheduled steps.

    Guidance runs through the same ``RewardGradientTeacher`` that produces
    RG-OPD targets, so a distilled student sees the guidance this loop applies.

    ``reference_step(latents, condition, sigma, next_sigma, **step_kwargs)``
    supplies the frozen generator transition. It defaults to
    ``register.adapter.reference_step`` for adapter-backed registers.
    """
    sigma_values = tuple(float(value) for value in sigmas)
    if len(sigma_values) < 2:
        raise ValueError("Sampling requires at least two sigma values")

    if reference_step is None:
        adapter = getattr(register, "adapter", None)
        reference_step = getattr(adapter, "reference_step", None)
        if reference_step is None:
            raise ValueError(
                "reward_guided_sample needs a reference_step: this register has no adapter "
                "providing one, so pass the frozen generator transition explicitly"
            )

    teacher = RewardGradientTeacher(
        register,
        schedule=schedule,
        heads=tuple(heads),
        guidance=guidance,
        head_weights=head_weights,
    )

    current = latents
    scales: list[float] = []
    guided = 0
    for sigma_value, next_sigma_value in zip(sigma_values[:-1], sigma_values[1:]):
        sigma = torch.full((current.shape[0],), sigma_value, device=current.device, dtype=torch.float32)
        next_sigma = torch.full_like(sigma, next_sigma_value)
        base_next = reference_step(current, condition, sigma, next_sigma, **(step_kwargs or {}))

        step = teacher.guided_step(
            latents=current,
            base_next=base_next,
            condition=condition,
            timesteps=sigma,
            sigma=sigma_value,
        )
        current = step.guided_next
        scales.append(step.scale)
        guided += int(step.applied)

    if return_trace:
        return current, SamplingTrace(len(scales), guided, tuple(scales))
    return current
