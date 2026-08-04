from __future__ import annotations

from dataclasses import dataclass

import torch

from .guidance import RewardGradientGuidance


@dataclass(frozen=True)
class RGOPDTarget:
    target: torch.Tensor
    reward_delta: torch.Tensor


def build_rgopd_target(
    *,
    latents: torch.Tensor,
    reference_next: torch.Tensor,
    gradient: torch.Tensor,
    reward_scale: float,
    guidance: RewardGradientGuidance,
) -> RGOPDTarget:
    reward_delta, _ = guidance.correction(
        latents=latents,
        base_next=reference_next,
        gradient=gradient,
        scale=reward_scale,
    )
    return RGOPDTarget(target=(reference_next + reward_delta).detach(), reward_delta=reward_delta.detach())


def rgopd_loss(student_next: torch.Tensor, target: torch.Tensor, transition_std: torch.Tensor | float) -> torch.Tensor:
    variance = torch.as_tensor(transition_std, device=student_next.device, dtype=student_next.dtype).square()
    return ((student_next - target).square() / (2.0 * variance.clamp_min(1e-12))).mean()

