from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .types import GuidanceDiagnostics


def rms(value: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, value.ndim))
    return value.float().square().mean(dim=dimensions).sqrt()


def _expand_batch(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], *([1] * (target.ndim - 1))).to(device=target.device, dtype=target.dtype)


def unit_rms(value: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    return value / _expand_batch(rms(value).clamp_min(epsilon), value)


@dataclass(frozen=True)
class GuidanceSchedule:
    bands: tuple[tuple[float, float], ...]

    def __post_init__(self):
        thresholds = [threshold for threshold, _ in self.bands]
        if thresholds != sorted(set(thresholds), reverse=True):
            raise ValueError("Guidance thresholds must be strictly ordered high to low")

    def at(self, sigma: float) -> float:
        for threshold, scale in self.bands:
            if sigma > threshold:
                return scale
        return 0.0

    @classmethod
    def from_pairs(cls, pairs: Sequence[Sequence[float]]) -> "GuidanceSchedule":
        return cls(tuple((float(threshold), float(scale)) for threshold, scale in pairs))


class RewardGradientGuidance:
    def __init__(self, *, epsilon: float = 1e-12, min_gradient_rms: float = 1e-6, max_scale_factor: float = 2.0):
        self.epsilon = epsilon
        self.min_gradient_rms = min_gradient_rms
        self.max_scale_factor = max_scale_factor

    def combine(self, gradients: Mapping[str, torch.Tensor], weights: Mapping[str, float] | None = None) -> torch.Tensor:
        if not gradients:
            raise ValueError("At least one reward gradient is required")
        weights = weights or {}
        combined = None
        for name, gradient in gradients.items():
            contribution = unit_rms(gradient, self.epsilon) * float(weights.get(name, 1.0))
            combined = contribution if combined is None else combined + contribution
        return combined

    def correction(
        self,
        *,
        latents: torch.Tensor,
        base_next: torch.Tensor,
        gradient: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, GuidanceDiagnostics]:
        base_delta = base_next - latents
        base_rms = rms(base_delta).clamp_min(self.epsilon)
        gradient_rms = rms(gradient)
        applied = gradient_rms > self.min_gradient_rms
        requested = float(scale) * base_rms
        correction = unit_rms(gradient, self.epsilon) * _expand_batch(requested, gradient)
        correction = correction * _expand_batch(applied, correction)
        correction_rms = rms(correction)
        maximum = self.max_scale_factor * float(scale) * base_rms
        clip_factor = torch.minimum(torch.ones_like(correction_rms), maximum / correction_rms.clamp_min(self.epsilon))
        clipped = clip_factor < 1.0
        correction = correction * _expand_batch(clip_factor, correction)
        diagnostics = GuidanceDiagnostics(
            base_delta_rms=base_rms.detach(),
            reward_delta_rms=rms(correction).detach(),
            gradient_rms=gradient_rms.detach(),
            applied=applied.detach(),
            clipped=clipped.detach(),
        )
        return correction, diagnostics

    def guided_step(self, **kwargs) -> tuple[torch.Tensor, GuidanceDiagnostics]:
        correction, diagnostics = self.correction(**kwargs)
        return kwargs["base_next"] + correction, diagnostics
