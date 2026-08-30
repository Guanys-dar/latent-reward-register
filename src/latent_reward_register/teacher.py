"""The reward-gradient teacher shared by RG-OPD and reward-guided sampling.

Both consume the same quantity: a guided next-state ``mu_base + reward_delta``,
where ``reward_delta`` points along ``d reward / d latent`` and is
magnitude-matched to a fraction of the base sampler step. RG-OPD regresses a
student onto that state; RGS commits it directly.

Keeping one implementation means the distilled student is trained against
exactly the guidance the sampler applies. Two copies would drift.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from .guidance import GuidanceSchedule, RewardGradientGuidance
from .types import GuidanceDiagnostics, RegisterCondition


class ScoringRegister(Protocol):
    """The register interface the teacher needs: gradients of reward w.r.t. latents."""

    def score_and_grad(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        timesteps: torch.Tensor,
        *,
        heads: tuple[str, ...] | None = ...,
    ): ...


@dataclass(frozen=True)
class TeacherStep:
    """One guided step: the target state plus what guidance did to produce it."""

    guided_next: torch.Tensor
    reward_delta: torch.Tensor
    scale: float
    applied: bool
    diagnostics: GuidanceDiagnostics | None = None


class RewardGradientTeacher:
    """Turns a register into guided next-states.

    ``schedule`` gates guidance by noise level: the sigma-banded schedule is the
    one the paper ships. Steps where it resolves to zero skip the register
    backward entirely, which is most of the trajectory for an early-only
    schedule and is where the reported efficiency comes from.
    """

    def __init__(
        self,
        register: ScoringRegister,
        *,
        schedule: GuidanceSchedule,
        heads: Sequence[str],
        guidance: RewardGradientGuidance | None = None,
        head_weights: Mapping[str, float] | None = None,
    ):
        if not heads:
            raise ValueError("At least one reward head is required")
        self.register = register
        self.schedule = schedule
        self.heads = tuple(heads)
        self.guidance = guidance or RewardGradientGuidance()
        self.head_weights = dict(head_weights or {})

    def scale_at(self, sigma: float) -> float:
        """Guidance strength at this noise level; 0.0 means skip the backward."""
        return self.schedule.at(float(sigma))

    def is_active(self, sigma: float) -> bool:
        return self.scale_at(sigma) > 0.0

    def reward_gradient(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Combined ``d reward / d latent`` over the configured heads.

        Per-head gradients are unit-RMS normalized before weighting, so no head
        dominates through scale alone.
        """
        convert = getattr(self.register, "timesteps_from_sigma", None)
        register_timesteps = convert(timesteps) if convert is not None else timesteps
        output = self.register.score_and_grad(latents, condition, register_timesteps, heads=self.heads)
        return self.guidance.combine(output.gradients, self.head_weights)

    def guided_step(
        self,
        *,
        latents: torch.Tensor,
        base_next: torch.Tensor,
        condition: RegisterCondition,
        timesteps: torch.Tensor,
        sigma: float,
        gradient: torch.Tensor | None = None,
    ) -> TeacherStep:
        """Guided next-state at this step, or the base step when guidance is off."""
        scale = self.scale_at(sigma)
        if scale == 0.0:
            return TeacherStep(
                guided_next=base_next,
                reward_delta=torch.zeros_like(base_next),
                scale=0.0,
                applied=False,
            )
        if gradient is None:
            gradient = self.reward_gradient(latents, condition, timesteps)
        delta, diagnostics = self.guidance.correction(
            latents=latents,
            base_next=base_next,
            gradient=gradient,
            scale=scale,
        )
        return TeacherStep(
            guided_next=(base_next + delta).detach(),
            reward_delta=delta.detach(),
            scale=scale,
            applied=True,
            diagnostics=diagnostics,
        )
