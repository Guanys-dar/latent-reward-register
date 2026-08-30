"""On-policy rollout driver for RG-OPD.

The student generates its own trajectory, the teacher labels each visited state
with a guided target, and the student regresses onto those targets. Two details
carry the method:

1. **On-policy.** Targets are built at states the *student* visited, so the
   rollout must run with the student's own weights, not a frozen generator's.
2. **Guidance is sparse.** The schedule turns guidance off in the low-noise
   tail, and those steps skip the register backward entirely. That skip is the
   reported efficiency, so it is measured here rather than assumed.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from .rgopd import rgopd_loss
from .teacher import RewardGradientTeacher
from .types import RegisterCondition


class StudentPolicy(Protocol):
    """One denoising transition, differentiable with respect to student weights.

    Returns ``(next_mean, transition_std)``: the predicted next state and the
    per-sample noise scale that turns the regression into the step's KL.
    """

    def __call__(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class RolloutTrace:
    """What the rollout did, for cost reporting and ablations."""

    steps: int = 0
    guided_steps: int = 0
    scales: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    @property
    def guidance_fraction(self) -> float:
        return self.guided_steps / self.steps if self.steps else 0.0

    @property
    def mean_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0


@dataclass(frozen=True)
class RolloutConfig:
    """Ten-step rollout optimizing the first nine steps, per the paper presets."""

    sigmas: tuple[float, ...]
    optimized_steps: int | None = None

    def __post_init__(self):
        if len(self.sigmas) < 2:
            raise ValueError("A rollout needs at least two sigma values")
        if self.optimized_steps is not None and self.optimized_steps < 1:
            raise ValueError("optimized_steps must be positive")

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1

    def optimizes(self, step_index: int) -> bool:
        limit = self.optimized_steps if self.optimized_steps is not None else self.steps
        return step_index < limit


@torch.no_grad()
def rollout_trajectory(
    *,
    student: StudentPolicy,
    latents: torch.Tensor,
    condition: RegisterCondition,
    config: RolloutConfig,
) -> list[torch.Tensor]:
    """States the student visits, ``steps + 1`` tensors starting from ``latents``.

    Runs under ``no_grad``: these states are where targets get built, and the
    gradient is taken later on a re-run of each transition.
    """
    states = [latents]
    current = latents
    for sigma_value, next_sigma_value in zip(config.sigmas[:-1], config.sigmas[1:]):
        sigma = torch.full((current.shape[0],), sigma_value, device=current.device, dtype=torch.float32)
        next_sigma = torch.full_like(sigma, next_sigma_value)
        current, _ = student(current, condition, sigma, next_sigma)
        states.append(current)
    return states


def train_rgopd_rollout(
    *,
    student: StudentPolicy,
    teacher: RewardGradientTeacher,
    reference_step: Callable[..., torch.Tensor],
    initial_latents: Sequence[torch.Tensor],
    condition: RegisterCondition,
    config: RolloutConfig,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float = 1.0,
    parameters: Sequence[torch.nn.Parameter] | None = None,
) -> RolloutTrace:
    """Train a student against guided targets along its own trajectories.

    ``reference_step`` is the frozen anchor transition. The teacher adds the
    reward delta to it; the student regresses onto that sum.
    """
    if parameters is None:
        raise ValueError("parameters is required: pass the student's trainable parameters")
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("RG-OPD student has no trainable parameters")

    trace = RolloutTrace()
    for latents in initial_latents:
        states = rollout_trajectory(
            student=student, latents=latents, condition=condition, config=config
        )

        for index, (sigma_value, next_sigma_value) in enumerate(
            zip(config.sigmas[:-1], config.sigmas[1:])
        ):
            if not config.optimizes(index):
                continue
            state = states[index].detach()
            sigma = torch.full((state.shape[0],), sigma_value, device=state.device, dtype=torch.float32)
            next_sigma = torch.full_like(sigma, next_sigma_value)

            with torch.no_grad():
                anchor_next = reference_step(state, condition, sigma, next_sigma)

            step = teacher.guided_step(
                latents=state,
                base_next=anchor_next,
                condition=condition,
                timesteps=sigma,
                sigma=sigma_value,
            )

            student_next, transition_std = student(state, condition, sigma, next_sigma)
            loss = rgopd_loss(student_next, step.guided_next, transition_std)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            trace.steps += 1
            trace.guided_steps += int(step.applied)
            trace.scales.append(step.scale)
            trace.losses.append(float(loss.detach()))

    if trace.steps == 0:
        raise ValueError("RG-OPD rollout optimized no steps: check sigmas and optimized_steps")
    return trace
