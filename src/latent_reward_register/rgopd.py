"""RG-OPD targets, loss, and the single-step trainer.

Two trainers exist and they are not interchangeable:

- :func:`latent_reward_register.rollout.train_rgopd_rollout` is **the paper
  path**. It is on-policy: the student generates its own trajectory and targets
  are built at the states it actually visits.
- :func:`train_rgopd` here takes a fixed batch of ``(state, reference_next,
  reward_gradient)`` triples and optimizes one transition each. It is off-policy
  by construction, since whoever produced those states was not this student. It
  is retained for ablations that need to hold the visited states fixed, and it
  is what the asset-free smoke exercises. Reported numbers do not come from it.

Both share :func:`rgopd_loss`, and both take their target from the same
:class:`~latent_reward_register.teacher.RewardGradientTeacher` correction, so
the two cannot disagree about what guidance is.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch

from .guidance import RewardGradientGuidance
from .teacher import RewardGradientTeacher, TeacherStep
from .types import RegisterCondition


@dataclass(frozen=True)
class RGOPDTarget:
    target: torch.Tensor
    reward_delta: torch.Tensor


@dataclass(frozen=True)
class RGOPDBatch:
    latents: torch.Tensor
    sigma: torch.Tensor
    next_sigma: torch.Tensor
    reference_next: torch.Tensor
    reward_gradient: torch.Tensor
    transition_std: torch.Tensor | float


@dataclass(frozen=True)
class RGOPDTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    reward_scale: float = 0.4


@dataclass(frozen=True)
class RGOPDTrainMetrics:
    steps: int
    mean_loss: float


class RGOPDStudent(Protocol):
    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def __call__(self, latents: torch.Tensor, sigma: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor: ...


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


def rollout_target(
    *,
    teacher: RewardGradientTeacher,
    latents: torch.Tensor,
    reference_next: torch.Tensor,
    condition: RegisterCondition,
    timesteps: torch.Tensor,
    sigma: float,
) -> TeacherStep:
    """Regression target for one rollout step, via the shared teacher.

    Using the teacher rather than recomputing the correction here is what keeps
    the distilled student aligned with the guidance the sampler actually
    applies: one implementation, so the two cannot drift.
    """
    return teacher.guided_step(
        latents=latents,
        base_next=reference_next,
        condition=condition,
        timesteps=timesteps,
        sigma=sigma,
    )


def rgopd_loss(student_next: torch.Tensor, target: torch.Tensor, transition_std: torch.Tensor | float) -> torch.Tensor:
    standard_deviation = torch.as_tensor(transition_std, device=student_next.device, dtype=student_next.dtype)
    if standard_deviation.ndim == 1 and student_next.ndim > 1:
        if standard_deviation.shape[0] != student_next.shape[0]:
            raise ValueError("Per-sample transition_std must match the batch dimension")
        standard_deviation = standard_deviation.reshape(-1, *([1] * (student_next.ndim - 1)))
    variance = standard_deviation.square()
    return ((student_next - target).square() / (2.0 * variance.clamp_min(1e-12))).mean()


def train_rgopd(
    *,
    student: RGOPDStudent,
    batches: Iterable[RGOPDBatch],
    config: RGOPDTrainConfig,
    guidance: RewardGradientGuidance,
) -> RGOPDTrainMetrics:
    """Optimize the student on pre-supplied transitions, one step per batch.

    Off-policy: the states come from the caller, not from this student. For the
    paper's on-policy training use
    :func:`latent_reward_register.rollout.train_rgopd_rollout` instead.
    """
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("RG-OPD student has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    student.train()
    loss_total = 0.0
    steps = 0
    for batch in batches:
        target = build_rgopd_target(
            latents=batch.latents,
            reference_next=batch.reference_next,
            gradient=batch.reward_gradient,
            reward_scale=config.reward_scale,
            guidance=guidance,
        )
        student_next = student(batch.latents, batch.sigma, batch.next_sigma)
        loss = rgopd_loss(student_next, target.target, batch.transition_std)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        steps += 1
        loss_total += float(loss.detach().item())
    if steps == 0:
        raise ValueError("RG-OPD training requires at least one batch")
    return RGOPDTrainMetrics(steps=steps, mean_loss=loss_total / steps)
