"""FlowMatch Euler transitions: the frozen anchor step RGS and RG-OPD wrap.

The generator is a flow model: ``dz/dsigma = v(z, sigma)``. One deterministic
Euler advance is

    z_next = z + (sigma_next - sigma) * v(z, sigma)

which is the released solver. The research sampler also implements ab2,
midpoint, heun, and rk4; those were exploratory and cost extra forward passes
per step, so only Euler is released.

This module owns the *arithmetic* and takes the velocity as a callable, so it
carries no dependency on any particular pipeline. A backbone supplies its own
CFG velocity function.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Protocol

import torch

from .types import RegisterCondition

# FlowMatch schedulers index timesteps as sigma * num_train_timesteps.
NUM_TRAIN_TIMESTEPS = 1000.0


class VelocityModel(Protocol):
    """Predicts flow velocity at a state. Classifier-free guidance lives here."""

    def __call__(
        self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor
    ) -> torch.Tensor: ...


def timesteps_for_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Scheduler timestep for a noise level."""
    return sigma * NUM_TRAIN_TIMESTEPS


def euler_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
) -> torch.Tensor:
    """One Euler advance. ``dt`` is negative while denoising."""
    dt = (next_sigma - sigma).reshape(-1, *([1] * (latents.ndim - 1)))
    return (latents.float() + dt.float() * velocity.float()).to(dtype=latents.dtype)


def make_reference_step(
    velocity_model: VelocityModel,
    *,
    context_factory: Callable[[], object] | None = None,
) -> Callable[..., torch.Tensor]:
    """Build the frozen anchor transition used by RGS and RG-OPD.

    The returned callable matches the ``reference_step`` signature both consume,
    and runs under ``no_grad``: the anchor is never trained through.
    """

    def reference_step(
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # The whole advance is inside no_grad, not just the velocity call: the
        # anchor must not carry a graph back to the incoming latent either.
        context = context_factory() if context_factory is not None else nullcontext()
        with torch.no_grad(), context:
            velocity = velocity_model(latents, condition, timesteps_for_sigma(sigma), **kwargs)
            return euler_step(latents, velocity, sigma, next_sigma)

    return reference_step


def make_student_policy(
    velocity_model: VelocityModel, *, noise_level: float = 0.0
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Build the differentiable student transition for RG-OPD.

    Returns ``(next_mean, transition_std)``. ``transition_std`` scales the
    regression into the step's KL; with ``noise_level=0`` the step is
    deterministic and the loss reduces to a plain squared error.

    Unlike the reference step this keeps the graph: the student is what trains.
    """
    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    def student_policy(
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = velocity_model(latents, condition, timesteps_for_sigma(sigma), **kwargs)
        next_mean = euler_step(latents, velocity, sigma, next_sigma)
        if noise_level == 0.0:
            transition_std = torch.ones_like(sigma)
        else:
            # Std grows with the step size, matching the sampler's noise scale.
            transition_std = (noise_level * (sigma - next_sigma).abs()).clamp_min(1e-6)
        return next_mean, transition_std

    return student_policy


def classifier_free_velocity(
    conditional: torch.Tensor, unconditional: torch.Tensor, guidance_scale: float
) -> torch.Tensor:
    """Standard CFG combination of two velocity predictions."""
    return unconditional + guidance_scale * (conditional - unconditional)


# FlowMatchEulerDiscreteScheduler's defaults. sigma_min is not 0: the linspace
# runs between these, and the terminal 0 is appended afterwards.
SIGMA_MAX = 1.0
SIGMA_MIN = 0.0029940120875835419


def sigma_schedule(steps: int, *, shift: float = 3.0) -> tuple[float, ...]:
    """FlowMatch sampling sigmas, matching ``FlowMatchEulerDiscreteScheduler``.

    ``steps`` transitions yield ``steps + 1`` values: ``steps`` linearly spaced
    points from ``SIGMA_MAX`` down to ``SIGMA_MIN``, reparameterized by ``shift``,
    with a terminal 0 appended. Reproducing the scheduler exactly matters because
    the guidance schedule gates on sigma bands, so a different sigma grid moves
    which steps get guided.

    ``shift`` is 3.0 for SD3 and FLUX at 1024x1024; ``shift=1.0`` is unshifted.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if steps == 1:
        linear = [SIGMA_MAX]
    else:
        span = SIGMA_MAX - SIGMA_MIN
        linear = [SIGMA_MAX - span * index / (steps - 1) for index in range(steps)]
    shifted = [shift * value / (1.0 + (shift - 1.0) * value) for value in linear]
    return tuple(shifted) + (0.0,)


def flux_sigma_schedule(steps: int, *, shift: float) -> tuple[float, ...]:
    """FLUX grid: ``linspace(1, 1/steps, steps)`` followed by dynamic shift."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    linear = [1.0 - (1.0 - 1.0 / steps) * index / max(1, steps - 1) for index in range(steps)]
    shifted = [shift * value / (1.0 + (shift - 1.0) * value) for value in linear]
    return tuple(shifted) + (0.0,)
