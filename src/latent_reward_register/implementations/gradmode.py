"""Recovering ``d reward / d latent`` from a frozen-trunk register.

The training forward runs each frozen backbone block inside
``torch.no_grad()``: correct while the register trains, because the trunk is
frozen and the register reads detached snapshots. But it hard-detaches
latents from the reward score, so a plain ``forward()`` produces a reward that
does not require grad with respect to the input latent.

Reward-guided sampling and RG-OPD consume exactly that gradient, so they need
the trunk to run with grad enabled while the trunk *weights* stay frozen.

The research code achieved this by rebinding the per-block helper to a
grad-enabled clone of its body. This module keeps the same math but makes the
choice explicit and reversible: the block bodies consult
``latent_reward_register.implementations.gradmode`` instead of hardcoding
``torch.no_grad()``, and callers opt in with :func:`latent_gradient_enabled`.

Trunk parameters are never unfrozen. Only the latent path is restored.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator

import torch

_state = threading.local()


def latent_gradient_is_enabled() -> bool:
    """True when the frozen trunk should run with autograd recording."""
    return getattr(_state, "enabled", False)


@contextlib.contextmanager
def latent_gradient_enabled(enabled: bool = True) -> Iterator[None]:
    """Run the frozen trunk with grad recording so latent gradients survive.

    Thread-local and re-entrant, so a scoring call cannot leak the mode into
    unrelated work. Requires more activation memory than the training path,
    since trunk activations must be kept for the backward.
    """
    previous = getattr(_state, "enabled", False)
    _state.enabled = bool(enabled)
    try:
        yield
    finally:
        _state.enabled = previous


def frozen_trunk_context():
    """Context for a frozen backbone block body.

    ``torch.no_grad()`` normally, a no-op when latent gradients are requested.
    """
    if latent_gradient_is_enabled():
        return contextlib.nullcontext()
    return torch.no_grad()
