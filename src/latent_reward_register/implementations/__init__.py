"""Checkpoint-faithful backbone implementations consolidated from the research workspaces."""

from .gradmode import latent_gradient_enabled, latent_gradient_is_enabled
from .loader import CheckpointRewardRegister, load_legacy_register

__all__ = [
    "CheckpointRewardRegister",
    "latent_gradient_enabled",
    "latent_gradient_is_enabled",
    "load_legacy_register",
]
