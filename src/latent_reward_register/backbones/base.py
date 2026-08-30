"""The feature-extraction interface behind ``ReferenceRewardRegister``.

Scope warning, because the name suggests more than it delivers: no released
SD3/FLUX/Z-Image backbone implements this. The research models are complete
registers that own their own trunk traversal and are reached through
``backbones.build_register`` — see ``diffusers.py`` and ``registry.py``. The only
implementation in the tree is the synthetic adapter in ``smoke.py``.

It is kept because it is the seam that makes the algorithm layer runnable with no
weights: an adapter is the smallest thing that can stand in for a backbone, which
is what the synthetic smoke test exploits. Treat it as the test seam, not as the
extension point for adding a real backbone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from latent_reward_register.types import RegisterCondition


@dataclass(frozen=True)
class BackboneFeatures:
    reward_tokens: torch.Tensor
    visual_features: tuple[torch.Tensor, ...]
    text_features: tuple[torch.Tensor, ...]
    timestep_embedding: torch.Tensor


class BackboneAdapter(ABC):
    """Exposes a backbone as reward-token features plus a reference transition."""

    name: str

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_features(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        *,
        reward_tokens: torch.Tensor,
        feature_layers: tuple[int, ...],
    ) -> BackboneFeatures:
        raise NotImplementedError

    @abstractmethod
    def reference_step(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
