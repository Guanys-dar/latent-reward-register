from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from latent_reward_register.types import RegisterCondition


@dataclass(frozen=True)
class BackboneFeatures:
    reward_tokens: torch.Tensor
    visual_features: tuple[torch.Tensor, ...]
    text_features: tuple[torch.Tensor, ...]
    timestep_embedding: torch.Tensor


class BackboneAdapter(ABC):
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

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {"backbone": self.name}

