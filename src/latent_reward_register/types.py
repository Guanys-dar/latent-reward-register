from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class RegisterCondition:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisterOutput:
    scores: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class RewardGradientOutput:
    scores: Mapping[str, torch.Tensor]
    gradients: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class GuidanceDiagnostics:
    base_delta_rms: torch.Tensor
    reward_delta_rms: torch.Tensor
    gradient_rms: torch.Tensor
    applied: torch.Tensor
    clipped: torch.Tensor

