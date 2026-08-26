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

    def expand_groups(self, group_size: int) -> "RegisterCondition":
        """Repeat each prompt ``group_size`` times to match flattened group latents.

        One prompt conditions every image in its group, so the conditioning is
        repeated rather than reshaped.
        """
        if group_size < 1:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if group_size == 1:
            return self

        def _expand(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.repeat_interleave(group_size, dim=0)

        return RegisterCondition(
            prompt_embeds=_expand(self.prompt_embeds),
            pooled_prompt_embeds=_expand(self.pooled_prompt_embeds),
            attention_mask=_expand(self.attention_mask),
            metadata=self.metadata,
        )


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

