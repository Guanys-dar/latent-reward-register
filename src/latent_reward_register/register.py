from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from .backbones.base import BackboneAdapter
from .types import RegisterCondition, RegisterOutput, RewardGradientOutput


@dataclass(frozen=True)
class RewardRegisterConfig:
    backbone: str
    head_names: tuple[str, ...]
    feature_layers: tuple[int, ...]
    num_register_tokens: int = 32
    num_attention_heads: int = 8
    pool_factor: int = 4
    hidden_factor: int = 2

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["head_names"] = list(self.head_names)
        payload["feature_layers"] = list(self.feature_layers)
        return payload


class RegisterReadout(nn.Module):
    def __init__(self, hidden_size: int, head_names: tuple[str, ...], hidden_factor: int):
        super().__init__()
        inner_size = hidden_size * hidden_factor
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, inner_size),
                    nn.SiLU(),
                    nn.Linear(inner_size, 1),
                )
                for name in head_names
            }
        )

    def forward(self, reward_features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = reward_features.mean(dim=1)
        return {name: head(pooled).squeeze(-1) for name, head in self.heads.items()}


class RewardRegister(nn.Module):
    def __init__(self, adapter: BackboneAdapter, config: RewardRegisterConfig):
        super().__init__()
        if adapter.name != config.backbone:
            raise ValueError(f"Adapter {adapter.name!r} does not match checkpoint backbone {config.backbone!r}")
        self.adapter = adapter
        self.config = config
        self.register_tokens = nn.Parameter(torch.empty(config.num_register_tokens, adapter.hidden_size))
        nn.init.normal_(self.register_tokens, std=0.02)
        self.readout = RegisterReadout(adapter.hidden_size, config.head_names, config.hidden_factor)

    def forward(
        self, latents: torch.Tensor, condition: RegisterCondition, sigma: torch.Tensor
    ) -> RegisterOutput:
        features = self.adapter.extract_features(
            latents,
            condition,
            sigma,
            reward_tokens=self.register_tokens,
            feature_layers=self.config.feature_layers,
        )
        return RegisterOutput(scores=self.readout(features.reward_tokens))

    def score(self, latents: torch.Tensor, condition: RegisterCondition, sigma: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self(latents, condition, sigma).scores

    def score_and_grad(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        *,
        heads: tuple[str, ...] | None = None,
    ) -> RewardGradientOutput:
        requested = heads or self.config.head_names
        unknown = set(requested).difference(self.config.head_names)
        if unknown:
            raise ValueError(f"Unknown reward heads: {sorted(unknown)}")
        differentiable_latents = latents.detach().requires_grad_(True)
        scores = self.score(differentiable_latents, condition, sigma)
        gradients = {}
        for index, name in enumerate(requested):
            gradients[name] = torch.autograd.grad(
                scores[name].sum(),
                differentiable_latents,
                retain_graph=index + 1 < len(requested),
                create_graph=False,
            )[0].detach()
        return RewardGradientOutput(scores={name: scores[name].detach() for name in requested}, gradients=gradients)

