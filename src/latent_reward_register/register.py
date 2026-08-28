"""Reference register architecture: register tokens plus a readout, over any adapter.

**This is not the production path.** Nothing here loads or produces a published
checkpoint. It is a self-contained, weight-free implementation of the register
idea — trainable tokens read a backbone's features, an MLP head turns them into
a reward — and its purpose is to make the algorithm layer (training loop,
preference scoring, RGS, RG-OPD) executable with no model weights at all. That
is what ``lrr smoke-release`` runs, via the synthetic adapter in ``smoke.py``.

For a real SD3/FLUX/Z-Image register use
:class:`~latent_reward_register.implementations.loader.CheckpointRewardRegister`,
built by ``backbones.build_register_from_config`` or loaded by
``load_legacy_register``. Those models own their own backbone traversal and are
checkpoint-compatible with the paper's weights; this class is neither.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from .backbones.base import BackboneAdapter
from .types import RegisterCondition, RegisterOutput, RewardGradientOutput


@dataclass(frozen=True)
class ReferenceRegisterConfig:
    backbone: str
    head_names: tuple[str, ...]
    feature_layers: tuple[int, ...]
    num_register_tokens: int = 32
    hidden_factor: int = 2

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["head_names"] = list(self.head_names)
        payload["feature_layers"] = list(self.feature_layers)
        return payload


class RegisterReadout(nn.Module):
    """Mean-pools the register tokens and maps them to one scalar per head."""

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


class ReferenceRewardRegister(nn.Module):
    """Register tokens over a :class:`BackboneAdapter`, with an MLP readout.

    Exposes the same scoring surface as ``CheckpointRewardRegister`` — ``score``,
    ``score_groups``, ``score_and_grad`` — so the algorithm layer can be driven
    without weights. See the module docstring: not checkpoint-compatible.
    """

    def __init__(self, adapter: BackboneAdapter, config: ReferenceRegisterConfig):
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

    def score_groups(
        self, latents: torch.Tensor, condition: RegisterCondition, sigma: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        """Score ``(batch, group_size, ...)`` latents, keeping the group dimension.

        The group dimension is folded into the batch for the backbone pass and
        restored afterwards, so the pairwise loss can compare within groups.

        Conditioning is expanded here, unlike ``CheckpointRewardRegister``, which
        passes it unexpanded because the research models repeat each prompt across
        its group internally. ``BackboneAdapter.extract_features`` does not, so
        this path must expand it.
        """
        if latents.ndim < 3:
            raise ValueError(f"Expected (batch, group_size, ...) latents, got {tuple(latents.shape)}")
        batch_size, group_size = latents.shape[:2]
        flat_latents = latents.reshape(batch_size * group_size, *latents.shape[2:])
        flat_sigma = sigma.reshape(-1)
        if flat_sigma.numel() == batch_size:
            flat_sigma = flat_sigma.repeat_interleave(group_size)
        elif flat_sigma.numel() != batch_size * group_size:
            raise ValueError(
                f"sigma must hold one value per group or per item, got {tuple(sigma.shape)}"
            )
        scores = self.score(flat_latents, condition.expand_groups(group_size), flat_sigma)
        return {name: value.reshape(batch_size, group_size) for name, value in scores.items()}

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
