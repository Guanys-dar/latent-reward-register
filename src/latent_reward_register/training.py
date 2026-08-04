from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import AdamW

from .checkpoint import CheckpointManifest, save_register_checkpoint
from .losses import multihead_pairwise_loss
from .register import RewardRegister
from .types import RegisterCondition


@dataclass
class PairBatch:
    preferred_latents: torch.Tensor
    rejected_latents: torch.Tensor
    condition: RegisterCondition
    sigma: torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 3
    ema_decay: float = 0.999


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            self.state[name].lerp_(value.detach(), 1.0 - self.decay)


def train_register(
    *,
    model: RewardRegister,
    batches: Iterable[PairBatch],
    config: TrainConfig,
    output_dir: str | Path,
    manifest: CheckpointManifest,
    head_weights: dict[str, float] | None = None,
) -> None:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    ema = EMA(model, config.ema_decay)
    model.train()
    for _ in range(config.epochs):
        for batch in batches:
            preferred = model.score(batch.preferred_latents, batch.condition, batch.sigma)
            rejected = model.score(batch.rejected_latents, batch.condition, batch.sigma)
            loss, _ = multihead_pairwise_loss(preferred, rejected, batch.sigma, head_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            ema.update(model)
    save_register_checkpoint(output_dir, model, manifest, {"train": config.__dict__, "register": model.config.to_dict()})
