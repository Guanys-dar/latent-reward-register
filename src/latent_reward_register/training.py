from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.optim import AdamW

from .checkpoint import CheckpointManifest, save_register_checkpoint
from .losses import multihead_group_loss
from .register import RewardRegister
from .types import RegisterCondition


@dataclass
class GroupBatch:
    """One batch of prompt groups.

    ``latents`` is ``(batch, group_size, ...)`` and ``targets`` maps each head
    to its ``(batch, group_size)`` teacher scores. Groups are the unit of the
    loss, so the group dimension must survive into the objective.
    """

    latents: torch.Tensor
    condition: RegisterCondition
    sigma: torch.Tensor
    targets: dict[str, torch.Tensor]
    group_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 3
    ema_decay: float = 0.999
    min_target_gap: float = 0.0


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
    batches: Iterable[GroupBatch],
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
            predictions = model.score_groups(batch.latents, batch.condition, batch.sigma)
            loss, _ = multihead_group_loss(
                {name: predictions[name] for name in batch.targets},
                batch.targets,
                sigmas=batch.sigma,
                head_weights=head_weights,
                group_mask=batch.group_mask,
                min_target_gap=config.min_target_gap,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            ema.update(model)
    save_register_checkpoint(output_dir, model, manifest, {"train": config.__dict__, "register": model.config.to_dict()})
