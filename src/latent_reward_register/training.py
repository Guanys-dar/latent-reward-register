"""The shared register training loop.

Backbone-agnostic on purpose: it needs only ``score_groups``, so it drives a
model-backed ``CheckpointRewardRegister`` and the weight-free
``ReferenceRewardRegister`` alike.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.optim import AdamW

from .checkpoint import CheckpointManifest, save_register_checkpoint
from .losses import multihead_group_loss
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


class TrainableRegister(Protocol):
    """What the loop needs: group-shaped scores and trainable parameters."""

    def score_groups(
        self, latents: torch.Tensor, condition: RegisterCondition, sigma: torch.Tensor
    ) -> Mapping[str, torch.Tensor]: ...

    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def state_dict(self): ...


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 3
    ema_decay: float = 0.999
    min_target_gap: float = 0.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.state[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.state[name].copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.state, strict=True)
        return backup


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _move_batch(batch: GroupBatch, device: torch.device) -> GroupBatch:
    return GroupBatch(
        latents=batch.latents.to(device),
        condition=RegisterCondition(
            prompt_embeds=batch.condition.prompt_embeds.to(device),
            pooled_prompt_embeds=(
                batch.condition.pooled_prompt_embeds.to(device)
                if batch.condition.pooled_prompt_embeds is not None
                else None
            ),
            metadata=batch.condition.metadata,
        ),
        sigma=batch.sigma.to(device),
        targets={name: value.to(device) for name, value in batch.targets.items()},
        group_mask=batch.group_mask.to(device) if batch.group_mask is not None else None,
    )


def _noise_batch(
    model: torch.nn.Module, batch: GroupBatch
) -> GroupBatch | tuple[GroupBatch, torch.Tensor]:
    research_model = getattr(model, "model", model)
    backbone = getattr(research_model, "backbone", None)
    if backbone is None or not hasattr(backbone, "sample_timesteps"):
        return batch
    timesteps, sigmas = backbone.sample_timesteps(
        batch_size=batch.latents.shape[0],
        device=batch.latents.device,
        n_dim=batch.latents.ndim,
        dtype=batch.latents.dtype,
        weighting_scheme="uniform",
    )
    noise = torch.randn_like(batch.latents[:, :1]).expand_as(batch.latents)
    noisy_latents = backbone.add_noise(batch.latents, noise, sigmas)
    loss_sigmas = sigmas.reshape(sigmas.shape[0], -1)[:, 0]
    return GroupBatch(
        latents=noisy_latents,
        condition=batch.condition,
        sigma=loss_sigmas,
        targets=batch.targets,
        group_mask=batch.group_mask,
    ), timesteps


def train_register(
    *,
    model: TrainableRegister,
    batches: Iterable[GroupBatch] | Callable[[], Iterable[GroupBatch]],
    config: TrainConfig,
    output_dir: str | Path,
    manifest: CheckpointManifest,
    register_config: Mapping[str, Any],
    head_weights: dict[str, float] | None = None,
) -> None:
    """Train a register on group batches and write a release checkpoint.

    ``register_config`` is the architecture record written into the checkpoint's
    ``config.yaml``. It is a parameter rather than read off the model because the
    two register classes carry their architecture differently, and a checkpoint
    without it cannot be rebuilt.
    """
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, float(step + 1) / max(1, config.warmup_steps))
        if config.warmup_steps > 0
        else 1.0,
    )
    ema = EMA(model, config.ema_decay)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(config.epochs):
        epoch_batches = batches() if callable(batches) else batches
        pending = 0
        for batch in epoch_batches:
            batch = _move_batch(batch, _model_device(model))
            prepared = _noise_batch(model, batch)
            if isinstance(prepared, tuple):
                batch, timesteps = prepared
            else:
                timesteps = batch.sigma
            predictions = model.score_groups(batch.latents, batch.condition, timesteps)
            loss, _ = multihead_group_loss(
                {name: predictions[name] for name in batch.targets},
                batch.targets,
                sigmas=batch.sigma,
                head_weights=head_weights,
                group_mask=batch.group_mask,
                min_target_gap=config.min_target_gap,
            )
            (loss / config.gradient_accumulation_steps).backward()
            pending += 1
            if pending == config.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
    raw_state = ema.copy_to(model)
    try:
        save_register_checkpoint(
            output_dir,
            model,
            manifest,
            {"train": config.__dict__, "register": dict(register_config)},
        )
    finally:
        model.load_state_dict(raw_state, strict=True)
