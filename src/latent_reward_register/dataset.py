"""Group manifest -> ``GroupBatch``: the missing link in register training.

``train_register`` consumes ``GroupBatch`` objects and ``read_group_manifest``
yields ``GroupRecord`` objects; this module is the loader between them. It reads
cached latents and prompt embeddings from disk — training never runs a VAE or a
text encoder, which is what makes it cheap.

Two invariants are enforced here rather than left to fail later:

- **Uniform group size.** The loss compares within groups, so a batch is a
  rectangular ``(batch, group_size, ...)`` tensor. Groups whose size differs
  from the requested one are truncated or skipped rather than silently padded,
  because a padded group would contribute meaningless pairs.
- **Positional head alignment.** ``targets[head]`` is built by asking each
  record for that head's score by name, so a reordered head list cannot shift
  targets onto the wrong head.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .data import GroupRecord, read_group_manifest
from .local_config import LocalConfig
from .training import GroupBatch
from .types import RegisterCondition


def load_tensor(path: str | Path) -> torch.Tensor:
    """Load one cached tensor, accepting both ``.pt`` and ``.safetensors``."""
    tensor_path = Path(path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Cached tensor not found: {tensor_path}")
    if tensor_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(tensor_path))
        if len(payload) != 1:
            raise ValueError(
                f"{tensor_path}: expected a single tensor, found keys {sorted(payload)}"
            )
        return next(iter(payload.values()))
    return torch.load(tensor_path, map_location="cpu", weights_only=True)


@dataclass(frozen=True)
class DatasetConfig:
    """How to turn manifest records into batches.

    ``group_size`` must match what the manifest actually holds: the register's
    pooling asserts a token count, and the loss needs at least two samples per
    group to form a pair.
    """

    group_size: int = 4
    batch_size: int = 2
    sigma: float = 1.0
    heads: tuple[str, ...] = ("preference",)
    require_pooled: bool = False

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError(f"group_size must be at least 2 to form a pair, got {self.group_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not self.heads:
            raise ValueError("At least one reward head is required")


def _group_tensors(
    record: GroupRecord, config: DatasetConfig, local: LocalConfig
) -> tuple[torch.Tensor, dict[str, list[float]]] | None:
    """Latents and per-head scores for one group, or None if it cannot be used."""
    records = record.image_records[: config.group_size]
    if len(records) < config.group_size:
        return None

    latents = torch.stack([load_tensor(local.resolve_data(item.latent_x0_path)) for item in records])
    targets = {head: [item.score_for_head(head) for item in records] for head in config.heads}
    return latents, targets


def iter_group_batches(
    manifest: str | Path,
    *,
    config: DatasetConfig,
    local: LocalConfig | None = None,
    limit: int | None = None,
) -> Iterator[GroupBatch]:
    """Stream ``GroupBatch`` objects from a prepared manifest.

    Streaming rather than materializing: a production manifest holds hundreds of
    thousands of cached latents, which does not fit in memory.
    """
    local = local or LocalConfig()
    pending: list[tuple[GroupRecord, torch.Tensor, dict[str, list[float]]]] = []
    produced = 0

    for record in read_group_manifest(manifest):
        prepared = _group_tensors(record, config, local)
        if prepared is None:
            continue
        latents, targets = prepared
        pending.append((record, latents, targets))

        if len(pending) == config.batch_size:
            yield _collate(pending, config, local)
            pending.clear()
            produced += 1
            if limit is not None and produced >= limit:
                return

    if pending and (limit is None or produced < limit):
        yield _collate(pending, config, local)


def _collate(
    pending: Sequence[tuple[GroupRecord, torch.Tensor, dict[str, list[float]]]],
    config: DatasetConfig,
    local: LocalConfig,
) -> GroupBatch:
    """Stack prepared groups into one rectangular batch.

    Prompt embeddings are stacked per group and passed unexpanded: the research
    models repeat each prompt across its group internally, so expanding here
    would double the group axis.
    """
    latents = torch.stack([item[1] for item in pending])
    prompt_embeds = torch.stack(
        [load_tensor(local.resolve_data(item[0].prompt_embeds_path)) for item in pending]
    )

    pooled = None
    if config.require_pooled:
        missing = [item[0].group_id for item in pending if not item[0].pooled_prompt_embeds_path]
        if missing:
            raise ValueError(
                f"This backbone needs pooled prompt embeddings; groups without them: {missing[:5]}"
            )
        pooled = torch.stack(
            [load_tensor(local.resolve_data(item[0].pooled_prompt_embeds_path)) for item in pending]
        )

    targets = {
        head: torch.tensor([item[2][head] for item in pending], dtype=torch.float32)
        for head in config.heads
    }
    return GroupBatch(
        latents=latents,
        condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled),
        sigma=torch.full((len(pending),), config.sigma, dtype=torch.float32),
        targets=targets,
    )


def count_usable_groups(
    manifest: str | Path, *, group_size: int, limit: int | None = None
) -> tuple[int, int]:
    """``(usable, total)`` groups, for reporting before a run starts.

    Cheap because it reads only the manifest, never the cached tensors.
    """
    usable = total = 0
    for record in read_group_manifest(manifest):
        total += 1
        if len(record.image_records) >= group_size:
            usable += 1
        if limit is not None and total >= limit:
            break
    return usable, total
