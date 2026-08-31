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

import bisect
import glob
import io
import json
import random
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import LocalConfig
from .training import GroupBatch
from .types import RegisterCondition

HEAD_SCORE_KEYS: Mapping[str, str] = {
    "preference": "teacher_score_zscore",
    "pickscore": "pickscore",
    "imagereward": "imagereward_score",
}


@dataclass(frozen=True)
class ImageRecord:
    sample_id: str
    latent_x0_path: str
    scores: Mapping[str, float]
    image_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ImageRecord:
        for key in ("sample_id", "latent_x0_path"):
            if key not in payload:
                raise ValueError(f"Image record missing field: {key}")
        scores = {
            key: float(value)
            for key, value in payload.items()
            if key in set(HEAD_SCORE_KEYS.values()) | {"teacher_score_raw"} and value is not None
        }
        if not scores:
            raise ValueError(f"Image record {payload['sample_id']} carries no teacher scores")
        return cls(
            sample_id=str(payload["sample_id"]),
            latent_x0_path=str(payload["latent_x0_path"]),
            scores=scores,
            image_path=str(payload["image_path"]) if payload.get("image_path") else None,
        )

    def score_for_head(self, head: str) -> float | None:
        try:
            key = HEAD_SCORE_KEYS[head]
        except KeyError:
            raise KeyError(
                f"Unknown reward head {head!r}; expected one of {sorted(HEAD_SCORE_KEYS)}"
            ) from None
        return self.scores.get(key)


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    prompt: str
    prompt_embeds_path: str
    image_records: tuple[ImageRecord, ...]
    pooled_prompt_embeds_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GroupRecord:
        required = ("group_id", "prompt", "prompt_embeds_path", "image_records")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Group record missing fields: {', '.join(missing)}")
        images = payload["image_records"]
        if not isinstance(images, Sequence) or len(images) < 2:
            raise ValueError(f"Group {payload['group_id']} needs at least two images to form a pair")
        return cls(
            group_id=str(payload["group_id"]),
            prompt=str(payload["prompt"]),
            prompt_embeds_path=str(payload["prompt_embeds_path"]),
            image_records=tuple(ImageRecord.from_mapping(item) for item in images),
            pooled_prompt_embeds_path=(
                str(payload["pooled_prompt_embeds_path"])
                if payload.get("pooled_prompt_embeds_path")
                else None
            ),
        )


def read_group_manifest(path: str | Path) -> Iterator[GroupRecord]:
    manifest_path = Path(path)
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield GroupRecord.from_mapping(json.loads(line))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid group record at {manifest_path}:{line_number}: {error}"
                ) from error


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
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping):
        for key in ("latent_x0", "prompt_embeds", "pooled_prompt_embeds", "tensor"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(f"{tensor_path}: unsupported cached tensor payload")


@dataclass(frozen=True)
class DatasetConfig:
    """How to turn manifest records into batches.

    ``group_size`` must match what the manifest actually holds: the register's
    pooling asserts a token count, and the loss needs at least two samples per
    group to form a pair.
    """

    group_size: int = 2
    batch_size: int = 2
    sigma: float = 1.0
    heads: tuple[str, ...] = ("preference",)
    require_pooled: bool = False
    seed: int = 42
    shuffle: bool = True
    shuffle_buffer_size: int = 1024
    drop_last: bool = True

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError(f"group_size must be at least 2 to form a pair, got {self.group_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not self.heads:
            raise ValueError("At least one reward head is required")
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")


def _buffered_shuffle(
    records: Iterator[GroupRecord], *, seed: int, buffer_size: int
) -> Iterator[GroupRecord]:
    random_generator = random.Random(seed)
    buffer: list[GroupRecord] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = random_generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    random_generator.shuffle(buffer)
    yield from buffer


def _group_tensors(
    record: GroupRecord, config: DatasetConfig, local: LocalConfig
) -> tuple[torch.Tensor, dict[str, list[float]], dict[str, list[bool]]] | None:
    """Latents and per-head scores for one group, or None if it cannot be used."""
    records = record.image_records[: config.group_size]
    if len(records) < config.group_size:
        return None

    latents = torch.stack([load_tensor(local.resolve_data(item.latent_x0_path)) for item in records])
    scores = {head: [item.score_for_head(head) for item in records] for head in config.heads}
    targets = {
        head: [float(value) if value is not None else 0.0 for value in values]
        for head, values in scores.items()
    }
    head_masks = {
        head: [value is not None for value in values]
        for head, values in scores.items()
    }
    return latents, targets, head_masks


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
    pending: list[
        tuple[GroupRecord, torch.Tensor, dict[str, list[float]], dict[str, list[bool]]]
    ] = []
    produced = 0

    records = read_group_manifest(manifest)
    if config.shuffle:
        records = _buffered_shuffle(
            records, seed=config.seed, buffer_size=config.shuffle_buffer_size
        )
    for record in records:
        prepared = _group_tensors(record, config, local)
        if prepared is None:
            continue
        latents, targets, head_masks = prepared
        pending.append((record, latents, targets, head_masks))

        if len(pending) == config.batch_size:
            yield _collate(pending, config, local)
            pending.clear()
            produced += 1
            if limit is not None and produced >= limit:
                return

    if pending and not config.drop_last and (limit is None or produced < limit):
        yield _collate(pending, config, local)


def _collate(
    pending: Sequence[
        tuple[GroupRecord, torch.Tensor, dict[str, list[float]], dict[str, list[bool]]]
    ],
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
    head_masks = {
        head: torch.tensor([item[3][head] for item in pending], dtype=torch.bool)
        for head in config.heads
    }
    return GroupBatch(
        latents=latents,
        condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled),
        sigma=torch.full((len(pending),), config.sigma, dtype=torch.float32),
        targets=targets,
        head_masks=head_masks,
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


@dataclass(frozen=True)
class _PairRowGroup:
    path: Path
    index: int
    start: int
    rows: int


class DinaPairDataset:
    """Random-access adapter for the original DiNa chosen/rejected parquet rows."""

    def __init__(
        self,
        parquet_paths: str | Path,
        *,
        manifest: str | Path,
        multihead_manifest: str | Path,
        heads: tuple[str, ...],
        local: LocalConfig | None = None,
        latents_from_manifest: bool = False,
    ) -> None:
        import pyarrow.parquet as pq

        self.local = local or LocalConfig()
        self.heads = heads
        self.latents_from_manifest = latents_from_manifest
        self.prompts = {record.group_id: record for record in read_group_manifest(manifest)}
        self.scores = {
            (record.group_id, image.sample_id): image
            for record in read_group_manifest(multihead_manifest)
            for image in record.image_records
        }
        paths = sorted(Path(path) for path in glob.glob(str(parquet_paths)))
        if not paths and Path(parquet_paths).is_file():
            paths = [Path(parquet_paths)]
        if not paths:
            raise FileNotFoundError(f"No parquet files matched: {parquet_paths}")
        self.row_groups: list[_PairRowGroup] = []
        self.starts: list[int] = []
        total = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            for index in range(parquet.num_row_groups):
                rows = parquet.metadata.row_group(index).num_rows
                self.starts.append(total)
                self.row_groups.append(_PairRowGroup(path, index, total, rows))
                total += rows
        self.total = total
        self.cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def _rows(self, offset: int) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        if offset in self.cache:
            self.cache.move_to_end(offset)
            return self.cache[offset]
        row_group = self.row_groups[offset]
        rows = pq.ParquetFile(row_group.path).read_row_group(row_group.index).to_pylist()
        self.cache[offset] = rows
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> GroupBatch:
        offset = bisect.bisect_right(self.starts, index) - 1
        row_group = self.row_groups[offset]
        row = self._rows(offset)[index - row_group.start]
        group_id = str(row["group_id"])
        prompt = self.prompts[group_id]

        import numpy as np

        sample_ids = (str(row["chosen_sample_id"]), str(row["rejected_sample_id"]))
        if self.latents_from_manifest:
            latents = torch.stack(
                [
                    load_tensor(self.local.resolve_data(self.scores[(group_id, sample_id)].latent_x0_path)).float()
                    for sample_id in sample_ids
                ]
            ).unsqueeze(0)
        else:
            latents = torch.stack(
                [
                    torch.from_numpy(np.load(io.BytesIO(bytes(row[key])), allow_pickle=False)).float()
                    for key in ("latent1", "latent2")
                ]
            ).unsqueeze(0)
        targets: dict[str, torch.Tensor] = {}
        head_masks: dict[str, torch.Tensor] = {}
        for head in self.heads:
            values = [self.scores.get((group_id, sample_id)) for sample_id in sample_ids]
            scores = [value.score_for_head(head) if value is not None else None for value in values]
            targets[head] = torch.tensor(
                [[float(score) if score is not None else 0.0 for score in scores]], dtype=torch.float32
            )
            head_masks[head] = torch.tensor([[score is not None for score in scores]], dtype=torch.bool)
        pooled = None
        if prompt.pooled_prompt_embeds_path:
            pooled = load_tensor(self.local.resolve_data(prompt.pooled_prompt_embeds_path)).unsqueeze(0)
        return GroupBatch(
            latents=latents,
            condition=RegisterCondition(
                prompt_embeds=load_tensor(self.local.resolve_data(prompt.prompt_embeds_path)).unsqueeze(0),
                pooled_prompt_embeds=pooled,
            ),
            sigma=torch.ones(1),
            targets=targets,
            group_mask=torch.ones((1, 2), dtype=torch.bool),
            head_masks=head_masks,
        )


def iter_dina_pair_batches(
    dataset: DinaPairDataset,
    *,
    batch_size: int,
    seed: int,
    drop_last: bool = True,
    limit: int | None = None,
) -> Iterator[GroupBatch]:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for produced, start in enumerate(range(0, len(indices), batch_size), start=1):
        selected = indices[start : start + batch_size]
        if len(selected) < batch_size and drop_last:
            break
        items = [dataset[index] for index in selected]
        pooled = [item.condition.pooled_prompt_embeds for item in items]
        if any(value is None for value in pooled):
            raise ValueError("DiNa pair batch requires pooled prompt embeddings for every item")
        yield GroupBatch(
            latents=torch.cat([item.latents for item in items]),
            condition=RegisterCondition(
                prompt_embeds=torch.cat([item.condition.prompt_embeds for item in items]),
                pooled_prompt_embeds=torch.cat([value for value in pooled if value is not None]),
            ),
            sigma=torch.cat([item.sigma for item in items]),
            targets={head: torch.cat([item.targets[head] for item in items]) for head in dataset.heads},
            group_mask=torch.cat([item.group_mask for item in items if item.group_mask is not None]),
            head_masks={
                head: torch.cat([item.head_masks[head] for item in items if item.head_masks is not None])
                for head in dataset.heads
            },
        )
        if limit is not None and produced >= limit:
            return
