from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# Score key for each reward head. Order is positional: the loss aligns
# targets[:, h] with head_names[h], so these must stay parallel.
HEAD_SCORE_KEYS: Mapping[str, str] = {
    "preference": "teacher_score_zscore",
    "pickscore": "pickscore",
    "imagereward": "imagereward_score",
}


@dataclass(frozen=True)
class ImageRecord:
    """One scored image inside a prompt group."""

    sample_id: str
    latent_x0_path: str
    scores: Mapping[str, float]
    image_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ImageRecord":
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

    def score_for_head(self, head: str) -> float:
        try:
            key = HEAD_SCORE_KEYS[head]
        except KeyError:
            raise KeyError(f"Unknown reward head {head!r}; expected one of {sorted(HEAD_SCORE_KEYS)}") from None
        if key not in self.scores:
            raise KeyError(f"Sample {self.sample_id} has no {key!r} for head {head!r}")
        return self.scores[key]


@dataclass(frozen=True)
class GroupRecord:
    """A prompt and its ranked images. Groups are the unit of the pairwise loss."""

    group_id: str
    prompt: str
    prompt_embeds_path: str
    image_records: tuple[ImageRecord, ...]
    pooled_prompt_embeds_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroupRecord":
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
                str(payload["pooled_prompt_embeds_path"]) if payload.get("pooled_prompt_embeds_path") else None
            ),
        )

    def requires_pooled_embeds(self, backbone: str) -> bool:
        return backbone == "sd3"


def read_group_manifest(path: str | Path) -> Iterator[GroupRecord]:
    """Stream a prepared group manifest, reporting the offending line on error."""
    manifest_path = Path(path)
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield GroupRecord.from_mapping(json.loads(line))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid group record at {manifest_path}:{line_number}: {error}") from error
