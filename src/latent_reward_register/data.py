from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    prompt: str
    latent_path: str
    prompt_embed_path: str
    rewards: Mapping[str, float]
    pooled_prompt_embed_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManifestRecord":
        required = ("sample_id", "prompt", "latent_path", "prompt_embed_path", "rewards")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Manifest record missing fields: {', '.join(missing)}")
        rewards = payload["rewards"]
        if not isinstance(rewards, dict) or not rewards:
            raise ValueError("Manifest rewards must be a non-empty mapping")
        return cls(
            sample_id=str(payload["sample_id"]),
            prompt=str(payload["prompt"]),
            latent_path=str(payload["latent_path"]),
            prompt_embed_path=str(payload["prompt_embed_path"]),
            rewards={str(key): float(value) for key, value in rewards.items()},
            pooled_prompt_embed_path=(
                str(payload["pooled_prompt_embed_path"]) if payload.get("pooled_prompt_embed_path") else None
            ),
        )


def read_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    manifest_path = Path(path)
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield ManifestRecord.from_mapping(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest record at {manifest_path}:{line_number}: {error}") from error

