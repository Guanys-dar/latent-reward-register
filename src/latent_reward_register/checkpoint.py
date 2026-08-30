from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: int
    backbone: str
    backbone_revision: str
    adapter: str
    head_names: tuple[str, ...]
    feature_layers: tuple[int, ...]
    noise_convention: str
    source_checkpoint: str | None = None


def save_register_checkpoint(directory: str | Path, model, manifest: CheckpointManifest, config: Mapping[str, Any]) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, output / "register.safetensors")
    (output / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    (output / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))


def load_register_checkpoint(directory: str | Path, model) -> tuple[CheckpointManifest, dict[str, Any]]:
    checkpoint = Path(directory)
    manifest_payload = json.loads((checkpoint / "manifest.json").read_text())
    manifest_payload["head_names"] = tuple(manifest_payload["head_names"])
    manifest_payload["feature_layers"] = tuple(manifest_payload["feature_layers"])
    manifest = CheckpointManifest(**manifest_payload)
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    model.load_state_dict(load_file(checkpoint / "register.safetensors"), strict=True)
    return manifest, config


def read_legacy_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise ValueError("Legacy checkpoint must contain self-describing 'config' and 'model' entries")
    return payload

