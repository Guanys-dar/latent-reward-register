from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def require_keys(payload: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")


DEFAULT_LOCAL_CONFIG = Path("configs/local.yaml")


@dataclass(frozen=True)
class LocalConfig:
    data_root: Path | None = None
    checkpoint_root: Path | None = None
    models: Mapping[str, str] = field(default_factory=dict)
    tracking: Mapping[str, str] = field(default_factory=dict)

    def resolve_data(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() or self.data_root is None else self.data_root / candidate

    def resolve_checkpoint(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or self.checkpoint_root is None:
            return candidate
        return self.checkpoint_root / candidate

    def model_path(self, backbone: str, fallback: str) -> str:
        return str(self.models.get(backbone, fallback))


def load_local_config(path: str | Path | None = None) -> LocalConfig:
    config_path = Path(path) if path is not None else DEFAULT_LOCAL_CONFIG
    if not config_path.exists():
        if path is not None:
            raise FileNotFoundError(f"Local config not found: {config_path}")
        return LocalConfig()
    payload = load_config(config_path)
    models = payload.get("models") or {}
    tracking = payload.get("tracking") or {}
    if not isinstance(models, dict):
        raise ValueError(f"{config_path}: 'models' must be a mapping of backbone to path")
    return LocalConfig(
        data_root=Path(payload["data_root"]) if payload.get("data_root") else None,
        checkpoint_root=Path(payload["checkpoint_root"]) if payload.get("checkpoint_root") else None,
        models={str(key): str(value) for key, value in models.items() if value},
        tracking={str(key): str(value) for key, value in tracking.items() if value},
    )
