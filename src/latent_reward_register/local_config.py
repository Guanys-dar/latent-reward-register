"""Machine-specific paths, kept out of every committed config.

Data roots, checkpoint roots, and local model snapshots differ per machine. If
they lived in the paper configs, the public and private copies of those files
would diverge, and the release invariant is that every committed file is
byte-identical between them. So they live in one uncommitted file instead:
``configs/local.yaml``, made from ``configs/local.yaml.example``.

Everything here is optional. With no local config, model identifiers resolve
from the Hugging Face Hub and manifest paths must be absolute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .config import load_config

DEFAULT_LOCAL_CONFIG = Path("configs/local.yaml")


@dataclass(frozen=True)
class LocalConfig:
    """Resolved machine-local settings."""

    data_root: Path | None = None
    checkpoint_root: Path | None = None
    models: Mapping[str, str] = field(default_factory=dict)
    tracking: Mapping[str, str] = field(default_factory=dict)

    def resolve_data(self, path: str | Path) -> Path:
        """Resolve a manifest-relative path against ``data_root``."""
        candidate = Path(path)
        if candidate.is_absolute() or self.data_root is None:
            return candidate
        return self.data_root / candidate

    def resolve_checkpoint(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or self.checkpoint_root is None:
            return candidate
        return self.checkpoint_root / candidate

    def model_path(self, backbone: str, fallback: str) -> str:
        """Local snapshot for ``backbone``, or ``fallback`` to resolve from the Hub."""
        return str(self.models.get(backbone, self.models.get(backbone.replace("-", ""), fallback)))


def load_local_config(path: str | Path | None = None) -> LocalConfig:
    """Read ``configs/local.yaml`` if present.

    A missing file is not an error: it means every path in play is absolute and
    every model resolves from the Hub. An explicitly named file that is missing
    *is* an error, since the caller asked for it.
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Local config not found: {config_path}")
    else:
        config_path = DEFAULT_LOCAL_CONFIG
        if not config_path.exists():
            return LocalConfig()

    payload = load_config(config_path)
    data_root = payload.get("data_root")
    checkpoint_root = payload.get("checkpoint_root")
    models = payload.get("models") or {}
    tracking = payload.get("tracking") or {}
    if not isinstance(models, dict):
        raise ValueError(f"{config_path}: 'models' must be a mapping of backbone to path")
    return LocalConfig(
        data_root=Path(data_root) if data_root else None,
        checkpoint_root=Path(checkpoint_root) if checkpoint_root else None,
        models={str(key): str(value) for key, value in models.items() if value},
        tracking={str(key): str(value) for key, value in tracking.items() if value},
    )
