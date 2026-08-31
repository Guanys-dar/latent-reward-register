"""Build the checkpoint-faithful SD3 and FLUX reward registers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from latent_reward_register.models.loader import CheckpointRewardRegister

# Default pipeline identifiers. Point these at local snapshots via config when
# running offline.
DEFAULT_MODELS: Mapping[str, str] = {
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.1-dev",
}

# Head names are experiment-specific and must come from config.
_ARCHITECTURE = {
    "sd3": "latent_reward_grid_pool_nope_multihead",
    "flux": "flux_latent_reward_grid_pool_nope_multihead",
}


def _require_diffusers() -> None:
    try:
        import diffusers  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Model-backed registers need the pinned diffusers revision. "
            "Install with: pip install -e '.[models]'"
        ) from error


def build_register(
    backbone: str,
    *,
    head_names: tuple[str, ...],
    feature_layers: tuple[int, ...],
    model_name_or_path: str | None = None,
    revision: str | None = None,
    text_layers: tuple[int, ...] | None = None,
    num_transformer_layers: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
    **architecture_kwargs: Any,
) -> CheckpointRewardRegister:
    """Build an untrained model-backed register for one backbone.

    ``feature_layers`` are the taps recorded in the release configs; they follow
    one rule across backbones (1/6, 1/3, 1/2 of depth, register stopping at
    half depth) rather than per-model magic numbers.
    """
    if backbone not in _ARCHITECTURE:
        raise ValueError(f"Unknown backbone {backbone!r}; expected one of {sorted(_ARCHITECTURE)}")
    if not head_names:
        raise ValueError("head_names is required and must match the training config order")
    if not feature_layers:
        raise ValueError("feature_layers is required")
    _require_diffusers()

    from latent_reward_register.models import (
        FluxLatentRewardGridPoolNoPEMultiHeadModel,
        FluxRewardBackbone,
        SD3LatentRewardGridPoolNoPEMultiHeadModel,
        SD3RewardBackbone,
    )

    path = model_name_or_path or DEFAULT_MODELS[backbone]
    backbone_kwargs = {"torch_dtype": dtype, "local_files_only": local_files_only}
    if revision and revision != "unknown":
        backbone_kwargs["revision"] = revision
    shared = {
        "head_names": tuple(head_names),
        "visual_layers": tuple(feature_layers),
        "text_layers": tuple(text_layers or feature_layers),
        "num_transformer_layers": num_transformer_layers,
        **architecture_kwargs,
    }

    if backbone == "flux":
        trunk = FluxRewardBackbone.from_pretrained(path, **backbone_kwargs)
        model = FluxLatentRewardGridPoolNoPEMultiHeadModel(trunk, **shared)
    else:
        trunk = SD3RewardBackbone.from_pretrained(path, **backbone_kwargs)
        model = SD3LatentRewardGridPoolNoPEMultiHeadModel(trunk, **shared)

    return CheckpointRewardRegister(model, backbone, tuple(head_names), compute_dtype=dtype)
