"""Model-backed register builders for SD3, FLUX, and Z-Image.

This is the production path. The research implementations under
``implementations/`` are complete reward registers: each owns its backbone
traversal, latent packing, conditioning, and reward-token side stream, and
returns ``dict[head -> (batch, group_size)]``. They are therefore wired as
registers, not as feature extractors behind ``BackboneAdapter`` — do not try to
route a real backbone through that interface.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from latent_reward_register.implementations.loader import CheckpointRewardRegister

from .registry import normalize_backbone_name, register_backbone

# Default pipeline identifiers. Point these at local snapshots via config when
# running offline.
DEFAULT_MODELS: Mapping[str, str] = {
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.1-dev",
    "z-image": "Tongyi-MAI/Z-Image-Turbo",
}

# head_names is intentionally absent: it is per-experiment (SD3 exp11 trains
# three heads, FLUX/Z-Image unified-v3 train two) and must come from config.
_ARCHITECTURE = {
    "sd3": "latent_reward_grid_pool_nope_multihead",
    "flux": "flux_latent_reward_grid_pool_nope_multihead",
    "z-image": "zimage_latent_reward_grid_pool_nope_multihead",
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

    from latent_reward_register.implementations.models import (
        FluxLatentRewardGridPoolNoPEMultiHeadModel,
        FluxRewardBackbone,
        SD3LatentRewardGridPoolNoPEMultiHeadModel,
        SD3RewardBackbone,
        ZImageLatentRewardGridPoolNoPEMultiHeadModel,
        ZImageRewardBackbone,
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
    elif backbone == "z-image":
        trunk = ZImageRewardBackbone.from_pretrained(path, **backbone_kwargs)
        model = ZImageLatentRewardGridPoolNoPEMultiHeadModel(trunk, **shared)
    else:
        trunk = SD3RewardBackbone.from_pretrained(path, **backbone_kwargs)
        model = SD3LatentRewardGridPoolNoPEMultiHeadModel(trunk, **shared)

    return CheckpointRewardRegister(model, backbone, tuple(head_names))


def build_register_from_config(config: Mapping[str, Any], **overrides: Any) -> CheckpointRewardRegister:
    """Build a register from a ``configs/register/<backbone>/paper.yaml`` mapping."""
    backbone_config = config.get("backbone")
    if not isinstance(backbone_config, Mapping):
        raise ValueError("Config requires a backbone mapping")
    register_config = config.get("register")
    if not isinstance(register_config, Mapping):
        raise ValueError("Config requires a register mapping")

    known = {"head_names", "feature_layers", "text_layers", "num_transformer_layers", "score_keys"}
    passthrough = {
        key: value
        for key, value in register_config.items()
        if key not in known and key != "architecture"
    }
    return build_register(
        normalize_backbone_name(str(backbone_config.get("name", ""))),
        head_names=tuple(register_config["head_names"]),
        feature_layers=tuple(register_config["feature_layers"]),
        text_layers=tuple(register_config["text_layers"]) if register_config.get("text_layers") else None,
        num_transformer_layers=register_config.get("num_transformer_layers"),
        model_name_or_path=backbone_config.get("model_name_or_path"),
        revision=backbone_config.get("revision"),
        **{**passthrough, **overrides},
    )


for _name in _ARCHITECTURE:
    register_backbone(_name, build_register)
