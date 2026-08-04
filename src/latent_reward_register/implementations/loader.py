from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from latent_reward_register.checkpoint import read_legacy_checkpoint
from latent_reward_register.types import RegisterCondition

def _architecture_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    register = config["reward_token"]
    feature_layers = register.get("feature_layers", register.get("visual_layers"))
    return {
        "head_names": register["head_names"],
        "head_hidden_factor": int(register.get("head_hidden_factor", 2)),
        "visual_layers": feature_layers,
        "text_layers": register.get("text_layers", feature_layers),
        "layer_index_base": int(register.get("layer_index_base", 1)),
        "num_transformer_layers": register.get("num_transformer_layers"),
        "vis_h": int(register.get("vis_h", 64)),
        "vis_w": int(register.get("vis_w", 64)),
        "width": int(register.get("width", -1)),
        "num_attn_heads": int(register.get("num_attn_heads", 8)),
        "dropout": float(register.get("dropout", 0.0)),
        "use_proj_in": bool(register.get("use_proj_in", False)),
        "use_self_attn": bool(register.get("use_self_attn", True)),
        "skip_attn2": bool(register.get("skip_attn2", True)),
        "pool_factor": int(register.get("pool_factor", 4)),
        "num_reward_tokens": int(register.get("num_reward_tokens", 32)),
        "disable_side_stream": bool(register.get("disable_side_stream", False)),
        "side_stream_ffn": bool(register.get("side_stream_ffn", False)),
        "freeze_q_proj": bool(register.get("freeze_q_proj", False)),
    }


class CheckpointRewardRegister(nn.Module):
    def __init__(self, model: nn.Module, backbone: str, head_names: tuple[str, ...]):
        super().__init__()
        self.model = model
        self.backbone = backbone
        self.head_names = head_names

    def score(self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor):
        grouped_latents = latents.unsqueeze(1)
        grouped_prompts = condition.prompt_embeds.unsqueeze(1)
        kwargs = {
            "prompt_embeds": grouped_prompts,
            "timesteps": timesteps,
        }
        if self.backbone != "z-image":
            if condition.pooled_prompt_embeds is None:
                raise ValueError(f"{self.backbone} requires pooled prompt embeddings")
            kwargs["pooled_prompt_embeds"] = condition.pooled_prompt_embeds.unsqueeze(1)
        scores = self.model(grouped_latents, **kwargs)
        return {name: value.reshape(latents.shape[0]) for name, value in scores.items()}

    def forward(self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor):
        return self.score(latents, condition, timesteps)


def load_legacy_register(
    checkpoint_path: str | Path,
    *,
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = True,
) -> CheckpointRewardRegister:
    payload = read_legacy_checkpoint(checkpoint_path)
    from .models import (
        FluxLatentRewardGridPoolNoPEMultiHeadModel,
        FluxRewardBackbone,
        SD3LatentRewardGridPoolNoPEMultiHeadModel,
        SD3RewardBackbone,
        ZImageLatentRewardGridPoolNoPEMultiHeadModel,
        ZImageRewardBackbone,
    )
    config = payload["config"]
    architecture = str(config["reward_token"]["architecture"]).lower()
    model_config = config.get("model", {})
    backbone_kwargs = {
        "torch_dtype": dtype,
        "max_sequence_length": int(model_config.get("max_sequence_length", 512)),
        "local_files_only": local_files_only,
    }
    architecture_kwargs = _architecture_kwargs(config)

    if architecture.startswith("flux_"):
        backbone_name = "flux"
        backbone = FluxRewardBackbone.from_pretrained(
            model_name_or_path,
            guidance_scale=float(model_config.get("guidance_scale", 3.5)),
            **backbone_kwargs,
        )
        model = FluxLatentRewardGridPoolNoPEMultiHeadModel(backbone, **architecture_kwargs)
    elif architecture.startswith("zimage_"):
        backbone_name = "z-image"
        backbone = ZImageRewardBackbone.from_pretrained(model_name_or_path, **backbone_kwargs)
        model = ZImageLatentRewardGridPoolNoPEMultiHeadModel(backbone, **architecture_kwargs)
    else:
        backbone_name = "sd3"
        backbone = SD3RewardBackbone.from_pretrained(model_name_or_path, **backbone_kwargs)
        architecture_kwargs["head_input_tokens"] = bool(config["reward_token"].get("head_input_tokens", False))
        model = SD3LatentRewardGridPoolNoPEMultiHeadModel(backbone, **architecture_kwargs)

    state = payload.get("ema_model", payload["model"])
    if hasattr(model, "load_checkpoint_state"):
        model.load_checkpoint_state(state)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return CheckpointRewardRegister(model, backbone_name, tuple(config["reward_token"]["head_names"]))
