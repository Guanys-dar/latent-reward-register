from __future__ import annotations

from typing import Any

import torch

from latent_reward_register.types import RegisterCondition

from .base import BackboneAdapter, BackboneFeatures
from .registry import register_backbone


class DiffusersBackboneAdapter(BackboneAdapter):
    pipeline_class_name: str

    def __init__(self, model_name_or_path: str, *, dtype: torch.dtype = torch.bfloat16, local_files_only: bool = False):
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.pipeline = self._load_pipeline()
        self.transformer = self.pipeline.transformer
        self.transformer.requires_grad_(False).eval()

    def _load_pipeline(self):
        import diffusers

        pipeline_class = getattr(diffusers, self.pipeline_class_name)
        return pipeline_class.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        )

    @property
    def hidden_size(self) -> int:
        if hasattr(self.transformer, "inner_dim"):
            return int(self.transformer.inner_dim)
        if hasattr(self.transformer.config, "dim"):
            return int(self.transformer.config.dim)
        config = self.transformer.config
        return int(config.num_attention_heads) * int(config.attention_head_dim)

    def extract_features(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        *,
        reward_tokens: torch.Tensor,
        feature_layers: tuple[int, ...],
    ) -> BackboneFeatures:
        extractor = getattr(self, "feature_extractor", None)
        if extractor is None:
            raise RuntimeError(
                f"{self.name} feature extraction requires the pinned compatibility implementation; "
                "install the repository rather than importing this adapter file in isolation"
            )
        return extractor(latents, condition, sigma, reward_tokens=reward_tokens, feature_layers=feature_layers)

    def reference_step(
        self,
        latents: torch.Tensor,
        condition: RegisterCondition,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        stepper = getattr(self, "sampler_step", None)
        if stepper is None:
            raise RuntimeError(f"{self.name} sampler compatibility implementation is not installed")
        return stepper(latents, condition, sigma, next_sigma, **kwargs)


class SD3Adapter(DiffusersBackboneAdapter):
    name = "sd3"
    pipeline_class_name = "StableDiffusion3Pipeline"


class FluxAdapter(DiffusersBackboneAdapter):
    name = "flux"
    pipeline_class_name = "FluxPipeline"


class ZImageAdapter(DiffusersBackboneAdapter):
    name = "z-image"
    pipeline_class_name = "ZImagePipeline"


register_backbone("sd3", SD3Adapter)
register_backbone("flux", FluxAdapter)
register_backbone("z-image", ZImageAdapter)
