"""FLUX.1-dev reward backbone (frozen MMDiT front-end for the reward register).

Mirrors the public surface of ``ZImageRewardBackbone`` (zimage_backbone.py) used by the
engine and the reward-register model, but wraps ``FluxTransformer2DModel``:

  * ``hidden_size``  -> transformer.inner_dim (3072)
  * ``num_layers``   -> 19 double-stream + 38 single-stream = 57 (the register model
    indexes this combined stack; blocks past ``stop_at_layer`` are never executed)
  * timestep/sigma helpers are the scheduler-generic SD3/Z-Image semantics;
    ``sample_timesteps`` returns the continuous flow level u = sigma in [0, 1].
  * ``build_flux_inputs`` runs the frozen front-end of ``FluxTransformer2DModel.forward``
    under ``no_grad``: x_embedder on 2x2-packed latents, context_embedder on T5 embeds,
    temb from (timestep, guidance, CLIP-pooled), and the shared [txt, img] RoPE.

Timestep convention (verified, transformer_flux.py:682): FLUX conditions on sigma
DIRECTLY — ``temb = time_text_embed(u * 1000, guidance * 1000, pooled)``. There is NO
``1 - u`` inversion (that is Z-Image's native convention; copying it here would
re-introduce the fixv2 inverted-t bug in mirror image). ``flux_parity_check.py`` gates
this against the stock forward.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from .. import flux_common


class FluxRewardBackbone(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler_config: dict,
        max_sequence_length: int = 512,
        guidance_scale: float = flux_common.GUIDANCE_SCALE,
    ):
        super().__init__()
        self.transformer = transformer
        # scheduler_config is expected to come from flux_common.flux_scheduler_config()
        # (static 1024px shift pinned, dynamic shifting off).
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(deepcopy(scheduler_config))
        self.max_sequence_length = max_sequence_length
        self.guidance_scale = float(guidance_scale)
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.trainable_layers: tuple[int, ...] = ()
        self.use_gradient_checkpointing = False
        self.freeze_all()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 512,
        local_files_only: bool = True,
        guidance_scale: float = flux_common.GUIDANCE_SCALE,
        revision: str | None = None,
        variant: str | None = None,
    ) -> "FluxRewardBackbone":
        del revision, variant
        model_name_or_path = str(pretrained_model_name_or_path or flux_common.FLUX_MODEL_PATH)
        # Load only the transformer (skip T5/CLIP/VAE — embeddings/latents are precached).
        transformer = flux_common.load_flux_transformer(
            dtype=torch_dtype,
            local_files_only=local_files_only,
            model_name_or_path=model_name_or_path,
        )
        return cls(
            transformer=transformer,
            scheduler_config=flux_common.flux_scheduler_config(model_name_or_path),
            max_sequence_length=max_sequence_length,
            guidance_scale=guidance_scale,
        )

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    @property
    def num_double_layers(self) -> int:
        return len(self.transformer.transformer_blocks)

    @property
    def num_single_layers(self) -> int:
        return len(self.transformer.single_transformer_blocks)

    @property
    def num_layers(self) -> int:
        # Combined double+single stack; the register model indexes into this.
        return self.num_double_layers + self.num_single_layers

    @property
    def hidden_size(self) -> int:
        return int(self.transformer.inner_dim)

    # ------------------------------------------------------------------
    # Freeze / train
    # ------------------------------------------------------------------
    def freeze_all(self) -> None:
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.trainable_layers = ()

    def set_trainable_layers(self, layer_indices: Iterable[int]) -> tuple[int, ...]:
        unique_layers = sorted({int(index) for index in layer_indices})
        for index in unique_layers:
            if index < 0 or index >= self.num_layers:
                raise ValueError(f"Layer index {index} out of range [0, {self.num_layers})")
        self.freeze_all()
        for index in unique_layers:
            block = self.get_block(index)
            block.requires_grad_(True)
        if unique_layers:
            self.transformer.train()
        self.trainable_layers = tuple(unique_layers)
        return self.trainable_layers

    def get_block(self, index: int) -> nn.Module:
        """0-indexed combined-stack lookup: 0..18 double-stream, 19..56 single-stream."""
        n_double = self.num_double_layers
        if index < n_double:
            return self.transformer.transformer_blocks[index]
        return self.transformer.single_transformer_blocks[index - n_double]

    def is_double_block(self, index: int) -> bool:
        return index < self.num_double_layers

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.use_gradient_checkpointing = enabled

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer.eval()  # backbone stays frozen/eval regardless
        return self

    # ------------------------------------------------------------------
    # Timestep / sigma (scheduler-generic; identical semantics to SD3/Z-Image)
    # ------------------------------------------------------------------
    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        *,
        n_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        schedule_timesteps = self.scheduler.timesteps.to(device=device)
        scheduler_sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(device=device)]
        sigma = scheduler_sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def sample_timesteps(
        self,
        *,
        batch_size: int,
        device: torch.device,
        n_dim: int,
        dtype: torch.dtype,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u_flat, sigmas)``: u = sigma in [0, 1] (what the engine passes to
        ``model(..., timesteps=)``) and the n_dim-broadcast sigma tensor for the loss."""
        density = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
            device=device,
        )
        indices = (density * self.num_train_timesteps).long()
        indices = indices.clamp_max(self.num_train_timesteps - 1)
        timesteps_discrete = self.scheduler.timesteps.to(device=device)[indices]
        sigmas = self.get_sigmas(timesteps_discrete, n_dim=n_dim, dtype=dtype, device=device)
        u_flat = sigmas.reshape(batch_size).to(dtype)
        return u_flat, sigmas

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        return (1.0 - sigmas) * latents + sigmas * noise

    # ------------------------------------------------------------------
    # Frozen front-end -> (img tokens, txt tokens, temb, rotary)
    # ------------------------------------------------------------------
    def build_flux_inputs(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        u: torch.Tensor,
    ):
        """Run the frozen front-end of ``FluxTransformer2DModel.forward`` under ``no_grad``.

        Args: ``noisy_latents [B, 16, H, W]`` (unpacked), ``prompt_embeds [B, L, 4096]``
        (T5, full padded, no mask), ``pooled_prompt_embeds [B, 768]`` (CLIP),
        ``u [B]`` = sigma in [0, 1].

        Returns ``(img, txt, temb, rotary, txt_len, img_len)`` where ``img [B, N_img,
        3072]`` is post-x_embedder, ``txt [B, L, 3072]`` post-context_embedder, ``temb
        [B, 3072]``, and ``rotary`` the (cos, sin) tuple over the ``[txt, img]`` sequence.
        """
        t = self.transformer
        device = noisy_latents.device
        with torch.no_grad():
            packed = flux_common.pack_latents(noisy_latents)
            img = t.x_embedder(packed)
            txt = t.context_embedder(prompt_embeds.to(dtype=img.dtype))

            # Verbatim transformer.forward semantics: timestep and guidance are the RAW
            # values (u = sigma, guidance = 3.5) scaled *1000 in the transformer's dtype.
            # u DIRECTLY — no ``1.0 - u`` (Flux != Z-Image; see module docstring).
            timestep = u.to(dtype=img.dtype) * 1000
            guidance = torch.full_like(u, self.guidance_scale).to(dtype=img.dtype) * 1000
            temb = t.time_text_embed(timestep, guidance, pooled_prompt_embeds.to(dtype=img.dtype))

            grid_h = noisy_latents.shape[2] // 2
            grid_w = noisy_latents.shape[3] // 2
            txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=img.dtype)
            img_ids = flux_common.prepare_latent_image_ids(grid_h, grid_w, device, img.dtype)
            ids = torch.cat((txt_ids, img_ids), dim=0)
            rotary = t.pos_embed(ids)

        return img, txt, temb, rotary, int(prompt_embeds.shape[1]), int(grid_h * grid_w)
