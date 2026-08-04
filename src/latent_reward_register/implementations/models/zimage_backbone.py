"""Z-Image reward backbone (frozen single-stream DiT front-end for the reward register).

Mirrors the public surface of ``SD3RewardBackbone`` (backbone.py) used by the engine and
the reward-register model, but wraps the Z-Image ``ZImageTransformer2DModel``:

  * ``hidden_size`` -> transformer.config.dim (3840)
  * ``num_layers``  -> len(transformer.layers) (30)
  * timestep/sigma helpers reuse the SD3 semantics (scheduler-generic), except
    ``sample_timesteps`` returns the continuous flow level u = sigma in [0,1] as the
    ``timesteps`` element (the Z-Image model forward consumes u, not a discrete step).
  * ``build_unified_inputs`` runs the frozen front-end of ``ZImageTransformer2DModel.forward``
    under ``no_grad``, reusing the transformer's own patchify/RoPE/refiner methods so the
    single unified [image, cap] sequence, its RoPE freqs, and the adaLN input are faithful.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling

from .. import zimage_common


class ZImageRewardBackbone(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler_config: dict,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        self.transformer = transformer
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(deepcopy(scheduler_config))
        # Pin the flow-match shift to 6.0 for train/infer consistency.
        try:
            self.scheduler.config.shift = zimage_common.SCHEDULER_SHIFT
        except Exception:
            pass
        self.max_sequence_length = max_sequence_length
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
        revision: str | None = None,
        variant: str | None = None,
    ) -> "ZImageRewardBackbone":
        del revision, variant
        pipeline = zimage_common.load_zimage_pipeline(
            dtype=torch_dtype,
            local_files_only=local_files_only,
            model_name_or_path=str(pretrained_model_name_or_path or zimage_common.Z_IMAGE_PATH),
        )
        transformer = pipeline.transformer
        scheduler_config = dict(pipeline.scheduler.config)
        # Drop the rest of the pipeline; keep only the transformer front-end.
        pipeline.vae = None
        pipeline.text_encoder = None
        return cls(
            transformer=transformer,
            scheduler_config=scheduler_config,
            max_sequence_length=max_sequence_length,
        )

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.transformer.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.transformer.config.dim)

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
            self.transformer.layers[index].requires_grad_(True)
        if unique_layers:
            self.transformer.train()
        self.trainable_layers = tuple(unique_layers)
        return self.trainable_layers

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.use_gradient_checkpointing = enabled

    def train(self, mode: bool = True):
        super().train(mode)
        self.transformer.eval()  # backbone stays frozen/eval regardless
        return self

    # ------------------------------------------------------------------
    # Timestep / sigma (scheduler-generic; identical semantics to SD3)
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
        """Return ``(u_flat, sigmas)`` where ``u_flat`` is the continuous flow level

        u = sigma in [0, 1] (shape ``[batch_size]``, the value the engine passes to
        ``model(..., timesteps=)``) and ``sigmas`` is the n_dim-broadcast sigma tensor
        the DiNA-Thurstone loss consumes for variance scaling (exactly as in SD3).
        """
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
    # Frozen front-end -> unified [image, cap] sequence
    # ------------------------------------------------------------------
    def build_unified_inputs(
        self,
        image_list: list[torch.Tensor],
        cap_list: list[torch.Tensor],
        u: torch.Tensor,
        *,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        """Run the frozen front-end of ``ZImageTransformer2DModel.forward`` (basic mode)
        under ``no_grad`` and return the unified sequence and its RoPE frequencies.

        Returns ``(unified, unified_freqs, unified_mask, adaln_input, x_seqlens, cap_seqlens)``.
        Basic-mode order is ``[x, cap]`` (image first): image span ``[0:N_img]``, caption
        span ``[N_img:N_img+N_cap]``.
        """
        t = self.transformer
        device = image_list[0].device
        with torch.no_grad():
            # 1) adaLN source from the continuous flow level (no pooled text).
            # Z-Image's native timestep convention is t = 1 - sigma (t=1 <=> clean):
            # the pipeline feeds (1000 - t_sched)/1000 and clean reference tokens get
            # t_embedder(ones). u here is sigma, so condition the frozen trunk on 1 - u.
            # (fixv2; the pre-fix run fed u = sigma directly — inverted conditioning.)
            adaln_input = t.t_embedder((1.0 - u) * t.config.t_scale).type_as(image_list[0])

            # 2) patchify (no embedding yet) for image + caption.
            (
                x,
                cap_feats,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = t.patchify_and_embed(image_list, cap_list, patch_size, f_patch_size)

            # 3) image embed + prepare (RoPE / pad) + noise_refiner.
            x_seqlens = [len(xi) for xi in x]
            x = t.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))
            x, x_freqs, x_mask, _, _ = t._prepare_sequence(
                list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, t.x_pad_token, None, device
            )
            for layer in t.noise_refiner:
                x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)

            # 4) caption embed + prepare + context_refiner (modulation=False).
            cap_seqlens = [len(ci) for ci in cap_feats]
            cap_feats = t.cap_embedder(torch.cat(cap_feats, dim=0))
            cap_feats, cap_freqs, cap_mask, _, _ = t._prepare_sequence(
                list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, t.cap_pad_token, None, device
            )
            for layer in t.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, cap_freqs)

            # 5) build the unified [x, cap] sequence.
            unified, unified_freqs, unified_mask, _ = t._build_unified_sequence(
                x, x_freqs, x_seqlens, None,
                cap_feats, cap_freqs, cap_seqlens, None,
                None, None, None, None,
                False, device,
            )

        return unified, unified_freqs, unified_mask, adaln_input, x_seqlens, cap_seqlens

    def build_register_freqs(
        self, num_reward_tokens: int, cap_end: int | list[int], device: torch.device
    ) -> torch.Tensor:
        """RoPE freqs for the position-free reward register.

        Positions: axis-0 = ``cap_end + 1 + i`` for i in 0..num_reward_tokens-1, other axes 0
        (the same axis-0 plane the image tokens start on: native image offset is cap_len + 1).
        ``cap_end`` may be a single int (uniform caps -> ``[1, R, D/2]``, broadcast over batch)
        or a per-sample list of rounded cap lengths (variable caps -> ``[B, R, D/2]``).
        """
        if isinstance(cap_end, int):
            cap_ends = [cap_end]
        else:
            cap_ends = [int(c) for c in cap_end]
        offsets = torch.arange(num_reward_tokens, dtype=torch.int32, device=device)
        pos = torch.zeros((len(cap_ends), num_reward_tokens, 3), dtype=torch.int32, device=device)
        for b, ce in enumerate(cap_ends):
            pos[b, :, 0] = offsets + ce + 1
        freqs = self.transformer.rope_embedder(pos.reshape(-1, 3))  # [B*R, D/2] complex
        return freqs.reshape(len(cap_ends), num_reward_tokens, -1)
