"""SD3 reward backbone: frozen MMDiT front-end for the reward register.

Ported from the research workspace; see docs/source-provenance.md. Kept
checkpoint-faithful: the architecture must stay byte-compatible with the
published checkpoints, so prefer provenance notes over refactoring here.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from diffusers.training_utils import compute_density_for_timestep_sampling


def _run_transformer_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        joint_attention_kwargs=None,
    )


def _reshape_heads(hidden_states: torch.Tensor, heads: int) -> torch.Tensor:
    batch_size, sequence_length, dim = hidden_states.shape
    head_dim = dim // heads
    return hidden_states.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)


def _merge_heads(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, heads, sequence_length, head_dim = hidden_states.shape
    return hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, heads * head_dim)


def load_sd3_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.float16,
    local_files_only: bool = True,
    revision: str | None = None,
    variant: str | None = None,
) -> StableDiffusion3Pipeline:
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        str(pretrained_model_name_or_path),
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        revision=revision,
        variant=variant,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


class SD3RewardBackbone(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler_config: dict,
        max_sequence_length: int = 256,
    ):
        super().__init__()
        self.transformer = transformer
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(deepcopy(scheduler_config))
        self.max_sequence_length = max_sequence_length
        self.trainable_layers: tuple[int, ...] = ()
        self.use_gradient_checkpointing = False
        self.freeze_all()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        torch_dtype: torch.dtype = torch.float16,
        max_sequence_length: int = 256,
        local_files_only: bool = True,
        revision: str | None = None,
        variant: str | None = None,
    ) -> "SD3RewardBackbone":
        pipeline = load_sd3_pipeline(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            revision=revision,
            variant=variant,
        )
        return cls(
            transformer=pipeline.transformer,
            scheduler_config=dict(pipeline.scheduler.config),
            max_sequence_length=max_sequence_length,
        )

    @property
    def num_layers(self) -> int:
        return len(self.transformer.transformer_blocks)

    @property
    def hidden_size(self) -> int:
        return self.transformer.inner_dim

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
            self.transformer.transformer_blocks[index].requires_grad_(True)

        if unique_layers:
            self.transformer.train()
        self.trainable_layers = tuple(unique_layers)
        return self.trainable_layers

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.use_gradient_checkpointing = enabled

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
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps.to(device=device)]
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
        density = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
            device=device,
        )
        indices = (density * self.scheduler.config.num_train_timesteps).long()
        indices = indices.clamp_max(self.scheduler.config.num_train_timesteps - 1)
        timesteps = self.scheduler.timesteps.to(device=device)[indices]
        sigmas = self.get_sigmas(timesteps, n_dim=n_dim, dtype=dtype, device=device)
        return timesteps, sigmas

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        return (1.0 - sigmas) * latents + sigmas * noise

    def extract_image_hidden_states(
        self,
        noisy_latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        selected_layers: Iterable[int],
        stop_at_layer: int | None = None,
    ) -> dict[int, torch.Tensor]:
        layer_list = sorted({int(index) for index in selected_layers})
        if not layer_list:
            raise ValueError("`selected_layers` must be non-empty")
        for index in layer_list:
            if index < 0 or index >= self.num_layers:
                raise ValueError(f"Layer index {index} out of range [0, {self.num_layers})")

        stop_index = stop_at_layer if stop_at_layer is not None else (max(layer_list) + 1)
        if stop_index <= 0 or stop_index > self.num_layers:
            raise ValueError(f"`stop_at_layer` must be in [1, {self.num_layers}], got {stop_index}")

        device = noisy_latents.device
        model_dtype = next(self.transformer.parameters()).dtype

        hidden_states = self.transformer.pos_embed(noisy_latents.to(dtype=model_dtype))
        temb = self.transformer.time_text_embed(
            timesteps.to(device=device),
            pooled_prompt_embeds.to(device=device, dtype=model_dtype),
        )
        encoder_hidden_states = self.transformer.context_embedder(
            prompt_embeds.to(device=device, dtype=model_dtype)
        )

        outputs: dict[int, torch.Tensor] = {}
        trainable_set = set(self.trainable_layers)
        for index, block in enumerate(self.transformer.transformer_blocks):
            if index >= stop_index:
                break

            checkpoint_block = (
                self.training
                and self.use_gradient_checkpointing
                and torch.is_grad_enabled()
                and index in trainable_set
                and not block.context_pre_only
            )

            if checkpoint_block:
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    partial(_run_transformer_block, block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=None,
                )

            if index in layer_list:
                outputs[index] = hidden_states

        return outputs

    def _run_joint_block_collect_cross(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        collect_cross: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run one frozen SD3 joint block by hand (mirrors latent_reward_grid's
        ``_run_frozen_sd3_block``) and, when ``collect_cross`` is set, additionally
        return the image->text cross-attention weights, mean-pooled over heads, as
        ``[B, N_img, N_txt]``.

        The diffusers ``JointAttnProcessor2_0`` runs fused SDPA and discards the
        attention weights, so we recompute the post-norm Q/K projections here. The
        full joint attention is still run to faithfully propagate the residual stream
        to deeper blocks.
        """
        heads = block.attn.heads
        cross_attn: torch.Tensor | None = None

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            block.norm1_context(encoder_hidden_states, emb=temb)
        )

        q_img = _reshape_heads(block.attn.to_q(norm_hidden_states), heads)
        k_img = _reshape_heads(block.attn.to_k(norm_hidden_states), heads)
        v_img = _reshape_heads(block.attn.to_v(norm_hidden_states), heads)
        q_txt = _reshape_heads(block.attn.add_q_proj(norm_encoder_hidden_states), heads)
        k_txt = _reshape_heads(block.attn.add_k_proj(norm_encoder_hidden_states), heads)
        v_txt = _reshape_heads(block.attn.add_v_proj(norm_encoder_hidden_states), heads)

        if block.attn.norm_q is not None:
            q_img = block.attn.norm_q(q_img)
        if block.attn.norm_k is not None:
            k_img = block.attn.norm_k(k_img)
        if block.attn.norm_added_q is not None:
            q_txt = block.attn.norm_added_q(q_txt)
        if block.attn.norm_added_k is not None:
            k_txt = block.attn.norm_added_k(k_txt)

        if collect_cross:
            # Image queries attend to text keys: softmax over the text axis, per head,
            # then mean over heads -> [B, N_img, N_txt]. Done in fp32 for stability.
            head_dim = q_img.shape[-1]
            scale = head_dim ** -0.5
            scores = torch.matmul(q_img.float(), k_txt.float().transpose(-1, -2)) * scale
            scores = torch.softmax(scores, dim=-1)
            cross_attn = scores.mean(dim=1)

        joint_query = torch.cat([q_img, q_txt], dim=2)
        joint_key = torch.cat([k_img, k_txt], dim=2)
        joint_value = torch.cat([v_img, v_txt], dim=2)
        joint_output = torch.nn.functional.scaled_dot_product_attention(
            joint_query, joint_key, joint_value, dropout_p=0.0, is_causal=False,
        )
        joint_output = _merge_heads(joint_output)
        attn_output, context_attn_output = joint_output.split(
            (hidden_states.shape[1], encoder_hidden_states.shape[1]), dim=1,
        )

        attn_output = block.attn.to_out[0](attn_output)
        attn_output = block.attn.to_out[1](attn_output)

        hidden_states = hidden_states + gate_msa[:, None] * attn_output
        norm_hidden_states_ff = block.norm2(hidden_states)
        norm_hidden_states_ff = norm_hidden_states_ff * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * block.ff(norm_hidden_states_ff)

        if not block.context_pre_only:
            context_attn_output = block.attn.to_add_out(context_attn_output)
            encoder_hidden_states = encoder_hidden_states + c_gate_msa[:, None] * context_attn_output
            norm_encoder_hidden_states_ff = block.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states_ff = (
                norm_encoder_hidden_states_ff * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            )
            encoder_hidden_states = (
                encoder_hidden_states
                + c_gate_mlp[:, None] * block.ff_context(norm_encoder_hidden_states_ff)
            )

        return hidden_states, encoder_hidden_states, cross_attn

    def extract_cross_attention_maps(
        self,
        noisy_latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        selected_layers: Iterable[int],
        stop_at_layer: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Capture image->text cross-attention maps from selected joint blocks.

        Returns ``{layer_index: tensor[B, N_txt, h_tok, w_tok]}`` where each text
        token has a spatial attention map over the image latent grid. The backbone
        is frozen, so this runs under ``no_grad``.
        """
        layer_list = sorted({int(index) for index in selected_layers})
        if not layer_list:
            raise ValueError("`selected_layers` must be non-empty")
        for index in layer_list:
            if index < 0 or index >= self.num_layers:
                raise ValueError(f"Layer index {index} out of range [0, {self.num_layers})")
            if self.transformer.transformer_blocks[index].context_pre_only:
                raise ValueError(
                    f"Layer {index} is the final context_pre_only block and has no text "
                    "stream to attend to; choose a lower layer."
                )

        stop_index = stop_at_layer if stop_at_layer is not None else (max(layer_list) + 1)
        if stop_index <= 0 or stop_index > self.num_layers:
            raise ValueError(f"`stop_at_layer` must be in [1, {self.num_layers}], got {stop_index}")

        device = noisy_latents.device
        model_dtype = next(self.transformer.parameters()).dtype
        patch_size = int(getattr(self.transformer.config, "patch_size", 2))
        h_tok = noisy_latents.shape[-2] // patch_size
        w_tok = noisy_latents.shape[-1] // patch_size

        layer_set = set(layer_list)
        with torch.no_grad():
            hidden_states = self.transformer.pos_embed(noisy_latents.to(dtype=model_dtype))
            temb = self.transformer.time_text_embed(
                timesteps.to(device=device),
                pooled_prompt_embeds.to(device=device, dtype=model_dtype),
            )
            encoder_hidden_states = self.transformer.context_embedder(
                prompt_embeds.to(device=device, dtype=model_dtype)
            )

            outputs: dict[int, torch.Tensor] = {}
            for index, block in enumerate(self.transformer.transformer_blocks):
                if index >= stop_index:
                    break
                collect = index in layer_set
                hidden_states, encoder_hidden_states, cross_attn = self._run_joint_block_collect_cross(
                    block, hidden_states, encoder_hidden_states, temb, collect_cross=collect
                )
                if collect and cross_attn is not None:
                    batch_size, num_img, num_txt = cross_attn.shape
                    if num_img != h_tok * w_tok:
                        raise ValueError(
                            f"image token count {num_img} != h_tok*w_tok ({h_tok}x{w_tok}); "
                            "check patch_size / latent shape"
                        )
                    # [B, N_img, N_txt] -> [B, N_txt, h_tok, w_tok]
                    outputs[index] = (
                        cross_attn.transpose(1, 2).reshape(batch_size, num_txt, h_tok, w_tok)
                    )

        return outputs

    def trainable_backbone_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            str(index): self.transformer.transformer_blocks[index].state_dict()
            for index in self.trainable_layers
        }

    def load_trainable_backbone_state_dict(self, state_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        for raw_index, block_state in state_dict.items():
            index = int(raw_index)
            self.transformer.transformer_blocks[index].load_state_dict(block_state)

