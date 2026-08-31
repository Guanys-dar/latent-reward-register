"""FLUX.1-dev integration contract.

Centralized pipeline/component loading, VAE encode/decode, and prompt encoding.

Load-bearing conventions (verified against the vendored diffusers 0.38 Flux code):

  * Timestep: FLUX conditions on sigma DIRECTLY (``pipeline_flux.py`` passes
    ``timestep = t/1000`` and ``FluxTransformer2DModel.forward`` rescales ``*1000``).
    There is NO ``1 - u`` inversion anywhere in the Flux stack.
  * Guidance: dev is guidance-distilled; the raw guidance value (3.5 = FluxPipeline
    default) is passed and the transformer scales it ``*1000`` internally.
  * VAE: 16 channels, shift 0.1159, scale 0.3611.
  * Text: T5-XXL at max_length 512 with NO attention mask (stock Flux attends
    padding), plus the REAL CLIP pooled [768] vector (consumed by the modulation).
  * Scheduler: the stock config uses dynamic shifting; at 1024px (4096 image tokens)
    FluxPipeline resolves mu = max_shift = 1.15 => shift = e^1.15. We pin that static
    shift for train/eval consistency.
"""
from __future__ import annotations

import json
import math
import os

import torch

FLUX_MODEL_PATH = os.environ.get("LRR_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
GUIDANCE_SCALE = 3.5  # FluxPipeline dev default; pinned for train + all eval
T5_MAX_LEN = 512
CLIP_MAX_LEN = 77
T5_EMBED_DIM = 4096
CLIP_POOLED_DIM = 768
# Effective static shift at 1024x1024 (image_seq_len=4096 -> mu=max_shift=1.15).
SCHEDULER_SHIFT = math.exp(1.15)  # 3.158193...

VAE_SCALING = 0.3611
VAE_SHIFT = 0.1159


def vae_encode_1024(vae, pixel_values_fp32: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(pixel_values_fp32).latent_dist.mode()
    return (latents - VAE_SHIFT) * VAE_SCALING


def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    images = vae.decode(latents / VAE_SCALING + VAE_SHIFT).sample
    return (images / 2 + 0.5).clamp(0, 1)


def load_flux_pipeline(dtype: torch.dtype = torch.bfloat16, local_files_only: bool = True, model_name_or_path: str = FLUX_MODEL_PATH):
    """Load the full FluxPipeline (transformer + VAE + CLIP + T5 + tokenizers)."""
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_flux_transformer(dtype: torch.dtype = torch.bfloat16, local_files_only: bool = True, model_name_or_path: str = FLUX_MODEL_PATH):
    """Load ONLY the FluxTransformer2DModel (skips T5/CLIP/VAE — training backbone path)."""
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )


def flux_scheduler_config(model_name_or_path: str = FLUX_MODEL_PATH) -> dict:
    """Stock scheduler config with the static 1024px shift pinned (dynamic shifting off)."""
    scheduler_path = os.path.join(model_name_or_path, "scheduler", "scheduler_config.json")
    if os.path.isfile(scheduler_path):
        with open(scheduler_path) as handle:
            cfg = {key: value for key, value in json.load(handle).items() if not key.startswith("_")}
    else:
        from diffusers import FlowMatchEulerDiscreteScheduler
        cfg = dict(FlowMatchEulerDiscreteScheduler().config)
    cfg["use_dynamic_shifting"] = False
    cfg["shift"] = SCHEDULER_SHIFT
    return cfg


def load_text_encoders(dtype: torch.dtype = torch.bfloat16, local_files_only: bool = True):
    """Load (clip_tokenizer, clip_encoder, t5_tokenizer, t5_encoder) without the transformer/VAE."""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    tok = CLIPTokenizer.from_pretrained(f"{FLUX_MODEL_PATH}/tokenizer", local_files_only=local_files_only)
    te = CLIPTextModel.from_pretrained(
        f"{FLUX_MODEL_PATH}/text_encoder", torch_dtype=dtype, local_files_only=local_files_only
    )
    tok2 = T5TokenizerFast.from_pretrained(f"{FLUX_MODEL_PATH}/tokenizer_2", local_files_only=local_files_only)
    te2 = T5EncoderModel.from_pretrained(
        f"{FLUX_MODEL_PATH}/text_encoder_2", torch_dtype=dtype, local_files_only=local_files_only
    )
    return tok, te, tok2, te2


def encode_prompt_flux(
    clip_tokenizer,
    clip_encoder,
    t5_tokenizer,
    t5_encoder,
    prompts,
    device,
    max_sequence_length: int = T5_MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts exactly as ``FluxPipeline.encode_prompt`` does.

    Returns ``(prompt_embeds [B, max_sequence_length, 4096], pooled [B, 768])``.
    The T5 sequence is FULL padded, no attention mask (stock Flux attends padding);
    the CLIP pooled output is the real modulation input (NOT a dummy).
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # T5 (_get_t5_prompt_embeds)
    t5_inputs = t5_tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = t5_encoder(t5_inputs.input_ids.to(device), output_hidden_states=False)[0]

    # CLIP pooled (_get_clip_prompt_embeds)
    clip_inputs = clip_tokenizer(
        prompts,
        padding="max_length",
        max_length=CLIP_MAX_LEN,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    pooled = clip_encoder(clip_inputs.input_ids.to(device), output_hidden_states=False).pooler_output

    return prompt_embeds, pooled


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, 16, H, W] -> [B, (H/2)*(W/2), 64] (verbatim FluxPipeline._pack_latents)."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(latents: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """[B, grid_h*grid_w, 64] -> [B, 16, 2*grid_h, 2*grid_w] (exact inverse of pack_latents)."""
    b, _, packed_c = latents.shape
    c = packed_c // 4
    latents = latents.view(b, grid_h, grid_w, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, grid_h * 2, grid_w * 2)


def prepare_latent_image_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    """[height*width, 3] RoPE ids (verbatim FluxPipeline._prepare_latent_image_ids,
    2D form — height/width are the PACKED grid dims, i.e. latent_h//2 x latent_w//2)."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)
