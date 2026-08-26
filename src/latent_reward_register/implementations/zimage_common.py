"""Z-Image integration contract.

Centralized pipeline load + VAE encode/decode + Qwen3 prompt-encode helpers used by
the cache scripts, the reward-model scoring, and the reward-register backbone so that
every code path shares one implementation. Single implementation shared by caching, scoring, and the reward register, so
those paths cannot drift apart.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from diffusers import ZImagePipeline
else:
    ZImagePipeline = Any

Z_IMAGE_PATH = os.environ.get("LRR_ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
VAE_SCALING = 0.3611
VAE_SHIFT = 0.1159
SCHEDULER_SHIFT = 6.0
CAP_MAX_LEN = 512  # fixed caption length for the reward stack (fixed by the reward stack)
T_SCALE = 1000.0  # transformer.config.t_scale


def load_zimage_pipeline(dtype: torch.dtype = torch.bfloat16, local_files_only: bool = True, model_name_or_path: str = Z_IMAGE_PATH) -> ZImagePipeline:
    """Load the Z-Image pipeline, pin the scheduler shift to 6.0, and disable the progress bar."""
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    # Pin the flow-match scheduler shift for train/infer consistency (checkpoint value).
    try:
        pipe.scheduler.config.shift = SCHEDULER_SHIFT
        if hasattr(pipe.scheduler, "set_shift"):
            pipe.scheduler.set_shift(SCHEDULER_SHIFT)
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe


def vae_encode_1024(vae, pixel_values_fp32: torch.Tensor) -> torch.Tensor:
    """Encode 1024px RGB pixels (fp32, CHW in [-1,1]) into Z-Image Flux-VAE latents.

    z = vae.encode(x).latent_dist.mode(); returns (z - VAE_SHIFT) * VAE_SCALING (fp32).
    """
    z = vae.encode(pixel_values_fp32).latent_dist.mode()
    return (z - VAE_SHIFT) * VAE_SCALING


def vae_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents back to [0,1] images (differentiable path; VAE in fp32)."""
    img = vae.decode(latents / VAE_SCALING + VAE_SHIFT).sample
    return (img / 2 + 0.5).clamp(0, 1)


_CAP_LEN_TOKENIZER = None
_CAP_LEN_CACHE: dict[str, int] = {}


def qwen3_cap_len(prompt: str, max_length: int = CAP_MAX_LEN) -> int:
    """Real (unpadded) token length of ``prompt`` under the exact templating used by
    ``encode_prompt_qwen3`` — i.e. how many leading rows of the cached full-padded
    ``[max_length, 2560]`` prompt-embed tensor are real tokens (right padding).

    Lazily loads the Z-Image Qwen tokenizer; results are memoized per prompt string.
    """
    global _CAP_LEN_TOKENIZER
    cached = _CAP_LEN_CACHE.get(prompt)
    if cached is not None:
        return cached
    if _CAP_LEN_TOKENIZER is None:
        from transformers import AutoTokenizer

        _CAP_LEN_TOKENIZER = AutoTokenizer.from_pretrained(
            f"{Z_IMAGE_PATH}/tokenizer", local_files_only=True
        )
    templated = _CAP_LEN_TOKENIZER.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    length = len(
        _CAP_LEN_TOKENIZER(templated, truncation=True, max_length=max_length).input_ids
    )
    _CAP_LEN_CACHE[prompt] = length
    return length


def encode_prompt_qwen3(
    text_encoder,
    tokenizer,
    prompts,
    device,
    max_length: int = CAP_MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with Qwen3 and return (hidden_states[-2], attention_mask).

    Applies the Qwen chat template (add_generation_prompt=True, enable_thinking=True),
    tokenizes with padding="max_length" to ``max_length``, and returns the FULL padded
    ``[B, max_length, 2560]`` hidden state (NOT trimmed by mask) plus the attention mask.
    Caption length is fixed so cached prompt embeddings stay interchangeable.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    templated = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in prompts
    ]
    text_inputs = tokenizer(
        templated,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask.bool(),
        output_hidden_states=True,
    )
    # Keep the full padded [B, max_length, 2560] tensor (do NOT trim by mask).
    return outputs.hidden_states[-2], attention_mask
