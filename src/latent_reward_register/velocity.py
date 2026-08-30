"""Backbone velocity models: the last piece between the algorithms and a real run.

`flowmatch` needs a callable returning the classifier-free-guided flow velocity;
`rollout` needs the same callable with LoRA weights in the graph. Both are thin
wrappers over a frozen transformer's standard forward, which is why they live
here rather than inside the sampler: nothing in `sampling.py` or `rollout.py`
should know which backbone it is driving.

Two conventions are easy to get wrong and are handled here once:

- **Timesteps, not sigmas.** FlowMatch transformers take `sigma * 1000`.
  `flowmatch.timesteps_for_sigma` does the conversion; these wrappers receive
  the already-scaled value.
- **CFG needs both branches.** A guided velocity is
  `uncond + scale * (cond - uncond)`, so the conditional and unconditional
  predictions must come from the same weights in the same call.
"""
from __future__ import annotations

from typing import Protocol

import torch

from .flowmatch import classifier_free_velocity
from .types import RegisterCondition


class Transformer(Protocol):
    """The subset of a diffusers transformer these wrappers use."""

    def __call__(self, **kwargs) -> object: ...


def _sample(output) -> torch.Tensor:
    """diffusers returns a dataclass with `.sample`; a bare tensor also works."""
    return output.sample if hasattr(output, "sample") else output[0]


class SD3VelocityModel:
    """Classifier-free-guided velocity from an SD3 MMDiT.

    ``negative`` supplies the unconditional branch. Omit it to run without CFG,
    which is what a distilled student does at deployment.
    """

    def __init__(
        self,
        transformer,
        *,
        guidance_scale: float = 4.5,
        negative: RegisterCondition | None = None,
    ):
        self.transformer = transformer
        self.guidance_scale = float(guidance_scale)
        self.negative = negative

    def _forward(self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor):
        if condition.pooled_prompt_embeds is None:
            raise ValueError("SD3 requires pooled prompt embeddings")
        dtype = next(self.transformer.parameters()).dtype
        return _sample(
            self.transformer(
                hidden_states=latents.to(dtype=dtype),
                encoder_hidden_states=condition.prompt_embeds.to(dtype=dtype),
                pooled_projections=condition.pooled_prompt_embeds.to(dtype=dtype),
                timestep=timesteps,
                return_dict=True,
            )
        )

    def __call__(
        self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del kwargs
        conditional = self._forward(latents, condition, timesteps)
        if self.negative is None or self.guidance_scale == 1.0:
            return conditional.float()
        unconditional = self._forward(latents, self.negative, timesteps)
        return classifier_free_velocity(
            conditional.float(), unconditional.float(), self.guidance_scale
        )


class FluxVelocityModel:
    """Velocity from a FLUX transformer.

    FLUX.1-dev is guidance-distilled: the scale is an *embedded* conditioning
    input, not a two-branch combination, so there is no unconditional forward.
    """

    def __init__(
        self,
        transformer,
        *,
        guidance_scale: float = 3.5,
        image_ids: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
    ):
        self.transformer = transformer
        self.guidance_scale = float(guidance_scale)
        self.image_ids = image_ids
        self.text_ids = text_ids

    def __call__(
        self, latents: torch.Tensor, condition: RegisterCondition, timesteps: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del kwargs
        if condition.pooled_prompt_embeds is None:
            raise ValueError("FLUX requires pooled prompt embeddings")
        dtype = next(self.transformer.parameters()).dtype
        guidance = torch.full(
            (latents.shape[0],), self.guidance_scale, device=latents.device, dtype=torch.float32
        )
        arguments = {
            "hidden_states": latents.to(dtype=dtype),
            "encoder_hidden_states": condition.prompt_embeds.to(dtype=dtype),
            "pooled_projections": condition.pooled_prompt_embeds.to(dtype=dtype),
            # FLUX expects timesteps normalized to [0, 1].
            "timestep": (timesteps / 1000.0).to(dtype=dtype),
            "guidance": guidance.to(dtype=dtype),
            "return_dict": True,
        }
        if self.image_ids is not None:
            arguments["img_ids"] = self.image_ids
        if self.text_ids is not None:
            arguments["txt_ids"] = self.text_ids
        return _sample(self.transformer(**arguments)).float()


def attach_lora_student(
    transformer,
    *,
    backbone: str = "sd3",
    rank: int = 32,
    alpha: int = 64,
    target_modules=None,
):
    """Wrap a frozen transformer in a LoRA adapter and return it with its parameters.

    RG-OPD trains only the adapter: the base transformer stays frozen, which is
    what makes the student cheap and keeps the teacher comparison meaningful.
    Presets use rank 32 / alpha 64.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LoRA students need peft. Install with: pip install -e '.[models]'"
        ) from error

    if target_modules is None:
        if backbone == "flux":
            target_modules = (
                r"(?:transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)|ff\.net\.(?:0\.proj|2)|ff_context\.net\.(?:0\.proj|2))|single_transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v)|proj_mlp|proj_out))"
            )
        else:
            target_modules = (
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            )
    transformer.requires_grad_(False)
    student = get_peft_model(
        transformer,
        LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules, init_lora_weights="gaussian"),
    )
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            f"LoRA attached no trainable parameters; target_modules={target_modules} "
            "probably match nothing in this transformer"
        )
    return student, trainable
