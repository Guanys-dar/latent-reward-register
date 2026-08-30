"""Assets on disk -> running task: the layer the CLI subcommands execute.

The algorithm modules take their dependencies as parameters — a velocity model
is a callable, a reference transition is a callable, an encoder is a callable —
which is what lets them be tested on CPU with no weights. Something still has to
supply the real ones. That is this module, and it is deliberately the only place
that imports diffusers pipelines.

Nothing here is needed to read the algorithms. Start at ``sampling.py`` or
``rollout.py`` for those.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .flowmatch import make_reference_step, sigma_schedule
from .guidance import GuidanceSchedule
from .local_config import LocalConfig
from .types import RegisterCondition
from .velocity import FluxVelocityModel, SD3VelocityModel

# Which pipeline class loads each backbone. Kept here so no algorithm module
# needs to know diffusers exists.
_PIPELINES: Mapping[str, tuple[str, str]] = {
    "sd3": ("diffusers", "StableDiffusion3Pipeline"),
    "flux": ("diffusers", "FluxPipeline"),
}

DTYPES: Mapping[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown precision {name!r}; expected one of {sorted(DTYPES)}") from None


def load_pipeline(
    backbone: str,
    *,
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
):
    """Load the generator pipeline for one backbone.

    Z-Image is absent on purpose: no released downstream experiment consumes a
    Z-Image register, so there is no sampling or distillation path for it.
    """
    try:
        module_name, class_name = _PIPELINES[backbone]
    except KeyError:
        raise ValueError(
            f"No sampling pipeline for backbone {backbone!r}; available: {sorted(_PIPELINES)}"
        ) from None
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Sampling needs the pinned diffusers revision. Install with: pip install -e '.[models]'"
        ) from error
    pipeline = getattr(module, class_name).from_pretrained(
        model_name_or_path, torch_dtype=dtype, local_files_only=local_files_only
    )
    return pipeline


@dataclass(frozen=True)
class PromptEncoding:
    """A conditioning pair: what the register scores and what the sampler steers."""

    condition: RegisterCondition
    negative: RegisterCondition | None = None


def encode_prompts(
    pipeline,
    prompts: list[str],
    *,
    backbone: str,
    negative_prompt: str = "",
    device: torch.device | str = "cpu",
) -> PromptEncoding:
    """Run the pipeline's text encoders once, outside the sampling loop.

    Encoding inside the loop would repeat identical work at every step and is a
    common source of wasted time in reward-guided sampling.
    """
    with torch.no_grad():
        if backbone == "sd3":
            prompt_embeds, negative_embeds, pooled, negative_pooled = pipeline.encode_prompt(
                prompt=prompts, prompt_2=prompts, prompt_3=prompts, device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt or None,
            )
            return PromptEncoding(
                condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled),
                negative=RegisterCondition(
                    prompt_embeds=negative_embeds, pooled_prompt_embeds=negative_pooled
                ),
            )
        if backbone == "flux":
            prompt_embeds, pooled, _text_ids = pipeline.encode_prompt(
                prompt=prompts, prompt_2=prompts, device=device, num_images_per_prompt=1
            )
            # FLUX.1-dev is guidance-distilled: the scale is an embedded input, so
            # there is no unconditional branch to encode.
            return PromptEncoding(
                condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled)
            )
    raise ValueError(f"No prompt encoder wired for backbone {backbone!r}")


def build_velocity_model(
    backbone: str,
    transformer,
    *,
    guidance_scale: float,
    negative: RegisterCondition | None = None,
):
    """The classifier-free-guided velocity callable for one backbone."""
    if backbone == "sd3":
        return SD3VelocityModel(transformer, guidance_scale=guidance_scale, negative=negative)
    if backbone == "flux":
        return FluxVelocityModel(transformer, guidance_scale=guidance_scale)
    raise ValueError(f"No velocity model for backbone {backbone!r}")


@dataclass(frozen=True)
class SampleRequest:
    """One reward-guided sampling run."""

    backbone: str
    prompts: list[str]
    steps: int
    resolution: tuple[int, int]
    heads: tuple[str, ...]
    schedule: GuidanceSchedule
    text_guidance_scale: float
    register_checkpoint: str | Path
    seed: int = 42
    shift: float = 3.0
    precision: str = "bf16"
    device: str = "cuda"
    local_files_only: bool = False


def _latent_shape(backbone: str, resolution: tuple[int, int], batch: int) -> tuple[int, ...]:
    """Latent shape for a target pixel resolution.

    Both backbones use an 8x VAE. SD3 keeps latents as a 16-channel image; FLUX
    packs 2x2 patches into a sequence of 64-wide tokens, which is the layout its
    transformer expects.
    """
    height, width = resolution
    if height % 16 or width % 16:
        raise ValueError(f"Resolution must be a multiple of 16, got {resolution}")
    latent_height, latent_width = height // 8, width // 8
    if backbone == "sd3":
        return (batch, 16, latent_height, latent_width)
    if backbone == "flux":
        return (batch, (latent_height // 2) * (latent_width // 2), 64)
    raise ValueError(f"Unknown latent layout for backbone {backbone!r}")


def run_sampling(request: SampleRequest, *, local: LocalConfig | None = None) -> dict[str, Any]:
    """Load real weights and sample with reward guidance. Returns run metadata.

    This is the executable form of ``configs/rgs/*/paper.yaml``. Everything it
    does is available piecewise through the Python API; this exists so a reader
    can reproduce a trajectory without writing glue code first.
    """
    from .implementations import load_legacy_register
    from .sampling import reward_guided_sample

    local = local or LocalConfig()
    dtype = resolve_dtype(request.precision)
    device = torch.device(request.device)
    model_path = local.model_path(request.backbone, "")
    if not model_path:
        raise ValueError(
            f"No model path for {request.backbone!r}: set models.{request.backbone} in "
            "configs/local.yaml, or pass --model-path"
        )

    pipeline = load_pipeline(
        request.backbone,
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    pipeline.to(device)

    register = load_legacy_register(
        local.resolve_checkpoint(request.register_checkpoint),
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    register.to(device).eval()

    encoding = encode_prompts(
        pipeline, request.prompts, backbone=request.backbone, device=device
    )
    velocity = build_velocity_model(
        request.backbone,
        pipeline.transformer,
        guidance_scale=request.text_guidance_scale,
        negative=encoding.negative,
    )

    generator = torch.Generator(device="cpu").manual_seed(request.seed)
    shape = _latent_shape(request.backbone, request.resolution, len(request.prompts))
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    sigmas = sigma_schedule(request.steps, shift=request.shift)
    final, trace = reward_guided_sample(
        register=register,
        latents=latents,
        condition=encoding.condition,
        sigmas=sigmas,
        heads=request.heads,
        schedule=request.schedule,
        reference_step=make_reference_step(velocity),
        return_trace=True,
    )
    # The pipeline is returned so the caller can decode without reloading the VAE.
    return {
        "backbone": request.backbone,
        "prompts": len(request.prompts),
        "steps": trace.steps,
        "guided_steps": trace.guided_steps,
        "guidance_fraction": trace.guidance_fraction,
        "scales": list(trace.scales),
        "latents": final,
        "pipeline": pipeline,
        "seed": request.seed,
    }


def decode_latents(pipeline, latents: torch.Tensor, *, backbone: str, resolution: tuple[int, int]):
    """VAE-decode sampled latents to PIL images."""
    height, width = resolution
    with torch.no_grad():
        if backbone == "flux":
            latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        scaling = pipeline.vae.config.scaling_factor
        shift = getattr(pipeline.vae.config, "shift_factor", None) or 0.0
        decoded = pipeline.vae.decode(
            latents.to(pipeline.vae.dtype) / scaling + shift, return_dict=False
        )[0]
    return pipeline.image_processor.postprocess(decoded, output_type="pil")


@dataclass(frozen=True)
class TrainRegisterRequest:
    """One register-training run over a prepared group manifest."""

    backbone: str
    manifest: str | Path
    output_directory: str | Path
    register_config: Mapping[str, Any]
    train_config: Mapping[str, Any]
    group_size: int = 4
    batch_size: int = 2
    max_batches: int | None = None
    precision: str = "bf16"
    device: str = "cuda"
    local_files_only: bool = False


def run_register_training(
    request: TrainRegisterRequest, *, local: LocalConfig | None = None
) -> dict[str, Any]:
    """Build a register from config and train it on a prepared manifest.

    The executable form of ``configs/register/*/paper.yaml``. Head names come
    from config and stay positional: ``head_names[i]`` pairs with
    ``score_keys[i]``, and the loss aligns targets the same way, so a reordering
    silently mistrains rather than failing.
    """
    from .backbones import build_register
    from .checkpoint import CheckpointManifest
    from .dataset import DatasetConfig, count_usable_groups, iter_group_batches
    from .training import TrainConfig, train_register

    local = local or LocalConfig()
    dtype = resolve_dtype(request.precision)
    head_names = tuple(request.register_config["head_names"])
    feature_layers = tuple(request.register_config["feature_layers"])
    manifest_path = local.resolve_data(request.manifest)

    usable, total = count_usable_groups(manifest_path, group_size=request.group_size)
    if not usable:
        raise ValueError(
            f"{manifest_path}: no group holds {request.group_size} images; "
            f"{total} groups read. Lower group_size or check the manifest."
        )

    architecture_keys = {
        key: value
        for key, value in request.register_config.items()
        if key not in {"head_names", "feature_layers", "score_keys", "architecture"}
    }
    register = build_register(
        request.backbone,
        head_names=head_names,
        feature_layers=feature_layers,
        model_name_or_path=local.model_path(request.backbone, "") or None,
        dtype=dtype,
        local_files_only=request.local_files_only,
        **architecture_keys,
    )
    register.to(torch.device(request.device))

    dataset_config = DatasetConfig(
        group_size=request.group_size,
        batch_size=request.batch_size,
        heads=head_names,
        require_pooled=request.backbone in {"sd3", "flux"},
    )
    batches = list(
        iter_group_batches(
            manifest_path, config=dataset_config, local=local, limit=request.max_batches
        )
    )

    train_register(
        model=register,
        batches=batches,
        config=TrainConfig(
            learning_rate=float(request.train_config.get("learning_rate", 5e-5)),
            weight_decay=float(request.train_config.get("weight_decay", 0.01)),
            epochs=int(request.train_config.get("epochs", 1)),
            ema_decay=float(request.train_config.get("ema_decay", 0.999)),
            max_grad_norm=float(request.train_config.get("max_grad_norm", 1.0)),
        ),
        output_dir=request.output_directory,
        manifest=CheckpointManifest(
            format_version=1,
            backbone=request.backbone,
            backbone_revision=str(request.register_config.get("revision", "unknown")),
            adapter=request.backbone,
            head_names=head_names,
            feature_layers=feature_layers,
            noise_convention="sigma",
        ),
        register_config=dict(request.register_config),
    )
    return {
        "backbone": request.backbone,
        "groups_usable": usable,
        "groups_total": total,
        "batches": len(batches),
        "heads": list(head_names),
        "output_directory": str(request.output_directory),
    }


@dataclass(frozen=True)
class TrainRGOPDRequest:
    """One RG-OPD distillation run."""

    backbone: str
    prompts: list[str]
    register_checkpoint: str | Path
    output_directory: str | Path
    schedule: GuidanceSchedule
    heads: tuple[str, ...]
    rollout_steps: int = 10
    optimized_steps: int = 9
    resolution: tuple[int, int] = (1024, 1024)
    text_guidance_scale: float = 4.5
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 1e-4
    rounds: int = 1
    batch_size: int = 1
    seed: int = 42
    shift: float = 3.0
    precision: str = "bf16"
    device: str = "cuda"
    local_files_only: bool = False


def run_rgopd_training(
    request: TrainRGOPDRequest, *, local: LocalConfig | None = None
) -> dict[str, Any]:
    """Distill reward guidance into a LoRA student along its own rollouts.

    The executable form of ``configs/rgopd/*/paper.yaml``. The teacher here is
    the same ``RewardGradientTeacher`` the sampler uses, so the student cannot be
    trained against guidance the sampler would not apply.
    """
    from .flowmatch import make_student_policy
    from .implementations import load_legacy_register
    from .rollout import RolloutConfig, train_rgopd_rollout
    from .teacher import RewardGradientTeacher
    from .velocity import attach_lora_student

    local = local or LocalConfig()
    dtype = resolve_dtype(request.precision)
    device = torch.device(request.device)
    model_path = local.model_path(request.backbone, "")
    if not model_path:
        raise ValueError(
            f"No model path for {request.backbone!r}: set models.{request.backbone} in "
            "configs/local.yaml, or pass --model-path"
        )

    pipeline = load_pipeline(
        request.backbone,
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    pipeline.to(device)

    register = load_legacy_register(
        local.resolve_checkpoint(request.register_checkpoint),
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    register.to(device).eval()

    encoding = encode_prompts(
        pipeline, request.prompts, backbone=request.backbone, device=device
    )
    # The frozen anchor and the student must share a velocity convention, so both
    # are built from the same transformer: the anchor from the base weights, the
    # student from the same weights plus the adapter.
    reference_velocity = build_velocity_model(
        request.backbone,
        pipeline.transformer,
        guidance_scale=request.text_guidance_scale,
        negative=encoding.negative,
    )
    reference_step = make_reference_step(reference_velocity)

    student_transformer, trainable = attach_lora_student(
        pipeline.transformer, rank=request.lora_rank, alpha=request.lora_alpha
    )
    student_velocity = build_velocity_model(
        request.backbone,
        student_transformer,
        guidance_scale=request.text_guidance_scale,
        negative=encoding.negative,
    )
    student = make_student_policy(student_velocity)

    teacher = RewardGradientTeacher(register, schedule=request.schedule, heads=request.heads)
    optimizer = torch.optim.AdamW(trainable, lr=request.learning_rate)
    sigmas = sigma_schedule(request.rollout_steps, shift=request.shift)
    shape = _latent_shape(request.backbone, request.resolution, request.batch_size)

    traces = []
    for round_index in range(request.rounds):
        generator = torch.Generator(device="cpu").manual_seed(request.seed + round_index)
        initial = [torch.randn(shape, generator=generator, dtype=torch.float32).to(device)]
        traces.append(
            train_rgopd_rollout(
                student=student,
                teacher=teacher,
                reference_step=reference_step,
                initial_latents=initial,
                condition=encoding.condition,
                config=RolloutConfig(sigmas=sigmas, optimized_steps=request.optimized_steps),
                optimizer=optimizer,
                parameters=trainable,
            )
        )

    output = Path(request.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    student_transformer.save_pretrained(str(output / "lora"))
    return {
        "backbone": request.backbone,
        "rounds": len(traces),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "mean_loss": sum(t.mean_loss for t in traces) / len(traces) if traces else 0.0,
        "guidance_fraction": traces[-1].guidance_fraction if traces else 0.0,
        "output_directory": str(output),
    }
