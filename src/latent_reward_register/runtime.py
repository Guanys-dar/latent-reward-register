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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .config import LocalConfig
from .flowmatch import flux_sigma_schedule, make_reference_step, sigma_schedule
from .guidance import GuidanceSchedule
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

_REGISTER_METADATA_KEYS = {
    "head_names", "feature_layers", "score_keys", "architecture", "revision"
}


def _register_architecture_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in _REGISTER_METADATA_KEYS}


def _gradient_accumulation_steps(global_batch_size: int, local_batch_size: int) -> int:
    if global_batch_size < local_batch_size or global_batch_size % local_batch_size:
        raise ValueError(
            "train.global_batch_size must be a positive multiple of --batch-size"
        )
    return global_batch_size // local_batch_size


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown precision {name!r}; expected one of {sorted(DTYPES)}") from None


def load_register_asset(
    checkpoint: str | Path,
    *,
    model_name_or_path: str,
    dtype: torch.dtype,
    local_files_only: bool,
):
    """Load either a released checkpoint directory or a legacy research ``.pt``."""
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        from .models.loader import load_legacy_register

        return load_legacy_register(
            checkpoint_path,
            model_name_or_path=model_name_or_path,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Register checkpoint not found: {checkpoint_path}")

    import json

    import yaml

    from .checkpoint import load_register_checkpoint
    from .models.builder import build_register

    manifest = json.loads((checkpoint_path / "manifest.json").read_text())
    saved_config = yaml.safe_load((checkpoint_path / "config.yaml").read_text())
    register_config = saved_config["register"]
    register = build_register(
        manifest["backbone"],
        head_names=tuple(register_config["head_names"]),
        feature_layers=tuple(register_config["feature_layers"]),
        model_name_or_path=model_name_or_path,
        revision=manifest.get("backbone_revision"),
        dtype=dtype,
        local_files_only=local_files_only,
        **_register_architecture_kwargs(register_config),
    )
    load_register_checkpoint(checkpoint_path, register)
    return register


def load_pipeline(
    backbone: str,
    *,
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
):
    """Load the generator pipeline for one backbone."""
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
    text_ids: torch.Tensor | None = None


def encode_prompts(
    pipeline,
    prompts: list[str],
    *,
    backbone: str,
    negative_prompt: str = "",
    device: torch.device | str = "cpu",
    max_sequence_length: int | None = None,
) -> PromptEncoding:
    """Run the pipeline's text encoders once, outside the sampling loop.

    Encoding inside the loop would repeat identical work at every step and is a
    common source of wasted time in reward-guided sampling.
    """
    with torch.no_grad():
        if backbone == "sd3":
            arguments = {
                "prompt": prompts,
                "prompt_2": prompts,
                "prompt_3": prompts,
                "device": device,
                "num_images_per_prompt": 1,
                "do_classifier_free_guidance": True,
                "negative_prompt": negative_prompt or None,
            }
            if max_sequence_length is not None:
                arguments["max_sequence_length"] = max_sequence_length
            prompt_embeds, negative_embeds, pooled, negative_pooled = pipeline.encode_prompt(**arguments)
            return PromptEncoding(
                condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled),
                negative=RegisterCondition(
                    prompt_embeds=negative_embeds, pooled_prompt_embeds=negative_pooled
                ),
            )
        if backbone == "flux":
            prompt_embeds, pooled, text_ids = pipeline.encode_prompt(
                prompt=prompts, prompt_2=prompts, device=device, num_images_per_prompt=1
            )
            # FLUX.1-dev is guidance-distilled: the scale is an embedded input, so
            # there is no unconditional branch to encode.
            return PromptEncoding(
                condition=RegisterCondition(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled),
                text_ids=text_ids,
            )
    raise ValueError(f"No prompt encoder wired for backbone {backbone!r}")


def build_velocity_model(
    backbone: str,
    transformer,
    *,
    guidance_scale: float,
    negative: RegisterCondition | None = None,
    image_ids: torch.Tensor | None = None,
    text_ids: torch.Tensor | None = None,
):
    """The classifier-free-guided velocity callable for one backbone."""
    if backbone == "sd3":
        return SD3VelocityModel(transformer, guidance_scale=guidance_scale, negative=negative)
    if backbone == "flux":
        return FluxVelocityModel(
            transformer,
            guidance_scale=guidance_scale,
            image_ids=image_ids,
            text_ids=text_ids,
        )
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

    register = load_register_asset(
        local.resolve_checkpoint(request.register_checkpoint),
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    register.to(device).eval()

    encoding = encode_prompts(
        pipeline, request.prompts, backbone=request.backbone, device=device
    )
    image_ids = None
    if request.backbone == "flux":
        height, width = request.resolution
        generator = torch.Generator(device=device).manual_seed(request.seed)
        channels = pipeline.transformer.config.in_channels // 4
        latents, image_ids = pipeline.prepare_latents(
            len(request.prompts), channels, height, width, torch.float32, device, generator, None
        )
        from .models.flux_utils import SCHEDULER_SHIFT

        schedule_shift = SCHEDULER_SHIFT
    else:
        shape = _latent_shape(request.backbone, request.resolution, 1)
        latents = torch.cat(
            [
                torch.randn(
                    shape,
                    generator=torch.Generator(device=device).manual_seed(request.seed),
                    device=device,
                    dtype=torch.float32,
                )
                for _ in request.prompts
            ]
        )
        schedule_shift = request.shift

    velocity = build_velocity_model(
        request.backbone,
        pipeline.transformer,
        guidance_scale=request.text_guidance_scale,
        negative=encoding.negative,
        image_ids=image_ids,
        text_ids=encoding.text_ids,
    )

    sigmas = (
        flux_sigma_schedule(request.steps, shift=schedule_shift)
        if request.backbone == "flux"
        else sigma_schedule(request.steps, shift=schedule_shift)
    )
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
    pair_parquet: str | Path | None = None
    multihead_manifest: str | Path | None = None
    latents_from_manifest: bool = False
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
    from .checkpoint import CheckpointManifest
    from .dataset import (
        DatasetConfig,
        DinaPairDataset,
        count_usable_groups,
        iter_dina_pair_batches,
        iter_group_batches,
    )
    from .models.builder import build_register
    from .training import TrainConfig, seed_everything, train_register

    local = local or LocalConfig()
    dtype = resolve_dtype(request.precision)
    seed = int(request.train_config.get("seed", 42))
    seed_everything(seed)
    head_names = tuple(request.register_config["head_names"])
    feature_layers = tuple(request.register_config["feature_layers"])
    manifest_path = local.resolve_data(request.manifest)

    if request.pair_parquet is not None:
        if request.multihead_manifest is None:
            raise ValueError("DiNa pair training requires --multihead-manifest")
        pair_dataset = DinaPairDataset(
            local.resolve_data(request.pair_parquet),
            manifest=manifest_path,
            multihead_manifest=local.resolve_data(request.multihead_manifest),
            heads=head_names,
            local=local,
            latents_from_manifest=request.latents_from_manifest,
        )
        usable = total = len(pair_dataset)
    else:
        pair_dataset = None
        usable, total = count_usable_groups(manifest_path, group_size=request.group_size)
    if not usable:
        raise ValueError(
            f"{manifest_path}: no group holds {request.group_size} images; "
            f"{total} groups read. Lower group_size or check the manifest."
        )

    register = build_register(
        request.backbone,
        head_names=head_names,
        feature_layers=feature_layers,
        model_name_or_path=local.model_path(request.backbone, "") or None,
        revision=request.register_config.get("revision"),
        dtype=dtype,
        local_files_only=request.local_files_only,
        **_register_architecture_kwargs(request.register_config),
    )
    register.to(torch.device(request.device))

    dataset_config = DatasetConfig(
        group_size=request.group_size,
        batch_size=request.batch_size,
        heads=head_names,
        require_pooled=request.backbone in {"sd3", "flux"},
        seed=seed,
        shuffle=bool(request.train_config.get("shuffle", True)),
        drop_last=bool(request.train_config.get("drop_last", True)),
    )
    epoch_index = 0

    def batches():
        nonlocal epoch_index
        epoch_config = replace(dataset_config, seed=seed + epoch_index)
        epoch_index += 1
        if pair_dataset is not None:
            return iter_dina_pair_batches(
                pair_dataset,
                batch_size=request.batch_size,
                seed=epoch_config.seed,
                drop_last=epoch_config.drop_last,
                limit=request.max_batches,
            )
        return iter_group_batches(manifest_path, config=epoch_config, local=local, limit=request.max_batches)

    train_register(
        model=register,
        batches=batches,
        config=TrainConfig(
            learning_rate=float(request.train_config.get("learning_rate", 5e-5)),
            weight_decay=float(request.train_config.get("weight_decay", 0.01)),
            epochs=int(request.train_config.get("epochs", 1)),
            ema_decay=float(request.train_config.get("ema_decay", 0.999)),
            max_grad_norm=float(request.train_config.get("max_grad_norm", 1.0)),
            warmup_steps=int(request.train_config.get("warmup_steps", 0)),
            gradient_accumulation_steps=_gradient_accumulation_steps(
                int(request.train_config.get("global_batch_size", request.batch_size)),
                request.batch_size,
            ),
            weighting_scheme=str(request.train_config.get("weighting_scheme", "uniform")),
            share_noise_within_group=bool(
                request.train_config.get("share_noise_within_group", True)
            ),
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
        "batches": min(
            (usable + request.batch_size - 1) // request.batch_size,
            request.max_batches if request.max_batches is not None else usable,
        ),
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
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    rounds: int = 300
    rollout_batches_per_update: int = 4
    batch_size: int = 2
    ema_decay: float = 0.9
    checkpoint_interval: int = 30
    selected_checkpoint_round: int | None = None
    distributed_world_size: int = 8
    prompt_repeats: int = 1
    seed: int = 42
    shift: float = 3.0
    precision: str = "bf16"
    device: str = "cuda"
    local_files_only: bool = False


def distributed_k_repeat_indices(
    *,
    dataset_size: int,
    batch_size: int,
    repeats: int,
    num_replicas: int,
    rank: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """One batch from the originals' ``DistributedKRepeatSampler``."""
    if dataset_size < 1:
        raise ValueError("RG-OPD prompt dataset must not be empty")
    total_samples = num_replicas * batch_size
    if repeats < 1 or total_samples % repeats:
        raise ValueError("prompt_repeats must divide distributed_world_size * batch_size")
    if not 0 <= rank < num_replicas:
        raise ValueError("rank must be in [0, distributed_world_size)")
    generator = torch.Generator().manual_seed(seed + epoch)
    unique_samples = total_samples // repeats
    indices = torch.randperm(dataset_size, generator=generator)[:unique_samples].tolist()
    repeated = [index for index in indices for _ in range(repeats)]
    order = torch.randperm(len(repeated), generator=generator).tolist()
    shuffled = [repeated[index] for index in order]
    start = rank * batch_size
    return shuffled[start : start + batch_size]


def run_rgopd_training(
    request: TrainRGOPDRequest, *, local: LocalConfig | None = None
) -> dict[str, Any]:
    """Distill reward guidance into a LoRA student along its own rollouts.

    The executable form of ``configs/rgopd/*/paper.yaml``. The teacher here is
    the same ``RewardGradientTeacher`` the sampler uses, so the student cannot be
    trained against guidance the sampler would not apply.
    """
    from .flowmatch import make_student_policy
    from .rollout import RolloutConfig, train_rgopd_rollout
    from .teacher import RewardGradientTeacher
    from .training import RGOPDEMA, seed_everything
    from .velocity import attach_lora_student

    local = local or LocalConfig()
    dtype = resolve_dtype(request.precision)
    device = torch.device(request.device)
    seed_everything(request.seed)
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

    register = load_register_asset(
        local.resolve_checkpoint(request.register_checkpoint),
        model_name_or_path=model_path,
        dtype=dtype,
        local_files_only=request.local_files_only,
    )
    register.to(device).eval()

    student_transformer, trainable = attach_lora_student(
        pipeline.transformer,
        backbone=request.backbone,
        rank=request.lora_rank,
        alpha=request.lora_alpha,
    )

    teacher = RewardGradientTeacher(register, schedule=request.schedule, heads=request.heads)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=request.learning_rate,
        betas=(request.adam_beta1, request.adam_beta2),
        weight_decay=request.weight_decay,
        eps=request.adam_epsilon,
    )
    student_ema = RGOPDEMA(student_transformer, request.ema_decay)
    if request.backbone == "flux":
        from .models.flux_utils import SCHEDULER_SHIFT

        schedule_shift = SCHEDULER_SHIFT
    else:
        schedule_shift = request.shift
    sigmas = (
        flux_sigma_schedule(request.rollout_steps, shift=schedule_shift)
        if request.backbone == "flux"
        else sigma_schedule(request.rollout_steps, shift=schedule_shift)
    )
    if request.distributed_world_size < 1:
        raise ValueError("distributed_world_size must be positive")
    rank_generators = [
        torch.Generator(device=device).manual_seed(request.seed + rank)
        for rank in range(request.distributed_world_size)
    ]
    traces = []
    for round_index in range(request.rounds):
        for rollout_batch_index in range(request.rollout_batches_per_update):
            sampler_epoch = round_index * request.rollout_batches_per_update + rollout_batch_index
            for rank in range(request.distributed_world_size):
                prompt_indices = distributed_k_repeat_indices(
                    dataset_size=len(request.prompts),
                    batch_size=request.batch_size,
                    repeats=request.prompt_repeats,
                    num_replicas=request.distributed_world_size,
                    rank=rank,
                    seed=request.seed,
                    epoch=sampler_epoch,
                )
                prompt_batch = [request.prompts[index] for index in prompt_indices]
                encoding = encode_prompts(
                    pipeline,
                    prompt_batch,
                    backbone=request.backbone,
                    device=device,
                    max_sequence_length=128 if request.backbone == "sd3" else None,
                )
                image_ids = None
                if request.backbone == "flux":
                    height, width = request.resolution
                    channels = pipeline.transformer.config.in_channels // 4
                    latents, image_ids = pipeline.prepare_latents(
                        request.batch_size,
                        channels,
                        height,
                        width,
                        torch.float32,
                        device,
                        rank_generators[rank],
                        None,
                    )
                    initial = [latents]
                else:
                    shape = _latent_shape(request.backbone, request.resolution, request.batch_size)
                    initial = [
                        torch.randn(
                            shape,
                            device=device,
                            dtype=torch.float32,
                            generator=rank_generators[rank],
                        )
                    ]

                reference_velocity = build_velocity_model(
                    request.backbone,
                    student_transformer,
                    guidance_scale=request.text_guidance_scale,
                    negative=encoding.negative,
                    image_ids=image_ids,
                    text_ids=encoding.text_ids,
                )
                student_velocity = build_velocity_model(
                    request.backbone,
                    student_transformer,
                    guidance_scale=request.text_guidance_scale,
                    negative=encoding.negative,
                    image_ids=image_ids,
                    text_ids=encoding.text_ids,
                )
                final_microbatch = (
                    rollout_batch_index + 1 == request.rollout_batches_per_update
                    and rank + 1 == request.distributed_world_size
                )
                traces.append(
                    train_rgopd_rollout(
                        student=make_student_policy(student_velocity),
                        teacher=teacher,
                        reference_step=make_reference_step(
                            reference_velocity,
                            context_factory=student_transformer.disable_adapter,
                        ),
                        initial_latents=initial,
                        condition=encoding.condition,
                        config=RolloutConfig(sigmas=sigmas, optimized_steps=request.optimized_steps),
                        optimizer=optimizer,
                        parameters=trainable,
                        reset_gradients=rollout_batch_index == 0 and rank == 0,
                        optimizer_step=final_microbatch,
                        loss_scale=1.0
                        / (request.rollout_batches_per_update * request.distributed_world_size),
                    )
                )
        for generator in rank_generators:
            torch.randperm(
                request.batch_size * request.rollout_batches_per_update,
                device=device,
                generator=generator,
            )

        completed_round = round_index + 1
        student_ema.update(student_transformer, completed_round)
        if request.checkpoint_interval > 0 and completed_round % request.checkpoint_interval == 0:
            backup = student_ema.copy_to(student_transformer)
            try:
                checkpoint_dir = Path(request.output_directory) / "checkpoints" / f"checkpoint-{completed_round}" / "lora"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                student_transformer.save_pretrained(str(checkpoint_dir))
            finally:
                student_transformer.load_state_dict(backup, strict=False)

    output = Path(request.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    selected_round = request.selected_checkpoint_round or request.rounds
    selected = output / "checkpoints" / f"checkpoint-{selected_round}" / "lora"
    if request.selected_checkpoint_round is not None and not selected.is_dir():
        raise ValueError(
            f"Selected checkpoint round {selected_round} was not saved; "
            f"checkpoint_interval={request.checkpoint_interval}"
        )
    if selected.is_dir():
        import shutil

        shutil.copytree(selected, output / "lora", dirs_exist_ok=True)
    else:
        backup = student_ema.copy_to(student_transformer)
        try:
            student_transformer.save_pretrained(str(output / "lora"))
        finally:
            student_transformer.load_state_dict(backup, strict=False)
    return {
        "backbone": request.backbone,
        "rounds": request.rounds,
        "rollout_batches": len(traces),
        "distributed_world_size": request.distributed_world_size,
        "selected_checkpoint_round": selected_round,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "mean_loss": sum(t.mean_loss for t in traces) / len(traces) if traces else 0.0,
        "guidance_fraction": traces[-1].guidance_fraction if traces else 0.0,
        "output_directory": str(output),
    }
