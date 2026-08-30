from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .workflows import load_workflow


def _read_prompts(path: str | None, inline: list[str] | None) -> list[str]:
    """Prompts from a file (one per line) or from --prompt flags."""
    if inline:
        return list(inline)
    if not path:
        raise ValueError("Provide prompts with --prompt-file or repeated --prompt")
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    prompts = [line for line in lines if line and not line.startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _schedule_from_config(config: dict):
    """The guidance gate for a task config, in either of the two shapes presets use."""
    from .guidance import GuidanceSchedule

    guidance = config.get("guidance") or {}
    bands = guidance.get("bands")
    if bands:
        return GuidanceSchedule(bands=tuple((float(edge), float(scale)) for edge, scale in bands))
    # RG-OPD presets express the same gate as a floor plus a single scale.
    return GuidanceSchedule(
        bands=(
            (float(config.get("active_sigma_min", 0.2)), float(config.get("reward_scale", 0.4))),
        )
    )


def _workflow_command(args: argparse.Namespace) -> int:
    """Run one of the three paper workflows."""
    workflow = load_workflow(args.config)
    plan = workflow.plan()
    requested = {
        key: value
        for key, value in {
            "register_checkpoint": getattr(args, "register_checkpoint", None),
            "prompt_file": getattr(args, "prompt_file", None),
            "training_manifest": getattr(args, "training_manifest", None),
            "output_directory": getattr(args, "output_directory", None),
        }.items()
        if value is not None
    }
    plan["requested_inputs"] = requested
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute_workflow(args, workflow)


def _execute_workflow(args: argparse.Namespace, workflow) -> int:
    """Dispatch a validated workflow to its runtime, with real assets."""
    from .local_config import load_local_config
    from .runtime import (
        SampleRequest,
        TrainRegisterRequest,
        TrainRGOPDRequest,
        decode_latents,
        run_register_training,
        run_rgopd_training,
        run_sampling,
    )

    config = load_config(args.config)
    local = load_local_config(args.local_config)
    if args.model_path:
        local = replace(local, models={**local.models, workflow.backbone: args.model_path})
    task = workflow.task

    if task == "train-register":
        manifest = args.training_manifest
        if not manifest:
            raise ValueError("train-register needs --training-manifest")
        if not args.output_directory:
            raise ValueError("train-register needs --output-directory")
        register_config = dict(config["register"])
        register_config.setdefault("revision", (config.get("backbone") or {}).get("revision", "unknown"))
        summary = run_register_training(
            TrainRegisterRequest(
                backbone=workflow.backbone,
                manifest=manifest,
                output_directory=args.output_directory,
                register_config=register_config,
                train_config=config.get("train") or {},
                group_size=args.group_size,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                precision=args.precision,
                device=args.device,
                local_files_only=args.local_files_only,
            ),
            local=local,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if task == "sample":
        if not args.register_checkpoint:
            raise ValueError("sample needs --register-checkpoint")
        prompts = _read_prompts(args.prompt_file, args.prompt)
        resolution = tuple(config.get("resolution", (1024, 1024)))
        result = run_sampling(
            SampleRequest(
                backbone=workflow.backbone,
                prompts=prompts,
                steps=args.steps or int(config.get("steps", 28)),
                resolution=resolution,
                heads=tuple(config.get("heads", ("preference",))),
                schedule=_schedule_from_config(config),
                text_guidance_scale=float(config.get("text_guidance_scale", 4.5)),
                register_checkpoint=args.register_checkpoint,
                seed=args.seed,
                precision=args.precision,
                device=args.device,
                local_files_only=args.local_files_only,
            ),
            local=local,
        )
        latents = result.pop("latents")
        pipeline = result.pop("pipeline")
        if args.output_directory:
            output = Path(args.output_directory)
            output.mkdir(parents=True, exist_ok=True)
            images = decode_latents(
                pipeline, latents, backbone=workflow.backbone, resolution=resolution
            )
            for index, image in enumerate(images):
                image.save(output / f"{index:04d}_seed{args.seed}.png")
            result["images_written"] = len(images)
        result["latent_shape"] = list(latents.shape)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if not args.register_checkpoint:
        raise ValueError("train-rgopd needs --register-checkpoint")
    if not args.output_directory:
        raise ValueError("train-rgopd needs --output-directory")
    student = config.get("student") or {}
    summary = run_rgopd_training(
        TrainRGOPDRequest(
            backbone=workflow.backbone,
            prompts=_read_prompts(args.prompt_file, args.prompt),
            register_checkpoint=args.register_checkpoint,
            output_directory=args.output_directory,
            schedule=_schedule_from_config(config),
            heads=tuple(config.get("reward_heads", ("preference",))),
            rollout_steps=int(config.get("rollout_steps", 10)),
            optimized_steps=int(config.get("optimized_steps", 9)),
            lora_rank=int(student.get("rank", 32)),
            lora_alpha=int(student.get("alpha", 64)),
            rounds=args.rounds,
            batch_size=args.batch_size,
            seed=args.seed,
            precision=args.precision,
            device=args.device,
            local_files_only=args.local_files_only,
        ),
        local=local,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", required=True)
    command.add_argument("--output-directory")
    command.add_argument("--local-config", help="machine paths; defaults to configs/local.yaml")
    command.add_argument("--model-path", help="override the backbone snapshot for this run")
    command.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    command.add_argument("--device", default="cuda")
    command.add_argument("--local-files-only", action="store_true")
    command.add_argument("--dry-run", action="store_true", help="validate without loading weights")


def _add_prompt_arguments(command: argparse.ArgumentParser) -> None:
    prompts = command.add_mutually_exclusive_group()
    prompts.add_argument("--prompt-file")
    prompts.add_argument("--prompt", action="append", help="inline prompt; repeatable")


def _add_register_command(subcommands) -> None:
    command = subcommands.add_parser("train-register", help="train a latent reward register")
    _add_common_arguments(command)
    command.add_argument("--training-manifest")
    command.add_argument("--group-size", type=int, default=4)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--max-batches", type=int, help="stop after N batches")
    command.set_defaults(handler=_workflow_command)


def _add_sample_command(subcommands) -> None:
    command = subcommands.add_parser("sample", help="sample with reward-gradient guidance")
    _add_common_arguments(command)
    _add_prompt_arguments(command)
    command.add_argument("--register-checkpoint")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--steps", type=int, help="override the configured sampling steps")
    command.set_defaults(handler=_workflow_command)


def _add_rgopd_command(subcommands) -> None:
    command = subcommands.add_parser("train-rgopd", help="distill guidance into a LoRA student")
    _add_common_arguments(command)
    _add_prompt_arguments(command)
    command.add_argument("--register-checkpoint")
    command.add_argument("--rounds", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--seed", type=int, default=42)
    command.set_defaults(handler=_workflow_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lrr", description="Latent Reward Register paper code")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subcommands = parser.add_subparsers(dest="command", required=True)
    _add_register_command(subcommands)
    _add_sample_command(subcommands)
    _add_rgopd_command(subcommands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, KeyError) as error:
        # A missing asset or a bad config is user error, not a crash: a traceback
        # here buries the one line that says what to fix.
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
