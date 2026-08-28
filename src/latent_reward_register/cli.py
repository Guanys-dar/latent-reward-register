from __future__ import annotations

import argparse
import json

from .backbones import available_backbones
from .checkpoint import read_legacy_checkpoint
from .config import load_config
from .workflows import load_workflow, validate_release


def _inspect_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


def _inspect_checkpoint(args: argparse.Namespace) -> int:
    payload = read_legacy_checkpoint(args.checkpoint)
    config = payload["config"]
    summary = {
        "global_step": payload.get("global_step"),
        "backbone": config.get("model", {}).get("backbone", config.get("reward_token", {}).get("architecture")),
        "reward_register": config.get("reward_token", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _plan(args: argparse.Namespace) -> int:
    workflow = load_workflow(args.config)
    print(json.dumps(workflow.plan(), indent=2, sort_keys=True))
    return 0


def _validate_release(args: argparse.Namespace) -> int:
    print(json.dumps(validate_release(args.root), indent=2, sort_keys=True))
    return 0


def _workflow_command(args: argparse.Namespace) -> int:
    workflow = load_workflow(args.config)
    plan = workflow.plan()
    plan["requested_inputs"] = {
        key: value
        for key, value in {
            "dataset_manifest": args.dataset_manifest,
            "register_checkpoint": args.register_checkpoint,
            "prompt_file": args.prompt_file,
            "training_manifest": args.training_manifest,
            "output_directory": args.output_directory,
        }.items()
        if value is not None
    }
    if not args.dry_run:
        raise RuntimeError(
            "Asset-backed execution is intentionally deferred; rerun with --dry-run or provide the published "
            "checkpoint/data integration for this workflow."
        )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _add_workflow_command(subcommands, name: str, help_text: str) -> None:
    command = subcommands.add_parser(name, help=help_text)
    command.add_argument("--config", required=True)
    command.add_argument("--output-directory")
    command.add_argument("--dataset-manifest")
    command.add_argument("--training-manifest")
    command.add_argument("--register-checkpoint")
    command.add_argument("--prompt-file")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(handler=_workflow_command)


def _build_register(args: argparse.Namespace) -> int:
    """Construct a register from a config against real weights.

    This is the cheapest command that exercises the model path. Config
    validation and --dry-run cannot catch a config key the model does not
    accept, or a shape error in the group plumbing; this can.
    """
    import torch

    from .backbones import build_register_from_config

    config = load_config(args.config)
    if args.model_path:
        config.setdefault("backbone", {})["model_name_or_path"] = args.model_path
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    register = build_register_from_config(
        config, dtype=dtype, local_files_only=args.local_files_only
    )
    trainable = sum(p.numel() for p in register.parameters() if p.requires_grad)
    total = sum(p.numel() for p in register.parameters())
    print(
        json.dumps(
            {
                "config": args.config,
                "model": type(register.model).__name__,
                "backbone": register.backbone,
                "head_names": list(register.head_names),
                "trainable_parameters": trainable,
                "total_parameters": total,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _list_backbones(_: argparse.Namespace) -> int:
    print("\n".join(available_backbones()))
    return 0


def _smoke_release(_: argparse.Namespace) -> int:
    from .smoke import run_release_smoke

    print(json.dumps(run_release_smoke(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lrr", description="Latent Reward Register research toolkit")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect_config = subcommands.add_parser("inspect-config")
    inspect_config.add_argument("config")
    inspect_config.set_defaults(handler=_inspect_config)
    inspect_checkpoint = subcommands.add_parser("inspect-checkpoint")
    inspect_checkpoint.add_argument("checkpoint")
    inspect_checkpoint.set_defaults(handler=_inspect_checkpoint)
    backbones = subcommands.add_parser("list-backbones")
    backbones.set_defaults(handler=_list_backbones)
    plan = subcommands.add_parser("plan", help="validate a workflow config and print its runtime inputs")
    plan.add_argument("config")
    plan.set_defaults(handler=_plan)
    validate = subcommands.add_parser("validate-release", help="validate all supported release workflows")
    validate.add_argument("--root", default=".")
    validate.set_defaults(handler=_validate_release)
    _add_workflow_command(subcommands, "train-register", "plan register training")
    _add_workflow_command(subcommands, "sample", "plan reward-guided sampling")
    _add_workflow_command(subcommands, "train-rgopd", "plan reward-guided OPD training")
    build = subcommands.add_parser(
        "build-register", help="construct a register from a config against real model weights"
    )
    build.add_argument("--config", required=True)
    build.add_argument("--model-path", help="local snapshot; overrides backbone.model_name_or_path")
    build.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    build.add_argument("--local-files-only", action="store_true")
    build.set_defaults(handler=_build_register)
    smoke = subcommands.add_parser("smoke-release", help="exercise all core algorithms without external assets")
    smoke.set_defaults(handler=_smoke_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
