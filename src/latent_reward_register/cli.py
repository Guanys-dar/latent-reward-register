from __future__ import annotations

import argparse
import json

from .backbones import available_backbones
from .checkpoint import read_legacy_checkpoint
from .config import load_config


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
    backbones.set_defaults(handler=lambda _: print("\n".join(available_backbones())) or 0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
