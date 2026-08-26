#!/usr/bin/env python3
"""Export this working repository to a clean public tree.

The public repository is a derived artifact, never a parallel branch: it is
rebuilt from here so no file can drift into two versions. Machine-specific
values live only in configs/local.yaml, which is excluded, so every exported
file is byte-identical to its source.

    python release/export.py --out /tmp/public --check

Exits non-zero if any exported file still carries a machine path, an internal
identifier, or a credential variable.
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXCLUDE_FILE = ROOT / "release" / "EXCLUDE.txt"

FORBIDDEN = {
    "machine path": ("/home/", "/kaimm-distill/"),
    "internal identifier": ("kuaishou", "kwaidc", "klingai", "oversea-squid", "aiplatform-bjx"),
    "credential": ("SWANLAB_API_KEY", "WANDB_API_KEY", "HF_TOKEN"),
}

# This file names the patterns it forbids, and the guard test does too.
SCAN_EXEMPT = {"release/export.py", "tests/test_release_hygiene.py"}


def load_patterns() -> list[str]:
    lines = EXCLUDE_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def is_excluded(relative: str, patterns: list[str]) -> bool:
    parts = Path(relative).parts
    for pattern in patterns:
        bare = pattern.rstrip("/")
        if pattern.endswith("/") and bare in parts:
            return True
        if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(Path(relative).name, pattern):
            return True
    return False


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def scan(paths: list[str]) -> list[str]:
    offenders = []
    for relative in paths:
        if relative in SCAN_EXEMPT:
            continue
        try:
            content = (ROOT / relative).read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for label, markers in FORBIDDEN.items():
            for marker in markers:
                if marker in content:
                    offenders.append(f"{relative}: {label} {marker!r}")
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="scan only; write nothing")
    args = parser.parse_args()

    patterns = load_patterns()
    exported = [f for f in tracked_files() if not is_excluded(f, patterns)]
    skipped = [f for f in tracked_files() if is_excluded(f, patterns)]

    offenders = scan(exported)
    if offenders:
        print(f"REFUSING to export: {len(offenders)} hygiene violation(s)", file=sys.stderr)
        for line in offenders:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"{len(exported)} files clean, {len(skipped)} excluded")
    for name in skipped:
        print(f"  excluded: {name}")
    if args.check:
        return 0

    if args.out.exists():
        shutil.rmtree(args.out)
    for relative in exported:
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    print(f"exported to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
