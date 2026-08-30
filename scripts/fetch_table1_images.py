#!/usr/bin/env python3
"""Reconstruct the Table 1 image roots.

The released pair file records each image as ``dataset`` + a path relative to
that dataset's root. The images themselves come from four third-party sets and
are not redistributed here, so this fetches them from their upstream sources.

    python scripts/fetch_table1_images.py --pairs all_table1_pairs.jsonl --out ./table1_images
    python scripts/fetch_table1_images.py --pairs all_table1_pairs.jsonl --out ./table1_images --verify

`--verify` checks that every path referenced by the pair file exists, which is
what you want before trusting a reported accuracy.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

# Upstream source per dataset. Each is a Hugging Face dataset repo.
SOURCES = {
    "ImageReward": "THUDM/ImageRewardDB",
    "HPDv2": "ymhao/HPDv2",
    "HPDv3": "MizzenAI/HPDv3",
    "GenAI-Bench": "BaiqiL/GenAI-Bench",
}


def read_pairs(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path}:{line_number}: {error}")


def verify(pairs, root: Path) -> int:
    missing = collections.Counter()
    total = collections.Counter()
    for record in pairs:
        dataset = record["dataset"]
        for key in ("image1", "image2"):
            total[dataset] += 1
            if not (root / dataset / record[key]).exists():
                missing[dataset] += 1

    for dataset in sorted(total):
        absent = missing[dataset]
        status = "OK" if absent == 0 else f"{absent} MISSING"
        print(f"  {dataset:14} {total[dataset]:6} referenced  {status}")

    if missing:
        print("\nSome images are absent. Reported accuracies computed against this")
        print("root would silently cover only part of the benchmark.")
        return 1
    print("\nEvery referenced image is present.")
    return 0


def download(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    failed = []
    for dataset, repo in SOURCES.items():
        target = root / dataset
        print(f"\n=== {dataset} <- {repo} ===")
        command = [
            "hf", "download", repo, "--repo-type", "dataset",
            "--local-dir", str(target),
        ]
        print(" ".join(command))
        # check=False: a gated dataset is an expected outcome, collected and
        # reported at the end rather than aborting the other downloads.
        if subprocess.run(command, check=False).returncode != 0:
            failed.append(dataset)

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        print("Some sets are gated and need `hf auth login` plus accepting their terms.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", action="store_true", help="check coverage; download nothing")
    args = parser.parse_args()

    if not args.pairs.exists():
        raise SystemExit(f"pair file not found: {args.pairs}")

    if args.verify:
        return verify(read_pairs(args.pairs), args.out)

    status = download(args.out)
    print("\n=== coverage ===")
    return verify(read_pairs(args.pairs), args.out) or status


if __name__ == "__main__":
    sys.exit(main())
