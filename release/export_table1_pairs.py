#!/usr/bin/env python3
"""Make the Table 1 pair file portable.

The research pair file stores absolute paths into four third-party image sets.
Those images are not ours to redistribute, so the released file keeps the pair
structure, labels, and prompts, and records each image by
``dataset`` + ``relative_path``. A download script reconstructs the roots.

    python release/export_table1_pairs.py --plan
    python release/export_table1_pairs.py --out /tmp/publish
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SOURCE = "/home/guanyuanshen/iclr-exp-matrix/evaluate-lrm/data/all_table1_pairs_fixed.jsonl"

# Each dataset's image root in the research workspace. Everything below the root
# becomes the portable relative path.
ROOTS = {
    "ImageReward": "/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/ImageReward",
    "HPDv2": "/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/HPDv2",
    # HPDv3 images live with the training data, not under test_bench.
    "HPDv3": "/kaimm-distill/ysguan/reward-token-image-dataset/HPDv3",
    "GenAI-Bench": "/kaimm-distill/ysguan/Image-Reward-Token-Exps/test_bench/GenAI-Bench",
}

UPSTREAM = {
    "ImageReward": "https://huggingface.co/datasets/THUDM/ImageRewardDB",
    "HPDv2": "https://huggingface.co/datasets/ymhao/HPDv2",
    "HPDv3": "https://huggingface.co/datasets/MizzenAI/HPDv3",
    "GenAI-Bench": "https://huggingface.co/datasets/BaiqiL/GenAI-Bench",
}


def relative_to_root(path: str, dataset: str) -> str | None:
    root = ROOTS.get(dataset)
    if root and path.startswith(root):
        return path[len(root) :].lstrip("/")
    return None


def convert(source: Path):
    records, unresolved = [], collections.Counter()
    for line_number, line in enumerate(source.open(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        dataset = payload["dataset"]
        first = relative_to_root(payload["path1"], dataset)
        second = relative_to_root(payload["path2"], dataset)
        if first is None or second is None:
            unresolved[dataset] += 1
            continue
        preferred = payload["preferred"]
        records.append(
            {
                "pair_id": payload["pair_id"],
                "dataset": dataset,
                "subset": payload["subset"],
                "prompt": payload["prompt"],
                "image1": first,
                "image2": second,
                # Normalized to an index so a consumer never re-parses a path.
                "preferred": 0 if preferred == "path1" else 1,
                # Drop any provenance field carrying a machine path (raw_file
                # pointed into a personal workspace).
                "source": {
                    key: value
                    for key, value in (payload.get("source") or {}).items()
                    if not (isinstance(value, str) and ("/home/" in value or "/kaimm-distill/" in value))
                },
            }
        )
    return records, unresolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not args.plan and args.out is None:
        parser.error("--out is required unless --plan")

    source = Path(SOURCE)
    if not source.exists():
        print(f"MISSING {source}")
        return 1

    records, unresolved = convert(source)
    counts = collections.Counter(r["dataset"] for r in records)
    print(f"{len(records)} pairs resolved")
    for dataset, count in sorted(counts.items()):
        print(f"  {dataset:14} {count:6}  root -> {UPSTREAM[dataset]}")
    for dataset, count in sorted(unresolved.items()):
        print(f"  UNRESOLVED {dataset}: {count}")
    if unresolved:
        print("Refusing: every path must resolve, or the released file is incomplete.")
        return 1
    if args.plan:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "all_table1_pairs.jsonl"
    with target.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.out / "TABLE1_MANIFEST.json").write_text(
        json.dumps(
            {
                "file": target.name,
                "pairs": len(records),
                "sha256": digest,
                "per_dataset": dict(sorted(counts.items())),
                "image_roots": UPSTREAM,
                "note": (
                    "Images are not redistributed. Set one root per dataset; "
                    "image1/image2 are relative to their dataset's root."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {target} ({len(records)} pairs, sha256 {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
