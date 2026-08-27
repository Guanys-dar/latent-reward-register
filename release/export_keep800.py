#!/usr/bin/env python3
"""Export the keep-800 evaluation key set.

Table 3 (and the FLUX Table 2 block) report metrics over **keep-800**: the
1000 generated images (500 prompts x seeds 42/43) minus the worst 200 by an
equal-weight z-sum of {hpsv2, hpsv3, imagereward, pickscore}. Dropping happens
once, on a *defining variant*, and the same key set is then applied to every
variant so comparisons stay paired.

Publishing the key set matters: a reader who re-derives the filter on their own
generations gets a different set of 800 and therefore different numbers.

    python release/export_keep800.py --plan
    python release/export_keep800.py --out /tmp/publish

Five byte-identical copies exist in the research workspace; any is fine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = (
    "/kaimm-distill/ysguan/z-image-reward-matrix/node1/outputs/eval_table2"
    "/drop_set_keep800.json"
)


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

    payload = json.loads(source.read_text())
    keep = [tuple(key) for key in payload["keep_keys"]]
    drop = [tuple(key) for key in payload["drop_keys"]]

    if len(keep) != payload["n_keep"] or len(drop) != payload["n_drop"]:
        print("Refusing: recorded counts disagree with the key lists.")
        return 1
    if len(keep) + len(drop) != payload["n_total"]:
        print("Refusing: keep + drop does not equal the total.")
        return 1
    if set(keep) & set(drop):
        print("Refusing: a key appears in both keep and drop.")
        return 1

    seeds = sorted({seed for _, seed in keep})
    prompts = sorted({index for index, _ in keep})
    print(f"keep {len(keep)} of {payload['n_total']} (drop {len(drop)})")
    print(f"  seeds: {seeds}")
    print(f"  distinct prompt indices in keep: {len(prompts)} (range {min(prompts)}-{max(prompts)})")
    print(f"  defining variant: {payload['defining_variant']}")

    if args.plan:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "keep800.json"
    target.write_text(
        json.dumps(
            {
                "name": "keep-800",
                "n_total": payload["n_total"],
                "n_keep": payload["n_keep"],
                "n_drop": payload["n_drop"],
                "seeds": seeds,
                "definition": payload["definition"],
                "defining_variant": payload["defining_variant"],
                "note": (
                    "Keys are [prompt_index, seed]. Apply this exact set to every "
                    "variant being compared; re-deriving the filter on new "
                    "generations yields a different set of 800 and different numbers."
                ),
                "keep_keys": [list(key) for key in keep],
                "drop_keys": [list(key) for key in drop],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.out / "KEEP800_MANIFEST.json").write_text(
        json.dumps({"file": target.name, "sha256": digest, "keys": len(keep)}, indent=2) + "\n"
    )
    print(f"\nwrote {target} (sha256 {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
