#!/usr/bin/env python3
"""Package research checkpoints for publication.

Research checkpoints carry optimizer and LR-scheduler state, which is most of
their size and is useless to anyone doing inference: an SD3 register is 2.3 GB
on disk but only 147,599,619 parameters (0.59 GB at fp32). This strips the
training state, keeps the self-describing config, and writes a checksum
manifest.

    python release/package_checkpoints.py --plan            # what would be done
    python release/package_checkpoints.py --out /tmp/publish

Nothing is uploaded. Point `hf upload` at the output directory afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Sources are named here rather than in the release tree: this script is a
# private packaging tool and is excluded from export.
REGISTERS = {
    "sd3": (
        "/kaimm-distill/ysguan/Image-Reward-Token-Exps/Reward-Token-Image-0420/outputs"
        "/exp11_multihead_noattn2_sd3_8gpu/checkpoints/reward_token_epoch_003_ema.pt"
    ),
    "flux": (
        "/kaimm-distill/ysguan/z-image-reward-matrix/node5/outputs"
        "/reward_register_flux_unified_v3/checkpoints/reward_token_final_ema.pt"
    ),
}

_FLUX_OPD = "/kaimm-distill/ysguan/z-image-reward-matrix/node5/outputs/flux_opd_unified_v3"
_SD3_OPD = "/home/guanyuanshen/rt-gradient-opd/RG-OPD/logs"

# Epoch chosen by the selection protocol (argmax HPSv3 s.t. CLIP-IQA >= 0.649).
# SD3 publishes two epochs to match FLUX coverage; FLUX publishes the selected ones.
OPD_RUNS = {
    "flux-hps": (f"{_FLUX_OPD}/[FLUX-OPD B + unified-v3 HPS rt0.80 sigma>0.2 e150]", [150]),
    "flux-imagereward": (
        f"{_FLUX_OPD}/[FLUX-OPD C1 + unified-v3 ImageReward rt0.80 sigma>0.2 e150]", [60]
    ),
    "flux-twohead": (f"{_FLUX_OPD}/[FLUX-OPD C2 + unified-v3 TwoHead rt0.80 sigma>0.2 e150]", [150]),
    "sd3-hps": (f"{_SD3_OPD}/[RG-OPD A + SD3-M + exp11ema HPS rt0.40]", [60, 90]),
}

# Training-only state: large, and useless for inference or re-scoring.
DROP_KEYS = ("optimizer", "lr_scheduler", "extra_state")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_register(source: Path, target: Path) -> dict:
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    kept = {key: value for key, value in payload.items() if key not in DROP_KEYS}
    state = kept.get("ema_model", kept.get("model"))

    def count(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel()
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, target)
    return {
        "parameters": count(state),
        "dropped": [key for key in DROP_KEYS if key in payload],
        "source_bytes": source.stat().st_size,
        "packaged_bytes": target.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true", help="report sizes; write nothing")
    parser.add_argument("--all-epochs", action="store_true", help="publish every epoch, not just selected")
    args = parser.parse_args()

    if not args.plan and args.out is None:
        parser.error("--out is required unless --plan")

    entries = []
    for name, source in REGISTERS.items():
        path = Path(source)
        entries.append(("register", name, path, path.exists()))
    for name, (run, selected) in OPD_RUNS.items():
        run_path = Path(run)
        found = sorted(
            (int(d.name.split("-")[-1]), d)
            for d in run_path.glob("checkpoints/checkpoint-*")
            if d.name.split("-")[-1].isdigit()
        )
        chosen = found if args.all_epochs else [(e, d) for e, d in found if e in selected]
        if not chosen and not args.all_epochs:
            print(f"MISSING opd/{name}: no epoch in {selected} under {run_path}")
        for epoch, directory in chosen:
            entries.append(("opd", f"{name}-epoch{epoch}", directory, directory.exists()))

    missing = [f"{kind}/{name}: {path}" for kind, name, path, ok in entries if not ok]
    for line in missing:
        print(f"MISSING {line}")

    if args.plan:
        for kind, name, path, ok in entries:
            if not ok:
                continue
            size = (
                sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir()
                else path.stat().st_size
            )
            print(f"{kind:9} {name:28} {size / 1e9:6.2f} GB  {path}")
        print(f"\n{len(entries) - len(missing)} artifact(s); registers shrink when optimizer is dropped.")
        return 1 if missing else 0

    manifest = []
    for kind, name, path, ok in entries:
        if not ok:
            continue
        if kind == "register":
            target = args.out / "registers" / f"{name}-register.pt"
            info = strip_register(path, target)
            manifest.append({"name": name, "kind": kind, "file": str(target.relative_to(args.out)),
                             "sha256": sha256(target), **info})
            print(f"packaged {name}: {info['source_bytes']/1e9:.2f} -> {info['packaged_bytes']/1e9:.2f} GB")
        else:
            target = args.out / "opd" / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
            files = sorted(f for f in target.rglob("*") if f.is_file())
            manifest.append({"name": name, "kind": kind, "directory": str(target.relative_to(args.out)),
                             "files": {str(f.relative_to(target)): sha256(f) for f in files}})
            print(f"copied {name}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "CHECKSUMS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}/CHECKSUMS.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
