#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST to a prepared group manifest (.jsonl)}"
OUTPUT="${OUTPUT:-outputs/register_flux}"

lrr train-register \
    --config configs/register/flux/paper.yaml \
    --training-manifest "$MANIFEST" \
    --output-directory "$OUTPUT" \
    --group-size 2 \
    --batch-size 2 \
    --precision bf16 \
    "$@"
