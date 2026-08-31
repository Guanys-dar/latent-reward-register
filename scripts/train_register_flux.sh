#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST to a prepared group manifest (.jsonl)}"
PARQUET="${PARQUET:?set PARQUET to the DiNa pair parquet path or glob}"
MULTIHEAD_MANIFEST="${MULTIHEAD_MANIFEST:?set MULTIHEAD_MANIFEST to the scored manifest}"
OUTPUT="${OUTPUT:-outputs/register_flux}"

lrr train-register \
    --config configs/register/flux/paper.yaml \
    --training-manifest "$MANIFEST" \
    --training-parquet "$PARQUET" \
    --multihead-manifest "$MULTIHEAD_MANIFEST" \
    --output-directory "$OUTPUT" \
    --precision bf16 \
    "$@"
