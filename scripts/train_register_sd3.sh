#!/usr/bin/env bash
# Train the SD3 latent reward register on a prepared group manifest.
#
# Prepare configs/local.yaml first (copy configs/local.yaml.example) so
# data_root and the SD3 snapshot resolve. Training reads cached latents and
# prompt embeddings: no VAE or text encoder runs here.
set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST to a prepared group manifest (.jsonl)}"
PARQUET="${PARQUET:?set PARQUET to the DiNa pair parquet path or glob}"
MULTIHEAD_MANIFEST="${MULTIHEAD_MANIFEST:?set MULTIHEAD_MANIFEST to the scored manifest}"
OUTPUT="${OUTPUT:-outputs/register_sd3}"

lrr train-register \
    --config configs/register/sd3/paper.yaml \
    --training-manifest "$MANIFEST" \
    --training-parquet "$PARQUET" \
    --multihead-manifest "$MULTIHEAD_MANIFEST" \
    --output-directory "$OUTPUT" \
    --precision bf16 \
    "$@"
