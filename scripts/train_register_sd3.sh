#!/usr/bin/env bash
# Train the SD3 latent reward register on a prepared group manifest.
#
# Prepare configs/local.yaml first (copy configs/local.yaml.example) so
# data_root and the SD3 snapshot resolve. Training reads cached latents and
# prompt embeddings: no VAE or text encoder runs here.
set -euo pipefail

MANIFEST="${MANIFEST:?set MANIFEST to a prepared group manifest (.jsonl)}"
OUTPUT="${OUTPUT:-outputs/register_sd3}"

lrr train-register \
    --config configs/register/sd3/paper.yaml \
    --training-manifest "$MANIFEST" \
    --output-directory "$OUTPUT" \
    --group-size 4 \
    --batch-size 2 \
    --precision bf16 \
    "$@"
