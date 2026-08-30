#!/usr/bin/env bash
# Distill FLUX reward guidance into a LoRA student (RG-OPD, paper preset).
#
# The FLUX paper preset uses reward scale 0.80.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the FLUX teacher register (unified-v3 EMA)}"
PROMPTS="${PROMPTS:?set PROMPTS to a file of prompts, one per line}"
OUTPUT="${OUTPUT:-outputs/rgopd_flux}"
ROUNDS="${ROUNDS:-100}"

lrr train-rgopd \
    --config configs/rgopd/flux/paper.yaml \
    --register-checkpoint "$CHECKPOINT" \
    --prompt-file "$PROMPTS" \
    --output-directory "$OUTPUT" \
    --rounds "$ROUNDS" \
    "$@"
