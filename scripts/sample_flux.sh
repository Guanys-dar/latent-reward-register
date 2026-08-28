#!/usr/bin/env bash
# Reward-guided sampling with the FLUX register.
#
# FLUX.1-dev is guidance-distilled: the text guidance scale is an embedded input
# and there is no unconditional branch, so a step costs one transformer forward
# rather than two.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a released register checkpoint}"
PROMPTS="${PROMPTS:?set PROMPTS to a file of prompts, one per line}"
OUTPUT="${OUTPUT:-outputs/rgs_flux}"

lrr sample \
    --config configs/rgs/flux/paper.yaml \
    --register-checkpoint "$CHECKPOINT" \
    --prompt-file "$PROMPTS" \
    --output-directory "$OUTPUT" \
    --seed 42 \
    "$@"
