#!/usr/bin/env bash
# Reward-guided sampling with the SD3 register (paper preset: 42 steps).
#
# Writes PNGs to OUTPUT and prints the guidance fraction, which is how much of
# the trajectory the schedule actually steered.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a released register checkpoint}"
PROMPTS="${PROMPTS:?set PROMPTS to a file of prompts, one per line}"
OUTPUT="${OUTPUT:-outputs/rgs_sd3}"

lrr sample \
    --config configs/rgs/sd3/paper.yaml \
    --register-checkpoint "$CHECKPOINT" \
    --prompt-file "$PROMPTS" \
    --output-directory "$OUTPUT" \
    --seed 42 \
    "$@"
