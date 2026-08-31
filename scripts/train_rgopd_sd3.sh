#!/usr/bin/env bash
# Distill SD3 reward guidance into a LoRA student (RG-OPD, paper preset).
#
# Ten-step rollouts, first nine optimized, against a frozen reference anchor.
# The teacher is the same one the sampler uses, so the student cannot learn
# guidance the sampler would not apply.
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the SD3 teacher register (exp11 EMA)}"
PROMPTS="${PROMPTS:?set PROMPTS to a file of prompts, one per line}"
OUTPUT="${OUTPUT:-outputs/rgopd_sd3}"
ROUNDS="${ROUNDS:-300}"

lrr train-rgopd \
    --config configs/rgopd/sd3/paper.yaml \
    --register-checkpoint "$CHECKPOINT" \
    --prompt-file "$PROMPTS" \
    --output-directory "$OUTPUT" \
    --rounds "$ROUNDS" \
    "$@"
