#!/usr/bin/env bash
# Weight-free checks for a fresh installation.
set -euo pipefail
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
LRR=(python -m latent_reward_register.cli)

echo "== validate paper configurations =="
for config in configs/register/*/paper.yaml; do
    "${LRR[@]}" train-register --config "$config" --dry-run > /dev/null && echo "ok  $config"
done
for config in configs/rgs/*/paper.yaml; do
    "${LRR[@]}" sample --config "$config" --dry-run > /dev/null && echo "ok  $config"
done
for config in configs/rgopd/*/paper.yaml; do
    "${LRR[@]}" train-rgopd --config "$config" --dry-run > /dev/null && echo "ok  $config"
done
