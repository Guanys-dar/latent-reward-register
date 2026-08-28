#!/usr/bin/env bash
# Everything that runs without model weights or data. Use this to check an install.
set -euo pipefail

echo "== algorithm paths on a synthetic backbone =="
lrr smoke-release

echo
echo "== all seven workflow presets =="
lrr validate-release --root .

echo
echo "== each preset resolves to a runnable plan =="
for config in configs/register/*/paper.yaml; do
    lrr train-register --config "$config" --dry-run > /dev/null && echo "ok  $config"
done
for config in configs/rgs/*/paper.yaml; do
    lrr sample --config "$config" --dry-run > /dev/null && echo "ok  $config"
done
for config in configs/rgopd/*/paper.yaml; do
    lrr train-rgopd --config "$config" --dry-run > /dev/null && echo "ok  $config"
done

echo
echo "== tests =="
pytest -q
