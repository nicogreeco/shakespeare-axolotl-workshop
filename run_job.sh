#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/data/workshop-llm
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$ROOT/runs/$RUN_ID"

test -s "$ROOT/data/train.jsonl"
test -s "$ROOT/data/validation.jsonl"
mkdir -p "$RUN_DIR/adapter"
printf 'Run directory: %s\n' "$RUN_DIR"

axolotl train "$ROOT/axolotl.yaml"

test -s /workspace/output/adapter_config.json
cp /workspace/output/adapter_config.json "$RUN_DIR/adapter/"
if [[ -s /workspace/output/adapter_model.safetensors ]]; then
  cp /workspace/output/adapter_model.safetensors "$RUN_DIR/adapter/"
elif [[ -s /workspace/output/adapter_model.bin ]]; then
  cp /workspace/output/adapter_model.bin "$RUN_DIR/adapter/"
else
  echo 'Axolotl completed, but no adapter weights were found.' >&2
  exit 1
fi

python3 "$ROOT/compare.py" \
  --adapter "$RUN_DIR/adapter" \
  --validation "$ROOT/data/validation.jsonl" \
  --output "$RUN_DIR/comparison.json"
printf 'Adapter and comparison saved to %s\n' "$RUN_DIR"
