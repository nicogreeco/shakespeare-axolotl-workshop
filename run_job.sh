#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == --compare-only ]]; then
  RUN_ID=${2:?Usage: run_job.sh --compare-only RUN_ID CONFIG OUTPUT_DIR}
  CONFIG=${3:?Usage: run_job.sh --compare-only RUN_ID CONFIG OUTPUT_DIR}
  OUTPUT_DIR=${4:?Usage: run_job.sh --compare-only RUN_ID CONFIG OUTPUT_DIR}
  [[ "$RUN_ID" =~ ^run-[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
  RUN_DIR="$OUTPUT_DIR/runs/$RUN_ID"
  test -s "$RUN_DIR/adapter/adapter_config.json" || { echo "Adapter missing: $RUN_DIR/adapter" >&2; exit 1; }
  python3 "$SCRIPT_DIR/compare.py" \
    --config "$CONFIG" \
    --adapter "$RUN_DIR/adapter" \
    --output "$RUN_DIR/comparison.json" \
    --markdown-output "$RUN_DIR/comparison.md"
  exit
fi

CONFIG=${1:?Usage: run_job.sh CONFIG OUTPUT_DIR}
OUTPUT_DIR=${2:?Usage: run_job.sh CONFIG OUTPUT_DIR}
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$OUTPUT_DIR/runs/$RUN_ID"

test -s "$CONFIG"
mkdir -p "$RUN_DIR/adapter"
printf 'Run directory: %s\n' "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/axolotl.yaml"

axolotl train "$CONFIG"

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

python3 "$SCRIPT_DIR/compare.py" \
  --config "$CONFIG" \
  --adapter "$RUN_DIR/adapter" \
  --output "$RUN_DIR/comparison.json" \
  --markdown-output "$RUN_DIR/comparison.md"
touch "$RUN_DIR/_SUCCESS"
printf 'Adapter and comparison saved to %s\n' "$RUN_DIR"
