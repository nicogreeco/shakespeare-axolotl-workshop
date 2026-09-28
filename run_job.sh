#!/usr/bin/env bash
set -euo pipefail

# Paths passed by the Nebius job command.
CONFIG_PATH=${1:?Usage: run_job.sh CONFIG_PATH GROUP_OUTPUT_DIR}
GROUP_OUTPUT_DIR=${2:?Usage: run_job.sh CONFIG_PATH GROUP_OUTPUT_DIR}

# Each training run gets its own folder in the group's output directory.
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_OUTPUT_DIR="$GROUP_OUTPUT_DIR/runs/$RUN_ID"
ADAPTER_DIR="$RUN_OUTPUT_DIR/adapter"
AXOLOTL_OUTPUT_DIR=/workspace/output
COMPARE_SCRIPT="$(dirname "$0")/compare.py"

echo "Run ID: $RUN_ID"
echo "Output directory: $RUN_OUTPUT_DIR"

if [[ ! -s "$CONFIG_PATH" ]]; then
  echo "Config file not found or empty: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$ADAPTER_DIR"
cp "$CONFIG_PATH" "$RUN_OUTPUT_DIR/axolotl.yaml"

echo "Starting Axolotl training..."
axolotl train "$CONFIG_PATH"

echo "Saving the trained adapter..."
if [[ ! -s "$AXOLOTL_OUTPUT_DIR/adapter_config.json" ]]; then
  echo "Adapter config not found in $AXOLOTL_OUTPUT_DIR" >&2
  exit 1
fi
cp "$AXOLOTL_OUTPUT_DIR/adapter_config.json" "$ADAPTER_DIR/"

if [[ -s "$AXOLOTL_OUTPUT_DIR/adapter_model.safetensors" ]]; then
  cp "$AXOLOTL_OUTPUT_DIR/adapter_model.safetensors" "$ADAPTER_DIR/"
elif [[ -s "$AXOLOTL_OUTPUT_DIR/adapter_model.bin" ]]; then
  cp "$AXOLOTL_OUTPUT_DIR/adapter_model.bin" "$ADAPTER_DIR/"
else
  echo "Adapter weights not found in $AXOLOTL_OUTPUT_DIR" >&2
  exit 1
fi

echo "Comparing the base model with the trained adapter..."
python3 "$COMPARE_SCRIPT" \
  --config "$CONFIG_PATH" \
  --adapter "$ADAPTER_DIR" \
  --output "$RUN_OUTPUT_DIR/comparison.json" \
  --markdown-output "$RUN_OUTPUT_DIR/comparison.md"

touch "$RUN_OUTPUT_DIR/_SUCCESS"
echo "Training completed. Results saved to $RUN_OUTPUT_DIR"
