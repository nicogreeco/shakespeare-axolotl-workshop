#!/usr/bin/env bash
set -euo pipefail

# Paths passed by the Nebius job command.
CONFIG_PATH=${1:?Usage: run_job.sh CONFIG_PATH GROUP_NAME [RUN_ID]}
GROUP_NAME=${2:?Usage: run_job.sh CONFIG_PATH GROUP_NAME [RUN_ID]}
REQUESTED_RUN_ID=${3:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-/outputs}

RUN_LABEL=${RUN_LABEL:-}
[[ -z "$RUN_LABEL" || "$RUN_LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid run label: $RUN_LABEL" >&2; exit 2; }
[[ "$GROUP_NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid group name: $GROUP_NAME (use lowercase letters, numbers, and hyphens)" >&2; exit 2; }

# Use a supplied run ID when requested; otherwise make one unique to this job.
if [[ -n "$REQUESTED_RUN_ID" ]]; then
  [[ "$REQUESTED_RUN_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid run ID: $REQUESTED_RUN_ID (use lowercase letters, numbers, and hyphens)" >&2; exit 2; }
  RUN_ID="$REQUESTED_RUN_ID"
else
  RUN_ID="run-${RUN_LABEL:+$RUN_LABEL-}$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
GROUP_OUTPUT_DIR="$OUTPUT_ROOT/$GROUP_NAME"
RUN_OUTPUT_DIR="$GROUP_OUTPUT_DIR/runs/$RUN_ID"
ADAPTER_DIR="$RUN_OUTPUT_DIR/adapter"
AXOLOTL_OUTPUT_DIR=/workspace/output
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Keep the helper scripts beside this runner, both locally and in the S3 release.
for helper in compare.py plot_losses.py; do
  [[ -f "$SCRIPT_DIR/$helper" ]] || { echo "Missing helper: $SCRIPT_DIR/$helper" >&2; exit 1; }
done
COMPARE_SCRIPT="$SCRIPT_DIR/compare.py"

echo "Run ID: $RUN_ID"
echo "Output directory: $RUN_OUTPUT_DIR"

if [[ ! -s "$CONFIG_PATH" ]]; then
  echo "Config file not found or empty: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$ADAPTER_DIR"
cp "$CONFIG_PATH" "$RUN_OUTPUT_DIR/axolotl.yaml"

echo "Starting Axolotl training..."
axolotl train "$CONFIG_PATH" 2>&1 | tee "$RUN_OUTPUT_DIR/training.log"

echo "Creating training loss artifacts..."
if python3 "$SCRIPT_DIR/plot_losses.py" \
  --input "$AXOLOTL_OUTPUT_DIR" \
  --csv "$RUN_OUTPUT_DIR/loss.csv" \
  --svg "$RUN_OUTPUT_DIR/loss.svg"; then
  echo "Loss CSV and plot saved."
else
  echo "Warning: loss artifacts could not be created." >&2
fi

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
