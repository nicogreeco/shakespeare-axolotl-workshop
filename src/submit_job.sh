#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ID=${NEBIUS_PARENT_ID:-project-e00kjyj0pr00mf5kcczh6n}
SUBNET_ID=${NEBIUS_SUBNET_ID:-vpcsubnet-e00epty88k87jrawhx}
INPUT_BUCKET_ID=${NEBIUS_INPUT_BUCKET_ID:-storagebucket-e009119238704004704039}
OUTPUT_BUCKET_ID=${NEBIUS_OUTPUT_BUCKET_ID:-storagebucket-e005649953926204087405}
GROUP_ID=${WORKSHOP_GROUP_ID:-demo}
RELEASE=${WORKSHOP_RELEASE:-v1}
RUN_LABEL=${WORKSHOP_RUN_LABEL:-}
RUN_ID=${WORKSHOP_RUN_ID:-}
CONFIG_PATH=${AXOLOTL_CONFIG_PATH:-$ROOT/training.yaml}
DISK_SIZE=${NEBIUS_DISK_SIZE:-200Gi}
PLATFORM=${NEBIUS_PLATFORM:-gpu-h100-sxm}
PRESET=${NEBIUS_PRESET:-1gpu-16vcpu-200gb}
IMAGE=docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1
[[ "$GROUP_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid group ID: $GROUP_ID" >&2; exit 2; }
[[ "$RELEASE" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid release: $RELEASE" >&2; exit 2; }
[[ -z "$RUN_LABEL" || "$RUN_LABEL" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid run label: $RUN_LABEL" >&2; exit 2; }
[[ -z "$RUN_ID" || "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }

JOB_PREFIX=${RUN_LABEL:-shakespeare}
JOB_NAME="$JOB_PREFIX-$GROUP_ID-$(date -u +%Y%m%d%H%M%S)"
JOB_ARGS="-c \"bash /inputs/releases/$RELEASE/run_job.sh /config/axolotl.yaml $GROUP_ID${RUN_ID:+ $RUN_ID}\""
test -s "$CONFIG_PATH" || { echo "Missing config: $CONFIG_PATH" >&2; exit 1; }

options=(
  --name "$JOB_NAME"
  --parent-id "$PROJECT_ID"
  --image "$IMAGE"
  --platform "$PLATFORM"
  --preset "$PRESET"
  --disk-size "$DISK_SIZE"
  --timeout 4h
  --volume "$INPUT_BUCKET_ID:/inputs:ro"
  --volume "$OUTPUT_BUCKET_ID:/outputs:rw"
  --inject-file "$CONFIG_PATH:/config/axolotl.yaml"
  --subnet-id "$SUBNET_ID"
  --container-command bash
  --args "$JOB_ARGS"
)
if [[ -n "$RUN_LABEL" ]]; then
  options+=(--env "RUN_LABEL=$RUN_LABEL")
fi
if [[ -n "${NEBIUS_PROFILE:-}" ]]; then
  options+=(--profile "$NEBIUS_PROFILE")
fi

printf 'Job name: %s\nProject ID: %s\nGroup: %s\nRun label: %s\nRun ID: %s\nRelease: %s\nDisk size: %s\n' \
  "$JOB_NAME" "$PROJECT_ID" "$GROUP_ID" "${RUN_LABEL:-none}" "${RUN_ID:-auto-generated}" "$RELEASE" "$DISK_SIZE"
printf 'Config: %s\n' "$CONFIG_PATH"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf 'Local command preview (no job submitted):\n'
  printf '%q ' nebius ai job create "${options[@]}" "$@"
  printf '\n'
else
  nebius ai job create "${options[@]}" "$@"
fi
