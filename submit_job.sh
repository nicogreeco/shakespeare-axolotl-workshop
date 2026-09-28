#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/hcls/workshop-llm
PROJECT_ID=${NEBIUS_PARENT_ID:-project-e00qv62ppr00qnw83re3az}
SUBNET_ID=${NEBIUS_SUBNET_ID:-vpcsubnet-e00pemmjzw1rtz7nz0}
INPUT_BUCKET_ID=${NEBIUS_INPUT_BUCKET_ID:?Set NEBIUS_INPUT_BUCKET_ID to the workshop input bucket ID}
OUTPUT_BUCKET_ID=${NEBIUS_OUTPUT_BUCKET_ID:?Set NEBIUS_OUTPUT_BUCKET_ID to the workshop output bucket ID}
GROUP_ID=${WORKSHOP_GROUP_ID:-demo}
RELEASE=${WORKSHOP_RELEASE:-v1}
IMAGE=docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1
[[ "$GROUP_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid group ID: $GROUP_ID" >&2; exit 2; }
[[ "$RELEASE" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid release: $RELEASE" >&2; exit 2; }

JOB_NAME="shakespeare-$GROUP_ID-$(date -u +%Y%m%d%H%M%S)"
JOB_ARGS="-c \"bash /inputs/releases/$RELEASE/run_job.sh /config/axolotl.yaml /outputs/$GROUP_ID\""
if [[ "${1:-}" == --compare-only ]]; then
  RUN_ID=${2:?Usage: submit_job.sh --compare-only RUN_ID [--dry-run]}
  [[ "$RUN_ID" =~ ^run-[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
  JOB_NAME="shakespeare-$GROUP_ID-compare-$(date -u +%Y%m%d%H%M%S)"
  JOB_ARGS="-c \"bash /inputs/releases/$RELEASE/run_job.sh --compare-only $RUN_ID /config/axolotl.yaml /outputs/$GROUP_ID\""
  shift 2
fi

test -s "$ROOT/axolotl.yaml" || { echo "Missing $ROOT/axolotl.yaml" >&2; exit 1; }

options=(
  --name "$JOB_NAME"
  --parent-id "$PROJECT_ID"
  --image "$IMAGE"
  --platform gpu-l40s-a
  --preset 1gpu-16vcpu-64gb
  --timeout 4h
  --volume "$INPUT_BUCKET_ID:/inputs:ro"
  --volume "$OUTPUT_BUCKET_ID:/outputs:rw"
  --inject-file "$ROOT/axolotl.yaml:/config/axolotl.yaml"
  --subnet-id "$SUBNET_ID"
  --container-command bash
  --args "$JOB_ARGS"
)
if [[ -n "${NEBIUS_PROFILE:-}" ]]; then
  options+=(--profile "$NEBIUS_PROFILE")
fi

printf 'Job name: %s\nProject ID: %s\nGroup: %s\nRelease: %s\n' "$JOB_NAME" "$PROJECT_ID" "$GROUP_ID" "$RELEASE"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf 'Local command preview (no job submitted):\n'
  printf '%q ' nebius ai job create "${options[@]}" "$@"
  printf '\n'
else
  nebius ai job create "${options[@]}" "$@"
fi
