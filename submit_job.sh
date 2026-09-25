#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/hcls/workshop-llm
FILESYSTEM_ID=computefilesystem-e00jp4z98aw5jyyyq4
PROJECT_ID=${NEBIUS_PARENT_ID:-project-e00qv62ppr00qnw83re3az}
SUBNET_ID=${NEBIUS_SUBNET_ID:-vpcsubnet-e00pemmjzw1rtz7nz0}
IMAGE=docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1
JOB_NAME="shakespeare-lora-$(date -u +%Y%m%d%H%M%S)"
JOB_ARGS='-c "bash /workspace/data/workshop-llm/run_job.sh"'
if [[ "${1:-}" == --compare-only ]]; then
  RUN_ID=${2:?Usage: submit_job.sh --compare-only RUN_ID [--dry-run]}
  [[ "$RUN_ID" =~ ^run-[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
  test -s "$ROOT/runs/$RUN_ID/adapter/adapter_config.json" || { echo "Adapter missing: $ROOT/runs/$RUN_ID/adapter" >&2; exit 1; }
  JOB_NAME="shakespeare-compare-$(date -u +%Y%m%d%H%M%S)"
  JOB_ARGS="-c \"bash /workspace/data/workshop-llm/run_job.sh --compare-only $RUN_ID\""
  shift 2
fi

test -s "$ROOT/data/train.jsonl" || { echo 'Run python3 prepare_data.py first.' >&2; exit 1; }
test -s "$ROOT/data/validation.jsonl" || { echo 'Run python3 prepare_data.py first.' >&2; exit 1; }
for file in axolotl.yaml run_job.sh compare.py; do
  test -s "$ROOT/$file" || { echo "Missing $ROOT/$file" >&2; exit 1; }
done

options=(
  --name "$JOB_NAME"
  --parent-id "$PROJECT_ID"
  --image "$IMAGE"
  --platform gpu-l40s-a
  --preset 1gpu-16vcpu-64gb
  --timeout 4h
  --volume "$FILESYSTEM_ID:/workspace/data"
  --subnet-id "$SUBNET_ID"
  --container-command bash
  --args "$JOB_ARGS"
)
if [[ -n "${NEBIUS_PROFILE:-}" ]]; then
  options+=(--profile "$NEBIUS_PROFILE")
fi

printf 'Job name: %s\nProject ID: %s\n' "$JOB_NAME" "$PROJECT_ID"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf 'Local command preview (no job submitted):\n'
  printf '%q ' nebius ai job create "${options[@]}" "$@"
  printf '\n'
else
  nebius ai job create "${options[@]}" "$@"
fi
