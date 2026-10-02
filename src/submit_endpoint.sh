#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${NEBIUS_PARENT_ID:-project-e00qv62ppr00qnw83re3az}
SUBNET_ID=${NEBIUS_SUBNET_ID:-vpcsubnet-e00pemmjzw1rtz7nz0}
OUTPUT_BUCKET_ID=${NEBIUS_OUTPUT_BUCKET_ID:-storagebucket-e003360829724144205545}
GROUP_ID=${WORKSHOP_GROUP_ID:-demo}
MODEL_ID=${VLLM_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
ADAPTER_NAME=${VLLM_ADAPTER_NAME:-$GROUP_ID}
IMAGE=${VLLM_IMAGE:-docker.io/vllm/vllm-openai:latest}
DISK_SIZE=${NEBIUS_DISK_SIZE:-100Gi}
ENDPOINT_AUTH=${NEBIUS_ENDPOINT_AUTH:-none}

RUN_ID=${1:?Usage: submit_endpoint.sh RUN_ID [NEBIUS_OPTIONS]}
shift

[[ "$GROUP_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid group ID: $GROUP_ID" >&2; exit 2; }
[[ "$RUN_ID" =~ ^run-([a-z0-9][a-z0-9-]*-)?[0-9]{8}T[0-9]{6}Z-[0-9]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
[[ "$MODEL_ID" =~ ^[a-zA-Z0-9._/-]+$ ]] || { echo "Invalid model ID: $MODEL_ID" >&2; exit 2; }
[[ "$ADAPTER_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid adapter name: $ADAPTER_NAME" >&2; exit 2; }
[[ "$ENDPOINT_AUTH" == none || "$ENDPOINT_AUTH" == token ]] || { echo "Invalid endpoint auth: $ENDPOINT_AUTH" >&2; exit 2; }

ENDPOINT_NAME="shakespeare-$GROUP_ID-$(date -u +%Y%m%d%H%M%S)"
ADAPTER_PATH="/outputs/$GROUP_ID/runs/$RUN_ID/adapter"
SERVER_ARGS="-c \"python3 -m vllm.entrypoints.openai.api_server --model $MODEL_ID --enable-lora --lora-modules $ADAPTER_NAME=$ADAPTER_PATH --host 0.0.0.0 --port 8000\""

options=(
  --name "$ENDPOINT_NAME"
  --parent-id "$PROJECT_ID"
  --image "$IMAGE"
  --platform gpu-l40s-a
  --preset 1gpu-16vcpu-64gb
  --disk-size "$DISK_SIZE"
  --volume "$OUTPUT_BUCKET_ID:/outputs:ro"
  --subnet-id "$SUBNET_ID"
  --container-command bash
  --args "$SERVER_ARGS"
  --container-port 8000/http
  --auth "$ENDPOINT_AUTH"
)

if [[ -n "${NEBIUS_PROFILE:-}" ]]; then
  options+=(--profile "$NEBIUS_PROFILE")
fi

printf 'Endpoint name: %s\nModel: %s\nAdapter: %s\nAdapter path: %s\n' \
  "$ENDPOINT_NAME" "$MODEL_ID" "$ADAPTER_NAME" "$ADAPTER_PATH"

nebius ai endpoint create "${options[@]}" "$@"
