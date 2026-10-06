#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${NEBIUS_PARENT_ID:-project-e00kjyj0pr00mf5kcczh6n}
SUBNET_ID=${NEBIUS_SUBNET_ID:-vpcsubnet-e00epty88k87jrawhx}
OUTPUT_BUCKET_ID=${NEBIUS_OUTPUT_BUCKET_ID:-storagebucket-e005649953926204087405}
GROUP_ID=${WORKSHOP_GROUP_ID:-demo}
MODEL_ID=${VLLM_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
ADAPTER_NAME=${VLLM_ADAPTER_NAME:-$GROUP_ID}
IMAGE=${VLLM_IMAGE:-docker.io/vllm/vllm-openai:latest}
DISK_SIZE=${NEBIUS_DISK_SIZE:-100Gi}
ENDPOINT_AUTH=${NEBIUS_ENDPOINT_AUTH:-none}

BASE_ONLY=false
if [[ "${1:-}" == --base-only ]]; then
  BASE_ONLY=true
  RUN_ID=
  shift
else
  RUN_ID=${1:?Usage: submit_endpoint.sh --base-only | RUN_ID [NEBIUS_OPTIONS]}
  shift
fi

[[ "$GROUP_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Invalid group ID: $GROUP_ID" >&2; exit 2; }
[[ -z "$RUN_ID" || "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]*$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
[[ "$MODEL_ID" =~ ^[a-zA-Z0-9._/-]+$ ]] || { echo "Invalid model ID: $MODEL_ID" >&2; exit 2; }
[[ "$ADAPTER_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid adapter name: $ADAPTER_NAME" >&2; exit 2; }
[[ "$ENDPOINT_AUTH" == none || "$ENDPOINT_AUTH" == token ]] || { echo "Invalid endpoint auth: $ENDPOINT_AUTH" >&2; exit 2; }

ENDPOINT_KIND=adapter
ENDPOINT_NAME="shakespeare-$GROUP_ID-$(date -u +%Y%m%d%H%M%S)"
if [[ "$BASE_ONLY" == true ]]; then
  ENDPOINT_KIND=base
  ENDPOINT_NAME="base-$GROUP_ID-$(date -u +%Y%m%d%H%M%S)"
  SERVER_ARGS="-c \"python3 -m vllm.entrypoints.openai.api_server --model $MODEL_ID --host 0.0.0.0 --port 8000\""
else
  ADAPTER_PATH="/outputs/$GROUP_ID/runs/$RUN_ID/adapter"
  SERVER_ARGS="-c \"python3 -m vllm.entrypoints.openai.api_server --model $MODEL_ID --enable-lora --lora-modules $ADAPTER_NAME=$ADAPTER_PATH --host 0.0.0.0 --port 8000\""
fi

options=(
  --name "$ENDPOINT_NAME"
  --parent-id "$PROJECT_ID"
  --image "$IMAGE"
  --platform gpu-l40s-a
  --preset 1gpu-16vcpu-64gb
  --disk-size "$DISK_SIZE"
  --subnet-id "$SUBNET_ID"
  --container-command bash
  --args "$SERVER_ARGS"
  --container-port 8000/http
  --auth "$ENDPOINT_AUTH"
)

if [[ "$BASE_ONLY" == false ]]; then
  options+=(--volume "$OUTPUT_BUCKET_ID:/outputs:ro")
fi

if [[ -n "${NEBIUS_PROFILE:-}" ]]; then
  options+=(--profile "$NEBIUS_PROFILE")
fi

printf 'Endpoint name: %s\nType: %s\nModel: %s\n' \
  "$ENDPOINT_NAME" "$ENDPOINT_KIND" "$MODEL_ID"
if [[ "$BASE_ONLY" == false ]]; then
  printf 'Adapter: %s\nAdapter path: %s\n' "$ADAPTER_NAME" "$ADAPTER_PATH"
fi

nebius ai endpoint create "${options[@]}" "$@"
