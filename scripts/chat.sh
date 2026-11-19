#!/usr/bin/env bash
# On this workstation: tunnel to llama-server on the TPU VM (scripts/serve.sh) and open the chat page.
# usage: scripts/chat.sh            then browse http://localhost:8765
# env:   TPU_NODE (ggml-tpu-burnin-node), TPU_ZONE (us-west4-a), TPU_PROJECT (aisprint-491218),
#        REMOTE_PORT (8090), LOCAL_PORT (8090), UI_PORT (8765), IAP=1 to tunnel through IAP
set -euo pipefail
NODE=${TPU_NODE:-ggml-tpu-burnin-node}
ZONE=${TPU_ZONE:-us-west4-a}
PROJECT=${TPU_PROJECT:-aisprint-491218}
LP=${LOCAL_PORT:-8090}
HERE=$(cd "$(dirname "$0")/.." && pwd)

ACC=$(gcloud compute tpus tpu-vm describe "$NODE" --zone="$ZONE" --project="$PROJECT" --format='value(acceleratorType)')
gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --project="$PROJECT" ${IAP:+--tunnel-through-iap} \
    -- -N -L "$LP:127.0.0.1:${REMOTE_PORT:-8090}" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null' EXIT

for _ in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:$LP/health" > /dev/null 2>&1 && break
    kill -0 $TUNNEL 2>/dev/null || { echo "tunnel exited" >&2; exit 1; }
    sleep 1
done
curl -fsS "http://127.0.0.1:$LP/health" > /dev/null || echo "warning: llama-server not answering yet on :$LP" >&2

python3 "$HERE/ui/chat_ui.py" --upstream "http://127.0.0.1:$LP" --port "${UI_PORT:-8765}" \
    --rig "ACCELERATOR_TYPE=$ACC" "ZONE=$ZONE" "GGML_TPU_EXECUTOR=${GGML_TPU_EXECUTOR:-pjrt}" \
          "GGML_TPU_QUANT=${GGML_TPU_QUANT:-bf16}" "GGML_TPU_PRECISION=${GGML_TPU_PRECISION:-DEFAULT}"
