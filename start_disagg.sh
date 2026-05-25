#!/usr/bin/env bash
# DISAGGREGATED (Scenario B): NVIDIA Dynamo, 1 prefill + 1 decode vLLM worker,
# KV transferred over NVIDIA NIXL. OpenAI API on :8000.
#
# Reuses the weights the NIM baseline already downloaded into $NIM_CACHE_PATH,
# so run ./start_baseline.sh at least once first.
set -euo pipefail

set -a; source .env; set +a

DYNAMO_IMG=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
NIM_CACHE_PATH=${NIM_CACHE_PATH:-$HOME/.cache/nim}
MODEL=${MODEL:-nvidia/nemotron-3-nano}

# Find the weight directory inside the NIM cache (HF-style snapshots layout).
# Prune the root-owned `tmp/` subtree and silence any remaining permission noise.
CFG=$(find "$NIM_CACHE_PATH" \
        -path '*/tmp' -prune -o \
        -path '*/snapshots/*' -name config.json -print \
        2>/dev/null | head -n1)
[[ -n "$CFG" ]] || { echo "ERROR: no weights in $NIM_CACHE_PATH. Run ./start_baseline.sh once first."; exit 1; }
MODEL_PATH="/weights${CFG#$NIM_CACHE_PATH}"
MODEL_PATH="${MODEL_PATH%/config.json}"

echo "==> DISAGG (Dynamo): $MODEL"
echo "    weights: $MODEL_PATH"

docker run --rm -it \
    --name dynamo-disagg \
    --gpus all \
    --shm-size=16g --ipc=host --network=host \
    -e MODEL="$MODEL" -e MODEL_PATH="$MODEL_PATH" \
    -v "$NIM_CACHE_PATH":/weights:ro \
    "$DYNAMO_IMG" bash -lc '
set -e
trap "kill 0" EXIT
KV='\''{"kv_connector":"NixlConnector","kv_role":"kv_both"}'\''

# Dynamo discovery + message bus (frontend/workers connect here).
nats-server -js -a 127.0.0.1 -p 4222 >/tmp/nats.log 2>&1 &
etcd --listen-client-urls http://127.0.0.1:2379 \
     --advertise-client-urls http://127.0.0.1:2379 \
     --data-dir /tmp/etcd-data >/tmp/etcd.log 2>&1 &

# Wait until both are accepting connections.
for i in $(seq 1 30); do
  (echo > /dev/tcp/127.0.0.1/4222) 2>/dev/null && \
  (echo > /dev/tcp/127.0.0.1/2379) 2>/dev/null && break
  sleep 1
done

export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379

python3 -m dynamo.frontend --http-port 8000 &

CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 python3 -m dynamo.vllm \
    --model "$MODEL_PATH" --served-model-name "$MODEL" \
    --disaggregation-mode prefill --kv-transfer-config "$KV" \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code &

CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=5601 python3 -m dynamo.vllm \
    --model "$MODEL_PATH" --served-model-name "$MODEL" \
    --disaggregation-mode decode --kv-transfer-config "$KV" \
    --no-disable-hybrid-kv-cache-manager \
    --trust-remote-code &

wait -n
'
