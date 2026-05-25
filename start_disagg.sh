#!/usr/bin/env bash
# DISAGGREGATED: prefill worker on GPU 0 + decode worker on GPU 1 + router on :8000.
# Same OpenAI-compatible endpoint as the baseline -> notebook works unchanged.
#
#   GPU 0  -> vLLM "prefill"  on :8100  (NixlConnector, kv_role=kv_both)
#   GPU 1  -> vLLM "decode"   on :8200  (NixlConnector, kv_role=kv_both)
#   host   -> toy_proxy_server.py on :8000
#
# NixlConnector is experimental. Both workers must run from the same vLLM image.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Load secrets/config from .env if present.
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
mkdir -p "$HF_CACHE"

VLLM_IMG=vllm/vllm-openai:latest

# KV transfer config (NixlConnector, both producer+consumer roles).
KV_CFG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Side-channel ports used by NIXL for handshakes.
PREFILL_SIDE_PORT=5559
DECODE_SIDE_PORT=5659

cleanup() {
    echo "Stopping containers..."
    docker rm -f vllm-prefill vllm-decode >/dev/null 2>&1 || true
    if [[ -n "${PROXY_PID:-}" ]]; then
        kill "$PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup INT TERM EXIT

echo "==> Starting PREFILL worker on GPU 0 (port 8100)..."
docker run -d --rm \
    --name vllm-prefill \
    --gpus '"device=0"' \
    --shm-size=16g \
    --ipc=host \
    --network=host \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SIDE_PORT \
    -e VLLM_KV_CACHE_LAYOUT=HND \
    -e UCX_NET_DEVICES=all \
    "$VLLM_IMG" \
    --model "$MODEL" \
    --port 8100 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --kv-transfer-config "$KV_CFG"

echo "==> Starting DECODE worker on GPU 1 (port 8200)..."
docker run -d --rm \
    --name vllm-decode \
    --gpus '"device=1"' \
    --shm-size=16g \
    --ipc=host \
    --network=host \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SIDE_PORT \
    -e VLLM_KV_CACHE_LAYOUT=HND \
    -e UCX_NET_DEVICES=all \
    "$VLLM_IMG" \
    --model "$MODEL" \
    --port 8200 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --kv-transfer-config "$KV_CFG"

echo "==> Waiting for both workers to become ready..."
for port in 8100 8200; do
    for i in {1..600}; do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "    port $port is ready."
            break
        fi
        sleep 2
        if [[ $i -eq 600 ]]; then
            echo "ERROR: worker on port $port never became ready."
            docker logs --tail=80 "vllm-prefill" || true
            docker logs --tail=80 "vllm-decode"  || true
            exit 1
        fi
    done
done

# Fetch the reference toy proxy from the pinned vLLM main branch.
PROXY_PY=/tmp/toy_proxy_server.py
if [[ ! -f "$PROXY_PY" ]]; then
    echo "==> Downloading toy_proxy_server.py from vllm main..."
    curl -fsSL \
        https://raw.githubusercontent.com/vllm-project/vllm/main/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
        -o "$PROXY_PY"
fi

# The proxy needs `aiohttp` + `quart` style deps; the script uses httpx+fastapi/uvicorn.
python3 -m pip install --quiet httpx fastapi uvicorn || true

echo "==> Starting router on :8000 (prefill=8100, decode=8200)..."
python3 "$PROXY_PY" \
    --port 8000 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200 &
PROXY_PID=$!

echo
echo "Disaggregated endpoint is now live at: http://localhost:8000/v1/chat/completions"
echo "Model id to send in payload: $MODEL"
echo "Press Ctrl+C to stop everything."
wait "$PROXY_PID"
