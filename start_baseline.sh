#!/usr/bin/env bash
# BASELINE: one vLLM server, tp=1 on GPU 0 only.
# Same per-stage compute budget as the disagg setup -> fair comparison.
# Exposes OpenAI API on http://localhost:8000.
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

echo "==> BASELINE: $MODEL on GPU 0 (tp=1), endpoint http://localhost:8000"

docker run --rm -it \
    --name vllm-baseline \
    --gpus '"device=0"' \
    --shm-size=16g \
    --ipc=host \
    -p 8000:8000 \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384
