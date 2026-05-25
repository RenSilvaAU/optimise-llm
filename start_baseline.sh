#!/usr/bin/env bash
# BASELINE (Scenario A): NVIDIA NIM for nvidia/nemotron-3-nano.
# All GPUs, auto-selected profile, NIM downloads weights itself on first run.
set -euo pipefail

set -a; source .env; set +a

NIM_IMG=nvcr.io/nim/nvidia/nemotron-3-nano:latest
NIM_CACHE_PATH=${NIM_CACHE_PATH:-$HOME/.cache/nim}
mkdir -p "$NIM_CACHE_PATH"
# NIM runs as a non-root user in group 0; make the cache writable for it.
chmod -R a+rwX "$NIM_CACHE_PATH"

echo "==> BASELINE (NIM): nvidia/nemotron-3-nano on all GPUs at http://localhost:8000"

docker run --rm -it \
    --name nim-baseline \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    -p 8000:8000 \
    -e NGC_API_KEY \
    -e NIM_CACHE_PATH=/opt/nim/.cache \
    -v "$NIM_CACHE_PATH":/opt/nim/.cache \
    "$NIM_IMG"
