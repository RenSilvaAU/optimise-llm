# Prefill and Decode Disaggregation Benchmark

Measure how splitting an LLM serving stack into a dedicated **prefill** worker
and a dedicated **decode** worker affects **TTFT** (time-to-first-token) and
**tail ITL** (inter-token latency) under concurrency, compared to a single
vLLM engine.

```
┌────────────┐                                ┌───────────────────────────────┐
│  baseline  │                                │         disaggregated         │
│            │                                │                               │
│   GPU 0 ───┴─► vLLM tp=1 :8000              │  GPU 0 ─► prefill vLLM :8100  │
│                                             │  GPU 1 ─► decode  vLLM :8200  │
│                                             │  host  ─► router       :8000  │
└─────────────────────────────────────────────┴───────────────────────────────┘
```

The notebook hits the same URL (`http://localhost:8000`) in both cases, so
the comparison is apples-to-apples.

## Honest caveat

vLLM's own docs say **disagg does NOT improve throughput**. On a 2-GPU box
disagg may even lose on absolute TTFT, because the baseline gets all the
compute on one engine while disagg has to split it. The metric where disagg
should win is **flat ITL under concurrency** — decode never stalls behind a
fresh prefill.

## Requirements

- 2× NVIDIA GPUs (tested on 2× A100 80GB).
- Docker with the NVIDIA Container Toolkit:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```

  Both GPUs should appear.
- Python 3.10+ (a venv is fine; the notebook self-bootstraps `pip`).

## Setup

```bash
git clone <this-repo>
cd disaggregation-experiment

cp .env.example .env
# Edit .env — fill in HF_TOKEN if you plan to use a gated model.
# Qwen (the default) is open and works with an empty HF_TOKEN.
```

`.env` is git-ignored. Never commit secrets.

## Running the experiment

The notebook walks you through it. Order:

1. **Setup** cells — install deps, smoke test, load helpers.
2. **Scenario A — Baseline.** In a terminal:
   ```bash
   ./start_baseline.sh
   ```
   Run the Scenario A cells in the notebook. Stop the server with `Ctrl+C`.
3. **Scenario B — Disaggregated.** In a terminal:
   ```bash
   ./start_disagg.sh
   ```
   Run the Scenario B cells.
4. **Compare** cell — plots baseline vs disagg.

Both servers bind port `8000`, so only one can run at a time.

## Files

| File | Purpose |
|---|---|
| `experiment.ipynb` | Benchmark + plots |
| `start_baseline.sh` | Single vLLM container (tp=1, GPU 0) |
| `start_disagg.sh` | Prefill (GPU 0) + decode (GPU 1) + NIXL router |
| `.env.example` | Template for `HF_TOKEN`, `MODEL`, `HF_CACHE` |
| `.env` | Your real values (git-ignored) |
| `.gitignore` | Excludes `.env`, caches, checkpoints |

## Switching models

Edit `MODEL` in `.env` (read by both scripts and re-set in the notebook).
Gated models (Llama, Mistral) also need `HF_TOKEN` and a one-time licence
acceptance on the Hugging Face web UI.

## Stopping everything

```bash
docker rm -f vllm-baseline vllm-prefill vllm-decode 2>/dev/null
pkill -f toy_proxy_server.py 2>/dev/null
```

## References

- vLLM disaggregated prefill docs:
  <https://docs.vllm.ai/en/latest/features/disagg_prefill.html>
- NIXL connector source:
  <https://github.com/vllm-project/vllm/tree/main/tests/v1/kv_connector/nixl_integration>

## Security notes

- Never commit `.env`. Use `.env.example` as the template.
- If you previously had API keys (NGC, HF) in any file in this repo,
  **revoke and regenerate** them now.
