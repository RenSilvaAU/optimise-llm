# Disaggregated Prefill/Decode on the NVIDIA Stack

Measure how splitting an LLM serving stack into a dedicated **prefill** worker
and a dedicated **decode** worker affects **TTFT** (time-to-first-token) and
**tail ITL** (inter-token latency) under concurrency, compared to a single
aggregated engine — using **only NVIDIA software and NVIDIA-hosted weights**.

```
┌───────────────────────────────┐   ┌─────────────────────────────────────────┐
│  Scenario A (baseline)        │   │  Scenario B (disaggregated)             │
│                               │   │                                         │
│  NVIDIA NIM container         │   │  NVIDIA Dynamo orchestrating vLLM:      │
│   nemotron-3-nano on GPU 0    │   │   GPU 0 -> dynamo.vllm prefill          │
│   (vllm-bf16-tp1-pp1)         │   │   GPU 1 -> dynamo.vllm decode           │
│   API: http://localhost:8000  │   │   host  -> dynamo.frontend  :8000       │
│                               │   │   KV transfer over NVIDIA NIXL          │
└───────────────────────────────┘   └─────────────────────────────────────────┘
```

The notebook hits the same URL (`http://localhost:8000`) in both cases, so the
comparison is apples-to-apples.

## Architecture: how the disaggregated stack is wired

In the baseline, vLLM owns `:8000` directly — one process, one engine, both
phases. In the disaggregated stack, **Dynamo's frontend is the front door** and
the vLLM workers sit behind it:

```
   client (notebook)
        │  POST /v1/chat/completions
        ▼
   ┌──────────────────────────────┐
   │  dynamo.frontend  :8000      │  ← OpenAI-compatible HTTP, tokenizer, SSE
   └────────────┬─────────────────┘
                │ internal RPC
                │ (NATS message bus + etcd discovery)
        ┌───────┴────────┐
        ▼                ▼
  ┌───────────┐    ┌───────────┐
  │ vLLM      │    │ vLLM      │
  │ prefill   │    │ decode    │
  │ GPU 0     │───►│ GPU 1     │  ← KV cache shipped via NVIDIA NIXL
  └───────────┘    └───────────┘
       :5600           :5601        ← NIXL side-channel ports
```

- **Frontend** owns `:8000`, looks up workers via **etcd**, routes requests
  over **NATS**, streams tokens back to the client.
- **Prefill worker** reads the prompt, builds the KV cache, ships it to decode
  over **NIXL** (RDMA/GPUDirect; the `:5600`/`:5601` side-channel is for the
  handshake — each worker needs its own port because both share the host
  network namespace).
- **Decode worker** generates tokens reusing that KV cache, never paused by an
  incoming prefill from another user.

`start_disagg.sh` runs **NATS + etcd inside the same Dynamo container** (both
binaries ship with the image), so the script remains one-command despite the
extra moving parts.

## Pure-NVIDIA stack

| Layer | Component | Source |
|---|---|---|
| Model | `nvidia/nemotron-3-nano` (NemotronH hybrid Mamba2/Transformer) | NVIDIA NGC |
| Weights | downloaded by the NIM's `download-to-cache` | `ngc.nvidia.com` |
| Baseline runtime | NVIDIA NIM (vLLM under nginx) | `nvcr.io/nim/nvidia/nemotron-3-nano` |
| Disagg orchestrator | NVIDIA Dynamo 1.1.1 (`dynamo.frontend` + `dynamo.vllm`) | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` |
| KV transfer | NVIDIA NIXL (`NixlConnector`) | bundled in the Dynamo image |
| Hardware | 2× NVIDIA A100 80GB PCIe | local |

**No Hugging Face tokens. No HF downloads.** The only credential needed is an
NGC API key.

## Honest caveats

1. Disaggregated serving does **not** improve throughput. On a 2-GPU box it may
   even lose on absolute TTFT (the baseline gets all the compute on one
   engine). The metric where disagg should win is **flat ITL under
   concurrency** — decode never stalls behind a fresh prefill.
2. `nemotron-3-nano` is a **hybrid Mamba/Transformer**. vLLM 0.19's
   `NixlConnector` is proven for pure-attention KV cache; SSM state transfer
   across GPUs is less battle-tested. If the disagg launch errors out with
   `mamba` / `ssm_state` messages, that itself is the finding — document it
   and fall back to a pure-Transformer model.

## Findings (run 2026-05-25)

The full stack stands up cleanly end-to-end:

- `start_baseline.sh` — NIM serves `nemotron-3-nano` on `:8000`. Real answers.
- `start_disagg.sh` — Dynamo frontend + NATS + etcd + prefill worker (GPU 0) +
  decode worker (GPU 1) all start, models load on both GPUs, requests reach
  the workers, HTTP 200, valid SSE frames, ~470 ms TTFT for a 16-token
  completion.

**But every output token decodes to `<unk>`.** Root cause: vLLM's
`NixlConnector` transfers the attention KV cache prefill→decode but does
**not** transfer the Mamba **SSM hidden state**. The decode worker therefore
runs the model with a zeroed SSM, producing nothing but the unknown-token id.

That is exactly caveat #2 above, confirmed empirically. The infrastructure
works; the **model choice** is incompatible with NIXL-based disaggregation
today.

### What was needed to get this far (post-mortem of `start_disagg.sh`)

The initial script was a one-liner that assumed Dynamo would just work. Real
fixes that went in along the way:

| Symptom | Root cause | Fix |
|---|---|---|
| `find: '.../nim/tmp/...': Permission denied` | NIM left a root-owned `tmp/` subdir inside the cache mount | `-path '*/tmp' -prune` in the `find` + `sudo rm -rf` the stale dir |
| `Failed to connect to NATS: Connection refused` | Dynamo frontend needs NATS (messaging) + etcd (discovery); script never started them | Launch `nats-server -js` and `etcd` **inside the container** before `dynamo.frontend` (both ship in the image) |
| `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` | vLLM force-disables hybrid KV cache when `--kv-transfer-config` is set; hybrid model can't unify Mamba + attention specs | `--no-disable-hybrid-kv-cache-manager` on both workers |
| `zmq.error.ZMQError: Address already in use` on `:5600` | Both workers tried to bind the same NIXL side-channel port (host networking) | `VLLM_NIXL_SIDE_CHANNEL_PORT=5600` for prefill, `5601` for decode |
| Dynamo `HTTP 400: invalid request parameters` from the notebook | `"cache_prompt": false` in the request body — vLLM-only extension, Dynamo validates strictly | Dropped from `run_request` in the notebook |
| All tokens come back as `<unk>` | NIXL transfers attention KV only; Mamba SSM state is **not** transferred | **Not fixable in this stack** — needs a pure-attention model |

See `TODO.md` for the planned path forward (swap to Llama/Qwen, re-run both
scenarios, then this comparison becomes meaningful).

## Requirements

- 2× NVIDIA GPUs (tested on 2× A100 80GB PCIe).
- Docker with the NVIDIA Container Toolkit:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
  ```
- An **NGC API key** (https://org.ngc.nvidia.com/setup/api-key).
- ~80 GB free disk for model weights + container images.
- Python 3.10+ for the notebook (the notebook self-bootstraps `pip`).

## Setup

```bash
cp .env.example .env
# Edit .env: set NGC_API_KEY=... at minimum.

# One-time login so docker can pull from nvcr.io
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Pull the two images we use
docker pull nvcr.io/nim/nvidia/nemotron-3-nano:latest
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1

# Pull weights from NGC into $NIM_CACHE_PATH (one-time, ~17 GB)
./download_weights.sh
```

`.env` is git-ignored. Never commit secrets.

## Running the experiment

1. **Setup** cells in `experiment.ipynb` — install deps, smoke test, helpers.
2. **Scenario A — Baseline NIM.** In a terminal:
   ```bash
   ./start_baseline.sh
   ```
   Run the Scenario A cells. Stop the server with `Ctrl+C`.
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
| `download_weights.sh` | Pull `nemotron-3-nano` weights from NGC (one-time) |
| `start_baseline.sh` | NVIDIA NIM container, tp=1 on GPU 0 |
| `start_disagg.sh` | NVIDIA Dynamo: prefill (GPU 0) + decode (GPU 1) + frontend on :8000 |
| `.env.example` | Template for `NGC_API_KEY`, `MODEL`, `NIM_CACHE_PATH` |
| `.env` | Your real values (git-ignored) |

## Stopping everything

```bash
docker rm -f nim-baseline dynamo-disagg 2>/dev/null
```

## References

- NVIDIA Dynamo: <https://github.com/ai-dynamo/dynamo>
- NVIDIA NIM for LLMs: <https://docs.nvidia.com/nim/large-language-models/latest/>
- NVIDIA NIXL: <https://github.com/ai-dynamo/nixl>
- Nemotron-3-Nano on NGC: <https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-nano>

## Security notes

- Never commit `.env`. Use `.env.example` as the template.
- If you previously had `HF_TOKEN` or any API key committed in this repo,
  **revoke and regenerate** it now.
