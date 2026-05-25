# TODO — make the disagg comparison actually meaningful

## Status (2026-05-25)

- ✅ Baseline (NIM, `nemotron-3-nano`, GPU 0) — works, real answers.
- ✅ Disagg infra (Dynamo frontend + NATS + etcd + 2 vLLM workers + NIXL) —
  stands up, requests reach decode, HTTP 200.
- ❌ Disagg **outputs** are pure `<unk>` tokens with the current model. NIXL
  ships the attention KV cache but not the Mamba SSM hidden state, so the
  decode worker runs from a zeroed SSM. See README "Findings".

## Required to unblock the comparison

### 1. Swap the model to a pure-attention one
Candidates (small enough for 2× A100 80GB, well-supported by vLLM + NixlConnector):
- `meta-llama/Llama-3.2-3B-Instruct`  ← easiest, smallest, recommended
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

NGC does **not** host these — pulling from Hugging Face is required, which
breaks the "no HF token" property the README currently brags about.

### 2. Replace the baseline
NIM only serves the Nemotron family in our current setup. To keep both
scenarios on the same model, switch the baseline to **plain vLLM in a
container** instead of NIM:

```bash
docker run --rm --gpus '"device=0"' --network=host \
  -v $HOME/.cache/hf:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.2-3B-Instruct --port 8000
```

### 3. Update the disagg script
- `MODEL=meta-llama/Llama-3.2-3B-Instruct`
- mount `$HOME/.cache/hf` instead of NIM cache, point `--model` at it
- everything else (NATS, etcd, ports 5600/5601, `--no-disable-hybrid-kv-cache-manager`)
  can stay — though the hybrid flag becomes a no-op on pure-attention models

### 4. Update the notebook
- `MODEL = "meta-llama/Llama-3.2-3B-Instruct"`
- the smoke test should once again return real text on *both* servers
- re-run `baseline_sweep`, `disagg_sweep`, the compare cell

### 5. Add the missing `.env` keys
- `HF_TOKEN` (for HF model pulls)
- `MODEL` default updated

## Stretch goals

- Try **mooncake** connector (`MooncakeConnector`) instead of NIXL — has
  better hybrid-model support claims.
- Try **TP=2** baseline vs disagg 1+1 — the fairer "same hardware" comparison.
- Capture per-request prefill/decode worker logs to confirm decode never
  pauses for prefill (the central claim).
- Add an "isolate decode tail" plot: ITL distribution overlay rather than just
  p50/p95/p99 means.

## Cleanup

- Delete the local NIM `tmp/` periodically (it's root-owned, leaks across
  runs): `sudo rm -rf $NIM_CACHE_PATH/tmp`.
- Stop containers between runs: `docker rm -f nim-baseline dynamo-disagg`.
