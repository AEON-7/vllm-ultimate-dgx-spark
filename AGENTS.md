# AGENTS.md — AEON vLLM Ultimate (DGX Spark / Blackwell)

Instructions for autonomous agents (Claude Code, OpenAI Codex, Cursor,
Aider, custom harnesses) on how to **pull, configure, serve, and
benchmark** this container correctly on a DGX Spark or other consumer
Blackwell host.

## What this container is

A patched build of `vllm==0.22.1` that:

- Uses **PR #44389**'s Triton software NVFP4 KV cache (~3× capacity
  vs FP8 at the same memory budget — value when serving long context
  or many concurrent streams).
- Compiles **natively for SM 12.1a** (DGX Spark GB10 / consumer
  Blackwell sm_120 fallback).
- Bundles **TurboQuant K8V4** (AEON-7 CUDA-graph-safe QJL fork),
  **DFlash speculative drafting**, and **transformers HEAD** (needed
  for `gemma4_unified` and other 2026-Q2 architectures).
- Applies three idempotent AEON sm_121a runtime patches that no-op
  when upstream merges the equivalent fix:
  1. `cuda_optional_import` — wrap MXFP8/MXFP6 SM100 kernels in RTLD_LAZY
  2. `kv_cache_utils` — strip `None` from hybrid linear+attention `min()` reductions
  3. `cudagraph_align` — PIECEWISE mode rounds spec-decode capture sizes

## Hard requirements

- **Host kernel**: ≥ 5.15 with `nvidia.ko` 580+ (NV driver branch that ships sm_121a)
- **Docker** ≥ 24, with `nvidia-container-toolkit` configured
- **GPU**: NVIDIA GB10 (DGX Spark) or any Blackwell consumer (sm_120) — Hopper sm_90 is **not supported by this image**
- **Memory**: the headline NVFP4 KV cache path needs ~22 GB free for a 27B-class NVFP4 model; do not exceed `--gpu-memory-utilization 0.88` on Spark (unified memory thrashes above that — see [feedback_dgx_spark_gpu_mem_cap.md])

## Pull

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
# or pin to a tagged build
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:v0.22.1-pr44389-spark
```

## Verify the image is healthy before serving

```bash
docker run --rm --gpus all ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  -c "python3 -c '
import vllm, torch, flashinfer
print(\"vllm:\", vllm.__version__)
print(\"torch:\", torch.__version__, torch.version.cuda)
print(\"cuda available:\", torch.cuda.is_available())
print(\"sm:\", torch.cuda.get_device_capability())
print(\"flashinfer:\", flashinfer.__version__)
'"
```

**Expected output**:
- `vllm: 0.22.1+pr44389.aeon`
- `torch: 2.11.0+cu130 13.0`
- `cuda available: True`
- `sm: (12, 1)` on GB10 or `(12, 0)` on consumer Blackwell
- `flashinfer: 0.6.8.post1`

If `sm: (9, 0)` (Hopper) or `cuda available: False`, **stop** — this is the wrong image for this host.

## Standard serve recipe (Qwen3.6, NVFP4 + MTP + NVFP4 KV)

```bash
docker run -d \
  --name aeon-vllm \
  --gpus all \
  --ipc=host --shm-size=16g \
  --net=host \
  -v /path/to/model:/model:ro \
  --entrypoint vllm \
  ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
    --served-model-name aeon \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype nvfp4 \
    --max-model-len 24576 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.78 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --mamba-block-size 256 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --trust-remote-code
```

**Notes**:
- `--kv-cache-dtype nvfp4` activates the PR #44389 path; pair only with **causal** speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`). With DFlash, switch to `--kv-cache-dtype auto` (BF16).
- `--quantization compressed-tensors` for `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4` (the DFlash-paired body, `format: nvfp4-pack-quantized`); `--quantization modelopt` for the `*-MTP-XS` variants.
- `--speculative-config` enables MTP (Qwen3.5/3.6) or DFlash drafter (Qwen3.5/3.6 paired with a separate drafter checkpoint — see below).
- `--mamba-block-size 256` is needed for any hybrid mamba+attention stack (Qwen3.6 included).

## Variant: DFlash drafter (Qwen3.5 + Qwen3.6)

The drafter is a **separate 3.3 GB BF16 checkpoint** trained on the matching base. Pull it once and bind-mount alongside the main model:

```bash
# 1. Pull drafter
huggingface-cli download z-lab/Qwen3.6-27B-DFlash --local-dir /models/Qwen3.6-27B-DFlash-drafter

# 2. CRITICAL — materialize the dir if HF stores symlinks into the cache
# vLLM bind-mounts can't follow symlinks that point outside the mount.
HF_CACHE=~/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash
SNAP=$(ls -d $HF_CACHE/snapshots/*/ | head -1)
cp -L $SNAP/* /models/Qwen3.6-27B-DFlash-drafter/

# 3. Serve with DFlash
docker run -d --gpus all --ipc=host --shm-size=16g --net=host \
  -v /models/Qwen3.6-27B-AEON-NVFP4:/model:ro \
  -v /models/Qwen3.6-27B-DFlash-drafter:/drafter:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
    --served-model-name aeon \
    --dtype auto \
    --quantization compressed-tensors \
    --kv-cache-dtype auto \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":4}' \
    --max-model-len 24576 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.78 \
    --enable-chunked-prefill --enable-prefix-caching --mamba-block-size 256 \
    --trust-remote-code
```

> ⚠️ **`method: "dflash"`** is the correct value (not `"speculators"`). And **`--kv-cache-dtype auto`** is required — in vLLM 0.22.1 the non-causal attention backends lost their FP8/NVFP4 KV kernels, so DFlash falls back to FLASH_ATTN + BF16 KV. To get FP8 KV with DFlash today, use the v3 production image (`ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3`).

## Variant: TurboQuant K8V4 4-bit KV (extreme memory budget)

```bash
--kv-cache-dtype fp8                # NVFP4 KV is incompatible with K8V4
ENV VLLM_USE_TURBOQUANT=1            # turn on K8V4
ENV TURBOQUANT_KV_BITS=4             # 4-bit K + 4-bit V
```

Pair with `--gpu-memory-utilization 0.78` and `--max-num-seqs 12+`. See [feedback_turboquant_cuda_graph_fix.md] for why the AEON-7 fork is required.

## Health probes

```bash
# Liveness — server up
curl -fsSL http://localhost:8000/health && echo OK

# Readiness — model loaded
curl -fsSL http://localhost:8000/v1/models | python3 -m json.tool

# Functional — one-shot completion
curl -fsSL http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"aeon","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## Benchmark recipe

```bash
# Replicate the published numbers (single-stream + concurrent x4, 200-token outputs)
pip install --quiet aiohttp
curl -sLO https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/bench_vllm.py
python3 bench_vllm.py \
  --base http://localhost:8000 \
  --model aeon \
  --label "MTP+NVFP4-KV" \
  --single-rounds 5 \
  --concurrent-streams 4 \
  --concurrent-rounds 3 \
  --max-tokens 200
```

Compare your numbers to the table in the README under "Benchmarks". A ±10% deviation is within run-to-run noise.

## Common failure modes

| Symptom | Diagnosis | Fix |
|---|---|---|
| `cuda_optional_import` warns about `_C_stable_libtorch` | Expected on sm_121 — MXFP8 SM100-only kernels are lazy-loaded | Ignore, model loads normally |
| `RuntimeError: mat1 and mat2 shapes` on Gemma-4-12B | Upstream PR #44389 multimodal fallback bug | Use `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest` instead (vLLM 0.20.1) |
| `quantization 'NVFP4_SVD' not recognized` | vLLM modelopt deserializer not yet updated | Load with `modelopt+transformers` directly, or use older AEON-7 image |
| `embed_vision.embedding_projection.weight` missing | PR #44389 `exclude_modules` wildcard bug | Same as above — use older image for multimodal Gemma-4 |
| `gpu-memory-utilization > 0.88` thrashes | DGX Spark unified memory limit | Cap at 0.88, drop `--max-model-len`, or enable TurboQuant K8V4 |
| `torch.compile takes 30-60s on first request` | Expected on first cold launch | Subsequent restarts are cached; ignore |

## Restart + recovery on Spark

If you're on a Spark service stack matching `~/svc_watchdog.sh`:

```bash
# Reset failing container without manual intervention
docker update --restart unless-stopped aeon-vllm
docker start aeon-vllm

# Watchdog (every 5 min): real TTS-gen test, restart+warm after 2 fails
# See reference_spark_service_stack.md
```

## What this container does NOT do

- Does **not** include the Qwen3-ASR or Qwen3-TTS sidecars — see `qwen3-asr` and `qwen3-tts` images separately.
- Does **not** include a model — bring your own (suggested: Qwen3.6 NVFP4 from `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4`).
- Does **not** ship `humming` (NVIDIA-internal) — a stub is bundled so vLLM's eager imports succeed; actual humming usage will raise.
- Does **not** support Hopper sm_90 — wrong arch.

## License + provenance

- vLLM Apache-2.0, PyTorch BSD-3-Clause, TurboQuant Apache-2.0, AEON patches MIT.
- Source: [`lesj0610/vllm@lesj/triton-nvfp4-kv-fork-20260602`](https://github.com/lesj0610/vllm/tree/lesj/triton-nvfp4-kv-fork-20260602) commit `e8c77b85`.
- Patches + Dockerfile: [`AEON-7/vllm-ultimate-dgx-spark`](https://github.com/AEON-7/vllm-ultimate-dgx-spark).

## Support the work

Tips welcomed via the addresses in the README. No obligation; useful releases keep coming either way.
