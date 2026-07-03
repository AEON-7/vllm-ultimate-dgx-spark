# AGENTS.md — AEON vLLM Ultimate (DGX Spark / Blackwell)

Instructions for autonomous agents (Claude Code, OpenAI Codex, Cursor,
Aider, custom harnesses) on how to **pull, configure, serve, and
benchmark** this container correctly on a DGX Spark or other consumer
Blackwell host.

> **Note**: The entire AEON fleet — **Qwen3.6-27B**, **Qwen3.6-35B-A3B**, and
> **Gemma-4-26B-A4B** — is now unified onto this single image,
> `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (= `:2026-07-01-v0.24.0`;
> rollback `:2026-06-18-v0.23.0-dflashfix`), served with **DFlash
> `num_speculative_tokens: 12`**. The old lineage (`omni-q36`, `vllm-spark-*`,
> `aeon-gemma-4-26b-a4b-dflash`, `vllm-aeon-ultimate-*`, `vllm-dflash`) is
> consolidated into this image — historical only. There is no longer a separate
> per-repo image.

## What this container is

A from-source build of **vLLM v0.24.0** (compiled for sm_121a) that:

- Uses **PR #44389**'s Triton software NVFP4 KV cache (~3× capacity
  vs FP8 at the same memory budget — value when serving long context
  or many concurrent streams).
- Compiles **natively for SM 12.1a** (DGX Spark GB10 / consumer
  Blackwell sm_120 fallback).
- Bundles **TurboQuant K8V4** (AEON-7 CUDA-graph-safe QJL fork),
  **DFlash speculative drafting**, and **transformers HEAD** (needed
  for `gemma4_unified` and other 2026-Q2 architectures).
- Applies **two** idempotent AEON sm_121a runtime patches that no-op
  when upstream merges the equivalent fix (the old `kv_cache_utils`
  patch was **dropped in 0.23.0** — `block_size` is now an `int`
  upstream, so the `min()`-over-`None` reduction no longer applies):
  1. `cuda_optional_import` — wrap MXFP8/MXFP6 SM100 kernels in RTLD_LAZY
  2. `cudagraph_align` — PIECEWISE mode rounds spec-decode capture sizes
- Plus the **new in-tree DFlash high-concurrency fix** (port of upstream
  **PR #43982**): slices the drafter's KV block-table to the unpadded
  batch so DFlash no longer **crashes at ≥32 concurrent requests**
  (padded-vs-unpadded block-table shape mismatch) and now scales to **c=64**.
- Carries three still-open upstream PRs in-tree (3-way merged): **#44389**
  (Triton NVFP4 KV), **#40898** (DFlash sliding-window attention), **#41703**
  (Gemma-4 DFlash prefix-cache-safe).

## Hard requirements

- **Host kernel**: ≥ 5.15 with `nvidia.ko` 580+ (NV driver branch that ships sm_121a)
- **Docker** ≥ 24, with `nvidia-container-toolkit` configured
- **GPU**: NVIDIA GB10 (DGX Spark) or any Blackwell consumer (sm_120) — Hopper sm_90 is **not supported by this image**
- **Memory**: the headline NVFP4 KV cache path needs ~22 GB free for a 27B-class NVFP4 model; do not exceed `--gpu-memory-utilization 0.88` on Spark (unified memory thrashes above that — see [feedback_dgx_spark_gpu_mem_cap.md])

## Pull

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
# or pin the current build (vLLM 0.24.0 + AEON DFlash fixes)
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0
# previous build kept for rollback
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix
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
- `vllm: 0.23.0+sm121a.aeon`
- `torch: 2.11.0+cu130 13.0`
- `cuda available: True`
- `sm: (12, 1)` on GB10 or `(12, 0)` on consumer Blackwell
- `flashinfer: 0.6.12`

If `sm: (9, 0)` (Hopper) or `cuda available: False`, **stop** — this is the wrong image for this host.

## Standard serve recipe (Qwen3.6, NVFP4 body + DFlash drafter + FP8 KV)

This is the **canonical daily-driver recipe** — identical to the
[Quickstart in the README](README.md#quickstart-dgx-spark-copy-paste). The
drafter is a **separate ~3.3 GB BF16 checkpoint** trained on the matching base;
clone the body and the drafter fresh, then bind-mount both:

```bash
# 1) Pull the NVFP4 body (compressed-tensors, ~26 GB) — fresh clone
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP \
  /models/Qwen3.6-27B-AEON-MM-MTP
( cd /models/Qwen3.6-27B-AEON-MM-MTP && git lfs pull )

# 2) Pull the DFlash drafter (z-lab 5-layer, ~3.3 GB) — fresh clone
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/z-lab/Qwen3.6-27B-DFlash \
  /models/Qwen3.6-27B-DFlash-drafter
( cd /models/Qwen3.6-27B-DFlash-drafter && git lfs pull )

# 3) Serve — DFlash drafter + FP8 KV
docker run -d --name aeon-vllm \
    --restart unless-stopped \
    --gpus all --ipc=host --shm-size=16g \
    --net=host \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -v /models/Qwen3.6-27B-AEON-MM-MTP:/model:ro \
    -v /models/Qwen3.6-27B-DFlash-drafter:/drafter:ro \
    --entrypoint vllm \
    ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /model \
        --served-model-name aeon aeon-fast aeon-deep aeon-ultimate qwen36-ultimate aeon-ultimate-xs \
        --dtype auto \
        --quantization modelopt \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend TRITON_ATTN \
        --max-model-len 229376 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 32768 \
        --gpu-memory-utilization 0.60 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --generation-config vllm \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --mm-encoder-tp-mode data \
        --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":12,"attention_backend":"TRITON_ATTN"}' \
        --trust-remote-code
```

**Notes**:
- `--quantization modelopt` for `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP`. Use `--quantization compressed-tensors` only for the older `*-NVFP4` production body (`format: nvfp4-pack-quantized`).
- `--kv-cache-dtype fp8_e4m3` — DFlash is **non-causal** and has no NVFP4/FP8-vs-non-causal KV kernel partner that *also* supports NVFP4 on sm_121a; FP8 KV is the working DFlash pairing on this build. NVFP4 KV (`--kv-cache-dtype nvfp4`, PR #44389) pairs only with **causal** speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`) — see the MTP variant.
- `--speculative-config '{"method":"dflash",...}'` — `method: "dflash"` is the native vLLM speculator (not `"speculators"`).
- `--attention-backend TRITON_ATTN` and `"attention_backend":"TRITON_ATTN"` inside the DFlash JSON are both required for this Qwen3.6 DFlash path. vLLM does not inherit target attention-backend settings into speculative drafters.
- `--max-model-len 229376` gives one near-full-context session while retaining KV headroom for output and smaller concurrent agents; `--max-num-seqs 16` and `--max-num-batched-tokens 32768` keep agent/gateway bursts usable.
- Leave `--mamba-block-size` unset. vLLM now derives the correct cache geometry for Qwen3.6's hybrid GatedDeltaNet + attention stack.
- `--gpu-memory-utilization 0.60` — sidecar-safe default when Qwen3-ASR and Qwen3-TTS share the Spark. **Never exceed 0.88 on Spark.** GB10's unified LPDDR5X pool is shared CPU+GPU, so anything above ~0.88 page-thrashes.
- If `git clone` leaves LFS pointer files, re-run `git lfs pull` in the model dir. If you instead use `huggingface-cli download` and it stores symlinks into the HF cache `blobs/` dir, vLLM's bind-mount can't follow them — pass `--local-dir-use-symlinks=False` or `cp -L $HF_CACHE/snapshots/<hash>/* /models/Qwen3.6-27B-DFlash-drafter/` so the files are real.

> ⚠️ **`method: "dflash"`** is the correct value (not `"speculators"`). On the
> v0.24.0 Qwen3.6 path, set `"attention_backend":"TRITON_ATTN"` inside the
> speculative config because vLLM does not inherit the target backend. For
> **Gemma-4** targets, follow the Gemma recipe's drafter backend notes. Use
> **`--kv-cache-dtype fp8_e4m3`** — the non-causal DFlash drafter cannot pair
> with NVFP4 KV on sm_121a today.
>
> **Why `num_speculative_tokens: 12` and why this image matters for long
> context**: the z-lab Qwen3.6-27B DFlash drafter is a sliding-window model —
> 4 of its 5 layers use sliding-window attention (window 2048). vLLM PR #40898
> (in `aeon-vllm-ultimate:latest`) runs those layers as proper SWA; earlier
> images ran them as full attention, so drafting collapsed once context grew
> past ~2048 tokens. PR #41703 additionally makes `--enable-prefix-caching`
> corruption-immune with DFlash. The new PR #43982 port stops the drafter
> crashing at ≥32 concurrent requests (scales to c=64). Net: long-context
> drafting holds up and high concurrency is stable; short-context (<2048, one
> window) is unchanged. n=12 won the n=8–15 sweep (statistically tied
> short-context, best long-context acceptance) and is the production default.

## Variant: MTP self-speculation + NVFP4 KV (capacity-bound workloads)

For workloads where **KV capacity is the bottleneck** (long context, many
concurrent streams) on dedicated-VRAM Blackwell, use the modelopt MTP-XS body
with NVFP4 KV cache — the only path that exercises PR #44389's ~3× KV gain:

```bash
docker run -d --name aeon-vllm \
  --gpus all --ipc=host --shm-size=16g --net=host \
  -v /models/Qwen3.6-27B-AEON-MTP-XS:/model:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
    --served-model-name aeon \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype nvfp4 \
    --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.60 \
    --enable-chunked-prefill --enable-prefix-caching \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
    --trust-remote-code
```

> ⚠️ **MTP underperforms DFlash on Spark** (Qwen3.6-27B: DFlash +56% median /
> +150% peak). Use MTP only when you need NVFP4 KV's ~3× capacity and can
> accept lower throughput. On **dedicated-VRAM Blackwell** (RTX PRO 6000,
> B100/B200) MTP is the right choice everywhere; on **Spark** prefer the DFlash
> standard recipe above. `--kv-cache-dtype nvfp4` pairs only with causal
> speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`).

## Variant: TurboQuant K8V4 4-bit KV (extreme memory budget)

```bash
--kv-cache-dtype fp8                # NVFP4 KV is incompatible with K8V4
ENV VLLM_USE_TURBOQUANT=1            # turn on K8V4
ENV TURBOQUANT_KV_BITS=4             # 4-bit K + 4-bit V
```

Pair with `--gpu-memory-utilization 0.60` when ASR/TTS sidecars share the Spark, or raise cautiously only when the LLM is the dominant GPU workload. See [feedback_turboquant_cuda_graph_fix.md] for why the AEON-7 fork is required.

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
  --label "DFlash+FP8-KV" \
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
| DFlash drafter crashes / `block_table must have shape …` at ≥32 concurrent requests | Pre-v0.23.0 image (padded-vs-unpadded KV block-table) | Upgrade to `:latest` (= `:2026-06-18-v0.23.0-dflashfix`) — carries the PR #43982 port; scales to c=64 |
| `RuntimeError: mat1 and mat2 shapes` on Gemma-4-12B | **Model-side** multimodal-fused QKV not handled by the Transformers fallback — fails on any vLLM, not container-specific | No container fix; use the correctly-quantized `Gemma-4-26B-A4B-it-Uncensored-NVFP4` for production |
| `quantization 'NVFP4_SVD' not recognized` | vLLM modelopt deserializer doesn't yet know ModelOpt's SVD+low-rank algo (model-side) | Re-quantize with a supported algo, or load via `modelopt+transformers` directly |
| `embed_vision.embedding_projection.weight` missing | **Badly-quantized variant only** (vision embedder was quantized) — model-side, fails on any vLLM | Use the correctly-quantized `Gemma-4-26B-A4B-it-Uncensored-NVFP4` (vision embedder excluded as BF16) |
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
- Source: **vLLM v0.23.0 compiled from source for sm_121a** (`TORCH_CUDA_ARCH_LIST=12.1a`) as a 3-way merge that preserves the AEON spec-decode tree; carries open upstream PRs #44389 (Triton NVFP4 KV), #40898 (DFlash SWA), #41703 (Gemma-4 DFlash prefix-cache-safe) plus the in-tree DFlash high-concurrency fix (port of PR #43982). The earlier `:2026-06-04-pr44389` build pinned [`lesj0610/vllm@lesj/triton-nvfp4-kv-fork-20260602`](https://github.com/lesj0610/vllm/tree/lesj/triton-nvfp4-kv-fork-20260602) commit `e8c77b85` (historical).
- Patches + Dockerfile: [`AEON-7/vllm-ultimate-dgx-spark`](https://github.com/AEON-7/vllm-ultimate-dgx-spark).

## Support the work

Tips welcomed via the addresses in the README. No obligation; useful releases keep coming either way.
