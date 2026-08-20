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

A from-source build of **vLLM v0.27.1** (compiled for sm_121a) that:

- Ships **DSpark speculative decoding with quantized Markov heads** (v0.27.1
  headline #50424, + top-k Markov projection #49969) — MRv2-only; see the
  DSpark recipe below.
- **No longer bakes `VLLM_USE_V2_MODEL_RUNNER=0`.** At 0.27.1 the config
  auto-routes: DSpark and mixed sliding/full DFlash drafters (all z-lab fleet
  drafters are 4-sliding+1-full) force Model-Runner-V2, where upstream
  #47914/#48113 run drafter SWA natively. Remove the env pin from old launch
  scripts. Re-pin `=0` per-service ONLY if you use `thinking_token_budget`
  (V2 silently ignores it).
- torch 2.13.0+cu130 / Triton 3.7.1 / FlashInfer 0.6.16.post3 (exact
  python+cubin+jit-cache trio) / NCCL 2.30.7 / transformers 5.14.1 /
  torchcodec 0.16.0 (video decode is back).

Carried forward from earlier builds (see SOURCE.md for the full ledger):

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
- `vllm: 0.27.1+aeon.sm121a.dspark`
- `torch: 2.13.0+cu130 13.0`
- `cuda available: True`
- `sm: (12, 1)` on GB10 or `(12, 0)` on consumer Blackwell
- `flashinfer: 0.6.16.post3`

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

## Variant: DSpark Markov-head speculation (NEW in v0.27.1)

DSpark is a semi-autoregressive **block** drafter: a DFlash-style parallel
block draft plus a low-rank **Markov head** that biases each in-block token on
the previously sampled one. v0.27.1 loads **quantized** Markov heads (#50424),
so it pairs with the NVFP4 fleet bodies. DSpark runs **only on
Model-Runner-V2** — do NOT set `VLLM_USE_V2_MODEL_RUNNER=0` with it (v0.27.1
raises rather than silently degrading).

Community drafters for the fleet (no Gemma-4-26B drafter exists yet — that
model stays on DFlash until one is trained):

- **Qwen3.6-27B** → `satgeze/Qwen3.6-27B-DSpark` (block 15; also consider
  `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft`, fine-tuned from the z-lab DFlash
  drafter — untested)
- **Qwen3.6-35B-A3B** → `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark`
  (speculators format, block 8)

```bash
docker run -d --name aeon-vllm-dspark \
  --gpus all --ipc=host --shm-size=16g --net=host \
  -v /models/Qwen3.6-27B-AEON-MM-MTP:/model:ro \
  -v /models/Qwen3.6-27B-DSpark-satgeze:/dspark:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
    --served-model-name aeon \
    --dtype auto --quantization modelopt \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend TRITON_ATTN \
    --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.60 \
    --enable-chunked-prefill \
    --mamba-cache-mode align \
    --speculative-config '{"method":"dspark","model":"/dspark","num_speculative_tokens":15}' \
    --trust-remote-code
```

**DSpark rules:**

- `num_speculative_tokens` must be **>= the drafter's `dspark_block_size`**
  (15 for satgeze, 8 for RedHatAI). Smaller values feed the Markov-head
  machinery an unsupported layout and produce **garbled output**, not merely
  lower acceptance.
- On the hybrid-GDN Qwen3.6-27B keep **`--mamba-cache-mode align`** whenever
  prefix caching is enabled (upstream crash #52317; proper fix lands post-0.27.1).
- NVFP4 **KV** (`--kv-cache-dtype nvfp4`) + DSpark is not upstream-validated —
  use FP8 KV with DSpark.
- Benchmark against the DFlash recipe at production concurrency before
  flipping a service; DFlash remains the validated default.

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
## Variant: dual-Spark TP=2 over RoCE — with cross-node CUDA graphs

Two DGX Sparks joined by a direct 200 GbE ConnectX cable serve one model split
across both GPUs: **double the unified memory** (56.4 GiB pooled KV measured at
`--max-model-len 65536`, vs ~35 GiB on one box) at **single-stream throughput
parity**. TP=2 on Spark is a *capacity* play — the interconnect is ~9-10 GB/s
host-staged (GB10 has no GPUDirect), so it buys headroom, not latency.

> **Cross-node CUDA graphs work on this image.** Upstream
> [#46253](https://github.com/vllm-project/vllm/issues/46253) reports multi-node
> GB10 clusters must run `--enforce-eager`. This image carries the
> [#48053](https://github.com/vllm-project/vllm/pull/48053) `thread_local`
> capture-error-mode fix extended to **all five** `torch.cuda.graph` sites, and
> with NCCL 2.30.7 + fusion off + custom-all-reduce off, capture **and** replay
> are stable across nodes. Validated: 35 piecewise + 16 full + 15 DSpark graphs
> captured, a 64-request / 19.5k-token soak, and a 6-cycle 8-way concurrent
> burn-in with **zero** NCCL errors. Worth **+26% single-stream** (33.7 vs 26.7
> tok/s) and **-23% TPOT** versus eager. If you hit instability on different
> hardware, add `--enforce-eager` to both nodes to fall back.

### 1) Fabric (once per boot — the addresses do not survive a reboot)

```bash
# node 0
docker run --rm --net=host --cap-add NET_ADMIN busybox \
  ip addr add 10.10.10.1/30 dev enp1s0f0np0
# node 1
docker run --rm --net=host --cap-add NET_ADMIN busybox \
  ip addr add 10.10.10.2/30 dev enp1s0f0np0
```

Confirm the RoCE device is live and find your GID index (**3** = the IPv4-mapped
RoCE v2 entry, which is what the `10.10.10.x` addresses use):

```bash
cat /sys/class/infiniband/rocep1s0f0/ports/1/state
cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3
```

Substitute your own device/interface names — `ls /sys/class/infiniband/` and
`ls /sys/class/infiniband/<dev>/device/net` map RDMA devices to netdevs.

### 2) Node 0 — the API server

```bash
docker run -d --name tp2-node0 --gpus all --ipc=host --shm-size=16g --net=host \
  -e VLLM_HOST_IP=10.10.10.1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA==rocep1s0f0:1 -e NCCL_IB_GID_INDEX=3 \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /models/YOUR-NVFP4-BODY:/model:ro \
  -v /models/YOUR-DSPARK-DRAFTER:/dspark:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest serve /model \
    --served-model-name aeon --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
    --master-addr 10.10.10.1 --master-port 29501 \
    --quantization compressed-tensors --kv-cache-dtype fp8_e4m3 \
    --attention-backend TRITON_ATTN \
    --max-model-len 65536 --max-num-seqs 16 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.60 \
    --disable-custom-all-reduce \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --mamba-cache-mode align \
    --speculative-config '{"method":"dspark","model":"/dspark","num_speculative_tokens":7}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --trust-remote-code
```

### 3) Node 1 — headless worker (start ~10 s later)

Identical **engine** flags with `--headless`, `--node-rank 1`, and
`VLLM_HOST_IP=10.10.10.2`. Omit the frontend flags (`--served-model-name`,
`--host/--port`, and the parsers) — those are API-server-only.

### 4) Confirm RDMA is actually in use

```bash
docker logs tp2-node0 2>&1 | grep -E "NET/IB|Using network"
```

You want `NCCL INFO NET/IB : Using [0]rocep1s0f0:1/RoCE`. If it says
`NET/Socket`, NCCL fell back to TCP — recheck `NCCL_IB_HCA`, the GID index, and
that `/dev/infiniband` is passed into **both** containers.

**Notes**

- `--disable-custom-all-reduce` is required: vLLM's custom all-reduce is a
  single-node/NVLink path and is part of what destabilizes cross-node capture.
- `--quantization` must match the checkpoint: `compressed-tensors` for
  LLM-Compressor builds, `modelopt` for ModelOpt (`*-MO`) builds. The wrong
  value fails at boot.
- Both nodes need the model **and** the drafter on local disk at their own
  paths; the mount points inside the containers must match.
- Multi-node uses the `mp` backend natively — no Ray required.


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
