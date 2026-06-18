# AEON vLLM Ultimate — DGX Spark / Blackwell

[![docker](https://img.shields.io/badge/ghcr.io-aeon--7%2Faeon--vllm--ultimate-blue?logo=docker)](https://ghcr.io/aeon-7/aeon-vllm-ultimate)
[![vLLM](https://img.shields.io/badge/vLLM-0.23.0%2Bsm__121a.aeon-orange)](https://github.com/vllm-project/vllm)
[![sm_121a](https://img.shields.io/badge/sm__121a-DGX%20Spark-green)](https://www.nvidia.com/en-us/data-center/dgx-spark/)

**One container, the whole fleet.** A single image — `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` — serves every AEON model on **NVIDIA DGX Spark (GB10, sm_121a)** and other consumer-Blackwell GPUs (RTX 50 series): **Gemma-4-26B-A4B**, **Qwen3.6-27B**, and **Qwen3.6-35B-A3B** all run on the same build, with DFlash speculative decoding, NVFP4 weights, NVFP4/FP8 KV cache, and the OpenAI-compatible gateway intact.

Built on **vLLM v0.23.0 compiled from source for sm_121a**, merged with the AEON speculative-decoding stack: Triton software **NVFP4 KV cache** (PR #44389) + **DFlash SWA / high-concurrency / prefix-cache fixes** (PR #40898, #41703, #43982-port) + the **AEON DGX Spark runtime patches** + **TurboQuant** + **DFlash speculative decoding**.

> 🆕 **2026-06-18 — `:latest` is now the v0.23.0 sm_121a build (`:2026-06-18-v0.23.0-dflashfix`).** Rebuilt from source on vLLM v0.23.0 as a 3-way merge that preserves the AEON spec-decode tree, and adds the **DFlash high-concurrency fix** (port of upstream PR #43982): the drafter previously **crashed at ≥32 concurrent requests** under speculative decoding (padded-vs-unpadded KV block-table shape mismatch) and now scales cleanly to **c=64**. Carries the still-open PR #44389 (NVFP4-KV), #40898 (DFlash SWA), #41703 (prefix-cache corruption). See [What we fixed for the DGX Spark](#what-we-fixed-for-the-dgx-spark) and the [v0.23.0 fleet benchmarks](#v0230-fleet-benchmarks--one-image-three-models). Rollback tag: `:2026-06-11-pr41703`.

## What's inside

| Component | Version | Why |
|---|---|---|
| **vLLM** | 0.23.0 + sm_121a build, AEON spec-decode 3-way merge | Built from source for GB10; carries PR #44389 (Triton NVFP4 KV) + #40898 (DFlash SWA) + #41703 (prefix-cache corruption) + #43982-port (DFlash high-concurrency fix, new 2026-06-18) |
| **PyTorch** | 2.11.0+cu130 | CUDA 13.0 with sm_121a (DGX Spark / GB10) compute capability |
| **transformers** | 5.10.0.dev0 (HEAD) | Recognizes `gemma4_unified`, `qwen3_5`, all bleeding-edge model classes |
| **flashinfer** | 0.6.8.post1 | NVFP4 GEMM kernels, sliding-window attention, MLA, custom attention |
| **TurboQuant** | 0.2.0 (AEON-7 fork) | CUDA-graph-safe QJL — 4-bit KV compression on top of vLLM's native KV cache |
| **modelopt** | available via pip if needed | Quantization framework (not bundled — image stays small for serving) |

## v0.23.0 fleet benchmarks — one image, three models

The whole point of this container is that **a single build runs the entire AEON fleet** on a DGX Spark. The three charts below are the *same* `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (vLLM 0.23.0 + AEON sm_121a + DFlash) serving three very different architectures — a Gemma-4 MoE, a Qwen3.6 hybrid GDN+attention dense model, and a Qwen3.6 A3B MoE — each scaling cleanly from 1 to **64 concurrent requests** with no crash (the pre-fix image died at c≥32 under speculative decoding).

Numbers are measured on **DGX Spark GB10 (sm_121a)** with DFlash speculative decoding, NVFP4 weights, FP8 KV cache, prefix caching on, p50 across ≥ samples per point.

### Gemma-4-26B-A4B-it-Uncensored (NVFP4)

<p align="center"><img src="assets/perf/gemma26b_concurrency.svg" width="100%" alt="Gemma-4-26B-A4B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 1937 tok/s at c=64"></p>

Single-stream (c=1), by category, on `aeon-vllm-ultimate:latest`:

| Category | 🟢 Decode tok/s | TTFT p50 | TPOT p50 | Prefill (PP) | DFlash accept |
|---|---:|---:|---:|---:|---:|
| Coding | **155.8** | 83 ms | 6.4 ms | 601 tok/s | 58.9% |
| Math | **127.8** | 145 ms | 7.8 ms | 420 tok/s | 48.7% |
| Reasoning | **118.9** | 105 ms | 8.4 ms | 439 tok/s | 43.9% |
| Prose | **49.8** | 105 ms | 20.1 ms | 324 tok/s | 11.1% |
| Natural language | **67.3** | 97 ms | 14.9 ms | 393 tok/s | 20.0% |
| Extraction / JSON | **202.4** | 85 ms | 4.9 ms | 602 tok/s | 77.5% |

Long-context hold (DFlash acceptance does **not** collapse as histories grow): at **~16k tokens** (c=1) Coding draft acceptance is **58.7%** (128 tok/s decode); at **~33k tokens** it holds **46.7%** (93 tok/s decode). That long-context acceptance hold is the SWA-fix win (PR #40898) — earlier images collapsed past ~2k tokens.

### Qwen3.6-35B-A3B-heretic (NVFP4)

<p align="center"><img src="assets/perf/qwen35b_concurrency.svg" width="100%" alt="Qwen3.6-35B-A3B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 740 tok/s at c=64"></p>

Single-stream (c=1), by category:

| Category | 🟢 Decode tok/s | TTFT p50 | TPOT p50 | Prefill (PP) | DFlash accept |
|---|---:|---:|---:|---:|---:|
| Coding | **91.7** | 88 ms | 10.9 ms | 509 tok/s | 32.5% |
| Math | **123.6** | 113 ms | 8.1 ms | 494 tok/s | 47.7% |
| Reasoning | **120.6** | 120 ms | 8.3 ms | 359 tok/s | 46.3% |
| Prose | **75.2** | 137 ms | 13.3 ms | 234 tok/s | 23.7% |
| Natural language | **91.8** | 104 ms | 10.9 ms | 326 tok/s | 32.3% |
| Extraction / JSON | **79.8** | 103 ms | 12.5 ms | 468 tok/s | 28.1% |

Long-context hold: Coding draft acceptance is **40.8%** at **~16k** (90.8 tok/s decode) and **42.8%** at **~33k** (79.3 tok/s decode) — the A3B drafter holds acceptance flat across context.

### Qwen3.6-27B-AEON-Ultimate (NVFP4 MTP-XS body + DFlash n=12)

<p align="center"><img src="assets/perf/qwen27b_concurrency.svg" width="100%" alt="Qwen3.6-27B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 344 tok/s at c=64"></p>

Single-stream (c=1), by category:

| Category | 🟢 Decode tok/s | TTFT p50 | TPOT p50 | Prefill (PP) | DFlash accept |
|---|---:|---:|---:|---:|---:|
| Coding | **41.8** | 140 ms | 23.9 ms | 322 tok/s | 34.5% |
| Math | **47.3** | 244 ms | 21.1 ms | 229 tok/s | 41.7% |
| Reasoning | **56.1** | 234 ms | 17.8 ms | 183 tok/s | 50.0% |
| Prose | **34.1** | 146 ms | 29.4 ms | 220 tok/s | 27.3% |
| Natural language | **38.3** | 137 ms | 26.1 ms | 248 tok/s | 31.3% |
| Extraction / JSON | **44.2** | 246 ms | 22.6 ms | 195 tok/s | 37.2% |

vs a **stock vanilla `vllm/vllm-openai:nightly` baseline of ~10.5 tok/s** (no DFlash, no sm_121a optimizations) → roughly **4–5× faster single-stream decode**. Long-context hold: Coding draft acceptance is **49.5%** at **~16k** and **29.1%** at **~33k** — long histories stay drafted on the SWA-fixed drafter.

> **About the stock baseline:** the "stock / un-optimized" comparison figure is from **stock vanilla vLLM** (default settings, no DFlash, no DGX-Spark / sm_121a optimizations — `vllm/vllm-openai:nightly` eager). It is **provisional and will be refreshed** once a fresh fully-vanilla benchmark completes on the current version. The optimized figures above are measured on the new `aeon-vllm-ultimate:latest` (vLLM 0.23.0) build. There is no published vanilla baseline for the 35B-A3B yet (pending re-bench).

---

## What we fixed for the DGX Spark

All three models above run on one unified container — **`ghcr.io/aeon-7/aeon-vllm-ultimate:latest`** (= `:2026-06-18-v0.23.0-dflashfix`; rollback `:2026-06-11-pr41703`) — vLLM v0.23.0 built from source for GB10 / sm_121a and merged with the AEON speculative-decoding stack. This is the centerpiece of the build: a set of fixes that take the default "it runs, but it crashes under load and drafting collapses on long context" behavior and turn it into a stable, long-context, high-concurrency local-agent server.

| Fix | What it does | Why it matters on GB10 |
|---|---|---|
| **DFlash high-concurrency fix** *(new 2026-06-18)* | Slices the speculative drafter's KV block-table to the unpadded batch (`block_table[:num_reqs]`) | The drafter previously **crashed at ≥32 concurrent requests** (padded-vs-unpadded block-table shape mismatch in FlashAttention varlen — the engine died at c=64 with `block_table must have shape …`). Now scales cleanly to **c=64**. A port of upstream PR #43982, which fixed this for MTP but never for DFlash — present and unfixed even in the prior image. |
| **Triton NVFP4 KV cache** (PR #44389) | Software NVFP4 KV-cache path | The **only** 4-bit KV path on sm_121a (upstream's is hard-gated to B200) → **~3× KV capacity** / longer context per GB of unified memory. |
| **DFlash sliding-window attention** (PR #40898) | Runs the drafter's SWA layers as true sliding-window | **Long-context draft acceptance holds** as agent histories grow (e.g. Gemma-26B Coding ≈ 59% at ~16k, ≈ 47% at ~33k) instead of collapsing past ~2k tokens. |
| **Prefix-cache corruption immunity** (PR #41703) | Masks rejected/invalid context KV slots so they are never written | Without it, `--enable-prefix-caching` + DFlash silently decays draft acceptance to **0% over minutes-to-hours** of traffic (engine-global, ~6× slowdown that only a restart healed). With it, prefix caching is **safe again under sustained production load**. |
| **sm_121a-native build** | `TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0` | Compiles the **SM120-family CUTLASS NVFP4/FP8 kernels** GB10 actually dispatches to — true 4-bit tensor-core throughput, no dead B200-only kernels. |
| **sm_121a boot + CUDA-graph patches** | RTLD-lazy `_C_stable_libtorch` load; spec-decode CUDA-graph capture-size alignment | Boots past MXFP4 (SM100-only) symbols absent on GB10; prevents `cudaErrorIllegalAddress` on partial-acceptance decode steps under speculative decoding. |
| **Unified-memory tuning** | `--gpu-memory-utilization ≤0.70–0.88`, FULL CUDA graphs, async scheduling, z-lab DFlash drafters | GB10 shares one LPDDR5X pool across CPU + GPU; conservative KV headroom avoids page-thrash while keeping FULL-graph + speculative-decode throughput. |

### The result

- **Scales to 64 concurrent requests** with no crash — the same image, on all three fleet models (the prior image crashed at c≥32 under speculative decoding).
- **Native NVFP4 4-bit compute** on Blackwell tensor cores — the speed of 4-bit with near-16-bit accuracy.
- **Speculative decoding (DFlash)** holds high draft acceptance from short prompts through long (16k–32k) agent histories.
- Roughly **4–5× faster single-stream decode** vs a stock un-optimized vanilla vLLM baseline (Qwen3.6-27B: ~10.5 → ~38–56 tok/s by category; provisional pending a fresh vanilla re-bench).

---

## Why this container for Blackwell + DGX Spark users

### 🚀 NVFP4 KV cache — up to **3× KV capacity** (Triton software path)
PR #44389 (lesj0610/vllm) adds a Triton software path that packs the KV cache as **E2M1 FP4 + E4M3 block scales**. Enable per-serve via `--kv-cache-dtype nvfp4`. Independent of native FP4 conversion instructions — works on any sm_120 / sm_121 / sm_100 / sm_90 GPU.

When activated:
- **3× KV cache capacity** on Qwen3.6-27B and Qwen3.6-35B-A3B (per PR author benchmarks)
- MRCR quality comparable to `auto` KV baseline — closer than TurboQuant 4bit_nc

Not activated by default. Pass `--kv-cache-dtype nvfp4` to opt in.

### 🛠️ AEON DGX Spark patches (sm_121a runtime fixes)

The container ships with our 4 idempotent runtime patches that ensure correctness on GB10 hardware until upstream fixes land:

| Patch | What it fixes |
|---|---|
| **patch_cuda_optional_import** | Wraps `import vllm._C_stable_libtorch` in `RTLD_LAZY` so the SM100-only `mxfp4_experts_quant` and `silu_and_mul_mxfp4_experts_quant` symbols are tolerated as unresolved until first call (they never fire on sm_121a workloads) |
| **patch_kv_cache_utils** | Filters `None` block_size values out of `min()` calls in `v1/core/kv_cache_utils.py`, `v1/engine/core.py`, `v1/worker/gpu_model_runner.py` — fixes hybrid linear/full attention models (Qwen3.5/3.6, Nemotron-Omni) that crash with `TypeError: '<' not supported between NoneType and NoneType` |
| **patch_cudagraph_align** | Drops the `cudagraph_mode==FULL`-only gate on the spec-decode capture-size alignment filter in `config/compilation.py` so PIECEWISE mode also rounds capture sizes to multiples of `(1 + num_speculative_tokens)` — eliminates `cudaErrorIllegalAddress` mid-decode on partial-acceptance steps |

All patches are idempotent — they no-op when upstream merges the equivalent fix.

### 🩹 DFlash drafter correctness fixes (PR #40898 + #41703, merged ahead of upstream)

Both PRs are **open upstream but required** for correct DFlash operation (the z-lab drafter README pins them); the v0.23.0 `:latest` build carries them in-tree (3-way merged), alongside the new DFlash high-concurrency fix (PR #43982 port). They fix three real defects we root-caused in production on DGX Spark:

| Defect | Symptom | Fix |
|---|---|---|
| **Rejected-token context-KV writes** — the `copy_and_expand_dflash_inputs_kernel` stored slot mappings for *rejected* draft tokens, writing garbage K/V into the drafter's paged KV cache (incl. shared blocks). With `--enable-prefix-caching` the corruption was **persistent and self-accelerating** | Draft acceptance decays 34–56% → **0.0%** over minutes-to-hours of traffic (scales with volume); sticky engine-global; ~6× decode slowdown (144 → 24 tok/s) that only a restart healed | #41703 masks rejected/invalid context slots (`-1`) so they are never written |
| **Drafter sliding-window ignored** — SWA drafters (e.g. the Gemma-4-26B drafter: 4 of 5 layers SWA-2048) ran all layers as full attention | Long-context requests (>2048 tok history) got ~0% acceptance *per-request* even on a healthy server | #40898 adds DFlash SWA support (per-layer sliding-window wiring + causal SWA drafting metadata) |
| **Missing Gemma-4 adapter pieces** — no sqrt(hidden) embedding normalizer or final-logit softcapping in the draft path; `flash_attn` drafter rejected on multimodal Gemma targets | Depressed acceptance ceiling (MAL 4.4–6.6 vs z-lab's published 6.1–8.6); forced onto `flex_attention` | #41703 adds both + `use_mm_prefix=False`, enabling the upstream-tested **`flash_attn` drafter** on Gemma-4 |

⚠️ **Recipe change on this image: the DFlash drafter must use `"attention_backend": "flash_attn"`.** The old `flex_attention` workaround crashes at the first request (`key_cache.view(...)` on a non-contiguous tensor — the PR's KV-sharing machinery is only exercised with flash_attn upstream). With these fixes, `--enable-prefix-caching` is **safe again with DFlash** — soak-validated under production fleet traffic.

### 🧠 TurboQuant K8V4 — 4-bit KV cache compression
[0xSero/turboquant](https://github.com/0xSero/turboquant) with the AEON-7 fork applying our [`fix/cuda-graph-safe-qjl-powers`](https://github.com/AEON-7/turboquant/tree/fix/cuda-graph-safe-qjl-powers) patch — caches the `[1, 2, 4, 8, 16, 32, 64, 128]` constant per-device once at module load instead of re-allocating per call. **Without this fix, TurboQuant crashes at boot during CUDA graph capture**; the lazy workaround `--enforce-eager` costs ~30% throughput.

Enable per-serve via `--kv-cache-dtype tq_k8v4`.

### ⚡ DFlash speculative decoding (native via `--speculative-config`)
DFlash and EAGLE3 drafters are supported natively via vLLM's `--speculative-config` flag — no extra package needed since vLLM 0.21. Pair with our [aeon-7 DFlash drafters on HF](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored) for 1.5-2.5× throughput on the Qwen3.x family.

### 🔬 Native Blackwell SM 12.1 sm_121a compute
Built for `TORCH_CUDA_ARCH_LIST="12.1a"` — the sm_121a target for the GB10 in DGX Spark. Also runs on RTX 5090 / RTX 5080 / RTX PRO 6000 Blackwell (sm_120) thanks to the same family matcher in vLLM main.

## Quick start

The canonical target is the **AEON-7 Qwen3.6 family** — see [Validated models](#validated-models) below. Pick the variant that matches your hardware, then follow the matching recipe.

### Pull the image

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
# or pin the current build (vLLM 0.23.0 + DFlash high-concurrency fix)
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix
# previous build (pre-v0.23.0 / pre-concurrency-fix) kept for rollback
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-11-pr41703
```

### Recipe A — DGX Spark, DFlash drafter + FP8 KV (recommended for daily-driver)

This is the **measured-best config** for DGX Spark per the AEON-7 Qwen3.6 routing memo: **DFlash drafter beats MTP-method by +56 % median / +150 % peak** on Spark's unified-memory GB10.

> ⚠️ **DFlash + NVFP4 KV is not yet compatible on sm_121a (still true on the v0.23.0 build).** The DFlash drafter uses non-causal attention (parallel candidate generation), and none of the currently-built backends pair non-causal with NVFP4 KV on Spark:
> - `FLASH_ATTN` — doesn't support NVFP4 KV
> - `FLASHINFER` — supports NVFP4 KV but requires **SM100** (we're on SM121)
> - `TRITON_ATTN` — supports NVFP4 KV but is **causal-only**
>
> Use **`--kv-cache-dtype fp8_e4m3`** with DFlash. NVFP4 KV works cleanly with causal speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`) — see Recipe B.

#### Step 1 — download the base + DFlash drafter

```bash
# 1) Base — compressed-tensors NVFP4 + DFlash production variant (26 GB)
huggingface-cli download \
  AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4 \
  --local-dir /models/Qwen3.6-27B-AEON-NVFP4

# 2) DFlash drafter — z-lab's 5-layer Qwen3.6 drafter (3.3 GB)
huggingface-cli download \
  z-lab/Qwen3.6-27B-DFlash \
  --local-dir /models/Qwen3.6-27B-DFlash-drafter
```

> ⚠️ **Materialize the drafter dir.** If `huggingface-cli` stores symlinks into the HF cache blob dir, vLLM's bind-mounted container can't follow them. Either pass `--local-dir-use-symlinks=False` (newer hf_hub) or `cp -L $HF_CACHE/snapshots/<hash>/* /models/Qwen3.6-27B-DFlash-drafter/` so the files are real.

#### Step 2 — serve with DFlash + NVFP4 KV

```bash
docker run -d --name aeon-vllm \
    --gpus all --ipc=host --shm-size=16g \
    --net=host \
    -v /models/Qwen3.6-27B-AEON-NVFP4:/model:ro \
    -v /models/Qwen3.6-27B-DFlash-drafter:/drafter:ro \
    --entrypoint vllm \
    ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /model \
        --served-model-name aeon \
        --dtype auto \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8_e4m3 \
        --max-model-len 24576 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.78 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --mamba-block-size 256 \
        --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":4}' \
        --trust-remote-code
```

**Key flags**:
- `--quantization compressed-tensors` — the NVFP4 production model is in compressed-tensors format (`format: nvfp4-pack-quantized`), not modelopt. Use `--quantization modelopt` for the `*-MTP-XS` variants.
- `--kv-cache-dtype fp8_e4m3` — DFlash is non-causal and incompatible with NVFP4 KV on Spark today (see Recipe B for NVFP4 KV with MTP).
- `--speculative-config '{"method":"dflash",...}'` — `method: "dflash"` is the native vLLM speculator (not `"speculators"`).
- `--max-num-batched-tokens 8192` — must accommodate `num_speculative_tokens × max_num_seqs` plus headroom (vLLM warns if too low).
- `--mamba-block-size 256` — needed for Qwen3.6's hybrid GatedDeltaNet + attention stack.

> 💡 **Drafter materialization note.** vLLM bind-mounts the drafter dir but can't follow symlinks that point **outside** the mount (e.g. into the HF cache `blobs/` dir). After `huggingface-cli download`, either pass `--local-dir-use-symlinks=False` *or* `cp -L $HF_CACHE/snapshots/<hash>/* /models/Qwen3.6-27B-DFlash-drafter/` so the files are real, not symlinks. This pitfall cost us 4 startup failures.

### Recipe B — MTP self-speculation + NVFP4 KV (capacity-bound workloads)

For workloads where **KV capacity is the bottleneck** (long context, many concurrent streams), use the modelopt MTP-XS body with NVFP4 KV cache. This is the only Spark recipe that exercises PR #44389's ~3× KV capacity gain today.

```bash
docker run -d --name aeon-vllm \
    --gpus all --ipc=host --shm-size=16g --net=host \
    -v /models/Qwen3.6-27B-AEON-MTP-XS:/model:ro \
    --entrypoint vllm \
    ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /model \
        --served-model-name aeon \
        --quantization modelopt \
        --kv-cache-dtype nvfp4 \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --max-model-len 32768 --max-num-seqs 8 \
        --gpu-memory-utilization 0.78 \
        --enable-chunked-prefill --enable-prefix-caching --mamba-block-size 256 \
        --trust-remote-code
```

> ⚠️ **MTP throughput is lower than DFlash on Spark.** Measured 2026-04-28: DFlash beats MTP by **+56 % median / +150 % peak** with the same XS body. Use MTP only when you need NVFP4 KV's ~3× capacity (long contexts or higher batch sizes) **and** can accept the lower throughput. For pure throughput on Spark, use Recipe A. For dedicated-VRAM Blackwell (RTX PRO 6000, B100/B200), MTP is the right choice everywhere.

### Recipe C — TurboQuant K8V4 (4-bit KV, extreme capacity)

```bash
docker run -d --name aeon-vllm \
    --gpus all --ipc=host --shm-size=16g --net=host \
    -e VLLM_USE_TURBOQUANT=1 \
    -e TURBOQUANT_KV_BITS=4 \
    -v /models/Qwen3.6-27B-AEON-NVFP4:/model:ro \
    --entrypoint vllm \
    ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /model \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8 \
        --max-num-seqs 16 \
        ...
```

> ⚠️ **Cannot mix** TurboQuant K8V4 with `--kv-cache-dtype nvfp4`. Pick one. K8V4 wins on raw capacity (4-bit K + 4-bit V); NVFP4 KV wins on quality at ~3× capacity.

### Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aeon",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "temperature": 0.0
  }' | jq .choices[0].message.content
```

## Benchmarks

> For the current **v0.23.0 per-model decode tables and 1→64 concurrency charts**, see [v0.23.0 fleet benchmarks](#v0230-fleet-benchmarks--one-image-three-models) above. The benchmarks below are **prior-image validation gates and config-selection A/Bs** (`2026-06-11-pr41703` and `2026-06-04` era) — kept for the DFlash-correctness story and the KV/speculator config comparison, which still hold on the v0.23.0 build.

### Gemma-4-26B-A4B + DFlash — DFlash-correctness validation gates (image `2026-06-11-pr41703`)

[AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) + the z-lab `gemma-4-26B-A4B-it-DFlash` drafter, production profile (`--gpu-memory-utilization 0.68 --max-model-len 184320 --max-num-seqs 32 --max-num-batched-tokens 32768 --enable-prefix-caching`, body `triton_attn`, drafter **`flash_attn`**, `num_speculative_tokens 10`). Validation gates measured before/after the PR #40898+#41703 fixes:

| Gate | pre-fix image | `2026-06-11-pr41703` |
|---|---|---|
| Long-context (~9k sys prompt) draft acceptance | ~0–7% (SWA defect) | **43.3% / MAL 5.3** |
| Prefix-caching ON + fleet-burst + 10-min production soak | acceptance collapses to 0% in ~25 min (corruption) | **52.0% / MAL 6.20** — *improves* under load |
| Single-stream coding (c=1, greedy) | 144 tok/s fresh-boot best, decaying to ~24 | **149–150 tok/s, sustained** |
| Long-context throughput | ~46 tok/s (APC unusable) | **78 tok/s** (APC accelerates the cached prefix) |
| Live production probe (voice fleet, post-deploy) | — | **60% acceptance / MAL 7.0** |

Mean acceptance length now lands in z-lab's published 6.1–8.6 range. KV at this profile: 726k tokens / 3.94× concurrency at 180k ctx. Serve command:

```bash
docker run -d --name gemma26b --gpus all --ipc=host --net=host --shm-size=16g \
  -v /models/Gemma-4-26B-A4B-it-Uncensored-NVFP4:/model:ro \
  -v /models/gemma-4-26B-A4B-it-DFlash:/drafter:ro \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass -e TORCH_CUDA_ARCH_LIST=12.1a \
  --entrypoint bash ghcr.io/aeon-7/aeon-vllm-ultimate:latest -lc 'exec vllm serve /model \
    --quantization compressed-tensors --trust-remote-code \
    --attention-backend triton_attn \
    --max-model-len 184320 --max-num-seqs 32 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.68 --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/drafter\",\"num_speculative_tokens\":10,\"attention_backend\":\"flash_attn\"}"'
```

### Qwen3.6 benchmarks (image `2026-06-04` era)

Measured on **DGX Spark GB10 (sm_121a)** with `--max-num-seqs 8
--max-model-len 8192 --gpu-memory-utilization 0.78 --enable-chunked-prefill
--enable-prefix-caching --mamba-block-size 256
--quantization {compressed-tensors|modelopt}`.

### 🏆 Production-style: greedy + n_spec=15, by prompt category

The headline single-stream config — **MTP-XS body + DFlash drafter (n_spec=15) + BF16 KV + greedy sampling** — on 24 prompts (4 per category), `max_tokens=400`:

| Category | n | TTFT median | TPOT median | decode tok/s mean | decode tok/s median | peak |
|---|---:|---:|---:|---:|---:|---:|
| **math** | 4 | 243 ms | 22.3 ms | **44.6** | **44.9** | **45.7** ⚡ |
| code | 4 | 243 ms | 24.1 ms | 40.4 | 41.6 | 44.4 |
| reasoning | 4 | 195 ms | 28.4 ms | 35.9 | 35.2 | 40.1 |
| summary | 4 | 242 ms | 33.1 ms | 31.3 | 30.4 | 37.5 |
| dialogue | 4 | 243 ms | 33.4 ms | 30.0 | 30.1 | 36.6 |
| prose | 4 | 132 ms | 37.5 ms | 26.2 | 26.9 | 29.6 |
| **OVERALL** | **24** | **242 ms** | **29.3 ms** | **34.7** | **34.1** | **45.7** |

Concurrent ×4 streams (mixed categories):

| Round | Wall | Agg tok/s | TTFT mean |
|---|---:|---:|---:|
| 1 (cold) | 19.05 s | 71.5 | 1222 ms |
| **2 (steady)** | **17.57 s** | **84.4** | **276 ms** |

**Key findings**:
- **Math and code hit 41–46 tok/s** because token sequences are predictable — DFlash's n=15 acceptance window stays full.
- **Prose is slowest at ~26 tok/s** — high-entropy creative text means fewer drafter tokens accepted.
- **Per-category headline matches the v3 production card** (38.5 median / 71.3 peak, thinking-on) — math/code peak ~45 tok/s aligns with field reports.
- **n_spec=15 cuts KV concurrency in half** (146k tokens at 8k ctx, 17.9× max concurrent vs ~37× at n=4). Trade per-stream peak throughput for concurrency.

### Apples-to-apples 4-config comparison (sampled, n_spec=4 — same settings for all)

Same 8 generic prompts, `temperature=0.7`, `max_tokens=200`, `n_spec=4`. Use this when comparing **speculator method** or **KV dtype** at identical settings.

> **Provenance / status:** this 4-config A/B was measured on the **earlier `2026-06-04` era image** and is preserved here as the canonical *config-selection* baseline (speculator method × KV dtype, all else equal): the **FP8-E4M3 KV** config (**17.35 tok/s median single-stream**) vs the winning **DFlash + XS-body + BF16 KV** config (**24.27 tok/s**, +40%). These are *config-vs-config on the AEON container*, not a stock-vanilla comparison. The vanilla-vLLM baselines used in the fleet section above are **provisional and pending a fresh fully-vanilla re-benchmark** on the current v0.23.0 version.

### Single-stream (mean of 5 rounds)

| Config | Body | KV cache | TTFT mean | TTFT median | TPOT mean | tok/s mean | tok/s median |
|---|---|---|---:|---:|---:|---:|---:|
| MTP self-spec (n=1) | XS (modelopt, 21 GB) | NVFP4 (PR #44389) | 139 ms | 121 ms | 57.76 ms/tok | 17.26 | 16.64 |
| MTP self-spec (n=1) | XS (modelopt, 21 GB) | FP8-E4M3 | 182 ms | 214 ms | 57.05 ms/tok | 17.35 | 17.40 |
| DFlash drafter (n=4) | NVFP4 (compressed-tensors, 26 GB) | BF16 (auto) | 299 ms | 298 ms | 50.21 ms/tok | 19.44 | 20.10 |
| **🏆 DFlash drafter (n=4)** | **XS (modelopt, 21 GB)** | **BF16 (auto)** | **174 ms** | **131 ms** | **40.84 ms/tok** | **24.27** | **23.73** |

### Concurrent × 4 streams (mean of 12 streams over 3 rounds)

| Config | Body | KV cache | TTFT median (steady) | TPOT mean | per-stream tok/s | aggregate peak |
|---|---|---|---:|---:|---:|---:|
| MTP self-spec (n=1) | XS body | NVFP4 | 286 ms | 61.10 ms/tok | 15.71 | ~64 tok/s |
| MTP self-spec (n=1) | XS body | FP8-E4M3 | 239 ms | 60.17 ms/tok | 15.84 | ~66 tok/s |
| DFlash drafter (n=4) | NVFP4 body | BF16 (auto) | 328 ms | 55.39 ms/tok | 15.98 | ~68 tok/s |
| **🏆 DFlash drafter (n=4)** | **XS body** | **BF16 (auto)** | **476 ms¹ / 259 ms²** | **44.21 ms/tok** | **19.59** | **~87 tok/s** |

¹round 2 (warm)  ²round 3 (fully steady)

### Headlines

- **🏆 The winning config on Spark is the MTP-XS body + DFlash drafter (n=4) + BF16 KV.** Even though the body name says "MTP", it works great with an external DFlash drafter — and the **smaller body (21 GB vs 26 GB) leaves more compute and KV headroom**. Results: **+40% single-stream tok/s and +24% concurrent throughput vs the FP8-KV baseline.** Aggregate peak hits ~87 tok/s on 4 concurrent streams.
- DFlash on the NVFP4 (compressed-tensors) body is also a big win (+12% single, +0.9% concurrent) but the heavier 26 GB body loses ground to the same drafter on the lighter XS body.
- **MTP + NVFP4 KV** is the only path to PR #44389's ~3× KV capacity gain. Use when capacity (long context, more streams) outweighs the ~30-40% lower throughput vs DFlash. NVFP4 KV is within ±1% of FP8 on throughput at this prompt size; the real benefit is **~3× more KV blocks** at the same memory budget.
- **TPOT story is the cleanest signal.** DFlash + XS-body hits **40.8 ms/tok single-stream**, which is **28% faster than MTP** (57 ms) and **18% faster than DFlash on the heavier NVFP4 body** (50 ms). The drafter's n=4 acceptance and the smaller body's bandwidth advantage compound.
- **Round-1 concurrent TTFT (~1.5–4.6 s) is cold-cache + spec-decode warm-up.** Steady-state TTFT is rounds 2–3 (typically ~250–500 ms).

### KV cache capacity by body

| Body | GPU KV cache size at 8k ctx | Max concurrency |
|---|---:|---:|
| NVFP4 (compressed-tensors, 26 GB) + DFlash + BF16 KV | 264,922 tokens | 32.3× |
| XS (modelopt, 21 GB) + DFlash + BF16 KV | 300,966 tokens | 36.7× |

Raw JSON summaries: [`bench_mtp_fp8kv.json`](bench_mtp_fp8kv.json),
[`bench_mtp_nvfp4kv.json`](bench_mtp_nvfp4kv.json),
[`bench_dflash_bf16kv.json`](bench_dflash_bf16kv.json),
[`bench_xs_dflash_bf16kv.json`](bench_xs_dflash_bf16kv.json).
Methodology + plotting in [`bench_summary.md`](bench_summary.md).

## Validated models

This image is **purpose-built around the AEON-7 Qwen3.6 family** for DGX Spark. Other Blackwell-class models work but are not the canonical target.

| Model | Quant format | Spec method | Status | Notes |
|---|---|---|---|---|
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ **Canonical Spark recipe** — benchmarked in this card | Pair with [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) as the drafter |
| [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ **Fleet-benchmarked** in the v0.23.0 section above | Drafter **must** use `attention_backend: flash_attn` on this image; pair with z-lab `gemma-4-26B-A4B-it-DFlash` |
| [AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ **Fleet-benchmarked** in the v0.23.0 section above | A3B MoE; 8-layer all-full-attn drafter (no SWA/`--mamba-block-size` needed); optimal n≈10–11 |
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) | modelopt NVFP4 | qwen3_5_mtp (native) | ✅ End-to-end working + MTP benchmark below | Dedicated-VRAM Blackwell only; MTP underperforms DFlash on Spark |
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP) | modelopt NVFP4 (GDN preserved BF16) | qwen3_5_mtp | ✅ Same recipe as XS, regular footprint | RTX PRO 6000 / B100/B200 |
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | BF16 5-layer drafter (3.3 GB) | — | ✅ Pairs with `…-NVFP4` above | Drafter for DFlash recipe |
| [AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-NVFP4](https://huggingface.co/AEON-7/Step-3.7-Flash-AEON-Ultimate-Abliterated-NVFP4) | NVFP4 (modelopt) | — | 🟡 Expected to work | 198B MoE — not yet smoke-tested in this image |

## Known issues (upstream vLLM)

These are **upstream PR #44389** or core-vLLM bugs that we **didn't introduce** and can't fix without substantial patching. They're documented here so users don't think the container is broken:

### NVFP4 KV cache requires a causal attention backend on SM121

PR #44389 lights up `--kv-cache-dtype nvfp4` via the Triton software path, but the Triton backend is **causal-only**. The FlashInfer NVFP4 KV path requires SM100 — on SM121 it falls back to FP8.

Practical impact: NVFP4 KV pairs cleanly with **causal** speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`) but **not** with non-causal drafters like **DFlash**. If you pick `--kv-cache-dtype nvfp4` + `method:"dflash"`, vLLM raises:

```
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(...
  kv_cache_dtype=nvfp4, ..., use_non_causal=True). Reasons:
    FLASH_ATTN: [kv_cache_dtype not supported],
    FLASHINFER: [non-causal attention not supported, nvfp4 KV cache in FlashInfer requires SM100],
    TRITON_ATTN: [non-causal attention not supported],
    FLEX_ATTENTION: [kv_cache_dtype not supported],
    TURBOQUANT: [kv_cache_dtype not supported, non-causal attention not supported]
```

**Workaround for DFlash**: use **`--kv-cache-dtype auto`** (BF16). FP8 KV also fails for DFlash in this build because FLASHINFER and TRITON_ATTN both lost their non-causal kernel path in PR #44389's refactor:

```
ValueError: ... kv_cache_dtype=fp8_e4m3, ..., use_non_causal=True. Reasons:
  FLASH_ATTN: [kv_cache_dtype not supported]   (BF16 only)
  FLASHINFER:  [non-causal attention not supported]
  TRITON_ATTN: [non-causal attention not supported]
  FLEX_ATTENTION: [kv_cache_dtype not supported]
  TURBOQUANT: [kv_cache_dtype not supported, non-causal attention not supported]
```

This is a **regression vs the v3 production image** (vLLM 0.20.0), which had a FLASHINFER kernel variant that handled non-causal + FP8 KV. NVFP4 KV will land for DFlash once either (a) the Triton backend gains a non-causal kernel or (b) FLASHINFER's non-causal+FP8 path returns. For FP8-KV-with-DFlash today, fall back to `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3`.

**Workaround for NVFP4 KV**: use a **causal** speculator (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`) — see Recipe B. The Triton NVFP4-KV path supports those.


### Gemma-4-12B-AEON variants

| Variant | Issue |
|---|---|
| `Gemma-4-12B-AEON-Abliterated-K4-BF16` | vLLM's `TransformersMultiModalForCausalLM` fallback hits a shape mismatch on `Gemma4UnifiedForConditionalGeneration`. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x4096 and 8192x3840)` in a linear projection during graph capture. Suspect a multimodal-fused QKV layer not handled by the fallback path. |
| `Gemma-4-12B-AEON-Abliterated-K4-NVFP4-SVDQuant` | vLLM only knows `NVFP4 / NVFP4_FP8_MHA / W4A16_NVFP4 / MXFP8 / MIXED_PRECISION`. Our model's `quant_algo=NVFP4_SVD` (ModelOpt's newer SVD+low-rank variant) isn't yet recognized. Awaiting a deserializer PR in vLLM's `model_executor.layers.quantization.modelopt`. |

### Gemma-4-26B-A4B-NVFP4

vLLM creates the `embed_vision.embedding_projection` as a quantized `ReplicatedLinear`, but the checkpoint has only the unquantized `embed_vision.embedding_projection.weight` (because we excluded `embed_vision*` during quantization). Weight-loading mismatch. Likely an `exclude_modules` wildcard handling bug in PR #44389's refactor.

### For Gemma-4 production today

The **correctly-quantized** `Gemma-4-26B-A4B-it-Uncensored-NVFP4` body + z-lab `gemma-4-26B-A4B-it-DFlash` drafter is fleet-benchmarked on the current v0.23.0 `:latest` image (see the [fleet section](#v0230-fleet-benchmarks--one-image-three-models)). The variants in the table above (`Gemma-4-12B-AEON-*`, `Gemma-4-26B-A4B-NVFP4` with a quantized vision embedder) fail for **model-side** reasons that are independent of this container — they fail on any vLLM build.

## Build provenance

Current `:latest` (= `:2026-06-18-v0.23.0-dflashfix`) built 2026-06-18 on DGX Spark (GB10, 128 GB unified memory) — **vLLM v0.23.0 compiled from source for sm_121a** as a 3-way merge that preserves the AEON spec-decode tree (`TORCH_CUDA_ARCH_LIST=12.1a`, full CUDA compile). Carries the still-open upstream PRs #44389 (Triton NVFP4 KV), #40898 (DFlash SWA), #41703 (prefix-cache corruption immunity), plus the new in-tree DFlash high-concurrency fix (port of upstream PR #43982). Rollback tag: `:2026-06-11-pr41703` (vLLM 0.22.1 era). Earlier `:2026-06-04-pr44389` source pin was [`lesj0610/vllm@lesj/triton-nvfp4-kv-fork-20260602`](https://github.com/lesj0610/vllm/tree/lesj/triton-nvfp4-kv-fork-20260602) commit `e8c77b85`.

Dockerfile + patches + verify script live in this repo ([AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark)).

## License

vLLM is Apache-2.0. PyTorch BSD-3-Clause. TurboQuant Apache-2.0. AEON patches MIT.

This container is provided "AS IS" — see the legal section below.

---

## Arbitration Clause

**By accessing, downloading, using, running inference on, fine-tuning, merging, quantizing, distributing, integrating, or otherwise interacting with this container or its outputs, you acknowledge and agree to the following:**

1. **Sole Responsibility.** You, the user, are **solely and exclusively responsible** for (a) every prompt issued to any model served by this container, (b) every response produced, (c) every downstream action taken in reliance on those responses, and (d) any harm — direct, indirect, consequential, foreseeable, or otherwise — that results.

2. **No Warranty.** This container is provided strictly **"AS IS"**, without warranty of any kind, express or implied, including warranties of merchantability, fitness for a particular purpose, non-infringement, safety, alignment, factual accuracy, performance, or legal compliance in any jurisdiction.

3. **Legal Compliance.** You are responsible for ensuring your use complies with all applicable laws, regulations, terms of service, and organizational policies in every jurisdiction in which you operate.

4. **Operational Safety.** When serving uncensored or abliterated models with this container, you are expected to implement appropriate downstream safety layers: input validation, output filtering, content moderation, audit logging, rate limiting, access controls, and human-in-the-loop review for high-risk workflows.

5. **No Endorsement.** The authors, contributors, and publishers do not endorse, adopt, or take responsibility for any specific output produced by models served via this container.

6. **Arbitration.** Any dispute, claim, or controversy arising out of or relating to the use of this container shall be resolved through **binding individual arbitration** under the rules of a mutually agreed arbitration body (or, absent agreement, the American Arbitration Association's Consumer Arbitration Rules), waiving any right to a jury trial, class action, representative action, or consolidated proceeding.

7. **Indemnification.** You agree to indemnify, defend, and hold harmless the authors, contributors, and publishers from and against any claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from or related to your use of the container or your breach of this clause.

8. **Severability.** If any provision is held unenforceable in a given jurisdiction, the remaining provisions remain in full force.

9. **Acceptance.** Your use of this container constitutes your acceptance of this clause in full. If you do not accept, do not use the container.

---

## ☕ Support the work

If this container saves you days of vLLM compile-and-patch on Spark, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="left">
  <tr><td align="left">
    <strong>₿ Bitcoin (BTC)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="QR" width="200"/><br/>
    <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>Ξ Ethereum (ETH)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="QR" width="200"/><br/>
    <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>◎ Solana (SOL)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="QR" width="200"/><br/>
    <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>ⓜ Monero (XMR)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="QR" width="200"/><br/>
    <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
  </td></tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.
