# AEON vLLM Ultimate — DGX Spark / Blackwell

[![docker](https://img.shields.io/badge/ghcr.io-aeon--7%2Faeon--vllm--ultimate-blue?logo=docker)](https://ghcr.io/aeon-7/aeon-vllm-ultimate)
[![vLLM](https://img.shields.io/badge/vLLM-0.25.0%2Bsm__121a.aeon-orange)](https://github.com/vllm-project/vllm)
[![sm_121a](https://img.shields.io/badge/sm__121a-DGX%20Spark-green)](https://www.nvidia.com/en-us/data-center/dgx-spark/)

**One container, the whole fleet.** A single image — `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` — serves every AEON model on **NVIDIA DGX Spark (GB10, sm_121a)** and other consumer-Blackwell GPUs (RTX 50 series): **Gemma-4-26B-A4B**, **Qwen3.6-27B**, and **Qwen3.6-35B-A3B** all run on the same build, with DFlash speculative decoding, NVFP4 weights, NVFP4/FP8 KV cache, and the OpenAI-compatible gateway intact.

Built on **vLLM v0.25.0 compiled from source for sm_121a**, merged with the AEON speculative-decoding stack: Triton software **NVFP4 KV cache** (PR #44389) + **DFlash SWA / high-concurrency / prefix-cache fixes** (PR #40898, #41703, #43982-port) + the **AEON DGX Spark runtime patches** + **TurboQuant** + **DFlash speculative decoding**.

> 🆕 **2026-07-14 — `:latest` is now the v0.25.0 sm_121a build (`:2026-07-14-v0.25.0`).** Rebuilt from source on vLLM **v0.25.0** (commit `702f4814`) as a 3-way merge onto the AEON tree. Still carries the three open upstream PRs (#44389 Triton NVFP4-KV, #40898 DFlash SWA, #41703 prefix-cache/batched-verify — all re-verified still unmerged) plus the source-baked fixes (DFlash block-table unpad, cudagraph spec-decode alignment, UMA negative-estimate clamp) and the maxsafe carries (**#47356** exclude `kv_cache_memory_bytes` from cache-hash, #45207 Mamba page-pad, #47053). Picks up v0.25.0's **#45739** (NVFP4 swizzled-scale zero-init → Blackwell decode) and **#46761** (DFlash per-layer RMSNorm fusion). Dropped the now-upstream #45544 tie_weights cherry-pick. Deps: **FlashInfer 0.6.13**, cutlass-dsl 4.5.2, torch 2.11.0 (unchanged). ⚠️ **The big one — MRv2:** v0.25.0 defaults dense models to **Model-Runner-V2** and whitelists `method=dflash` for MRv2 **with no fallback**, which would silently route Qwen3.6-27B to an *unpatched* V2 DFlash tree (no sliding-window, none of our fixes). This image bakes **`VLLM_USE_V2_MODEL_RUNNER=0`** so every model stays on the carried V1 runner. **TP=2 is wired in (opt-in via `--tensor-parallel-size 2`) but UNTESTED** — no second Spark; TP=1 is the tested, supported default. Two merge bugs a plain merge would ship were caught + fixed (the `KVQuantMode.NVFP4` enum renumber 4→5 that silently disables NVFP4 KV; two auto-merge SyntaxErrors in qwen3_dflash.py). **Full-fleet A/B before push: v0.25.0 at throughput parity with v0.24.0 across Gemma-4-26B / Qwen3.6-35B-A3B / Qwen3.6-27B**, DFlash on V1, tools working, acceptance healthy. Rollback tag: `:2026-07-08-v0.24.0-maxsafe`.

> ℹ️ **torchcodec (v0.25.0 video-decode dep) is intentionally omitted** — 0.14.0 is ABI-incompatible with torch 2.11 and vLLM guards the import (graceful degrade); text/image/audio/voice are unaffected (same as the prior image). **NVFP4 backend note:** checkpoints with per-input global scales (e.g. Gemma-4-26B, Qwen3.6-27B) run native **CUTLASS** FP4; the Ornith-35B checkpoint lacks them so it correctly uses **Marlin** — forcing `--linear-backend flashinfer_cutlass` on it crashes (missing `input_global_scale_inv`); re-quantize with CUTLASS-compatible scales to unlock it.

> 🆕 **2026-07-02 — `:latest` is now the v0.24.0 sm_121a build (`:2026-07-01-v0.24.0`).** Rebuilt from source on vLLM v0.24.0 as a 3-way merge that preserves the AEON spec-decode tree. Still carries the three open upstream PRs (#44389 NVFP4-KV, #40898 DFlash SWA, #41703 prefix-cache corruption — all re-verified still unmerged), now **bakes the runtime patches into the source** (DFlash block-table unpad, spec-decode cudagraph alignment / open twin PR #46324), and adds three post-tag fixes: the **UMA negative-cudagraph-estimate clamp** (port of open PR #46932 — negative estimates on unified-memory GPUs silently inflated the KV budget), the **tied-embedding fix for ModelOpt checkpoints** (cherry-pick of merged-post-tag PR #45544 — without it every tied Gemma-4 crashes at load), and a **`use_mm_prefix` signature fix** for the carried NVFP4-KV backend overrides. New in the v0.24.0 base for Spark users: DFlash on the FlashInfer backend (#43081, non-causal prefill), UMA memory-pressure release during weight loading (#45179), `--moe-backend`/`--linear-backend` selection incl. the SM12x `flashinfer_b12x` CuteDSL backends, Dynamic SD per-batch-size draft lengths (#32374), async scheduling default-on, FlashInfer 0.6.12, pinned transformers 5.12.1 (replaces git-HEAD). ⚠️ **Breaking:** v0.24.0 removed `VLLM_NVFP4_GEMM_BACKEND` and the `VLLM_USE_FLASHINFER_MOE_*` env vars — use `--linear-backend flashinfer_cutlass` / `--moe-backend cutlass` instead (recipes below updated). Validated on the full fleet before push: 35B A/B at throughput parity with v0.23.0, DFlash concurrency clean through c=64, Triton NVFP4-KV boot + generation, 26B voice stack healthy (DFlash pos0 acceptance 60–86%). Rollback tag: `:2026-06-18-v0.23.0-dflashfix`.

> 🆕 **2026-06-18 — `:latest` is now the v0.23.0 sm_121a build (`:2026-06-18-v0.23.0-dflashfix`).** Rebuilt from source on vLLM v0.23.0 as a 3-way merge that preserves the AEON spec-decode tree, and adds the **DFlash high-concurrency fix** (port of upstream PR #43982): the drafter previously **crashed at ≥32 concurrent requests** under speculative decoding (padded-vs-unpadded KV block-table shape mismatch) and now scales cleanly to **c=64**. Carries the still-open PR #44389 (NVFP4-KV), #40898 (DFlash SWA), #41703 (prefix-cache corruption). See [What we fixed for the DGX Spark](#what-we-fixed-for-the-dgx-spark) and the [v0.23.0 fleet benchmarks](#v0230-fleet-benchmarks--one-image-three-models). Rollback tag: `:2026-06-11-pr41703`.

## Quickstart (DGX Spark, copy-paste)

The canonical Spark recipe: **Qwen3.6-27B Multimodal-NVFP4-MTP body + z-lab DFlash drafter + FP8 KV** — the measured-best daily-driver config (parity speed with the smaller XS body, higher quality-eval scores). One block pulls the container, the model, and the drafter, then serves on `:8000`. (Full deployment matrix — MTP/NVFP4-KV, TurboQuant, Gemma-4-26B, dedicated-VRAM Blackwell — is in [Deployment recipes](#deployment-recipes) further down.)

```bash
# 1) Pull the unified container (vLLM 0.25.0 + sm_121a + DFlash on a pinned V1 runner)
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest

# 2) Pull the recommended body — Multimodal-NVFP4-MTP (modelopt NVFP4, image+video capable;
#    parity speed with the XS variant + higher quality-eval scores) — fresh clone
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP \
  /models/Qwen3.6-27B-AEON-MM-MTP
( cd /models/Qwen3.6-27B-AEON-MM-MTP && git lfs pull )

# 3) Pull the DFlash drafter (z-lab 5-layer, ~3.3 GB) — fresh clone (DFlash only)
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/z-lab/Qwen3.6-27B-DFlash \
  /models/Qwen3.6-27B-DFlash-drafter
( cd /models/Qwen3.6-27B-DFlash-drafter && git lfs pull )

# 4) Serve — DFlash drafter + FP8 KV (mounts body at /model, drafter at /drafter)
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

# 5) Smoke test
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"aeon","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"temperature":0.0}' \
  | jq .choices[0].message.content
```

> **Why these flags:** `--quantization modelopt` matches the recommended Multimodal-NVFP4-MTP body; `--kv-cache-dtype fp8_e4m3` is the stable DFlash pairing on GB10; `--gpu-memory-utilization 0.60` leaves room for Qwen3-ASR/Qwen3-TTS sidecars on the same Spark; and `--attention-backend TRITON_ATTN` must be set both on the target model and inside the DFlash JSON because vLLM does not inherit target attention-backend settings into speculative drafters. `--max-model-len 229376` gives one near-full-context session while still leaving KV headroom for output and smaller concurrent agents; `--max-num-seqs 16` and `--max-num-batched-tokens 32768` keep the agent/gateway burst path usable. Leave `--mamba-block-size` unset and let vLLM derive the hybrid GDN geometry. If `git clone` leaves LFS pointer files, re-run `git lfs pull` in the model dir so vLLM sees real weights.

## What's new in vLLM 0.25.0 — features & unlocks

Rebased from vLLM 0.24.0 onto **vLLM 0.25.0** (`702f4814`) as a 3-way merge that preserves the entire AEON spec-decode stack. What the rebase brings:

### ⚡ Decode & kernel wins (automatic — no flag)
- **NVFP4 swizzled-scale zero-init restored** (#45739) — recovers Blackwell NVFP4 **decode** throughput that a prior upstream regression had degraded. (Net effect on sm_121a is small — GB10 decode is memory-bandwidth-bound — but the fix is in.)
- **DFlash per-layer RMSNorm fusion** (#46761) — fuses the drafter's RMSNorm into one batched CUDA kernel, cutting per-step launch overhead on the **Qwen3.6** DFlash drafters (27B / 35B-A3B).
- **FlashInfer 0.6.13** (from 0.6.12) — FP8-KV FA2 prefill fix + sm12x gating + a Mamba SSU cross-warp race fix relevant to the 26B hybrid. Stale 0.6.12 JIT-cache purged so 0.6.13 kernels aren't shadowed.
- **`skip cooperative top-K on SM120/121`** (#47164) and **DeepGEMM SM120** (#47304) land in-tree — remove a `cudaErrorInvalidValue` launch-failure class and fix an FP8-block-scaled boot crash on Spark-class Blackwell.
- **cutlass-dsl 4.5.2** pinned for the FA4 cute-DSL path; **`enable_cutedsl_warmup`** on by default.

### 🧊 KV / memory / stability
- **NVFP4-KV skip-layers sliding-window page unification** (#42890) — cleaner KV page sizing for hybrid SWA models, integrated with our carried Triton NVFP4-KV.
- **`kv_cache_memory_bytes` excluded from the cache-config hash** (#47356) — carried from the maxsafe tree; directly relevant to the GB10 unified-memory work in issue #9.
- **Mamba-hybrid prefix caching** (#42406), **Mamba2 crash fix** (#47428), **hybrid FA block-size restriction removed** (#36701) — all available for the GDN/NemotronH hybrids (candidate fix for the 35B stale-context APC issue; being validated).
- **UMA negative-cudagraph-estimate clamp** + **cudagraph spec-decode alignment** + **DFlash block-table unpad** — AEON fixes, now source-baked.

### 🧬 Speculative decoding — more methods available
- **DFlash** remains the AEON default (SWA drafter + ctx-mask + high-concurrency fix), running on the **pinned V1 runner** (see below).
- **Universal spec decode across heterogeneous vocabularies (TLI)** (#38174), **dynamic spec decode compatible with full CUDA graphs** (#45953), **block verification for rejection sampling** (#46781), and the **DSpark** drafter (#46995/#47093) are all present in-tree — and as of the `2026-07-16` build the MRv2 spec-decode loaders they run on are **correct** (lm_head-sharing fix, see below). DSpark still needs a trained per-model drafter that doesn't exist for this fleet, so it remains R&D.

### 🖥️ Model Runner V2 — audited, fixed, still pinned to V1 (by choice)
v0.25.0 makes **Model-Runner-V2 the default for dense models** and whitelists `method=dflash` for MRv2 **with no fallback**. This image still bakes **`VLLM_USE_V2_MODEL_RUNNER=0`** — but as of the `2026-07-16` build the pin is a *choice*, not a necessity. The Phase-2 MRv2 audit + port is done:

- **Fixed the one real MRv2 bug — spec-decode lm_head sharing** (upstream [#47914](https://github.com/vllm-project/vllm/pull/47914), ported at all 3 sites: `dflash`, `eagle`, `dspark`). MRv2 read `lm_head` off the outer `*ForConditionalGeneration` wrapper → `None` → the draft head stayed **zero-initialised** → **0% acceptance at every position** on multimodal-wrapped targets like Qwen3.6-27B. With the fix baked: 0% → healthy, **full V1 parity** in a same-image A/B (short prompts *and* 23.8k-token long-context with exact needle retrieval, both runners).
- **The other V1 carries turn out not to be MRv2 blockers** (adversarially-verified source audit + hardware A/B): the high-concurrency block-table crash is *structurally impossible* on MRv2 (everything is sliced to one consistent `num_reqs_padded`); the #41703 rejected-slot masking difference is *benign* there (the same-step drafter forward overwrites those exact slots); and the missing #40898 SWA wiring (the 5-layer drafter runs non-causal on MRv2) measured **zero acceptance impact**.
- **Why the pin stays anyway:** it turns out dropping it wouldn't move *any* fleet model by default — the 27B is a **75%-linear-attention hybrid** (`Qwen3_5ForConditionalGeneration` is `IsHybrid`, 48/64 layers linear), and hybrids are excluded from default-V2 exactly like the two MoE models. The pin guards against upstream routing-default changes and accidental opt-ins (e.g. `method=dspark` force-routing). Explicitly moving a model (env `=1`) also buys zero gain — DSpark needs a trained per-model drafter that doesn't exist for this fleet, dynamic-SD measured **−12–16%** on GB10, hybrid-APC is gated out of default-V2, and MRv2 still silently ignores `thinking_token_budget`. V1 keeps the fully-carried, battle-tested path; MRv2 is now a **correct fallback** instead of a broken one.
- **Opting in anyway (MRv2 / DSpark):** the pin is a default env var, not a build-time removal — everything MRv2 ships in the image. Add `-e VLLM_USE_V2_MODEL_RUNNER=1` to run any model on MRv2 (DFlash on MRv2 is validated working with the lm_head fix). **DSpark** additionally *requires* that override (the baked `=0` wins over DSpark's V2-forcing and fails validation loudly) plus a trained DSpark drafter (`Qwen3DSparkModel` arch, or a DeepSeek-V4 checkpoint with in-target drafter weights) — the loader is head-fix-covered and ready when such a drafter exists.

### 🔀 TP=2 — wired in, opt-in, untested
Tensor parallelism is available via `--tensor-parallel-size 2` and carries #47081 (blocking CUDA events → no driver-lock busy-polling under TP collective contention, neutral at TP=1). **TP>1 ships UNVALIDATED** — there is no second Spark to test it; **TP=1 is the supported, tested default.** Two Sparks have no NVLink (ConnectX/RoCE only), and the whole fleet fits one 121 GB Spark, so TP=2 is only relevant for >121 GB models (e.g. Step-3.7). Do **not** enable NCCL symmetric memory on SM121.

### 🧰 Build hardening (this rebase)
- **Two silent-killer merge bugs caught + fixed** that a plain merge would ship: v0.25.0 renumbered the `KVQuantMode.NVFP4` enum 4→5 while our kernel hard-coded `== 4` in a non-conflict region (would have silently disabled **all** NVFP4 KV); plus two auto-merge SyntaxErrors in `qwen3_dflash.py`.
- **torchcodec skipped** — v0.25.0 lists `torchcodec>=0.14`, but 0.14.0 is ABI-incompatible with torch 2.11; vLLM guards the import (`PlaceholderModule`), so absence degrades gracefully (video-decode unavailable) while text/image/audio/voice are unaffected. Revisit when a torch-2.11 torchcodec ships.
- Kept **`TORCH_CUDA_ARCH_LIST=12.1a`** (bare `12.1` would `#ifdef` out the FP4 kernels → Marlin fallback), GCC 12, transformers 5.12.1, xgrammar ≥ 0.2.1.
- Dropped the now-upstream #45544 tie_weights cherry-pick.

### 📊 Full-fleet A/B before promotion (GB10, vs the prior v0.24.0 image)
| Model | v0.25.0 | v0.24.0 | NVFP4 backend |
|---|---|---|---|
| Gemma-4-26B-A4B | 505–511 tok/s @c16 · ~73 single | 559 @c16 · ~71 single | CUTLASS |
| Qwen3.6-35B-A3B (Ornith) | 341 @c12 · 77–79 single | 348 @c12 · 72–76 single | Marlin¹ |
| Qwen3.6-27B (AEON-Ultimate) | 88 @c8 · ~22 single | 100 @c8 · ~23 single | CUTLASS |

All three: DFlash on the V1 runner, tool-calling working, DFlash acceptance healthy — **throughput parity, no regressions.** ¹The Ornith-35B checkpoint lacks per-input global scales, so it correctly uses Marlin; forcing `--linear-backend flashinfer_cutlass` crashes it. Checkpoints **with** those scales (Gemma-26B, Qwen-27B) run native CUTLASS FP4. To move the 35B to CUTLASS, re-quantize with CUTLASS-compatible scales.

> 💡 **Speed tips:** GB10 decode is memory-bandwidth-bound, so the real levers are (1) **FP8 KV cache** (`--kv-cache-dtype fp8` with the drafter also on `triton_attn`) — halves KV reads, ~2× KV capacity, DFlash acceptance intact; and (2) mounting a **persistent vLLM cache** (`-v …:/root/.cache/vllm`) so a restart warms in ~2 min instead of re-running the ~13 min cold autotune.

## What's inside

| Component | Version | Why |
|---|---|---|
| **vLLM** | 0.25.1 + sm_121a build, AEON spec-decode 3-way merge | Built from source for GB10; carries PR #44389 (Triton NVFP4 KV) + #40898 (DFlash SWA) + #41703 (prefix-cache corruption) + #43982-port (DFlash high-concurrency fix) + #47914-port (MRv2 spec-decode lm_head sharing, new 2026-07-16) |
| **PyTorch** | 2.11.0+cu130 | CUDA 13.0 with sm_121a (DGX Spark / GB10) compute capability |
| **transformers** | 5.12.1 | Recognizes `gemma4_unified`, `qwen3_5`, all bleeding-edge model classes |
| **flashinfer** | 0.6.13 | NVFP4 GEMM kernels, sliding-window attention, MLA, custom attention; FP8-KV FA2 prefill fix + sm12x gating |
| **TurboQuant** | 0.2.0 (AEON-7 fork) | CUDA-graph-safe QJL — 4-bit KV compression on top of vLLM's native KV cache |
| **modelopt** | available via pip if needed | Quantization framework (not bundled — image stays small for serving) |

## v0.23.0 fleet benchmarks — one image, three models

The whole point of this container is that **a single build runs the entire AEON fleet** on a DGX Spark. The three charts below are the *same* `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (vLLM 0.23.0 + AEON sm_121a + DFlash) serving three very different architectures — a Gemma-4 MoE, a Qwen3.6 hybrid GDN+attention dense model, and a Qwen3.6 A3B MoE — each scaling cleanly from 1 to **64 concurrent requests** with no crash (the pre-fix image died at c≥32 under speculative decoding).

Numbers are measured on **DGX Spark GB10 (sm_121a)** with DFlash speculative decoding, NVFP4 weights, FP8 KV cache, prefix caching on, p50 across ≥ samples per point.

### Gemma-4-26B-A4B-it-Uncensored (NVFP4)

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/gemma26b_concurrency.svg" width="100%" alt="Gemma-4-26B-A4B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 1937 tok/s at c=64"></p>

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

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/gemma26b_longcontext.svg" width="100%" alt="Gemma-4-26B-A4B DFlash draft acceptance and decode throughput holding flat from short prompts to ~33k-token histories on aeon-vllm-ultimate:latest"></p>

Stock-vs-optimized single-stream contrast on this build:

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/gemma26b_stock_vs_optimized.svg" width="100%" alt="Gemma-4-26B-A4B stock vanilla vLLM vs aeon-vllm-ultimate:latest single-stream decode throughput by category"></p>

> **Provisional contrast.** The stock / un-optimized bars are from **stock vanilla vLLM** (default settings, no DFlash, no DGX-Spark / sm_121a optimizations) and are **provisional, pending a fresh fully-vanilla re-bench** on the current v0.23.0 version.

### Qwen3.6-35B-A3B-heretic (NVFP4)

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/qwen35b_concurrency.svg" width="100%" alt="Qwen3.6-35B-A3B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 740 tok/s at c=64"></p>

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

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/qwen35b_longcontext.svg" width="100%" alt="Qwen3.6-35B-A3B DFlash draft acceptance and decode throughput holding flat from short prompts to ~33k-token histories on aeon-vllm-ultimate:latest"></p>

### Qwen3.6-27B-AEON-Ultimate (NVFP4 MTP-XS body + DFlash n=12)

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/qwen27b_concurrency.svg" width="100%" alt="Qwen3.6-27B aggregate throughput scaling from 1 to 64 concurrent requests on aeon-vllm-ultimate:latest — up to 344 tok/s at c=64"></p>

Single-stream (c=1), by category:

| Category | 🟢 Decode tok/s | TTFT p50 | TPOT p50 | Prefill (PP) | DFlash accept |
|---|---:|---:|---:|---:|---:|
| Coding | **41.8** | 140 ms | 23.9 ms | 322 tok/s | 34.5% |
| Math | **47.3** | 244 ms | 21.1 ms | 229 tok/s | 41.7% |
| Reasoning | **56.1** | 234 ms | 17.8 ms | 183 tok/s | 50.0% |
| Prose | **34.1** | 146 ms | 29.4 ms | 220 tok/s | 27.3% |
| Natural language | **38.3** | 137 ms | 26.1 ms | 248 tok/s | 31.3% |
| Extraction / JSON | **44.2** | 246 ms | 22.6 ms | 195 tok/s | 37.2% |

vs a **stock vanilla `vllm/vllm-openai:nightly` baseline of ~10.5 tok/s** (no DFlash, no sm_121a optimizations) → optimized hits **~38–56 tok/s by category ≈ 4–5× single-stream** decode.

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/qwen27b_stock_vs_optimized.svg" width="100%" alt="Qwen3.6-27B stock vanilla vLLM (~10.5 tok/s) vs aeon-vllm-ultimate:latest single-stream decode throughput by category"></p>

Long-context hold: Coding draft acceptance is **49.5%** at **~16k** and **29.1%** at **~33k** — long histories stay drafted on the SWA-fixed drafter.

<p align="center"><img src="https://raw.githubusercontent.com/AEON-7/vllm-ultimate-dgx-spark/main/assets/perf/qwen27b_longcontext.svg" width="100%" alt="Qwen3.6-27B DFlash draft acceptance and decode throughput across short-to-~33k-token histories on aeon-vllm-ultimate:latest"></p>

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

The container ships with our 3 idempotent runtime patches that ensure correctness on GB10 hardware until upstream fixes land:

| Patch | What it fixes |
|---|---|
| **patch_cuda_optional_import** | Wraps `import vllm._C_stable_libtorch` in `RTLD_LAZY` so the SM100-only `mxfp4_experts_quant` and `silu_and_mul_mxfp4_experts_quant` symbols are tolerated as unresolved until first call (they never fire on sm_121a workloads) |
| **patch_cudagraph_align** | Drops the `cudagraph_mode==FULL`-only gate on the spec-decode capture-size alignment filter in `config/compilation.py` so PIECEWISE mode also rounds capture sizes to multiples of `(1 + num_speculative_tokens)` — eliminates `cudaErrorIllegalAddress` mid-decode on partial-acceptance steps |

All patches are idempotent — they no-op when upstream merges the equivalent fix.

### 🩹 DFlash drafter correctness fixes (PR #40898 + #41703, merged ahead of upstream)

Both PRs are **open upstream but required** for correct DFlash operation (the z-lab drafter README pins them); the v0.23.0 `:latest` build carries them in-tree (3-way merged), alongside the new DFlash high-concurrency fix (PR #43982 port). They fix three real defects we root-caused in production on DGX Spark:

| Defect | Symptom | Fix |
|---|---|---|
| **Rejected-token context-KV writes** — the `copy_and_expand_dflash_inputs_kernel` stored slot mappings for *rejected* draft tokens, writing garbage K/V into the drafter's paged KV cache (incl. shared blocks). With `--enable-prefix-caching` the corruption was **persistent and self-accelerating** | Draft acceptance decays 34–56% → **0.0%** over minutes-to-hours of traffic (scales with volume); sticky engine-global; ~6× decode slowdown (144 → 24 tok/s) that only a restart healed | #41703 masks rejected/invalid context slots (`-1`) so they are never written |
| **Drafter sliding-window ignored** — SWA drafters (e.g. the Gemma-4-26B drafter: 4 of 5 layers SWA-2048) ran all layers as full attention | Long-context requests (>2048 tok history) got ~0% acceptance *per-request* even on a healthy server | #40898 adds DFlash SWA support (per-layer sliding-window wiring + causal SWA drafting metadata) |
| **Missing Gemma-4 adapter pieces** — no sqrt(hidden) embedding normalizer or final-logit softcapping in the draft path; `flash_attn` drafter rejected on multimodal Gemma targets | Depressed acceptance ceiling (MAL 4.4–6.6 vs z-lab's published 6.1–8.6); forced onto `flex_attention` | #41703 adds both + `use_mm_prefix=False`, enabling the upstream-tested **`flash_attn` drafter** on Gemma-4 |

⚠️ **Recipe note:** Qwen3.6 DFlash on the v0.24.0 Spark image uses `"attention_backend":"TRITON_ATTN"` in the speculative config, matching the top quickstart. Gemma-4 DFlash recipes remain a separate path and may specify `"attention_backend":"flash_attn"` where called out. With these fixes, `--enable-prefix-caching` is **safe again with DFlash** — soak-validated under production fleet traffic.

### 🧠 TurboQuant K8V4 — 4-bit KV cache compression
[0xSero/turboquant](https://github.com/0xSero/turboquant) with the AEON-7 fork applying our [`fix/cuda-graph-safe-qjl-powers`](https://github.com/AEON-7/turboquant/tree/fix/cuda-graph-safe-qjl-powers) patch — caches the `[1, 2, 4, 8, 16, 32, 64, 128]` constant per-device once at module load instead of re-allocating per call. **Without this fix, TurboQuant crashes at boot during CUDA graph capture**; the lazy workaround `--enforce-eager` costs ~30% throughput.

Enable per-serve via `--kv-cache-dtype tq_k8v4`.

### ⚡ DFlash speculative decoding (native via `--speculative-config`)
DFlash and EAGLE3 drafters are supported natively via vLLM's `--speculative-config` flag — no extra package needed since vLLM 0.21. Pair with our [aeon-7 DFlash drafters on HF](https://huggingface.co/AEON-7/DFlash-Qwen3.5-27B-Uncensored) for 1.5-2.5× throughput on the Qwen3.x family.

### 🔬 Native Blackwell SM 12.1 sm_121a compute
Built for `TORCH_CUDA_ARCH_LIST="12.1a"` — the sm_121a target for the GB10 in DGX Spark. Also runs on RTX 5090 / RTX 5080 / RTX PRO 6000 Blackwell (sm_120) thanks to the same family matcher in vLLM main.

## Deployment recipes

The [Quickstart](#quickstart-dgx-spark-copy-paste) at the top is the canonical daily-driver path (Recipe A). This section is the full deployment matrix: the canonical target is the **AEON-7 Qwen3.6 family** — see [Validated models](#validated-models) below. Pick the variant that matches your hardware, then follow the matching recipe.

### Pull the image

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest
# or pin the current build (vLLM 0.25.1 + sm_121a + MRv2 lm_head fix)
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-16-v0.25.1
# previous builds kept for rollback
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-14-v0.25.0
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-08-v0.24.0-maxsafe
```

### Recipe A — DGX Spark, DFlash drafter + FP8 KV (recommended for daily-driver)

This is the **measured-best config** for DGX Spark per the AEON-7 Qwen3.6 routing memo: on the **Qwen3.6-27B MTP-XS body**, the **DFlash drafter beats the MTP-method by +56 % median / +150 % peak** on Spark's unified-memory GB10 (measured 2026-04-28). Note this is a 27B-XS-body result — **Qwen3.6-35B-A3B is at parity** (no DFlash win; its 8-layer all-full-attention drafter draws even with MTP-style decoding).

> ⚠️ **DFlash + NVFP4 KV is not yet compatible on sm_121a (still true on the v0.23.0 build).** The DFlash drafter uses non-causal attention (parallel candidate generation), and none of the currently-built backends pair non-causal with NVFP4 KV on Spark:
> - `FLASH_ATTN` — doesn't support NVFP4 KV
> - `FLASHINFER` — supports NVFP4 KV but requires **SM100** (we're on SM121)
> - `TRITON_ATTN` — supports NVFP4 KV but is **causal-only**
>
> Use **`--kv-cache-dtype fp8_e4m3`** with DFlash. NVFP4 KV works cleanly with causal speculators (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`) — see Recipe B.

This is the recipe in the [top Quickstart](#quickstart-dgx-spark-copy-paste) — the full pull-container + pull-model + pull-drafter + serve block is there; don't duplicate it here. The serve flags below are the same ones; this section just explains each.

**Key flags**:
- `--quantization modelopt` — the recommended **Multimodal-NVFP4-MTP** body is a modelopt NVFP4 checkpoint. (The older `-NVFP4` production body is compressed-tensors `format: nvfp4-pack-quantized` → use `--quantization compressed-tensors` for that one.)
- `--kv-cache-dtype fp8_e4m3` — DFlash is non-causal and incompatible with NVFP4 KV on Spark today (see Recipe B for NVFP4 KV with MTP).
- `--speculative-config '{"method":"dflash",...}'` — `method: "dflash"` is the native vLLM speculator (not `"speculators"`).
- `--attention-backend TRITON_ATTN` plus `"attention_backend":"TRITON_ATTN"` inside the DFlash JSON — vLLM does not inherit target attention-backend settings into speculative drafters.
- `--max-num-batched-tokens 32768` — must accommodate long agent startup prompts plus `num_speculative_tokens × max_num_seqs` headroom.
- Leave `--mamba-block-size` unset — vLLM now derives the hybrid GatedDeltaNet + attention cache geometry correctly.
- `--gpu-memory-utilization` — **keep this ≤ 0.88 on DGX Spark.** The quickstart uses `0.60` so Qwen3-ASR/Qwen3-TTS sidecars can share the GPU without forcing unified-memory pressure. Raise toward `0.75-0.85` only when the LLM is the dominant workload.

> 💡 **Drafter materialization note.** vLLM bind-mounts the drafter dir but can't follow symlinks that point **outside** the mount (e.g. into the HF cache `blobs/` dir). After `huggingface-cli download`, either pass `--local-dir-use-symlinks=False` *or* `cp -L $HF_CACHE/snapshots/<hash>/* /models/Qwen3.6-27B-DFlash-drafter/` so the files are real, not symlinks. This pitfall cost us 4 startup failures.

### Recipe B — MTP self-speculation + NVFP4 KV (capacity-bound workloads)

For workloads where **KV capacity is the bottleneck** (long context, many concurrent streams), use the modelopt **MTP-XS** body with NVFP4 KV cache — the smaller body maximizes KV headroom. This is the only Spark recipe that exercises PR #44389's ~3× KV capacity gain today. *(If you have VRAM to spare and want higher-quality output, the full **Multimodal-NVFP4-MTP** body drops in here too, at a small KV-headroom cost.)*

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
        --gpu-memory-utilization 0.60 \
        --enable-chunked-prefill --enable-prefix-caching \
        --trust-remote-code
```

> ⚠️ **MTP throughput is lower than DFlash on Spark (Qwen3.6-27B).** Measured 2026-04-28 on the **Qwen3.6-27B MTP-XS body**: DFlash beats MTP by **+56 % median / +150 % peak** with the same XS body. (On **Qwen3.6-35B-A3B** the two are at **parity** — its 8-layer all-full-attention drafter has no DFlash win.) Use MTP only when you need NVFP4 KV's ~3× capacity (long contexts or higher batch sizes) **and** can accept the lower throughput. For pure throughput on Spark, use Recipe A. For dedicated-VRAM Blackwell (RTX PRO 6000, B100/B200), MTP is the right choice everywhere.

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
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  --entrypoint bash ghcr.io/aeon-7/aeon-vllm-ultimate:latest -lc 'exec vllm serve /model \
    --quantization compressed-tensors --trust-remote-code \
    --attention-backend triton_attn \
    --linear-backend flashinfer_cutlass \
    --max-model-len 184320 --max-num-seqs 32 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.68 --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/drafter\",\"num_speculative_tokens\":10,\"attention_backend\":\"flash_attn\"}"'
```

### Qwen3.6 benchmarks (image `2026-06-04` era)

Measured on **DGX Spark GB10 (sm_121a)** with the historical 2026-06-04-era
recipe: `--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.78
--enable-chunked-prefill --enable-prefix-caching --mamba-block-size 256
--quantization {compressed-tensors|modelopt}`. Current v0.24 quickstarts use
`--gpu-memory-utilization 0.60` for ASR/TTS sidecar headroom and leave
`--mamba-block-size` unset.

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

- **🏆 The winning speed pattern on Spark is an MTP-format body + DFlash drafter (n=4) + BF16 KV.** Even though the body name says "MTP", it works great with an external DFlash drafter — and an MTP body **leaves more compute and KV headroom than the 26 GB compressed-tensors body**. Results vs the FP8-KV baseline: **+40% single-stream tok/s and +24% concurrent throughput**; aggregate peak ~87 tok/s on 4 concurrent streams. *(Speed rows below were measured on the smaller `-MTP-XS` body.)*
- **✅ Recommended daily-driver body: `Multimodal-NVFP4-MTP` (the full, non-XS body).** In AEON-7's follow-up evals it runs at **parity speed with the XS body** while scoring **materially higher on quality benchmarks** — so it's the default when you have the VRAM (it's only slightly larger than XS). Keep the **`-MTP-XS`** body when footprint is tight. Both use `--quantization modelopt` and pair with the same z-lab DFlash drafter.
- DFlash on the NVFP4 (compressed-tensors) body is also a big win (+12% single, +0.9% concurrent) but the heavier 26 GB body loses ground to the same drafter on the lighter MTP bodies.
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
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP) | modelopt NVFP4 (GDN preserved BF16) | DFlash drafter (or native MTP) | ✅ **Recommended Spark daily-driver** — parity speed with XS, higher quality-eval scores | `--quantization modelopt`; pair with [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash); image+video capable |
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ Benchmarked in this card (heavier 26 GB body) | Prefer the Multimodal-NVFP4-MTP body above; pair with [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) |
| [AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ **Fleet-benchmarked** in the v0.23.0 section above | Drafter **must** use `attention_backend: flash_attn` on this image; pair with z-lab `gemma-4-26B-A4B-it-DFlash` |
| [AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4) | compressed-tensors `nvfp4-pack-quantized` | DFlash drafter | ✅ **Fleet-benchmarked** in the v0.23.0 section above | A3B MoE; 8-layer all-full-attn drafter (no SWA/`--mamba-block-size` needed); optimal n≈10–11 |
| [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) | modelopt NVFP4 | DFlash drafter (or native MTP) | ✅ Smallest footprint — speed rows benchmarked below | Min-VRAM option; step up to the full **Multimodal-NVFP4-MTP** for higher quality if VRAM allows. Native MTP-method underperforms DFlash on Spark |
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

This is a **current-state limitation of the v0.23.0 build on sm_121a**: DFlash's non-causal (parallel candidate) attention has no FP8/NVFP4 KV kernel partner on GB10 today — `FLASH_ATTN` is BF16-KV only, and both `FLASHINFER` and `TRITON_ATTN` dropped their non-causal path in PR #44389's refactor (FlashInfer's NVFP4 KV also needs SM100). NVFP4/FP8 KV will pair with DFlash once either (a) the Triton backend gains a non-causal kernel or (b) FLASHINFER's non-causal + FP8 path returns. Until then, run DFlash with **`--kv-cache-dtype auto`** (BF16), or use a causal speculator (Recipe B) for NVFP4 KV.

**Workaround for NVFP4 KV**: use a **causal** speculator (`mtp`, `qwen3_5_mtp`, `eagle3`, `ngram`, `ngram_gpu`) — see Recipe B. The Triton NVFP4-KV path supports those.


### Gemma-4-12B-AEON variants

| Variant | Issue |
|---|---|
| `Gemma-4-12B-AEON-Abliterated-K4-BF16` | vLLM's `TransformersMultiModalForCausalLM` fallback hits a shape mismatch on `Gemma4UnifiedForConditionalGeneration`. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x4096 and 8192x3840)` in a linear projection during graph capture. Suspect a multimodal-fused QKV layer not handled by the fallback path. |
| `Gemma-4-12B-AEON-Abliterated-K4-NVFP4-SVDQuant` | vLLM only knows `NVFP4 / NVFP4_FP8_MHA / W4A16_NVFP4 / MXFP8 / MIXED_PRECISION`. Our model's `quant_algo=NVFP4_SVD` (ModelOpt's newer SVD+low-rank variant) isn't yet recognized. Awaiting a deserializer PR in vLLM's `model_executor.layers.quantization.modelopt`. |

### Gemma-4-26B-A4B-NVFP4 — *badly-quantized vision-embedder variant only*

> This entry is specific to the **variant whose vision embedder was quantized**. The **correctly-quantized** [`Gemma-4-26B-A4B-it-Uncensored-NVFP4`](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4) (vision embedder excluded as BF16) is **fleet-benchmarked and works** on the v0.23.0 `:latest` image — see the [fleet section](#v0230-fleet-benchmarks--one-image-three-models).

For the badly-quantized variant, vLLM creates the `embed_vision.embedding_projection` as a quantized `ReplicatedLinear`, but the checkpoint has only the unquantized `embed_vision.embedding_projection.weight` (because `embed_vision*` was excluded during quantization). Weight-loading mismatch. Likely an `exclude_modules` wildcard handling bug in PR #44389's refactor.

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
