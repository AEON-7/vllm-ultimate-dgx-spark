## 2026-08-17 — SLIM image: 50.6 GB → 18.3 GB (`:2026-08-16-v0.27.1-slim`)

Same vLLM build, same userland, same gate results — **32.3 GB smaller on disk and 13.2 GB smaller to
pull** (21.4 GB → ~8.2 GB compressed). Built by `Dockerfile.slim`.

- **Where the fat was.** Two places, both measured rather than guessed. (1) The inherited
  `aeon-gemma-4-26b-a4b-dflash` base carried a **dead vLLM-0.20.1-era stack under layer whiteouts** —
  bytes that never appear at runtime but download on every pull; the shipped image's *live* filesystem
  was only ~22 GB of its 50.6 GB. (2) `/usr/local/cuda-13.0` was ~4.3 GB of which ~3.8 GB is dead:
  2.1 GB of static `.a` archives plus 1.7 GB of `.so` that **exactly duplicate the pip `nvidia-*`
  wheels**. Proven with `/proc/self/maps`: torch resolves libcublas/libcudart/libnccl/libcudnn from
  `site-packages/nvidia/*`, never from `/usr/local/cuda`. A DT_NEEDED census over every native lib in
  site-packages confirmed the pip wheels cover all CUDA-13 sonames.
- **Base: `nvidia/cuda:13.0.2-base-ubuntu22.04` (arm64)** — 383 MB live. `-base` beats `-runtime`
  (2.52 GB) by 2.1 GB precisely because of that duplication. **jammy, not noble:** the interpreter is a
  from-source CPython 3.12.13 at `/usr/local` built against glibc 2.35; Ubuntu 24.04 (glibc 2.39) would
  force a full revalidation of all 242 native wheels for zero gain, and its distro `python3.12` installs
  to `/usr/bin` with `dist-packages` — it would not even see our site-packages.
- **The JIT toolchain is load-bearing at runtime** and is copied explicitly (247 MB): `ptxas`
  (`TRITON_PTXAS_PATH` pins Triton to the toolkit copy — GB10 fix #32704), plus `nvcc` + `nvvm/cicc`
  (FlashInfer's JIT drives nvcc for kernels absent from the cubin wheels; torch `cpp_extension` needs it).
  **Neither `-base` nor `-runtime` ships them** — only `libnvrtc`.
- **`ld.so.conf` is required, not cosmetic:** ~1000 site-packages `.so` resolve `libcudart.so.13` &c.
  through the loader path rather than RPATH, so `/etc/ld.so.conf.d/000-pip-nvidia.conf` points at the
  pip wheel dirs.
- **Five things that silently break a rebase** (each hit as a real failure during probing, not
  theorized): `ENTRYPOINT ["/bin/bash"]`; the `/usr/bin/pip` + `/usr/bin/python3` symlinks; `ninja`;
  the **unversioned** `gcc`/`g++`/`cc`/`c++`/`make` aliases (Triton and torch `cpp_extension` shell out
  to bare names at runtime, and jammy ships none of them when only `gcc-12` is installed); and the OCI
  labels. The build now **fails** if any load-bearing binary is missing.
- **Validated:** `verify.py` GREEN, `nvfp4_kv_gate.py` PASSED (NHD + HND, 3 shapes each), full AEON
  smoke battery (carries present, DSpark quantized Markov heads present, MRv2 routing intact), **and a
  real serve** — Gemma-4-26B-A4B NVFP4 + DFlash n=10 booted, captured piecewise/full/dflash CUDA
  graphs, and generated coherently.
- Not a regression, pre-existing and identical in the shipped image: `nvidia-modelopt` is absent, and
  the `deep_ep` extension fails its guarded import (stale vs the torch-2.13 ABI). Neither affects
  single-node serving; rebuild `deep_ep` if EP across two Sparks is ever wanted.
- Intermediate artifact kept for reference: `Dockerfile.multistage` (38.1 GB) builds vLLM as a wheel in
  a discarded builder stage so the 11.5 GB of in-tree CMake objects never ship.

## 2026-08-17 — Cross-node CUDA graphs validated on dual-Spark TP=2 (no image change)

Upstream [#46253](https://github.com/vllm-project/vllm/issues/46253) reports that multi-node GB10 clusters
must serve with `--enforce-eager`: CUDA-graph capture aborts, and a separate replay-time desync hangs the
engine. **Both failure modes are absent on this image**, so dual-Spark TP=2 runs with graphs ON.

- **Why it works here:** our rebased [#48053](https://github.com/vllm-project/vllm/pull/48053) carry (closed
  unmerged upstream) sets `capture_error_mode="thread_local"` at **all five** `torch.cuda.graph` sites, so
  NCCL proxy-thread work can't abort an in-flight capture. Paired with NCCL **2.30.7**, allreduce fusion off
  (`fuse_allreduce_rms: False`), `--disable-custom-all-reduce` (custom AR is a single-node/NVLink path), and
  NCCL pinned to the RoCE device.
- **Validated (2× GB10, direct 200 GbE ConnectX, RoCE v2 `rocep1s0f0`, GID index 3):** 35 piecewise + 16 full
  + 15 DSpark graphs captured in 26 s; a 64-request / 18.5k-token concurrent soak; a 6-cycle 8-way concurrent
  burn-in; **0** NCCL timeout/abort/watchdog hits across a 90-minute log sweep.
- **Measured:** single-stream **33.7 tok/s vs 26.7** eager (**+26%**), TPOT **-23%** (29.6 ms vs 38.4 ms),
  ≈ 80 tok/s aggregate at concurrency 8, 56.4 GiB pooled KV (24.9× concurrency @ 65k).
- Recipe: see *Variant: dual-Spark TP=2 over RoCE* in [AGENTS.md](AGENTS.md). Fall back to `--enforce-eager`
  on other hardware if capture misbehaves.
- Still true: GB10 has no GPUDirect, so NCCL host-stages (≈9-10 GB/s inter-Spark). TP=2 is a **capacity**
  play — double the unified memory at single-stream parity — not a latency win.

## 2026-08-16 — vLLM v0.27.1 rebuild (`:2026-08-16-v0.27.1` = `:latest`)

- Tree: `aeon-v0.27.1` = 3-way merge of tag `v0.27.1` (6e448d0ea9; **577 commits** over merge-base `dcfebf93f`) onto `aeon-v0.26.0` (merge `f780154715`). Only **7 conflicts** this cycle (vs 16–23 previously). NOTE: no upstream v0.26.1 ever shipped (only rc0); v0.27.1 = v0.27.0 + #50424.
- **THE RUNNER SWITCH:** `VLLM_USE_V2_MODEL_RUNNER=0` is **no longer baked**. DSpark is MRv2-only (v0.27.1 raises rather than falling back to V1), and mixed sliding/full DFlash drafters — **all** fleet z-lab drafters are 4-sliding+1-full — now auto-force MRv2 (`_dflash_needs_multi_kv_group`), where upstream #47914/#48113 run SWA natively via multi-KV-groups. Fleet spec decode therefore routes to MRv2; the V1 carries stay in-tree as a fallback. Re-pin `=0` per-service **only** for `thinking_token_budget` users (V2 *silently ignores* the budget).
- **Carries kept:** #44389 Triton NVFP4-KV (NHD-safe slicing views + `nvfp4_split_data_scale` upstream-name alias — upstream's as_strided math is *still* NHD-unsafe at 0.27.1; `nvfp4_kv_gate.py` now committed to this repo), #41703 ctx-slot masking (V1), #40898 SWA-on-V1 (dormant fallback), cudagraph-align widening (V1-scoped — MRv2 sizing fixed upstream by #45953), #46932 UMA clamp (issue #44740 still open; survived upstream's #49208 profiling rewrite), capture_error_mode carry **corrected to its real identity #48053** (closed-unmerged; "#46253" is the *issue*) and extended to **all 5** `torch.cuda.graph` sites for the 2×-Spark TP=2 goal. **Dropped (now upstream):** #43982 port, #50065, #49659.
- **Cherry-picks (8, merged upstream post-tag):** #50276 packed-KV zeroer stride (**correctness-critical** under our NHD packed NVFP4-KV — #47574's zeroer writes wrong addresses when block stride ≠ page size), #51812 GDN gate/spec-token alignment (Qwen3.6-27B + DFlash), #51843 hybrid-KV fine-grained prefix-cache-hit fix, DSpark set #51602/#50693/#52288/#50910/#49969 (top-k Markov projection). **Skipped:** #47808 + #52436 (depend on post-0.27.1 MRv2 refactors — #51917, routed-experts plumbing; both arrive coherently with v0.27.2), #49718 XQA-on-SM12x (GPQA accuracy regression measured on DGX Spark nightly; pending revert #51987).
- **Silent-killer caught in auto-merge** (the 0.25.0 lesson pays again): upstream 0.27.1 changed the *speculators-format* DFlash layer-id convention (`eagle_aux = aux` direct) while our tree had `eagle_aux = aux+1`; the auto-merge combined our `+1` with upstream's new `target = aux−1` → `eagle = target+2` → silent acceptance collapse for speculators-format checkpoints. Resolved to upstream wholesale after verifying **all fleet drafters are native-format** (`dflash_config.target_layer_ids`) whose `[i+1]` fallback in `eagle3_utils` is byte-identical across 0.26.0→0.27.1 — fleet semantics unchanged. Our SWA config-key propagation loop kept.
- **Deps (breaking env change #48155):** torch 2.11.0→**2.13.0+cu130** (Triton **3.7.1**, torchvision 0.28.0; torchaudio stays 2.11.0 like upstream), FlashInfer 0.6.14→**0.6.16.post3** as an exact python+cubin+**jit-cache** trio (aarch64 cu130 jit-cache wheel now exists — prebuilt JIT cache on GB10, no more purge ritual), tvm-ffi 0.1.11, quack-kernels **==0.6.1 exact** (hard-pins cutlass-dsl 4.6.0 — do **not** bump to 4.7), tilelang 0.1.12 (new hard dep), transformers 5.12.1→**5.14.1**, **nvidia-nccl-cu13==2.30.7** (TP=2 dual-Spark floor 2.30.4), torchcodec **0.16.0 re-added** (stable-ABI; build import-gates it and removes-if-broken). New env: `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (#32704). FlashInfer **0.6.17** is the queued follow-up candidate (SM121 NVFP4 attention, NVFP4-MoE accuracy fixes, GDN WY-decode SM121 fix, disk JIT cache) — one minor above upstream's pin, adopt only after an accuracy A/B.
- **DSpark bring-up notes:** heads ship in drafter repos (or in-target for DSv4/K3). Fleet coverage on the Hub: Qwen3.6-27B → `satgeze` community drafter (`num_speculative_tokens≥15`), Qwen3.6-35B-A3B → RedHatAI speculators drafter (`≥8`), **Gemma-4-26B-A4B has none** (needs in-house training — SpecForge/torchspec). `num_speculative_tokens` must be ≥ the checkpoint's `dspark_block_size` or output *garbles* (not just lower acceptance). Qwen3.6-27B hybrid + DSpark + APC: pass `--mamba-cache-mode align` (crash #52317; upstream fix #52460 pending). TP=2: #49731 (Markov-head TP replication) is in-tag; cross-node CUDA graphs are reported broken upstream (issue #46253) but are **validated WORKING on this image** — see the 2026-08-17 entry below.
- **Validated (all on GB10, all on MRv2 auto-routing — `Using V2 Model Runner` confirmed in logs):** Gemma-4-26B-A4B voice recipe (FP8 KV, triton/triton, n=10): boot clean, 64.8–87.1 tok/s single-stream greedy, DFlash mean-accept-len **2.88–3.38** @ 18.8–23.8% — matches the 0.26.0 band (3.25/20.5%), 52.4 GiB KV / 9.15× @131k. Qwen3.6-27B Multimodal-NVFP4-MTP + z-lab DFlash (n=12): 21.5–30.5 tok/s single-stream, mean-accept-len **3.40–4.82** @ 20.0–31.8%, 16/16 FULL dflash cudagraphs. **DSpark first light** (satgeze drafter, n=15, `--mamba-cache-mode align`): `Qwen3DSparkModel` resolved, quantized Markov head loaded beside the modelopt_fp4 body, `dspark_head` compiled, 16/16 FULL dspark cudagraphs, 17.1–25.6 tok/s, mean-accept-len **3.05–4.08** @ 13.7–20.6% — DFlash remains the tuned production default; DSpark is now available for tuning (AEON-tuned drafter `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` already on disk for A/B). NVFP4-KV MTP path (Multimodal-NVFP4-MTP body, `--kv-cache-dtype nvfp4`, qwen3_5_mtp n=3): boots + generates coherently, **1,054,037 KV tokens in 40.1 GiB = 32.2× concurrency @32k** (vs 525k tokens / 35.7 GiB under FP8 on the same body — the capacity unlock, live). Known-stale: base-image `deep_ep` .so (torch-2.11 ABI) fails its guarded import — harmless single-node; rebuild if EP over 2 Sparks is ever wanted.
- Rollback: `:2026-07-27-v0.26.0`.

## 2026-07-27 — vLLM v0.26.0 rebuild (`:2026-07-27-v0.26.0`)

- Tree: `aeon-v0.26.0` = 3-way merge of tag `v0.26.0` (568afb3a1, **429 commits**) onto `aeon-v0.25.0`. 16 conflicts. NOTE: the true merge base is `6db31c8e7`, NOT v0.25.1 — v0.25.1 is a release-branch tag and its backports appear as "ours", explaining ~5 conflicts in files we never touched.
- **Carries kept:** #44389 Triton NVFP4-KV (**REWRITTEN** for the new 4-D layout), #40898 DFlash SWA on V1, #41703 ctx-mask + Gemma4 bits, dflash-blocktable-unpad, cudagraph_align, UMA clamp, use_mm_prefix, requires_eagle_cache_drop. **New:** NVFP4_AWQ support, #46253 cudagraph `capture_error_mode`, #50065 padded-batch draft buffers, #49659 MoE-gated router warmup. **Dropped (now upstream):** our lm_head fix (#47914 merged), #47053, #47356, #45207, #48330.
- **#44455 re-based the KV layout** 5-D -> 4-D packed. Our NVFP4-KV views are re-derived using plain SLICING, never upstream's `as_strided` offset math: that math only tiles correctly under HND, and **GB10 defaults to NHD** (FlashInfer forces HND only when `capability.major == 10`, i.e. SM100). Measured: upstream's formulation covers 7760/9216 bytes with **5 overlapping views** = silent KV corruption; ours covers 9216/9216 with none, under BOTH layouts. Pinned by a build gate (`nvfp4_kv_gate.py`) that fails the image on regression.
- **Upstream now REJECTS our drafters on V1** (`qwen3_dflash.py` raises NotImplementedError for mixed sliding/full DFlash), and both our drafters are 4-sliding+1-full. Resolved as a union: our #40898 path stays (their raise is defined but has zero callers) plus their `_SLIDING_ATTENTION` / `_dflash_layer_causal` / `dflash_has_any_non_causal`, which the auto-merged V1 proposer imports.
- **Build gotchas (each caught only by a real model boot, never by import gates):** v0.26.0 bumps `cutlass-dsl` 4.5.2->4.6.0 **together with** `quack-kernels` (base's 0.4.1 satisfies `>=0.4.0` but calls `cute.core.ThrMma`, removed in 4.6 -> AttributeError) **and** `apache-tvm-ffi` 0.1.9->0.1.10 (missing it kills `_warmup_ll_bf16_router_gemm` for every model). Pin all three; the build now asserts the ABI. FlashInfer 0.6.14 needs `--extra-index-url https://flashinfer.ai/whl/` (PyPI carries cubin only to 0.6.13).
- **`VLLM_ENABLE_STARTUP_PLAN=1` is boot-breaking on GB10** (NVML has no memory support on unified memory).
- **Validated:** Gemma-26B 1151.5 tok/s @128 concurrent at `max-num-seqs 128` (+46% vs the 32 cap), 0 errors, acceptance 3.25/20.5%; Qwen3.6-27B-MTP per-category 16.0-39.6 tok/s; Gemma-4-31B-DECKARD NVFP4_AWQ 120/120 pre_quant_scale + coherent output. Rollback: `:2026-07-16-v0.25.1`.

# vLLM source pin

Build was against:

- **Repo**: `lesj0610/vllm`
- **Branch**: `lesj/triton-nvfp4-kv-fork-20260602`
- **Commit**: `e8c77b85`
- **Upstream PR**: [vllm-project/vllm#44389](https://github.com/vllm-project/vllm/pull/44389) — Triton software NVFP4 KV cache (~3× capacity)

To reproduce the build:

```bash
git clone https://github.com/lesj0610/vllm.git vllm-src
cd vllm-src
git checkout lesj/triton-nvfp4-kv-fork-20260602
git checkout e8c77b85
cd ..
# Then docker build -t aeon-vllm-ultimate:latest .
```

The full source is not vendored in this repo (~140 MB) — only the patches, Dockerfile, humming-stub, verify script, bench tooling, and bench artifacts.

## 2026-07-16 — v0.25.1 + MRv2 lm_head fix build (`:2026-07-16-v0.25.1` = `:latest`)

- vLLM tree: `aeon-v0.25.0` @ `03cfc9d50` = the 2026-07-14 merge (`e876fec71`) + **MRv2 spec-decode lm_head sharing fix** (`9e005dca5`) + **merge of tag `v0.25.1`** (752a3a504; conflict-free, zero file overlap with carries). v0.25.1 upstream fixes: [#47888](https://github.com/vllm-project/vllm/pull/47888) torchcodec import no longer blocks startup when FFmpeg is absent (moot for us — torchcodec intentionally not installed); [#48330](https://github.com/vllm-project/vllm/pull/48330) guards mixed-dtype FlashInfer allreduce+RMSNorm+quant fusions that corrupted hidden state into `!!!!!` garbage on **NVFP4 Gemma/Qwen-style models under TP>1** — exactly the guard our dormant wired-in TP=2 path needed for the NVFP4 fleet (no effect at TP=1). Version string: `0.25.1+aeon.sm121a.dflash`.
- **MRv2 spec-decode lm_head sharing fix** (port of upstream [#47914](https://github.com/vllm-project/vllm/pull/47914)). The V2 model-runner loaders read `getattr(target_model, "lm_head")` off the outer module; `*ForConditionalGeneration` wrappers (Qwen3.6-27B) nest it under `language_model` → `None` → draft head stays zero-initialised → **0% DFlash acceptance on MRv2**. Fixed to prefer `target_language_model` at all 3 sites: `v1/worker/gpu/spec_decode/{dflash,eagle,dspark}/utils.py`. MTP unaffected (heads ship in its checkpoint).
- **Phase-2 MRv2 audit outcome** (adversarially-verified source audit + same-image hardware A/B on GB10): the lm_head bug was the *only* real MRv2 blocker. Blocktable-unpad — structurally impossible on MRv2 (single `num_reqs_padded` slicing). #41703 rejected-slot masking — benign on MRv2 (same-step drafter forward overwrites those slots). #40898 SWA — absent on MRv2 (drafter runs non-causal) but **zero measured acceptance impact**. A/B: V1 vs V2 mean-accept-len 2.75–4.52 vs 2.82–4.28 (short), 23.8k-token long-context healthy with exact needle retrieval on both.
- **`VLLM_USE_V2_MODEL_RUNNER=0` stays baked — by choice.** Correction to the audit: the 27B is `IsHybrid` (48/64 linear-attention layers) → excluded from default-V2 like the MoE models, so even unpinned, NO fleet model routes to V2 by default; the pin guards against upstream default changes + forced opt-ins. Explicit V2 (env=1) buys zero gain (DSpark lacks a fleet drafter; dynamic-SD −12–16% on GB10; hybrid-APC gated out of default-V2; `thinking_token_budget` silently ignored on V2). The fix makes MRv2 a correct fallback instead of a broken one.
- Prod (V1) behavior is **unchanged** from `:2026-07-14-v0.25.0` — the fixed code paths only execute on MRv2. Validated before push: 27B on MRv2 with the *baked* fix (0% → healthy acceptance), Gemma-26B voice recipe on V1 (boot + gen + DFlash acceptance). Rollback: `:2026-07-14-v0.25.0`.

## 2026-07-14 — v0.25.0 rebuild (`:2026-07-14-v0.25.0`)

- vLLM tree: `aeon-v0.25.0` = 3-way merge of tag `v0.25.0` (702f4814) onto `codex/aeon-v0.24.0-maxsafe-20260708` (v0.24.0 + prod cherry-picks). 23 conflicts: 12 pure-upstream new-model files (MiniMax-M3, moss_audio) taken as-is; 11 carry-critical files (triton NVFP4-KV cluster + runner/spec-decode cluster + docs) integrated by adversarially-verified cluster resolution.
- **Carries kept:** #44389 (Triton NVFP4-KV), #40898 (DFlash SWA), #41703 (ctx-mask/Gemma4 batched-verify), dflash-blocktable-unpad, cudagraph_align_all_modes, UMA negative-estimate clamp, plus maxsafe carries #47356 (kv_cache_memory_bytes cache-hash exclusion), #45207 (Mamba page-pad), #47053. **Dropped:** #45544 tie_weights (now upstream). **Integrated from v0.25.0:** #42890 (NVFP4-KV SWA page-unify), #46761 (DFlash RMSNorm fusion), #45739 (NVFP4 swizzled-scale zero-init).
- **Two silent-killer merge bugs caught + fixed:** #42890 renumbered `KVQuantMode.NVFP4` 4→5, our kernel's hard-coded `USE_NVFP4 = KV_QUANT_MODE == 4` sat in a non-conflict region → would have disabled ALL NVFP4 KV (fixed to `== 5`, enum value independently confirmed); two auto-merge SyntaxErrors in qwen3_dflash.py (duplicated `sliding_window` param + kwarg).
- **MRv2 pin:** `VLLM_USE_V2_MODEL_RUNNER=0` baked (Phase 1) — v0.25.0 whitelists `method=dflash` for MRv2 with no fallback, so without the pin dense models silently lose our DFlash patches. (Phase 2 verdict, 2026-07-16: pin KEPT by choice after the MRv2 audit — see the entry above.)
- **Deps:** FlashInfer 0.6.12→0.6.13 (purge stale jit-cache), cutlass-dsl 4.5.2, torch 2.11.0 unchanged. torchcodec omitted (0.14 ABI-broken on torch 2.11; vLLM guards the import). Kept 12.1a arch (bare 12.1 → Marlin fallback), GCC 12, transformers 5.12.1, xgrammar>=0.2.1. humming-stub made permissive (v0.25.0 registry touches new humming dtypes/schema submodules).
- **A/B before push (GB10, vs :2026-07-08-v0.24.0-maxsafe):** Gemma-4-26B 505–511 vs 559 tok/s @c16 (parity, CUTLASS FP4); Qwen3.6-35B-A3B 341 vs 348 @c12 (parity, Marlin — checkpoint lacks CUTLASS scales); Qwen3.6-27B 88 vs 100 @c8 (parity, CUTLASS). All: DFlash on V1, tools working, acceptance healthy. Artifacts: `AB_SUMMARY_v0250.md`. Rollback: `:2026-07-08-v0.24.0-maxsafe`.

## 2026-07-02 — v0.24.0 rebuild (`:2026-07-01-v0.24.0` = `:latest`)

- vLLM tree: `aeon-v0.24.0` branch = 3-way merge of tag `v0.24.0` (ee0da84ab) into
  `aeon-dflash-fix` (the v0.23.0-based AEON tree). 11 conflicted files, all in the
  carried #44389 (triton_attn/unified-attention NVFP4-KV) and #40898/#41703
  (runner/scheduler/warmup) footprints; resolutions integrate BOTH sides (all AEON
  NVFP4/DFlash machinery + upstream's non-causal Triton, causal bool|Tensor plumbing,
  fused multi-group staged block-table writes).
- Former runtime patches now COMMITTED IN SOURCE (patch scripts retired):
  - `dflash-blocktable-unpad` — `[: cad.num_reqs]` slice in `_get_dflash_block_table`
    (port of merged PR #43982, which only fixed the gemma4-MTP proposer).
  - `cudagraph_align_spec_decode_all_modes` — widen spec-decode capture-size alignment
    beyond `decode_mode()==FULL` (open upstream twin: PR #46324).
- `patch_cuda_optional_import` DROPPED: the v0.24.0 stable-ABI migration arch-gates the
  formerly ungated sm_100-only kernel registrations; replaced by a build-time dlopen
  smoke test against the CUDA driver stub (catches regressions at build, not serve).
- Post-tag fixes carried:
  - Cherry-pick `ad28d605e` (merged PR #45544): default `tie_weights` to weight sharing —
    without it every tied-embedding ModelOpt checkpoint (all Gemma-4) crashes at load
    with `NotImplementedError`.
  - Port of open PR #46932: clamp cudagraph memory estimates to >= 0 on unified-memory
    GPUs (GB10 issue #44740 — negative estimates inflate the KV budget and OOM).
  - `use_mm_prefix` added to the carried `supports_combination` overrides in
    `triton_attn.py`/`flashinfer.py` (upstream widened the base signature; the stale
    overrides crashed backend validation with a TypeError at engine start).
- Dependency changes: FlashInfer 0.6.8.post1 → **0.6.12** (v0.24.0 pin; 0.24 lazy-imports
  `flashinfer.fused_moe` b12x symbols absent from 0.6.8), transformers git-HEAD →
  **pinned 5.12.1** (first stable release covering the whole fleet; smoke-tested against
  every fleet architecture before the build), GCC 12 host compiler (#44923 C++20).
- Validation before push (all on GB10): Ornith-35B (GDN hybrid MoE NVFP4 + DFlash
  multi-KV-group) A/B vs the v0.23.0 image at parity (c=1: 81.6 vs 82.7 tok/s; c=12:
  441.8 vs 457.9 agg — within run-to-run variance); DFlash concurrency sweep clean at
  c=16/32/64 (no block-table crash); Gemma-4-12B K4-MIXED with `--kv-cache-dtype nvfp4`
  on the Triton backend boots + generates; Gemma-4-26B-A4B voice stack (triton_attn +
  DFlash flash_attn drafter n=10 + `--linear-backend flashinfer_cutlass`) healthy,
  pos0 acceptance 60–86%. Sweep artifacts: `sweep_prodcfg_v0240.json`,
  `sweep_conc_v0240.json`, `sweep_v0230_ab.json`.

## 2026-06-11 — PR #40898 + #41703 overlay (`:2026-06-11-pr41703` = `:latest`)

DFlash drafter fixes merged ahead of upstream (both PRs open at merge time; the z-lab
drafter README pins the #41703 revision):
- vLLM tree: `aeon-dflash-fix` branch = `main@2026-06-05 merge (542fe78)` + merge of
  `pull/41703/head` (contains #40898). 5 conflicts resolved; key resolution: kept the PR's
  KV-shape helper structure but re-grafted PR #44389's per-spec KV dtype
  (`get_attn_backend_cache_dtype_str`) at both `_get_attention_kv_cache_shape` call sites,
  and re-established `shape_block_size`/`cache_dtype_str` for the MLA `page_size_padded` branch.
- Both PRs touch only Python (the DFlash kernel is Triton), so the image is a thin overlay:
  see `Dockerfile.pr41703-layer` (copies 11 files into site-packages, re-applies the AEON
  patches — the merge touches `kv_cache_utils.py` — and smoke-asserts the fixes are present).
- ⚠️ Drafter `attention_backend` must be `flash_attn` on this image; `flex_attention` crashes
  on a non-contiguous KV view (upstream's KV-sharing path is only tested with flash_attn).

## Build it yourself (advanced)

Most users should just pull the prebuilt image (`docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest`). To reproduce it from source:

**Prereqs:** a DGX Spark (GB10 / sm_121a) or another Blackwell sm_120/121 box, ~30 GB free disk, ~60–90 min wall clock. The `12.1a` arch tag means the resulting image runs on the sm_121a GPU it was built for.

**Build:** clone the vLLM source per the pin above into `vllm-src/`, then build against the `Dockerfile` in this repo (the `Dockerfile.pr41703-layer` overlay carries the PR #40898/#41703 DFlash fixes — see the dated section above and the README's *Build provenance* for which source/overlay maps to which tag):

```bash
docker build -t aeon-vllm-ultimate:latest .
```

**Build knobs** (defaults are set in the Dockerfile, tuned for a ~20-core / 128 GB Spark):

| Env var | Default | Notes |
|---|---|---|
| `MAX_JOBS` | `12` | Compile parallelism. **Lower to 8/6 if the build OOMs.** |
| `NVCC_THREADS` | `2` | Per-`nvcc` threads. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `8` | CMake parallelism. |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | GB10 / sm_121a target. |
| `ENABLE_NVFP4_SM100` | `0` | Skips SM100-only NVFP4 kernels that fail to compile on SM121. |

The Dockerfile installs the CUDA 13.0 dev headers (`cuda-nvrtc-dev-13-0`, `libcusparse/cublas/cusolver/cufft/curand/nvjitlink-dev-13-0`), builds vLLM from the COPY'd `vllm-src/`, applies the three idempotent AEON sm_121a patches (`patch_cuda_optional_import`, `patch_kv_cache_utils`, `patch_cudagraph_align`), then layers TurboQuant (AEON-7 fork) + transformers HEAD + the `humming-stub`.

**Build troubleshooting:**

- `nvcc fatal: Unsupported gpu architecture` — your CUDA toolkit is too old; this build needs **CUDA ≥ 13.0** (the Dockerfile installs the `*-dev-13-0` headers).
- `RuntimeError: CUDA out of memory` during compile — lower `MAX_JOBS` (e.g. `--build-arg MAX_JOBS=8`).
- First build appears to "hang" generating CUDA stubs — that's normal (nvcc is compiling hundreds of objects); confirm progress with `docker stats` / `htop`.

**Verify** (the build runs `verify.py` automatically; to re-check manually):

```bash
docker run --rm aeon-vllm-ultimate:latest python3 -c "import vllm; print(vllm.__version__)"
```

No registry patch is needed — unlike the old `vllm-spark-omni-q36` image, the unified build loads the Qwen3.5/3.6 and Gemma-4 multimodal classes natively.
