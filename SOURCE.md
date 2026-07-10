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
- Hikari07jp / DSpark Qwen3.6 DFlash compatibility overlay added in
  `overlays/hikari-dflash/`: ports the Markov transition-bias head
  (`markov_head.markov_w1.weight` / `markov_head.markov_w2.weight`) into the current
  `qwen3_dflash.py` while preserving the v0.24.0 proposer, multi-KV-cache-group, and
  cudagraph paths. Smoke-validated on DGX Spark with
  `Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft`: model loads, CUDA graphs capture, and a
  real `/v1/chat/completions` decode returns HTTP 200.
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

The Dockerfile installs the CUDA 13.0 dev headers (`cuda-nvrtc-dev-13-0`, `libcusparse/cublas/cusolver/cufft/curand/nvjitlink-dev-13-0`), builds vLLM from the COPY'd `vllm-src/`, copies the Hikari DFlash Markov compatibility overlay into the installed package, then layers FlashInfer 0.6.12, TurboQuant (AEON-7 fork), pinned transformers, xgrammar, and the `humming-stub`.

**Build troubleshooting:**

- `nvcc fatal: Unsupported gpu architecture` — your CUDA toolkit is too old; this build needs **CUDA ≥ 13.0** (the Dockerfile installs the `*-dev-13-0` headers).
- `RuntimeError: CUDA out of memory` during compile — lower `MAX_JOBS` (e.g. `--build-arg MAX_JOBS=8`).
- First build appears to "hang" generating CUDA stubs — that's normal (nvcc is compiling hundreds of objects); confirm progress with `docker stats` / `htop`.

**Verify** (the build runs `verify.py` automatically; to re-check manually):

```bash
docker run --rm aeon-vllm-ultimate:latest python3 -c "import vllm; print(vllm.__version__)"
```

No registry patch is needed — unlike the old `vllm-spark-omni-q36` image, the unified build loads the Qwen3.5/3.6 and Gemma-4 multimodal classes natively.
