# RELEASE — aeon-vllm-ultimate 0.29.0-omni

**Tag:** `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-11-v0.29.0-omni`  
**Digest:** `sha256:2421bb1228a85370c1c50adb31f605c4361acf4d48d65282fcb919e74f34fae7`  
**Base:** upstream vLLM **v0.29.0**, compiled for **sm_121a** (DGX Spark / GB10)  
**Prior tip:** `2026-09-07-reasoning-eos` (v0.27.1-omni + reasoning EOS + ModelOpt fold)

## What this image is for

Same job as every AEON Ultimate Spark image: one container that runs the fleet on GB10 without fighting the hardware. This cut moves the engine from **0.27.1** to **0.29.0**, keeps the Spark carries that actually matter, and makes **two-Spark tensor-parallel** stop turning into gibberish when you set the right knobs.

## What changed (plain language)

### 1) Upstream leap: vLLM 0.29.0
You get the 0.29 engine — newer kernels, newer runner defaults, and a pile of TP / all-reduce / speculative fixes that landed between 0.27 and 0.29. The Spark bake still compiles **native sm_121a** so NVFP4 stays on CUTLASS instead of quietly falling back to Marlin.

### 2) Dual-Spark TP=2 that stays coherent
On 0.27-era dual Spark we kept hitting “looks alive, speaks salad.” On this 0.29 image, the B0 recipe that cleared it is:

- `VLLM_ALLREDUCE_USE_FLASHINFER=0` (0.29 turns FlashInfer all-reduce **on** by default — leave it off across RoCE until you prove otherwise)
- `--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'`
- `--disable-custom-all-reduce`
- `NCCL_IB_HCA=<roce-dev>:1` with a **single** `=` (a double `==` silently breaks the path)
- ModelOpt `#54367`-style bind on **both** ranks for MIXED
- Optional: `--enforce-eager` while you validate; CUDA-graph TP=2 remains a later tighten

Smoke on that recipe answered cleanly (exact string, capital, counting) — not the old TP=2 cake-spam. Full God Mode on that lane is the long soak; pin the dated tag and use the B0 knobs for TP=2 today.

### 3) Single-Spark TP=1: as good or better than 0907
Same-box A/B against `2026-09-07-reasoning-eos` on MIXED: quality held, decode slightly up (~9.3 → ~9.6 tok/s on the fixed-weights smoke). Not a regression cut.

### 4) What still rides with you
- DFlash / DFlash2 speculative path (and Dynamic DFlash2 for Perf ladders)
- Reasoning EOS force-end behavior from the 0907 line
- NVFP4 / ModelOpt MIXED serving (see caveat below)
- Multimodal hooks and the Omni layer (see caveat below)
- `#48053` thread_local CUDA-graph capture carry for cross-node graphs

## Honest caveats (read these)

1. **MIXED still wants the ModelOpt bind on 0.29.** In-tree ModelOpt on this bake does **not** load the MIXED lattice alone (`MergedColumnParallelLinear` / `.data`). Bind the `#54367`-style `modelopt.py` the same way 0907 recipes did, on every rank.
2. **Omni speech plugin may warn about `aenum`.** Text / vision serve still run; if you need Omni speech-out, install/fix `aenum` or wait for the next bake that folds it.
3. **`:latest` promotion.** Intended tip is this dated tag. Until GHCR write auth finishes the retag, **pin** `2026-09-11-v0.29.0-omni` (or pull by digest). Do not assume Docker Hub / GHCR `:latest` has moved until you verify the digest.

## Pull

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-11-v0.29.0-omni
# digest: sha256:2421bb1228a85370c1c50adb31f605c4361acf4d48d65282fcb919e74f34fae7
```

## Minimal TP=2 sanity knobs

```bash
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
# NCCL_IB_HCA=rocep1s0f0:1   # one equals sign
# --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
# --disable-custom-all-reduce
# bind modelopt.py on BOTH ranks for MIXED
```

## Rollback

```bash
docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-07-reasoning-eos
```

---

Built for people who own the box and the weights. Pin digests in production; dated tags beat folklore.
