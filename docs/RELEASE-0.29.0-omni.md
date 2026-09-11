# AEON vLLM Ultimate — `0.29.0-omni`

**Rebase target:** upstream `vllm-project/vllm` tag `v0.29.0` (2026-09-09).
**Branch:** `rebase/0.29.0-omni` from fleet tip `feat/reasoning-eos-force-end` @ `931d21fc`.

## Hard keeps (Spark + TP=2 + MM)

- sm_121a (`TORCH_CUDA_ARCH_LIST=12.1a`), CUDA 13, aarch64 GB10
- `#48053` `capture_error_mode="thread_local"` at CUDA-graph capture sites
- Triton NVFP4-KV NHD-safe splitter (no `as_strided`)
- DFlash2 / DSpark carries retained where still required after 0.29 ancestry check
- ModelOpt MIXED (`modelopt_mixed` / `FP8_PER_CHANNEL_PER_TOKEN`) — in upstream 0.29
- `reasoning_eos_policy=force_end` (AEON default)
- Dual-Spark RoCE recipe: `--disable-custom-all-reduce`, mp nnodes,
  `NCCL_IB_HCA=<dev>:1` (single `=`), `VLLM_ALLREDUCE_USE_FLASHINFER=0`,
  `fuse_allreduce_rms=False` (#51292-class / #52998 default-on FI AR opt-out)
- FULL multimodal: vllm-omni layer, vision, audio extras, 24→16 kHz `load_audio` bake gate,
  torchcodec/speech — no text-only regression (`VLLM_PLUGINS=` still works)

## Dep bumps vs 0.27.1-omni

- FlashInfer trio `0.6.16.post3` → `0.6.18`
- cutlass-dsl `4.6.0` → `4.6.2`, quack-kernels `0.6.1` → `0.6.4`
- torch stays `2.13.0+cu130` per upstream cuda.txt

## Image tag

`ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-11-v0.29.0-omni` (do not retag `:latest` until A/B passes)
