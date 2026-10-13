# reasoning_eos mid-think EOS fix (AEON ship 2026-09-07)

## Upstream
- Issue: https://github.com/vllm-project/vllm/issues/55420
- PR: https://github.com/vllm-project/vllm/pull/55562
- Head commit: d91bb5c02fcbd65241e6f0b1b9bbcaf25eba9970
- Title: [Feature] Add reasoning_eos_policy to close think on mid-reasoning EOS

## Symptom
Qwen3.8 can sample `<|im_end|>` inside `<think>` → finish_reason=stop with empty content.

## AEON backport (thin layer on 0.27.1-omni)
- Python-only overlay (V1 thinking_budget_state path; V2 runner off)
- `reasoning_eos_policy` sampling param; **AEON default `force_end`** (upstream PR defaults to `stop`)
- Folded #54367 `modelopt.py` into same layer (no bind-mount)

## Image
- `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-09-07-reasoning-eos`
- Did **not** retag `:latest` (dated-tag practice)

## Serve bounce
- Container: `qwen38-l8gateup-mse-bench`
- Recipe: util 0.70, max-model-len 131072, max-num-seqs 16, max-num-batched-tokens 16384, kv fp8, TRITON_ATTN, chunked prefill, no prefix cache, DFlash n=7, VLLM_USE_V2_MODEL_RUNNER=0
- No modelopt bind-mount

## Attested
- Held pending mothership queue cleanup (do not launch yet)
