# AGENTS.md — Gemma-4-12B-it AEON Abliterated K=4 (NVFP4)

Instructions for autonomous agents on how to **download, serve, prompt, and
verify** this NVFP4 model. The companion to the README QuickStart.

## What this is

A 4-bit **NVFP4** (ModelOpt `NVFP4_DEFAULT_CFG`) quantization of
[AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16),
a K=4 multi-direction biprojection abliteration of `google/gemma-4-12B-it`.
~8.5 GB. Loads natively in vLLM ≥ 0.22.2 via `Gemma4UnifiedForConditionalGeneration`.

## Hard requirements

- **GPU**: Blackwell — DGX Spark GB10 (`sm_121a`), B100/B200 (`sm_100`), or RTX 50-series (`sm_120`). NOT Hopper/Ampere (FP4 dequantizes to BF16, no benefit → use the BF16 sibling).
- **vLLM ≥ 0.22.2** — earlier versions lack the `gemma4_unified` loader. The [AEON vLLM Ultimate](https://github.com/AEON-7/vllm-ultimate-dgx-spark) container ships it for `sm_121a`.
- **nvidia-modelopt ≥ 0.43** — vLLM's `--quantization modelopt` deserializer.
- **transformers ≥ 5.10** — for the `gemma4_unified` architecture.
- **disk** ≥ 9 GB.

## Download

```bash
huggingface-cli download AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4 \
  --local-dir ./Gemma-4-12B-AEON-K4-NVFP4
```

The checkpoint is **self-contained** — vision keys are already in vLLM's
expected layout, the ignore list already excludes the vision embedder, and
the tokenizer is bundled. No post-processing needed.

## Serve (Docker — recommended)

```bash
docker run -d --name aeon-gemma12b --gpus all --ipc=host --shm-size=16g --net=host \
  -v $(pwd)/Gemma-4-12B-AEON-K4-NVFP4:/model:ro \
  --entrypoint vllm \
  ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model \
    --served-model-name gemma12b \
    --quantization modelopt \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 8192 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --trust-remote-code
```

**Key flags**:
- `--quantization modelopt` — REQUIRED. The checkpoint is ModelOpt-format NVFP4.
- `--kv-cache-dtype fp8_e4m3` — recommended; matches the published benchmarks.
- `--max-num-seqs 16` — the model fits ~155× concurrency at 8k ctx on a Spark; raise freely.
- `--tool-call-parser gemma4` — enables structured tool calls.

## Health probes

```bash
curl -fsSL http://localhost:8000/health && echo OK            # liveness
curl -fsSL http://localhost:8000/v1/models | python3 -m json.tool  # readiness
curl -fsSL http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma12b","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## Expected boot sequence (so you know it's healthy)

```
Resolved architecture: Gemma4UnifiedForConditionalGeneration
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
Loading safetensors checkpoint shards: 100%
GPU KV cache size: ~1,270,000 tokens
Maximum concurrency for 8,192 tokens per request: ~155x
Application startup complete.
```

If you see `GPU KV cache size: ~1.27M tokens` and `155x concurrency`, the NVFP4 weights loaded correctly. If KV cache is ~540k / 65x, you accidentally loaded the BF16 model.

## Throughput expectations (DGX Spark GB10, FP8 KV)

- Concurrent ×16 steady-state: **~338 tok/s aggregate** (2.35× the BF16 model)
- Steady-state TTFT: ~180 ms
- Per-stream decode: 15-29 tok/s depending on prompt category (summary fastest, math slowest)

## Prompt expectations

- **Voice**: Gemma-4 instruct — markdown-rich, English-reliable.
- **Refusal**: removed. On topics the base would decline, expect a 1-3 sentence disclaimer preamble, then full compliance.
- **Sampling**: `temperature=0` for deterministic; `0.7 / top_p=0.9` matches base chat behavior.
- **Tool calls**: produces Gemma-4 tool-call JSON without hesitation.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ModelOpt currently only supports [...]` | vLLM < 0.22.2 or stale modelopt | Use `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` |
| `no module named Gemma4UnifiedForConditionalGeneration` | vLLM < 0.22.2 | Same — needs the `gemma4_unified` loader |
| shape mismatch on `vision_embedder.patch_dense` | hand-edited the checkpoint and broke the ignore list | Re-download; the published copy is correct |
| `Can't load feature extractor` | missing `processor_config.json` | Re-download; it's bundled |
| slow / no FP4 speedup | running on Hopper/Ampere | Use the BF16 sibling instead |

## Reproducing the quantization

The full recipe + post-processing scripts live in
[github.com/AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark):

```bash
# modelopt 0.43.0, NVFP4_DEFAULT_CFG, 2048 CNN/DailyMail calib samples @ 1024 tok
python3 quantize_k4_nvfp4.py --src <bf16> --dst <out> \
  --config-name NVFP4_DEFAULT_CFG --num-calibration-samples 2048
# then make it vLLM-Gemma4Unified-ready (vision keys + ignore list + tokenizer)
python3 make_vllm_ready.py --model <out> --src <bf16>
```

## Fine-tuning

Don't fine-tune the NVFP4 weights directly — start from the [BF16 sibling](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16), train, then re-quantize.

## License + safety

- Inherits the [Gemma license](https://ai.google.dev/gemma/terms).
- Operator-side safety layers are **required** for production — see the README arbitration clause.
- Refusal removal means the duty of care is yours, not the model's.

## Support the work

Tip-jar addresses are in the README. Compute donations keep more Blackwell-native quants coming.
