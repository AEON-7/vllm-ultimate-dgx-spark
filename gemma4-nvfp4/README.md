# Gemma-4-12B NVFP4 quantization (for vLLM Gemma4Unified loader)

Working NVFP4 recipe for Google's encoder-free **Gemma-4-12B**
(`Gemma4UnifiedForConditionalGeneration`), validated end-to-end in vLLM
0.22.2 on DGX Spark GB10 (sm_121a) at **341 tok/s aggregate** (concurrent×16,
2.37× the BF16 model).

## The two-step recipe

```bash
# 1. Quantize: modelopt 0.43.0, NVFP4_DEFAULT_CFG, 2048 CNN/DailyMail calib @ 1024 tok
python3 quantize_k4_nvfp4.py \
    --src  /path/to/Gemma-4-12B-AEON-K4-BF16 \
    --dst  /path/to/Gemma-4-12B-AEON-K4-NVFP4 \
    --config-name NVFP4_DEFAULT_CFG \
    --num-calibration-samples 2048

# 2. Make it vLLM-Gemma4Unified-ready (vision keys + ignore list + tokenizer)
python3 make_vllm_ready.py \
    --model /path/to/Gemma-4-12B-AEON-K4-NVFP4 \
    --src   /path/to/Gemma-4-12B-AEON-K4-BF16
```

Then serve with `vllm serve <dst> --quantization modelopt`.

## Why step 2 is needed

Google's Gemma-4-12B uses the **encoder-free** `Gemma4UnifiedForConditionalGeneration`
architecture. ModelOpt's HF export produces two naming/structure quirks that
vLLM's native `gemma4_unified` loader rejects:

1. **Vision key naming** — ModelOpt wraps `embed_vision.embedding_projection`
   in a `multimodal_embedder` sublayer and keeps the patch embedder under
   `embed_vision.*`; vLLM wants the projection flat and the patch embedder
   under `vision_embedder.*`.
2. **Ignore-list gap** — after the rename, `model.vision_embedder*` must be in
   the quantization `ignore` list so vLLM keeps the patch embedder at BF16.
   Otherwise vLLM builds a packed-FP4 `patch_dense` layer (input dim halved:
   6912→3456) that mismatches the unquantized BF16 weights in the checkpoint.

`make_vllm_ready.py` fixes both + copies the tokenizer (which ModelOpt's
export omits). The diagnosis came from instrumenting vLLM's weight loader to
print the exact shape mismatch — the smoking gun was
`vision_embedder.patch_dense param.shape=(3840,3456) loaded.shape=(3840,6912)`.

## Requirements

- vLLM ≥ 0.22.2 (for the `gemma4_unified` loader)
- nvidia-modelopt 0.43.0
- transformers ≥ 5.10
- Blackwell GPU (sm_121a / sm_120 / sm_100)

Published model: [AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4)
