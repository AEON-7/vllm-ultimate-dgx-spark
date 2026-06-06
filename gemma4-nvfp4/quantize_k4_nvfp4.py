"""Quality-preserving NVFP4 quantization of Gemma-4-12B-AEON-K4-BF16.

Recipe: modelopt 0.43.0 + NVFP4_AWQ_LITE_CFG
  - AWQ_LITE adds per-channel pre-quantization scaling that absorbs
    Gemma-4's documented per-channel attention outliers (AxionML choice
    to use weight-only FP8 was specifically to handle this; AWQ scaling
    inside NVFP4 addresses the same issue at FP4 block resolution).
  - block_size=16 (NVFP4 default) keeps quantization error localized.
  - E4M3 dynamic block scales preserve dynamic range.

Excludes (all kept BF16):
  - lm_head (tied to embed_tokens, mandatory exclude)
  - model.embed_audio* (audio embedder — vLLM-incompatible dim anyway)
  - model.embed_vision* (vision embedder — preserves multimodal)
  - model.language_model.embed_tokens (input embedding, NVIDIA best-practice)
  - vision_embedder* + embed_vision* (defensive — covers any key remap)

Calibration:
  - CNN/DailyMail validation, 2048 samples × 1024 tokens (NVIDIA standard)
  - Native sm_121a calibration on DGX Spark for hardware-accurate scales

Output: ~6.5 GB safetensors with quant_algo=NVFP4 (NOT NVFP4_SVD).
        Loads in stock vLLM with --quantization modelopt.
"""
import argparse
import gc
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration
from datasets import load_dataset
import modelopt.torch.quantization as mtq
import modelopt.torch.export as mte
import modelopt.torch.opt as mto


EXCLUDE_PATTERNS = [
    "lm_head",
    "model.embed_audio*",
    "model.embed_vision*",
    "model.language_model.embed_tokens",
    "vision_embedder*",
    "embed_vision*",
]


def install_excludes(config: dict, patterns: list[str]):
    """Append exclude rules to the modelopt config in whatever schema applies."""
    quant_cfg = config.get("quant_cfg")
    if isinstance(quant_cfg, list):
        for pat in patterns:
            quant_cfg.append({"quantizer_name": pat, "enable": False})
        return f"appended {len(patterns)} exclude rules to list-form quant_cfg"
    elif isinstance(quant_cfg, dict):
        for pat in patterns:
            quant_cfg[pat] = {"enable": False}
        return f"merged {len(patterns)} exclude rules into dict-form quant_cfg"
    return f"WARN: quant_cfg has unexpected type {type(quant_cfg)}"


def load_model(src: str):
    print(f"[load] {src}", flush=True)
    t0 = time.perf_counter()
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"  load_time={time.perf_counter()-t0:.1f}s  device={model.device}", flush=True)
    return model


def make_calibration_loop(tok, calib_samples: list[str]):
    def loop(model):
        n = len(calib_samples)
        for i, text in enumerate(calib_samples):
            inp = tok(
                text, return_tensors="pt",
                max_length=1024, truncation=True, padding=False,
            )
            inp = {k: v.to(model.device) for k, v in inp.items()}
            with torch.no_grad():
                model(**inp)
            if (i + 1) % 100 == 0:
                # release fragmenting activation memory
                torch.cuda.empty_cache()
                gc.collect()
                print(f"  calib {i+1}/{n}  GPU mem {torch.cuda.memory_allocated()/1e9:.1f}GB",
                      flush=True)
        # final cleanup
        torch.cuda.empty_cache()
        gc.collect()
    return loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/albert/models/Gemma-4-12B-AEON-K4")
    ap.add_argument("--dst", default="/home/albert/models/Gemma-4-12B-AEON-K4-NVFP4")
    ap.add_argument("--config-name", default="NVFP4_AWQ_LITE_CFG",
                    choices=["NVFP4_AWQ_LITE_CFG", "NVFP4_AWQ_CLIP_CFG",
                             "NVFP4_AWQ_FULL_CFG", "NVFP4_DEFAULT_CFG", "NVFP4_MLP_ONLY_CFG", "NVFP4_MLP_WEIGHT_ONLY_CFG"])
    ap.add_argument("--num-calibration-samples", type=int, default=2048)
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    # 1. Calibration dataset
    print("[calib] loading CNN/DailyMail validation split (streaming)", flush=True)
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0",
                      split="validation", streaming=True)
    samples = []
    for x in ds:
        if len(samples) >= args.num_calibration_samples:
            break
        # Keep articles reasonably-sized; tokenizer will truncate to max_length anyway
        samples.append(x["article"][:8000])
    print(f"  collected {len(samples)} calibration samples", flush=True)

    # 2. Tokenizer
    tok = AutoTokenizer.from_pretrained(args.src)

    # 3. Model
    model = load_model(args.src)

    # 4. Configure quantization
    if not hasattr(mtq, args.config_name):
        sys.exit(f"modelopt has no config {args.config_name}")
    config = deepcopy(getattr(mtq, args.config_name))
    info = install_excludes(config, EXCLUDE_PATTERNS)
    print(f"[config] using {args.config_name}; {info}", flush=True)
    print(f"  excludes: {EXCLUDE_PATTERNS}", flush=True)

    # 5. Calibrate + quantize
    print(f"[quantize] starting — this is the long step (1-3h on DGX Spark)", flush=True)
    t0 = time.perf_counter()
    mtq.quantize(model, config, forward_loop=make_calibration_loop(tok, samples))
    elapsed = time.perf_counter() - t0
    print(f"[quantize] done in {elapsed/60:.1f} min", flush=True)

    # 6. Quick post-quant sanity: peek at q_proj weight shape on layer 0
    try:
        layer0 = model.model.language_model.layers[0]
        qw = layer0.self_attn.q_proj.weight
        print(f"[sanity] layer 0 q_proj.weight: shape={list(qw.shape)} dtype={qw.dtype}", flush=True)
    except Exception as e:
        print(f"[sanity] could not inspect layer 0: {e}", flush=True)

    # 7. Export to HF unified-checkpoint format (vllm --quantization modelopt loadable)
    print(f"[export] writing to {args.dst}", flush=True)
    t0 = time.perf_counter()
    mto.enable_huggingface_checkpointing()
    mte.export_hf_checkpoint(model, export_dir=args.dst)
    print(f"[export] done in {time.perf_counter()-t0:.1f}s", flush=True)

    # 8. Copy sidecar files (tokenizer, chat template, processor_config) from src
    for f in ["chat_template.jinja", "generation_config.json",
              "processor_config.json", "abliteration_meta.json"]:
        sp = Path(args.src) / f
        dp = Path(args.dst) / f
        if sp.exists() and not dp.exists():
            import shutil
            shutil.copy2(sp, dp)
            print(f"[sidecar] copied {f}", flush=True)

    # 9. Final report
    print(f"\n=== DONE ===")
    print(f"src:    {args.src}")
    print(f"dst:    {args.dst}")
    print(f"size:   {sum(p.stat().st_size for p in Path(args.dst).rglob('*') if p.is_file())/1e9:.2f} GB")

    # Show hf_quant_config.json result
    qc = Path(args.dst) / "hf_quant_config.json"
    if qc.exists():
        print(f"\nhf_quant_config.json:")
        print(qc.read_text())


if __name__ == "__main__":
    main()
