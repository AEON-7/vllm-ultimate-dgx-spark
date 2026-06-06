"""Make a ModelOpt NVFP4 Gemma-4-12B (Unified) checkpoint load cleanly in vLLM.

Applies three post-export fixes in one shot so the published model is
self-contained (loads with `--quantization modelopt`, no manual steps):

  1. Vision key naming — collapse embed_vision.multimodal_embedder.* and
     move the patch embedder from embed_vision.* to vision_embedder.*
     (what vLLM's Gemma4UnifiedForConditionalGeneration loader expects).
  2. Ignore list — add model.vision_embedder* to quantization_config.ignore
     (config.json) and exclude_modules (hf_quant_config.json) so vLLM keeps
     the patch embedder at BF16 instead of building a packed-FP4 layer.
  3. Tokenizer — copy tokenizer.json / tokenizer_config.json from the source
     BF16 model (ModelOpt export omits them).

Usage:
  python3 make_vllm_ready.py --model /models/Gemma-4-12B-AEON-K4-NVFP4 \\
                             --src   /models/Gemma-4-12B-AEON-K4
"""
import argparse
import json
import os
import shutil
import time

from safetensors import safe_open
from safetensors.torch import save_file

VISION_EMBEDDER_BASES = ("patch_dense", "patch_ln1", "patch_ln2",
                         "pos_embedding", "pos_norm")


def fix_vision_keys(model_dir: str):
    path = os.path.join(model_dir, "model.safetensors")
    dst = path + ".new"
    print(f"[1/3 vision-keys] reading {path}", flush=True)
    out = {}
    n_collapsed = n_moved = 0
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            new_k = k
            if "embed_vision.multimodal_embedder.embedding_projection" in k:
                new_k = k.replace("embed_vision.multimodal_embedder.", "embed_vision.")
                n_collapsed += 1
            elif "embed_vision." in k:
                base = k.split("embed_vision.", 1)[1].split(".")[0]
                if base in VISION_EMBEDDER_BASES:
                    new_k = k.replace("embed_vision.", "vision_embedder.")
                    n_moved += 1
            out[new_k] = f.get_tensor(k)
    print(f"  collapsed={n_collapsed} multimodal_embedder, moved={n_moved} to vision_embedder", flush=True)
    t0 = time.perf_counter()
    save_file(out, dst, metadata={"format": "pt"})
    os.replace(dst, path)
    print(f"  rewrote model.safetensors in {time.perf_counter()-t0:.1f}s", flush=True)


def fix_ignore_lists(model_dir: str):
    print("[2/3 ignore-lists]", flush=True)
    p = os.path.join(model_dir, "config.json")
    c = json.load(open(p))
    qc = c.get("quantization_config", {})
    ignore = qc.get("ignore", [])
    for a in ("model.vision_embedder*", "re:.*vision_embedder.*"):
        if a not in ignore:
            ignore.append(a)
    qc["ignore"] = ignore
    c["quantization_config"] = qc
    json.dump(c, open(p, "w"), indent=2)
    print(f"  config.json ignore: {ignore}", flush=True)

    p2 = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.exists(p2):
        h = json.load(open(p2))
        ex = h.get("quantization", {}).get("exclude_modules", [])
        if "model.vision_embedder*" not in ex:
            ex.append("model.vision_embedder*")
        h["quantization"]["exclude_modules"] = ex
        json.dump(h, open(p2, "w"), indent=2)
        print(f"  hf_quant_config exclude: {ex}", flush=True)


def copy_tokenizer(model_dir: str, src_dir: str):
    print("[3/3 tokenizer]", flush=True)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        sp = os.path.join(src_dir, f)
        dp = os.path.join(model_dir, f)
        if os.path.exists(sp) and not os.path.exists(dp):
            shutil.copy2(sp, dp)
            print(f"  copied {f}", flush=True)
        elif os.path.exists(dp):
            print(f"  {f} already present", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="NVFP4 model dir to fix in place")
    ap.add_argument("--src", required=True, help="source BF16 model dir (for tokenizer)")
    args = ap.parse_args()
    fix_vision_keys(args.model)
    fix_ignore_lists(args.model)
    copy_tokenizer(args.model, args.src)
    print("\n[DONE] model is vLLM-ready", flush=True)


if __name__ == "__main__":
    main()
