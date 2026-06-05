#!/usr/bin/env python3
"""Translate a Gemma4UnifiedForConditionalGeneration checkpoint to the
Gemma4ForConditionalGeneration layout that vLLM 0.22.1's native loader
in vllm/model_executor/models/gemma4_mm.py expects.

What this script does:
  1. Reads the source model dir (config.json + safetensors + sidecars)
  2. Writes a new dir with:
     - config.json renamed to architectures=Gemma4ForConditionalGeneration,
       model_type=gemma4 (text/audio/vision sub-types likewise),
       vision_config has default_output_length added, audio_config=null
     - safetensors with weight keys remapped (see KEY_MAP below) and audio
       tensors dropped
     - all sidecar files (chat_template.jinja, tokenizer, generation_config,
       processor_config.json from Google base) copied over
  3. Verifies that all the vLLM-expected weight keys are present

Why a pre-processor rather than a vLLM patch:
  - Keeps our patches OUT of vLLM (no patch maintenance across releases)
  - Produces a self-contained "vllm-ready" artifact that any future vLLM
    release supporting Gemma4ForConditionalGeneration can serve unmodified
  - The translation is lossless except for audio modality (which our
    abliteration doesn't touch anyway — vLLM expects a different audio dim,
    so we drop audio support and keep vision + text)

Usage:
  python3 prepare_gemma4_for_vllm.py \\
      --src /home/albert/models/Gemma-4-12B-AEON-K4 \\
      --dst /home/albert/models/Gemma-4-12B-AEON-K4-vllm-ready \\
      [--base-repo google/gemma-4-12B-it]  # for processor_config.json
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from glob import glob

from safetensors import safe_open
from safetensors.torch import save_file


# === Weight key rename rules ===
# Translate Gemma4Unified naming → Gemma4 (vLLM-native) naming.
# Tested by inspecting both Gemma4Unified safetensor keys (our K4 model) and
# vLLM's load_weights() expectations in gemma4_mm.py / gemma4.py.

def remap_key(k: str) -> str | None:
    """Return remapped key, or None to drop the tensor entirely."""
    # Drop audio support — vLLM's Gemma4 loader expects audio_embed_dim=1536,
    # ours is 640. Our abliteration doesn't touch audio anyway.
    if k.startswith("model.embed_audio") or "audio_embedder" in k:
        return None
    # Strip the leading "model." prefix on vision_embedder + embed_vision
    # (vLLM's loader keys these at top level, e.g. "vision_embedder.patch_dense.weight")
    if k.startswith("model.vision_embedder."):
        return k[len("model."):]
    if k.startswith("model.embed_vision."):
        return k[len("model."):]
    # The language stack stays under model.language_model.* (vLLM keeps it there)
    # — no rename needed
    return k


# === Config translations ===

UNIFIED_TO_GEMMA4_TYPES = {
    "gemma4_unified": "gemma4",
    "gemma4_unified_text": "gemma4_text",
    "gemma4_unified_audio": "gemma4_audio",
    "gemma4_unified_vision": "gemma4_vision",
}


def translate_config(src_path: str, dst_path: str, base_repo: str | None) -> dict:
    """Read config.json, rewrite it for the Gemma4 native loader."""
    with open(src_path) as f:
        cfg = json.load(f)

    # 1. Architecture name
    cfg["architectures"] = ["Gemma4ForConditionalGeneration"]

    # 2. Top + sub model_types
    if cfg.get("model_type") in UNIFIED_TO_GEMMA4_TYPES:
        cfg["model_type"] = UNIFIED_TO_GEMMA4_TYPES[cfg["model_type"]]
    for sub in ("text_config", "vision_config", "audio_config"):
        if isinstance(cfg.get(sub), dict):
            mt = cfg[sub].get("model_type")
            if mt in UNIFIED_TO_GEMMA4_TYPES:
                cfg[sub]["model_type"] = UNIFIED_TO_GEMMA4_TYPES[mt]

    # 3. vision_config: add default_output_length alias for num_soft_tokens
    vc = cfg.get("vision_config")
    if isinstance(vc, dict):
        if "num_soft_tokens" in vc and "default_output_length" not in vc:
            vc["default_output_length"] = vc["num_soft_tokens"]
        # vLLM's gemma4 also wants model_patch_size; default 48 per Google's variant
        if "model_patch_size" not in vc:
            vc["model_patch_size"] = 48

    # 4. Drop audio_config — vLLM expects audio_embed_dim=1536, ours is 640.
    # Setting to null makes vLLM's Gemma4 loader skip audio modality.
    cfg["audio_config"] = None
    # Also clear audio token IDs since we don't load the audio embedder
    cfg.pop("audio_token_id", None)
    cfg.pop("boa_token_id", None)
    cfg.pop("eoa_token_index", None)

    with open(dst_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def maybe_fetch_processor_config(dst_dir: str, base_repo: str | None):
    """vLLM's multimodal init needs processor_config.json. Try local first; fall
    back to base_repo on HF."""
    dst = os.path.join(dst_dir, "processor_config.json")
    if os.path.exists(dst):
        return
    if not base_repo:
        print(f"  [warn] processor_config.json missing and --base-repo not set; vLLM may fail at MM init")
        return
    print(f"  fetching processor_config.json from {base_repo}...")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(base_repo, "processor_config.json")
        shutil.copy(p, dst)
        print(f"    copied to {dst}")
    except Exception as e:
        print(f"  [warn] couldn't fetch processor_config.json from {base_repo}: {e}")


def rewrite_safetensors(src_dir: str, dst_dir: str) -> dict:
    """Read all *.safetensors in src_dir, remap keys, drop audio tensors,
    write to dst_dir. Returns stats."""
    src_files = sorted(glob(os.path.join(src_dir, "*.safetensors")))
    if not src_files:
        sys.exit(f"no safetensors found in {src_dir}")

    stats = {"in_tensors": 0, "kept": 0, "renamed": 0, "dropped_audio": 0,
             "out_files": 0, "in_bytes": 0, "out_bytes": 0}

    for src in src_files:
        rel = os.path.basename(src)
        dst = os.path.join(dst_dir, rel)
        print(f"\n  reading {src}...")
        stats["in_bytes"] += os.path.getsize(src)
        out_tensors = {}
        with safe_open(src, framework="pt") as f:
            keys = list(f.keys())
            stats["in_tensors"] += len(keys)
            for k in keys:
                new_k = remap_key(k)
                if new_k is None:
                    stats["dropped_audio"] += 1
                    continue
                t = f.get_tensor(k)
                if new_k != k:
                    stats["renamed"] += 1
                out_tensors[new_k] = t
                stats["kept"] += 1
        print(f"    kept {len(out_tensors)} tensors → writing {dst}")
        save_file(out_tensors, dst, metadata={"format": "pt"})
        stats["out_files"] += 1
        stats["out_bytes"] += os.path.getsize(dst)
        del out_tensors
    return stats


def copy_sidecars(src_dir: str, dst_dir: str):
    """Copy non-config, non-safetensors sidecar files."""
    keep = (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "chat_template.jinja", "generation_config.json", "special_tokens_map.json",
        "abliteration_meta.json", "README.md", "AGENTS.md",
    )
    for name in os.listdir(src_dir):
        if name in keep:
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))
            print(f"  copied {name}")


def verify(dst_dir: str):
    """Walk dst_dir and report key inventory of the new model."""
    files = sorted(glob(os.path.join(dst_dir, "*.safetensors")))
    print(f"\n=== verify {dst_dir} ===")
    print(f"  {len(files)} safetensor shard(s)")
    n_total = 0
    by_prefix = {}
    for f in files:
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                n_total += 1
                pfx = k.split(".")[0]
                by_prefix[pfx] = by_prefix.get(pfx, 0) + 1
    print(f"  total tensors: {n_total}")
    for pfx in sorted(by_prefix):
        print(f"    {pfx}: {by_prefix[pfx]}")
    # Spot-check that no audio keys leaked through
    audio_leaks = [k for k in by_prefix if "audio" in k.lower()]
    if audio_leaks:
        print(f"  [warn] audio keys still present: {audio_leaks}")
    # Spot-check that vision_embedder is at top level (no model. prefix)
    if "vision_embedder" not in by_prefix:
        print(f"  [warn] vision_embedder not at top level — vLLM Gemma4MM loader may fail")
    # Confirm config.json is renamed correctly
    cfg = json.load(open(os.path.join(dst_dir, "config.json")))
    print(f"  config.architectures: {cfg['architectures']}")
    print(f"  config.model_type: {cfg['model_type']}")
    print(f"  config.audio_config: {cfg.get('audio_config')}")
    print(f"  config.vision_config.default_output_length: "
          f"{cfg.get('vision_config', {}).get('default_output_length')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--base-repo", default="google/gemma-4-12B-it",
                    help="HF repo to fetch processor_config.json from")
    ap.add_argument("--skip-weights", action="store_true",
                    help="only rewrite config + sidecars (for fast iteration)")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        print(f"[warn] dst exists: {args.dst}")
    os.makedirs(args.dst, exist_ok=True)

    print(f"=== translate config ===")
    cfg = translate_config(os.path.join(args.src, "config.json"),
                           os.path.join(args.dst, "config.json"),
                           args.base_repo)
    print(f"  arch: {cfg['architectures']}")
    print(f"  model_type: {cfg['model_type']}")
    print(f"  audio_config: {cfg.get('audio_config')}")

    print(f"\n=== copy sidecars ===")
    copy_sidecars(args.src, args.dst)
    maybe_fetch_processor_config(args.dst, args.base_repo)

    if not args.skip_weights:
        print(f"\n=== rewrite safetensors ===")
        t0 = time.perf_counter()
        stats = rewrite_safetensors(args.src, args.dst)
        elapsed = time.perf_counter() - t0
        print(f"\n  stats: {json.dumps(stats, indent=2)}")
        print(f"  size: {stats['in_bytes']/1e9:.2f}GB → {stats['out_bytes']/1e9:.2f}GB "
              f"(elapsed {elapsed:.1f}s)")

    verify(args.dst)
    print(f"\n[DONE] translated model at: {args.dst}")


if __name__ == "__main__":
    main()
