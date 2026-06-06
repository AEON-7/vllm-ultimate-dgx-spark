"""Fix modelopt's vision-key renaming so vLLM's Gemma4Unified loader accepts them.

modelopt HF export does two things vLLM doesn't expect:
  1. Wraps embed_vision.embedding_projection inside a multimodal_embedder layer:
       embed_vision.multimodal_embedder.embedding_projection.weight
     vLLM wants: embed_vision.embedding_projection.weight
  2. Keeps the patch embedder under embed_vision.* instead of vision_embedder.*:
       embed_vision.patch_dense / patch_ln1 / patch_ln2 / pos_embedding / pos_norm
     vLLM wants these under vision_embedder.*

This script rewrites the single safetensors shard in place.
"""
import sys
import time
from safetensors import safe_open
from safetensors.torch import save_file

PATH = sys.argv[1] if len(sys.argv) > 1 else "/model/model.safetensors"
DST = PATH + ".new"

VISION_EMBEDDER_BASES = ("patch_dense", "patch_ln1", "patch_ln2", "pos_embedding", "pos_norm")

print(f"[fix-vision] reading {PATH}", flush=True)
out = {}
n_collapsed = n_moved = 0
with safe_open(PATH, framework="pt") as f:
    for k in f.keys():
        new_k = k
        if "embed_vision.multimodal_embedder.embedding_projection" in k:
            new_k = k.replace("embed_vision.multimodal_embedder.", "embed_vision.")
            n_collapsed += 1
        elif "embed_vision." in k:
            stripped = k.split("embed_vision.", 1)[1]
            base = stripped.split(".")[0]
            if base in VISION_EMBEDDER_BASES:
                new_k = k.replace("embed_vision.", "vision_embedder.")
                n_moved += 1
        out[new_k] = f.get_tensor(k)

print(f"[fix-vision] collapsed={n_collapsed} multimodal_embedder, "
      f"moved={n_moved} to vision_embedder", flush=True)
t0 = time.perf_counter()
save_file(out, DST, metadata={"format": "pt"})
print(f"[fix-vision] wrote {DST} in {time.perf_counter()-t0:.1f}s", flush=True)

import os
os.replace(DST, PATH)
print(f"[fix-vision] replaced {PATH}", flush=True)
