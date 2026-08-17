"""AEON build gate: NVFP4 KV data/scale views must be disjoint under NHD AND HND.

#44455 packed K/V into the content dim. Upstream's nvfp4_split_data_scale uses
as_strided offset math that only tiles correctly when each page is physically
[all data | all scale] -- true under HND, FALSE under NHD. GB10 defaults to NHD
(FlashInfer forces HND only when capability.major == 10, i.e. SM100), so the
upstream formulation silently corrupts KV here. Our splitter uses plain slicing.
Fails the build if that ever regresses.
"""
import sys
import torch
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
)


def elements(t):
    idx = torch.arange(t.untyped_storage().size(), dtype=torch.int64)
    v = torch.as_strided(idx, t.shape, t.stride(), storage_offset=t.storage_offset())
    return set(v.reshape(-1).tolist())


def main():
    failures = []
    for layout in ("NHD", "HND"):
        for nb, H, N, hs in [(2, 2, 16, 128), (3, 8, 32, 128), (1, 1, 16, 64)]:
            fd = nvfp4_kv_cache_full_dim(hs)
            if layout == "NHD":
                kv = torch.zeros((nb, N, H, 2 * fd), dtype=torch.uint8).permute(0, 2, 1, 3)
            else:
                kv = torch.zeros((nb, H, N, 2 * fd), dtype=torch.uint8)
            k_side, v_side = kv.transpose(1, 2).split(fd, dim=-1)
            (kd,), (ks,) = nvfp4_kv_cache_split_views(k_side)
            (vd,), (vs,) = nvfp4_kv_cache_split_views(v_side)
            S = {"k_data": elements(kd), "k_scale": elements(ks),
                 "v_data": elements(vd), "v_scale": elements(vs)}
            names = list(S)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ov = S[names[i]] & S[names[j]]
                    if ov:
                        failures.append(
                            f"[{layout}] {nb}x{H}x{N}x{hs}: {names[i]} and "
                            f"{names[j]} OVERLAP on {len(ov)} bytes"
                        )
            cov = len(set().union(*S.values()))
            tot = nb * H * N * 2 * fd
            if cov != tot:
                failures.append(f"[{layout}] {nb}x{H}x{N}x{hs}: covers {cov}/{tot} bytes")
    if failures:
        print("[aeon] NVFP4 KV view gate FAILED:")
        for f in failures:
            print("   ", f)
        sys.exit(1)
    print("[aeon] NVFP4 KV view-disjointness gate PASSED (NHD + HND, 3 shapes each)")


if __name__ == "__main__":
    main()
