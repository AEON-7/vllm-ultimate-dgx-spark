#!/usr/bin/env python3
"""Patch vLLM hybrid-attention KV-cache None-handling in 4 files.

Background:
  Hybrid attention models (Qwen3.6, Nemotron-Omni, etc.) mix linear_attention
  layers (mamba state, block_size=None) with full_attention layers (block_size=int).
  vLLM HEAD calls `min(block_size for group in groups)` in multiple places,
  crashing with `TypeError: '<' not supported between NoneType and NoneType`
  when None values slip through.

  Fix: filter None values before min(), default to 1 (or int 16 for mamba) if all None.

Idempotent — safe to run multiple times. Locates vllm dynamically.
"""
import sys
from pathlib import Path

import vllm
VLLM_ROOT = Path(vllm.__file__).parent


def patch_kv_cache_utils() -> None:
    target = VLLM_ROOT / "v1" / "core" / "kv_cache_utils.py"
    if not target.exists():
        print(f"[kv_cache_utils.py] not found — skipping")
        return
    src = target.read_text()
    if "# kv_cache_utils_min_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "    min_block_size = min(\n"
        "        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]\n"
        "    )"
    )
    new = (
        "    # kv_cache_utils_min_none_safe\n"
        "    _block_sizes = [\n"
        "        group.kv_cache_spec.block_size\n"
        "        for group in kv_cache_config.kv_cache_groups\n"
        "        if group.kv_cache_spec.block_size is not None\n"
        "    ]\n"
        "    min_block_size = min(_block_sizes) if _block_sizes else 1"
    )
    if old not in src:
        print(f"[{target.name}] anchor not found — likely upstream merged equivalent; skipping")
        return
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe min()")


def patch_engine_core() -> None:
    target = VLLM_ROOT / "v1" / "engine" / "core.py"
    if not target.exists():
        print(f"[engine/core.py] not found — skipping")
        return
    src = target.read_text()
    if "# engine_core_block_size_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "            vllm_config.cache_config.block_size = min(\n"
        "                g.kv_cache_spec.block_size for g in kv_cache_groups\n"
        "            )"
    )
    new = (
        "            # engine_core_block_size_none_safe\n"
        "            _bs = [g.kv_cache_spec.block_size for g in kv_cache_groups if g.kv_cache_spec.block_size is not None]\n"
        "            if _bs:\n"
        "                vllm_config.cache_config.block_size = min(_bs)"
    )
    if old not in src:
        print(f"[{target.name}] anchor not found — likely upstream merged equivalent; skipping")
        return
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe min()")


def patch_gpu_model_runner() -> None:
    target = VLLM_ROOT / "v1" / "worker" / "gpu_model_runner.py"
    if not target.exists():
        print(f"[v1/worker/gpu_model_runner.py] not found — skipping")
        return
    src = target.read_text()
    if "# gpu_model_runner_block_size_none_safe" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "            block_size = kv_cache_group.kv_cache_spec.block_size\n"
        "            block_sizes.append(block_size)\n"
        "            max_num_blocks_per_req = cdiv(\n"
        "                max_model_len, block_size * get_total_cp_world_size()\n"
        "            )"
    )
    new = (
        "            block_size = kv_cache_group.kv_cache_spec.block_size\n"
        "            block_sizes.append(block_size)\n"
        "            # gpu_model_runner_block_size_none_safe\n"
        "            if block_size is None:\n"
        "                # MambaSpec / linear-attention groups: block-based KV doesn't apply.\n"
        "                # MambaSpec branch below overrides max_num_blocks_per_req anyway.\n"
        "                max_num_blocks_per_req = 0\n"
        "            else:\n"
        "                max_num_blocks_per_req = cdiv(\n"
        "                    max_model_len, block_size * get_total_cp_world_size()\n"
        "                )"
    )
    if old not in src:
        print(f"[{target.name}] anchor not found — likely upstream merged equivalent; skipping")
        return
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied None-safe block_size handling")


def patch_mamba_abstract() -> None:
    """Default mamba_block_size to attention block_size or 16 when None.
    Per memory feedback_vllm_v0_20_0_aeon_patch_status: this sub-patch is OBSOLETE
    as of v0.20.0 (upstream added `assert mamba_block_size is not None`). Kept
    here as no-op-on-mismatch so we discover when/if the anchor reappears."""
    target = VLLM_ROOT / "model_executor" / "layers" / "mamba" / "abstract.py"
    if not target.exists():
        print(f"[mamba/abstract.py] not found — skipping (likely upstream restructured)")
        return
    src = target.read_text()
    if "# mamba_abstract_block_size_default" in src:
        print(f"[{target.name}] already applied")
        return

    old = (
        "        mamba_block_size = vllm_config.cache_config.mamba_block_size\n"
        "        page_size_padded = vllm_config.cache_config.mamba_page_size_padded"
    )
    new = (
        "        mamba_block_size = vllm_config.cache_config.mamba_block_size\n"
        "        # mamba_abstract_block_size_default — defensive fallback\n"
        "        if mamba_block_size is None:\n"
        "            mamba_block_size = vllm_config.cache_config.block_size or 16\n"
        "        page_size_padded = vllm_config.cache_config.mamba_page_size_padded"
    )
    if old not in src:
        print(f"[{target.name}] anchor not found — upstream restructured (expected, obsolete since v0.20.0); skipping")
        return
    target.write_text(src.replace(old, new, 1))
    print(f"[{target.name}] applied mamba_block_size default")


def main() -> None:
    print(f"[patch_kv_cache_utils] vllm at {VLLM_ROOT}")
    patch_mamba_abstract()
    patch_kv_cache_utils()
    patch_engine_core()
    patch_gpu_model_runner()


if __name__ == "__main__":
    main()
