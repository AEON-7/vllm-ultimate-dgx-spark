#!/usr/bin/env python3
"""Patch vLLM compilation.py to apply the spec-decode capture-size alignment
filter for ALL cudagraph modes (not just FULL).

Background:
  vLLM's CUDA graph capture builds graphs at specific batch sizes. When
  speculative decoding is on (num_speculative_tokens=K), every decode step
  processes (1+K) tokens and the captured graphs must be sized as multiples
  of (1+K) — otherwise on partial-acceptance steps, vLLM dispatches to a
  cached graph keyed on a misaligned size; the kernel writes
  slot_mapping/positions tensors at offsets [0..num_query_per_req-1] but the
  replayed kernel reads at offsets matching the wrong (smaller) graph.
  → cudaErrorIllegalAddress mid-decode.

  vllm/config/compilation.py has a filter that adjusts capture sizes to
  multiples of (1+K), BUT it's gated to cudagraph_mode=FULL only. The
  default PIECEWISE mode silently skips it.

  See: vLLM #28015, #28207, #29091, PR #29102, PR #23679.

Idempotent — safe to run multiple times. Locates vllm dynamically.
"""
import sys
from pathlib import Path

import vllm
VLLM_ROOT = Path(vllm.__file__).parent
TARGET = VLLM_ROOT / "config" / "compilation.py"

if not TARGET.exists():
    print(f"[patch_cudagraph_align] {TARGET} not found — skipping")
    sys.exit(0)

src = TARGET.read_text()

if "# cudagraph_align_spec_decode_all_modes" in src:
    print(f"[{TARGET.name}] already applied")
    sys.exit(0)

# Anchor on the multi-line if statement.
OLD = (
    "        if (\n"
    "            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)
NEW = (
    "        # cudagraph_align_spec_decode_all_modes\n"
    "        # Original: gated to cudagraph_mode=FULL only (vllm bug — PIECEWISE\n"
    "        # silently skips alignment, causing cudaErrorIllegalAddress on\n"
    "        # partial-acceptance decode steps when capture sizes aren't multiples\n"
    "        # of (1 + num_speculative_tokens). Apply for any non-NONE mode.\n"
    "        if (\n"
    "            cudagraph_mode != CUDAGraphMode.NONE\n"
    "            and uniform_decode_query_len > 1\n"
    "        ):\n"
    "            self.adjust_cudagraph_sizes_for_spec_decode(\n"
    "                uniform_decode_query_len,\n"
    "                tensor_parallel_size,\n"
    "            )"
)

if OLD not in src:
    print(f"[{TARGET.name}] anchor not found — likely upstream merged equivalent; skipping (idempotent)")
    sys.exit(0)

new_src = src.replace(OLD, NEW, 1)
TARGET.write_text(new_src)
print(f"[{TARGET.name}] applied spec-decode capture-size alignment for all cudagraph modes")
