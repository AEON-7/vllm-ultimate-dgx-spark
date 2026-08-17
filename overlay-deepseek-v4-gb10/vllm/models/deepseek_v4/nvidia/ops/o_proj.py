# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PATCHED (sm12x / GB10): adapt the o_proj fp8_einsum call to DeepGEMM nv_dev (PR #324 sm120 kernels).
# AEON's SM100 path passes int32-packed UE8M0 activation scales, a 2D (h*d, r) weight, and a 2D e8m0
# weight scale. nv_dev's fp8_einsum('bhr,hdr->bhd') expects unpacked fp32 scales and a 3D (h,d,r)
# weight (see DeepGEMM tests/test_einsum.py::test_fp8_bhr_hdr_bhd). This shim converts on sm12x only.
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum

_USE_SM12X = None


def _sm12x() -> bool:
    global _USE_SM12X
    if _USE_SM12X is None:
        cap = current_platform.get_device_capability()
        _USE_SM12X = cap is not None and cap.major >= 12
    return _USE_SM12X


def _ue8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 -> fp32 (exact): place the 8-bit exponent into the fp32 exponent field.
    Handles int32-packed (4 UE8M0 bytes per int32, little-endian) and raw float8_e8m0fnu."""
    e8 = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype == torch.int32:
        u8 = scale.contiguous().view(torch.uint8)      # [..., k*4] bytes in logical order
    elif e8 is not None and scale.dtype == e8:
        u8 = scale.contiguous().view(torch.uint8)
    else:
        return scale.to(torch.float32)
    return (u8.to(torch.int32) << 23).view(torch.float32)


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] -> sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] -> sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b."""
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    if _sm12x():
        # nv_dev sm120 einsum: unpacked fp32 scales + 3D weight.
        b, h, r = o_fp8.shape
        d = o_lora_rank
        o_s = _ue8m0_to_fp32(o_scale).reshape(b, h, -1)[:, :, : (r + 127) // 128]
        w3 = getattr(wo_a, "_sm12x_w3", None)
        if w3 is None:
            w3 = wo_a.weight.reshape(h, d, r)
            ws = (
                _ue8m0_to_fp32(wo_a.weight_scale_inv)
                .reshape(h, (d + 127) // 128, (r + 127) // 128)
                .contiguous()
            )
            wo_a._sm12x_w3, wo_a._sm12x_ws = w3, ws
        fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s.contiguous()), (w3, wo_a._sm12x_ws), z)
    else:
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (wo_a.weight, wo_a.weight_scale_inv),
            z,
            recipe=einsum_recipe,
        )
    return wo_b(z.flatten(1))
