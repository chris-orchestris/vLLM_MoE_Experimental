# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ampere sparse decode for DeepSeek V4 with FP8 KV cache.

Strategy: dequantize FP8 UE8M0 KV cache pages to BF16 on the fly,
then reuse the BF16 sparse MLA attention kernel (xpu_sparse_mla_bf16).
This keeps the external KV cache layout identical to CUDA/ROCm.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.xpu_mla_sparse import (
    triton_bf16_mla_sparse_interface,
)

# FP8 DS MLA cache layout constants
TOKEN_FP8_DIM = 448  # NoPE portion in FP8
TOKEN_BF16_DIM = 64  # RoPE portion in BF16
TOKEN_SCALE_DIM = 8  # UE8M0 scales per token
QUANT_BLOCK_SIZE = 64  # Elements per quant block
OUTPUT_DIM = 512  # = TOKEN_FP8_DIM + TOKEN_BF16_DIM after dequant
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2  # 576 bytes per token


@triton.jit
def _dequant_gather_slots_kernel(
    # Output workspace: [total_slots, OUTPUT_DIM] bf16
    out_ptr,
    # FP8 paged cache base pointer (uint8 flat)
    cache_ptr,
    # Global slot indices: [total_slots] int32
    indices_ptr,
    # Cache geometry
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,  # 576
    block_stride: tl.int64,  # k_cache.stride(0) — total uint8 per block
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    output_dim: tl.constexpr,  # 512
    n_quant_blocks: tl.constexpr,  # 7
):
    """Dequantize scattered FP8 slots into a flat BF16 workspace.

    Grid: [total_slots] — one program per slot to gather.

    Cache block layout (block_size tokens):
      [0, block_size*576): Token data, each token 448 FP8 + 128 BF16
      [block_size*576, block_size*576 + block_size*8): Scales
    """
    pid = tl.program_id(0)

    # Load global slot index
    slot_idx = tl.load(indices_ptr + pid).to(tl.int64)

    # Output pointer for this slot
    out_row_ptr = out_ptr + pid * output_dim

    # Handle invalid slots (index < 0): write zeros
    if slot_idx < 0:
        zero = tl.zeros([quant_block], dtype=tl.bfloat16)
        for i in tl.static_range(0, 512, 64):
            offsets = i + tl.arange(0, quant_block)
            mask = offsets < output_dim
            tl.store(out_row_ptr + offsets, zero, mask=mask)
        return

    # Compute block and position within block
    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Block base pointer
    block_base = cache_ptr + block_idx * block_stride

    # Token data: at offset pos_in_block * token_data_size within block
    token_data_ptr = block_base + pos_in_block * token_data_size

    # Scale: after all token data, at offset
    # cache_block_size * token_data_size + pos_in_block * scale_dim
    scale_region_offset = tl.cast(cache_block_size, tl.int64) * token_data_size
    token_scale_ptr = block_base + scale_region_offset + pos_in_block * scale_dim

    # ========== Dequantize FP8 portion (448 elements) ==========
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        offsets = qblock_start + tl.arange(0, quant_block)
        mask = offsets < fp8_dim

        # Load FP8 e4m3fn bytes. SM8x Triton cannot bitcast to fp8e4nv, so
        # decode the e4m3fn format manually (sign.4-bit exp.3-bit mantissa,
        # bias 7): normal = (1 + m/8) * 2^(e-7); subnormal (e==0) = m/8 * 2^-6.
        x_uint8 = tl.load(token_data_ptr + offsets, mask=mask, other=0)
        xi = x_uint8.to(tl.int32)
        sign = (xi >> 7) & 1
        exp = (xi >> 3) & 0xF
        mant = (xi & 0x7).to(tl.float32)
        normal = (1.0 + mant * 0.125) * tl.exp2(exp.to(tl.float32) - 7.0)
        subnorm = mant * 0.125 * tl.exp2(-6.0)
        x_float = tl.where(exp == 0, subnorm, normal) * (
            1.0 - 2.0 * sign.to(tl.float32)
        )

        # Load UE8M0 scale: scale = 2^(stored_value - 127)
        encoded_scale = tl.load(token_scale_ptr + qblock_idx)
        exponent = encoded_scale.to(tl.float32) - 127.0
        scale = tl.exp2(exponent)

        # Dequantize and store as bf16
        x_dequant = x_float * scale
        tl.store(out_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

    # ========== Copy BF16 portion (64 elements) directly ==========
    bf16_src_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    bf16_out_ptr = (out_row_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))

    for j in tl.static_range(bf16_dim // 16):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_src_ptr + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def apply_attn_sink(
    out: torch.Tensor,  # [T, H, Dv]
    lse: torch.Tensor,  # [T, H] natural-log logsumexp from the BF16 kernel
    attn_sink: torch.Tensor,  # [H] (or padded) learned per-head sink logit
) -> torch.Tensor:
    """Fold the learned per-head attention sink into a sink-less kernel output.

    ``triton_bf16_mla_sparse_interface`` has no sink slot; the DeepSeek-V4 sink
    is a per-head logit that contributes ``exp(s_h - LSE)`` to the softmax
    denominator with no value vector. Correct post-hoc:
    ``out' = out / (1 + exp(s_h - LSE))``, broadcast per (token, head) over the
    value dim. ``attn_sink`` shares the scaled-logit units of ``LSE``; padded or
    ``-inf`` sinks are inert (``exp -> 0`` leaves the row unchanged).
    """
    import os

    if os.environ.get("VLLM_SM86_SINK", "1") != "1":
        return out
    sink = attn_sink[: out.shape[1]].to(lse.dtype)
    delta = sink.unsqueeze(0) - lse
    # -inf sink and/or -inf LSE (empty row) can yield inf-inf = NaN; treat any
    # non-finite delta as -inf so exp(delta)=0 leaves the row unchanged.
    delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("-inf")))
    denom = 1.0 + torch.exp(delta)
    return out / denom.unsqueeze(-1)


def dequant_gather_slots(
    out: torch.Tensor,  # [total_slots, 512] bf16, pre-allocated
    cache: torch.Tensor,  # [num_blocks, block_size, head_bytes] uint8
    indices: torch.Tensor,  # [total_slots] int32, global slot IDs
    cache_block_size: int,  # block_size for this cache
) -> None:
    """Dequantize FP8 UE8M0 pages at scattered slot indices into bf16."""
    total_slots = indices.shape[0]
    if total_slots == 0:
        return

    block_stride = cache.stride(0)

    _dequant_gather_slots_kernel[(total_slots,)](
        out,
        cache,
        indices,
        cache_block_size=cache_block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        output_dim=OUTPUT_DIM,
        n_quant_blocks=7,
    )


def ampere_sparse_decode_fp8(
    q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
    kv_cache: torch.Tensor | None,  # [num_blocks, block_size, head_bytes] uint8
    swa_kv_cache: torch.Tensor,  # [num_blocks, swa_block_size, head_bytes] uint8
    swa_only: bool,
    topk_indices: torch.Tensor | None,  # [num_tokens, 1, topk] global slot IDs
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,  # [num_tokens, 1, swa_k] global slot IDs
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    out: torch.Tensor,  # [num_tokens, num_heads, head_dim]
) -> None:
    """Ampere decode: dequant FP8 pages to BF16, then BF16 sparse MLA attention.

    Keeps external FP8 KV cache layout identical to CUDA/ROCm.
    Performance is slower due to on-the-fly dequant, but correctness is
    guaranteed by reusing the validated BF16 attention kernel.
    """
    num_tokens = q.shape[0]
    device = q.device

    # Determine max topk and swa widths
    if not swa_only and topk_indices is not None:
        topk_idx_2d = (
            topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices
        )
        max_topk = topk_idx_2d.shape[1]
    else:
        topk_idx_2d = None
        max_topk = 0

    swa_idx_2d = swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices
    max_swa = swa_idx_2d.shape[1]

    K_total = max_topk + max_swa

    # Allocate flat workspace: [num_tokens * K_total, 512] bf16
    workspace = torch.empty(
        (num_tokens * K_total, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    ws_3d = workspace.view(num_tokens, K_total, OUTPUT_DIM)

    # Dequant+gather topk slots from compressed cache
    if not swa_only and topk_idx_2d is not None and kv_cache is not None:
        topk_flat = topk_idx_2d.reshape(-1).to(torch.int32)
        topk_buf = torch.empty(
            (num_tokens * max_topk, OUTPUT_DIM), dtype=torch.bfloat16, device=device
        )
        compressed_block_size = kv_cache.shape[1]
        dequant_gather_slots(topk_buf, kv_cache, topk_flat, compressed_block_size)
        ws_3d[:, :max_topk, :] = topk_buf.view(num_tokens, max_topk, OUTPUT_DIM)

    # Dequant+gather SWA slots
    swa_flat = swa_idx_2d.reshape(-1).to(torch.int32)
    swa_buf = torch.empty(
        (num_tokens * max_swa, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    swa_block_size = swa_kv_cache.shape[1]
    dequant_gather_slots(swa_buf, swa_kv_cache, swa_flat, swa_block_size)

    ws_3d[:, max_topk:, :] = swa_buf.view(num_tokens, max_swa, OUTPUT_DIM)

    # Build combined indices into the flat workspace, FIXED-WIDTH layout:
    # token t's row = [topk slots 0..max_topk) | swa slots 0..max_swa) | pad],
    # invalid entries = -1. The BF16 kernel masks (indices >= 0) per entry and
    # iterates the full index width, so holes are a no-op; nothing requires
    # contiguous packing. This keeps the build free of .item()/host syncs and
    # Python loops, making the decode path cudagraph-capturable.
    _BLOCK_N = 16
    K_padded = ((K_total + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N
    combined_indices = torch.full(
        (num_tokens, K_padded),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    token_offsets = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) * K_total
    )  # [B]

    if not swa_only and topk_lens is not None:
        topk_range = torch.arange(
            max_topk, device=device, dtype=torch.int32
        ).unsqueeze(0)
        topk_valid = topk_range < topk_lens.unsqueeze(1)
        combined_indices[:, :max_topk] = torch.where(
            topk_valid, token_offsets.unsqueeze(1) + topk_range, -1
        )
    swa_range = torch.arange(max_swa, device=device, dtype=torch.int32).unsqueeze(0)
    swa_valid = swa_range < swa_lens.unsqueeze(1)
    combined_indices[:, max_topk:K_total] = torch.where(
        swa_valid, token_offsets.unsqueeze(1) + max_topk + swa_range, -1
    )

    # Call BF16 sparse MLA kernel
    out_attn, _, lse = triton_bf16_mla_sparse_interface(
        q=q,
        kv=workspace.unsqueeze(1),
        indices=combined_indices.unsqueeze(1),
        sm_scale=softmax_scale,
        d_v=q.shape[-1],
        block_dpe=0,
    )
    out.copy_(apply_attn_sink(out_attn, lse, attn_sink))
