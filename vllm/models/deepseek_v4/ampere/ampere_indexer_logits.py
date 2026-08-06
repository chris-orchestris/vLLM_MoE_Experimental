# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8x (Ampere) kernels and references for sparse-indexer MQA logits.

DeepGEMM's ``fp8_fp4_{paged_,}mqa_logits`` assert "Unsupported architecture"
on SM8x. Ampere has no native FP8 tensor cores, so these convert FP8 Q/K to
BF16 and run the MQA logits on BF16 tensor cores. The decode path converts each
Q once and shares every decoded K tile across all speculative atoms.
"""

import os

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3fn_bits_to_bf16(x):
    """Convert OCP E4M3FN bytes to exact BF16 bit patterns."""
    x = x.to(tl.int32)
    sign = ((x >> 7) & 1) << 15
    exp = (x >> 3) & 0xF
    mant = x & 0x7

    normal_bits = sign | ((exp + 120) << 7) | (mant << 4)
    subnormal_bits = tl.where(
        mant == 0,
        0x0000,
        tl.where(
            mant == 1,
            0x3B00,
            tl.where(
                mant == 2,
                0x3B80,
                tl.where(
                    mant == 3,
                    0x3BC0,
                    tl.where(
                        mant == 4,
                        0x3C00,
                        tl.where(
                            mant == 5,
                            0x3C20,
                            tl.where(mant == 6, 0x3C40, 0x3C60),
                        ),
                    ),
                ),
            ),
        ),
    )
    bits = tl.where(exp == 0, sign | subnormal_bits, normal_bits)
    # E4M3FN has no infinity; exponent=15, mantissa=7 is its NaN encoding.
    bits = tl.where((exp == 0xF) & (mant == 0x7), 0x7FFF, bits)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _fp8_e4m3fn_to_bf16_kernel(
    src_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, _e4m3fn_bits_to_bf16(values), mask=mask)


def fp8_e4m3fn_to_bf16_sm86_triton(values: torch.Tensor) -> torch.Tensor:
    """Convert an E4M3FN tensor to BF16 without SM89 FP8 instructions."""
    out = torch.empty(values.shape, dtype=torch.bfloat16, device=values.device)
    n_elements = values.numel()
    if n_elements == 0:
        return out
    block_size = 256
    _fp8_e4m3fn_to_bf16_kernel[(triton.cdiv(n_elements, block_size),)](
        values.view(torch.uint8),
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out


@triton.jit
def _fp8_paged_mqa_logits_single_atom_kernel(
    q_ptr,  # [M, H, D] bf16
    kv_ptr,  # [num_blocks, block_size, D+4] uint8 (D fp8 K + 4B f32 scale)
    w_ptr,  # [M, H] float32
    ctx_ptr,  # [M] int32
    bt_ptr,  # [M, max_blocks] int32
    out_ptr,  # [M, max_model_len] float32 (pre-filled)
    kv_stride0,
    kv_stride1,
    bt_stride0,
    out_stride0,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Specialized flattened decode row with no speculative-atom state."""
    m = tl.program_id(0)
    tile = tl.program_id(1)
    ctx = tl.load(ctx_ptr + m)
    tile_start = tile * BLOCK_P
    if tile_start >= ctx:
        return

    pos = tile_start + tl.arange(0, BLOCK_P)
    pos_mask = pos < ctx
    blk = tl.load(bt_ptr + m * bt_stride0 + pos // block_size, mask=pos_mask, other=0)
    row_base = (
        blk.to(tl.int64) * kv_stride0 + (pos % block_size).to(tl.int64) * kv_stride1
    )
    k_offs = row_base[:, None] + tl.arange(0, D)[None, :]
    # Keep the legacy K conversion expression in this compatibility path. It
    # preserves the original kernel's instruction/reduction trajectory while Q
    # is converted once outside the per-position tile grid.
    ki = tl.load(kv_ptr + k_offs, mask=pos_mask[:, None], other=0).to(tl.int32)
    k_sign = (ki >> 7) & 1
    k_exp = (ki >> 3) & 0xF
    k_mant = (ki & 0x7).to(tl.float32)
    k_normal = (1.0 + k_mant * 0.125) * tl.exp2(k_exp.to(tl.float32) - 7.0)
    k_subnorm = k_mant * 0.125 * tl.exp2(-6.0)
    k_f = tl.where(k_exp == 0, k_subnorm, k_normal) * (
        1.0 - 2.0 * k_sign.to(tl.float32)
    )
    k_bf = k_f.to(tl.bfloat16)

    kv_f32 = kv_ptr.to(tl.pointer_type(tl.float32))
    scale = tl.load(kv_f32 + (row_base + D) // 4, mask=pos_mask, other=0.0)
    q_offs = m * H * D + tl.arange(0, H)[:, None] * D + tl.arange(0, D)[None, :]
    q_bf = tl.load(q_ptr + q_offs)
    qk = tl.dot(q_bf, tl.trans(k_bf))
    w = tl.load(w_ptr + m * H + tl.arange(0, H)).to(tl.float32)
    logits = tl.sum(w[:, None] * tl.maximum(qk, 0.0), axis=0) * scale
    tl.store(out_ptr + m * out_stride0 + pos, logits, mask=pos_mask)


@triton.jit
def _fp8_paged_mqa_logits_multi_atom_kernel(
    q_ptr,  # [B, NEXT_N, H, D] bf16
    kv_ptr,  # [num_blocks, block_size, D+4] uint8 (D fp8 K + 4B f32 scale)
    w_ptr,  # [M, H] float32
    ctx_ptr,  # [M] int32 per-row context length
    bt_ptr,  # [B, max_blocks] int32
    out_ptr,  # [M, max_model_len] float32 (pre-filled)
    kv_stride0,  # bytes per cache block
    kv_stride1,  # bytes per token row (D+4)
    bt_stride0,
    out_stride0,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    NEXT_N: tl.constexpr,
):
    """Score one request's speculative atoms while sharing each K tile."""
    b = tl.program_id(0)
    tile = tl.program_id(1)

    max_ctx = tl.load(ctx_ptr + b * NEXT_N)
    for atom in tl.static_range(1, NEXT_N):
        max_ctx = tl.maximum(max_ctx, tl.load(ctx_ptr + b * NEXT_N + atom))
    tile_start = tile * BLOCK_P
    if tile_start >= max_ctx:
        return

    pos = tile_start + tl.arange(0, BLOCK_P)
    pos_mask = pos < max_ctx

    blk = tl.load(bt_ptr + b * bt_stride0 + pos // block_size, mask=pos_mask, other=0)
    row_base = (
        blk.to(tl.int64) * kv_stride0 + (pos % block_size).to(tl.int64) * kv_stride1
    )
    k_offs = row_base[:, None] + tl.arange(0, D)[None, :]
    k_bytes = tl.load(kv_ptr + k_offs, mask=pos_mask[:, None], other=0)
    k_bf = _e4m3fn_bits_to_bf16(k_bytes)

    kv_f32 = kv_ptr.to(tl.pointer_type(tl.float32))
    scale = tl.load(kv_f32 + (row_base + D) // 4, mask=pos_mask, other=0.0)

    for atom in tl.static_range(0, NEXT_N):
        m = b * NEXT_N + atom
        ctx = tl.load(ctx_ptr + m)
        q_offs = m * H * D + tl.arange(0, H)[:, None] * D + tl.arange(0, D)[None, :]
        q_bf = tl.load(q_ptr + q_offs)
        qk = tl.dot(q_bf, tl.trans(k_bf))
        w = tl.load(w_ptr + m * H + tl.arange(0, H)).to(tl.float32)
        logits = tl.sum(w[:, None] * tl.maximum(qk, 0.0), axis=0) * scale
        tl.store(out_ptr + m * out_stride0 + pos, logits, mask=pos < ctx)


def fp8_paged_mqa_logits_sm86_triton(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Capture-safe SM8x replacement for fp8_fp4_paged_mqa_logits (decode).

    Q is converted once per row. The logits kernel then loads and converts each
    K tile once per request and reuses it for every speculative atom.
    """
    B, next_n, H, D = q_values.shape
    M = B * next_n
    kv3 = kv_cache.reshape(kv_cache.shape[0], kv_cache.shape[1], -1)
    block_size = kv3.shape[1]
    assert kv3.shape[-1] == D + 4, f"cache dim {kv3.shape[-1]} != D+4 ({D + 4})"
    s0, s1 = kv3.stride(0), kv3.stride(1)
    assert s0 % 4 == 0 and s1 % 4 == 0 and D % 4 == 0, "scale reads need 4B align"

    if context_lens.dim() == 2:
        ctx_flat = context_lens.reshape(M).to(torch.int32)
    else:
        ctx_flat = context_lens.to(torch.int32).repeat_interleave(next_n)

    fill = float("-inf") if clean_logits else 0.0
    logits = torch.full(
        (M, max_model_len), fill, dtype=torch.float32, device=q_values.device
    )
    q_bf = fp8_e4m3fn_to_bf16_sm86_triton(q_values)

    BLOCK_P = 64
    if next_n == 1:
        grid = (M, triton.cdiv(max_model_len, BLOCK_P))
        _fp8_paged_mqa_logits_single_atom_kernel[grid](
            q_bf,
            kv3,
            weights.to(torch.float32),
            ctx_flat,
            block_tables,
            logits,
            kv_stride0=s0,
            kv_stride1=s1,
            bt_stride0=block_tables.stride(0),
            out_stride0=logits.stride(0),
            block_size=block_size,
            H=H,
            D=D,
            BLOCK_P=BLOCK_P,
            num_warps=4,
        )
    else:
        grid = (B, triton.cdiv(max_model_len, BLOCK_P))
        _fp8_paged_mqa_logits_multi_atom_kernel[grid](
            q_bf,
            kv3,
            weights.to(torch.float32),
            ctx_flat,
            block_tables,
            logits,
            kv_stride0=s0,
            kv_stride1=s1,
            bt_stride0=block_tables.stride(0),
            out_stride0=logits.stride(0),
            block_size=block_size,
            H=H,
            D=D,
            BLOCK_P=BLOCK_P,
            NEXT_N=next_n,
            num_warps=4,
        )
    return logits


def _fp8_mqa_logits_pyref(
    q_values: torch.Tensor,
    k_packed: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """SM8x fp8 reference for fp8_fp4_mqa_logits (prefill; fp8 q, fp8 k).

    q_values: [M, H, D] float8_e4m3fn; k_packed: [N, D] float8_e4m3fn;
    k_scales: [N] float32; weights: [M, H] float32. Returns [M, N] float32.
    """
    M, H, D = q_values.shape
    N = k_packed.shape[0]
    k_bf = k_packed.to(torch.bfloat16) * k_scales.to(torch.bfloat16).unsqueeze(-1)
    out = torch.empty(M, N, dtype=torch.float32, device=q_values.device)
    n_idx = torch.arange(N, device=q_values.device, dtype=cu_seqlen_ks.dtype)
    fill = float("-inf") if clean_logits else 0.0
    # Tile over query rows so H*BM*N stays under a fixed element budget
    # (the full [M,H,N] + its fp32 copy is multi-GB for a long prefill chunk).
    budget = int(os.environ.get("VLLM_SM86_INDEXER_TILE_ELEMS", str(24_000_000)))
    BM = max(1, min(M, budget // max(1, H * N)))
    for m0 in range(0, M, BM):
        m1 = min(m0 + BM, M)
        q_bf = q_values[m0:m1].to(torch.bfloat16)
        scores = torch.einsum("mhd,nd->mhn", q_bf, k_bf)
        # Lightning indexer: per-head ReLU before the weighted head-sum, matching
        # deep_gemm fmaxf(accum, 0). k_scale (>0) is folded into k_bf, so
        # relu(scale*x) == scale*relu(x); clamp on scores is equivalent.
        logits = (
            weights[m0:m1].to(torch.float32).unsqueeze(-1)
            * scores.to(torch.float32).clamp_min(0.0)
        ).sum(dim=1)
        valid = (n_idx.unsqueeze(0) >= cu_seqlen_ks[m0:m1].unsqueeze(1)) & (
            n_idx.unsqueeze(0) < cu_seqlen_ke[m0:m1].unsqueeze(1)
        )
        out[m0:m1] = torch.where(valid, logits, torch.full_like(logits, fill))
    return out


def _fp8_paged_mqa_logits_pyref(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """SM8x fp8 reference for fp8_fp4_paged_mqa_logits (decode; fp8 q + fp8 paged KV).

    q_values: [B, next_n, H, D] float8_e4m3fn;
    kv_cache: [num_blocks, block_size, 1, D+4] uint8 (D fp8 K + 4 fp32 scale);
    weights: [B*next_n, H] float32; context_lens: [B] or [B, next_n] int32;
    block_tables: [B, max_blocks] int32. Returns [B*next_n, max_model_len] f32.
    """
    B, next_n, H, D = q_values.shape
    M = B * next_n
    block_size = kv_cache.shape[1]
    fill = float("-inf") if clean_logits else 0.0
    logits = torch.full(
        (M, max_model_len), fill, dtype=torch.float32, device=q_values.device
    )
    q_bf = q_values.to(torch.bfloat16).reshape(M, H, D)
    if context_lens.dim() == 2:
        ctx_per_row = context_lens
    else:
        ctx_per_row = context_lens.unsqueeze(-1).expand(-1, next_n)
    ctx_per_batch = ctx_per_row.amax(dim=-1)
    cache_dim = kv_cache.shape[-1]
    assert cache_dim == D + 4, (
        f"kv_cache last dim {cache_dim} != D+4 ({D + 4}); layout mismatch"
    )
    for b in range(B):
        ctx = int(ctx_per_batch[b].item())
        if ctx <= 0:
            continue
        n_blocks = (ctx + block_size - 1) // block_size
        bt = block_tables[b, :n_blocks].to(torch.long)
        rows = kv_cache[bt].reshape(n_blocks * block_size, cache_dim)[:ctx].contiguous()
        k_bytes = rows[:, :D].contiguous()
        scale_bytes = rows[:, D : D + 4].contiguous()
        scales = scale_bytes.view(torch.float32).squeeze(-1).to(torch.bfloat16)
        k_bf = k_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16) * scales.unsqueeze(
            -1
        )
        m_start = b * next_n
        m_end = m_start + next_n
        scores = torch.einsum("mhd,nd->mhn", q_bf[m_start:m_end], k_bf)
        w_b = weights[m_start:m_end].to(torch.float32)
        # Per-head ReLU before weighted head-sum (deep_gemm fmaxf(accum, 0)).
        batch_logits = (
            w_b.unsqueeze(-1) * scores.to(torch.float32).clamp_min(0.0)
        ).sum(dim=1)
        for atom in range(next_n):
            atom_ctx = int(ctx_per_row[b, atom].item())
            logits[m_start + atom, :atom_ctx] = batch_logits[atom, :atom_ctx]
    return logits
