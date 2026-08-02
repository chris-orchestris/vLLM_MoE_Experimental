# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _mhc_sinkhorn_kernel(
    in_ptr,
    out_ptr,
    n_tokens,
    eps: tl.constexpr,
    hc: tl.constexpr,
    iterations: tl.constexpr,
):
    """Fuse MHC softmax and Sinkhorn normalization for one token."""
    token_idx = tl.program_id(0)
    if token_idx >= n_tokens:
        return

    rows = tl.arange(0, hc)
    cols = tl.arange(0, hc)
    offsets = token_idx * hc * hc + rows[:, None] * hc + cols[None, :]
    comb_mix = tl.load(in_ptr + offsets).to(tl.float32)

    row_max = tl.max(comb_mix, axis=1)
    comb_mix = tl.exp(comb_mix - row_max[:, None])
    row_sum = tl.sum(comb_mix, axis=1)
    comb_mix = comb_mix / row_sum[:, None] + eps

    col_sum = tl.sum(comb_mix, axis=0)
    comb_mix = comb_mix / (col_sum[None, :] + eps)

    for _ in range(iterations - 1):
        row_sum = tl.sum(comb_mix, axis=1)
        comb_mix = comb_mix / (row_sum[:, None] + eps)
        col_sum = tl.sum(comb_mix, axis=0)
        comb_mix = comb_mix / (col_sum[None, :] + eps)

    tl.store(out_ptr + offsets, comb_mix)


def mhc_sinkhorn_triton(
    comb_logits: torch.Tensor,
    eps: float,
    iterations: int,
) -> torch.Tensor:
    """Run MHC softmax and all Sinkhorn iterations in one Triton kernel."""
    if comb_logits.ndim != 3 or comb_logits.shape[-2] != comb_logits.shape[-1]:
        raise ValueError("comb_logits must have shape (tokens, hc, hc)")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    n_tokens, hc, _ = comb_logits.shape
    logits = comb_logits.contiguous()
    out = torch.empty_like(logits)
    _mhc_sinkhorn_kernel[(n_tokens,)](
        logits,
        out,
        n_tokens,
        eps=eps,
        hc=hc,
        iterations=iterations,
        num_warps=1,
    )
    return out


@triton.jit
def _mhc_post_per_token_kernel(
    comb_ptr,
    residual_ptr,
    post_ptr,
    x_ptr,
    out_ptr,
    hc: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse the MHC post residual contraction and post-layer contribution."""
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1).to(tl.int64)
    hidden_offsets = hidden_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size

    hc_offsets = tl.arange(0, hc)
    comb_offsets = token_idx * hc * hc + hc_offsets[:, None] * hc + hc_offsets[None, :]
    comb = tl.load(comb_ptr + comb_offsets).to(tl.float32)
    post = tl.load(post_ptr + token_idx * hc + hc_offsets).to(tl.float32)
    x = tl.load(
        x_ptr + token_idx * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    residual_offsets = (
        token_idx * hc * hidden_size
        + hc_offsets[:, None] * hidden_size
        + hidden_offsets[None, :]
    )
    residual = tl.load(
        residual_ptr + residual_offsets,
        mask=hidden_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    products = tl.trans(comb)[:, :, None] * residual[None, :, :]
    mixed = tl.sum(products, axis=1)
    result = mixed + post[:, None] * x[None, :]

    out_offsets = (
        token_idx * hc * hidden_size
        + hc_offsets[:, None] * hidden_size
        + hidden_offsets[None, :]
    )
    tl.store(out_ptr + out_offsets, result, mask=hidden_mask[None, :])


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """SM8x MHC post path derived from the Ampere reference implementation."""
    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    n_tokens = residual.numel() // (hc * hidden_size)

    residual_flat = residual.reshape(n_tokens, hc, hidden_size).contiguous()
    comb_flat = comb_res_mix.reshape(n_tokens, hc, hc).float().contiguous()
    post_flat = post_layer_mix.reshape(n_tokens, hc).float().contiguous()
    x_flat = x.reshape(n_tokens, hidden_size).contiguous()
    out = torch.empty_like(residual_flat)

    block_h = 256 if hidden_size >= 256 else 128
    _mhc_post_per_token_kernel[(n_tokens, triton.cdiv(hidden_size, block_h))](
        comb_flat,
        residual_flat,
        post_flat,
        x_flat,
        out,
        hc=hc,
        hidden_size=hidden_size,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out.view(*outer_shape, hc, hidden_size)


def mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SM8x MHC pre path with the launch-heavy Sinkhorn loop fused."""
    del n_splits
    hc = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc, hidden_size)
    n_tokens = residual_flat.shape[0]

    residual_fp32 = residual_flat.view(n_tokens, hc * hidden_size).float()
    mixes = torch.matmul(residual_fp32, fn.t())
    square_sum = residual_fp32.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(square_sum / (hc * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc] * hc_scale[0] + hc_base[:hc]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_logits = mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value
    comb_logits = mixes[:, 2 * hc :].view(n_tokens, hc, hc) * hc_scale[2] + hc_base[
        2 * hc :
    ].view(1, hc, hc)
    comb_mix = mhc_sinkhorn_triton(
        comb_logits,
        hc_sinkhorn_eps,
        sinkhorn_repeat,
    )

    layer_input = torch.einsum(
        "bi,bih->bh",
        pre_mix,
        residual_fp32.view(n_tokens, hc, hidden_size),
    ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc, 1),
        comb_mix.view(*outer_shape, hc, hc),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_fused_post_pre_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the two SM8x Triton MHC kernels without Torch fallbacks."""
    residual_cur = mhc_post_triton(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    post_mix, comb_mix, layer_input = mhc_pre_triton(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return residual_cur, post_mix, comb_mix, layer_input


@triton.jit
def _rmsnorm_nw_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    D,
    eps,
    RBLOCK: tl.constexpr,
):
    """Weight-free RMSNorm Triton kernel: out = x * rsqrt(mean(x², -1) + eps)."""
    row = tl.program_id(0)
    cols = tl.arange(0, RBLOCK)
    mask = cols < D

    x = tl.load(
        x_ptr + row * stride_row + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 0) / D
    rstd = tl.rsqrt(var + eps)

    out = (x * rstd).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * D + cols, out, mask=mask, eviction_policy="evict_first")


def rmsnorm_nw(x: Tensor, eps: float) -> Tensor:
    """Weight-free RMSNorm over the last dimension.

    Treats *x* as ``[num_rows, D]`` where ``num_rows = product(shape[:-1])``.
    Returns a contiguous tensor with the same shape and dtype as *x*.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x_2d = x.reshape(-1, D)
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    RBLOCK = triton.next_power_of_2(D)

    _rmsnorm_nw_kernel[(num_rows,)](
        x_2d,
        out,
        x_2d.stride(0),
        D,
        eps,
        RBLOCK=RBLOCK,
        num_warps=1 if RBLOCK <= 512 else (4 if RBLOCK <= 4096 else 8),
    )
    return out.view(orig_shape)


@triton.jit
def _hc_head_reduce_store_kernel(
    pre_ptr,
    x_ptr,
    out_ptr,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    pre_stride_t: tl.constexpr,
    pre_stride_m: tl.constexpr,
    x_stride_t: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for mix_idx in tl.static_range(0, hc_mult):
        pre = tl.load(pre_ptr + token_idx * pre_stride_t + mix_idx * pre_stride_m).to(
            tl.float32
        )
        x = tl.load(
            x_ptr
            + token_idx * x_stride_t
            + mix_idx * x_stride_m
            + offsets * x_stride_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre * x

    tl.store(
        out_ptr + token_idx * out_stride_t + offsets * out_stride_h,
        acc,
        mask=mask,
    )


def hc_head_reduce_triton_kernel(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> None:
    x_flat = x.flatten(-2)
    x_normed = rmsnorm_nw(x_flat, norm_eps)
    mixes = F.linear(x_normed.float(), hc_fn)
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    hidden_size = x.shape[-1]
    hc_mult = x.shape[-2]
    block_h = 1024
    _hc_head_reduce_store_kernel[(x.shape[0], (hidden_size + block_h - 1) // block_h)](
        pre,
        x,
        out,
        hidden_size,
        hc_mult,
        pre.stride(0),
        pre.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )


def _hc_head_triton(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Fill pre-allocated `out` (T, H) in-place with the hc_head result."""
    if hs_flat.shape[0] == 0:
        return

    hc_head_reduce_triton_kernel(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        rms_eps,
        hc_eps,
    )
    return


direct_register_custom_op(
    op_name="hc_head_triton",
    op_func=_hc_head_triton,
    mutates_args=["out"],
)
