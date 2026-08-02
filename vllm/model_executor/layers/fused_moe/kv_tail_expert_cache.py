# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Safe, one-way lending of untouched packed KV tail blocks to MoE caches."""

import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass
class KVTailExpertCacheTier:
    name: str
    fence_block: int
    bank_index: int
    active: bool = True


@dataclass
class KVTailExpertCacheState:
    # Lowest active fence, retained for scheduler diagnostics and one-tier
    # source compatibility.
    fence_block: int
    providers: list[Any]
    tiers: list[KVTailExpertCacheTier]
    active: bool = True
    allocation_high_water: int = -1
    logged_allocation_bucket: int = -1


def attach_kv_tail_expert_cache(
    model: torch.nn.Module,
    packed_backing: torch.Tensor | None,
    kv_cache_config: KVCacheConfig,
) -> KVTailExpertCacheState | None:
    """Carve auxiliary expert banks from never-yet-used high KV blocks."""
    base_capacity = int(os.getenv("VLLM_MOE_KV_TAIL_CACHE_SIZE", "0"))
    burst_capacity = int(os.getenv("VLLM_MOE_KV_TAIL_BURST_CACHE_SIZE", "0"))
    if base_capacity <= 0:
        return None
    if burst_capacity < 0:
        raise ValueError("VLLM_MOE_KV_TAIL_BURST_CACHE_SIZE must be >= 0")
    min_tokens = int(os.getenv("VLLM_MOE_KV_TAIL_CACHE_MIN_TOKENS", "4096"))
    layer_limit = int(os.getenv("VLLM_MOE_KV_TAIL_CACHE_LAYER_LIMIT", "0"))
    if min_tokens < 0:
        raise ValueError("VLLM_MOE_KV_TAIL_CACHE_MIN_TOKENS must be >= 0")
    if packed_backing is None:
        raise RuntimeError(
            "KV-tail expert cache requires the packed DeepSeek V4 KV layout"
        )

    block_strides = {
        tensor.block_stride
        for tensor in kv_cache_config.kv_cache_tensors
        if tensor.block_stride > 0
    }
    if len(block_strides) != 1:
        raise RuntimeError("KV-tail expert cache requires one packed block stride")
    block_stride = next(iter(block_strides))

    providers = []
    seen_provider_ids: set[int] = set()
    for module in model.modules():
        provider = getattr(module, "expert_weight_provider", None)
        if provider is None or id(provider) in seen_provider_ids:
            continue
        if layer_limit > 0 and ".layers." in provider.name:
            layer_index = int(provider.name.split(".layers.", 1)[1].split(".", 1)[0])
            if layer_index >= layer_limit:
                continue
        seen_provider_ids.add(id(provider))
        providers.append(provider)
    providers.sort(key=lambda provider: provider.name)
    if not providers:
        raise RuntimeError("KV-tail expert cache found no cached MoE providers")

    alignment = 256

    def align(value: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    def make_layout(
        capacity: int,
    ) -> tuple[list[list[tuple[torch.dtype, tuple[int, ...], int]]], int]:
        layouts: list[list[tuple[torch.dtype, tuple[int, ...], int]]] = []
        required_bytes = 0
        for provider in providers:
            provider_layout = []
            for dtype, shape in provider.borrowed_tensor_specs(capacity):
                item_size = torch.empty((), dtype=dtype).element_size()
                tensor_bytes = math.prod(shape) * item_size
                required_bytes = align(required_bytes)
                provider_layout.append((dtype, shape, tensor_bytes))
                required_bytes += tensor_bytes
            layouts.append(provider_layout)
        return layouts, align(required_bytes)

    base_layouts, base_required_bytes = make_layout(base_capacity)
    base_borrowed_blocks = (
        base_required_bytes + block_stride - 1
    ) // block_stride
    base_fence_block = kv_cache_config.num_blocks - base_borrowed_blocks

    burst_layouts: list[
        list[tuple[torch.dtype, tuple[int, ...], int]]
    ] = []
    burst_required_bytes = 0
    burst_borrowed_blocks = 0
    if burst_capacity > 0:
        burst_layouts, burst_required_bytes = make_layout(burst_capacity)
        burst_borrowed_blocks = (
            burst_required_bytes + block_stride - 1
        ) // block_stride

    fence_block = base_fence_block - burst_borrowed_blocks
    total_borrowed_blocks = base_borrowed_blocks + burst_borrowed_blocks
    if fence_block <= 1:
        raise RuntimeError(
            "KV-tail expert cache would consume all usable KV blocks: "
            f"need {total_borrowed_blocks} of {kv_cache_config.num_blocks}"
        )
    if base_fence_block * block_stride + base_required_bytes > packed_backing.numel():
        raise RuntimeError("KV-tail expert cache layout exceeds packed backing")
    if (
        burst_capacity > 0
        and fence_block * block_stride + burst_required_bytes
        > base_fence_block * block_stride
    ):
        raise RuntimeError("KV-tail burst tier overlaps the durable tier")
    configured_fence = int(
        os.getenv("DSV4_MOE_KV_TAIL_CACHE_FENCE_BLOCK", "0")
    )
    if configured_fence and configured_fence != fence_block:
        raise RuntimeError(
            "KV-tail worker/scheduler fence mismatch: "
            f"computed {fence_block}, configured {configured_fence}"
        )
    configured_base_fence = int(
        os.getenv("DSV4_MOE_KV_TAIL_BASE_FENCE_BLOCK", "0")
    )
    if configured_base_fence and configured_base_fence != base_fence_block:
        raise RuntimeError(
            "KV-tail durable-tier fence mismatch: "
            f"computed {base_fence_block}, configured {configured_base_fence}"
        )

    def attach_tier(
        layouts: list[list[tuple[torch.dtype, tuple[int, ...], int]]],
        tier_start: int,
    ) -> int:
        cursor = tier_start
        bank_index: int | None = None
        for provider, provider_layout in zip(providers, layouts):
            buffers = []
            for dtype, shape, tensor_bytes in provider_layout:
                cursor = align(cursor)
                raw = packed_backing.narrow(0, cursor, tensor_bytes)
                buffers.append(raw.view(dtype).view(shape))
                cursor += tensor_bytes
            attached_index = provider.attach_borrowed_buffers(
                buffers, min_tokens=min_tokens
            )
            if bank_index is None:
                bank_index = attached_index
            elif bank_index != attached_index:
                raise RuntimeError("KV-tail provider bank indices diverged")
        assert bank_index is not None
        return bank_index

    # Attach the highest, durable tier first so it remains bank zero for the
    # existing provider API.  The lower burst tier can then be revoked without
    # disturbing the durable tier's buffers or residency metadata.
    base_bank_index = attach_tier(
        base_layouts, base_fence_block * block_stride
    )
    tiers = [
        KVTailExpertCacheTier(
            name="base",
            fence_block=base_fence_block,
            bank_index=base_bank_index,
        )
    ]
    if burst_capacity > 0:
        burst_bank_index = attach_tier(
            burst_layouts, fence_block * block_stride
        )
        tiers.append(
            KVTailExpertCacheTier(
                name="burst",
                fence_block=fence_block,
                bank_index=burst_bank_index,
            )
        )

    logger.info(
        "KV-tail expert cache attached: %d providers, base=%d experts "
        "(%d bytes/%d blocks, fence=%d), burst=%d experts "
        "(%d bytes/%d blocks, fence=%d), total fence=%d/%d",
        len(providers),
        base_capacity,
        base_required_bytes,
        base_borrowed_blocks,
        base_fence_block,
        burst_capacity,
        burst_required_bytes,
        burst_borrowed_blocks,
        fence_block,
        fence_block,
        kv_cache_config.num_blocks,
    )
    return KVTailExpertCacheState(
        fence_block=fence_block,
        providers=providers,
        tiers=tiers,
    )


def maybe_disable_kv_tail_expert_cache(
    state: KVTailExpertCacheState | None, scheduler_output: Any
) -> None:
    """Release the borrowed tail before vLLM zeroes its first KV block."""
    if state is None or not state.active:
        return

    allocated_block_ids = list(scheduler_output.new_block_ids_to_zero or [])
    for new_request in scheduler_output.scheduled_new_reqs:
        for group_block_ids in new_request.block_ids:
            allocated_block_ids.extend(group_block_ids)
    for new_group_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
        if new_group_block_ids is None:
            continue
        for group_block_ids in new_group_block_ids:
            allocated_block_ids.extend(group_block_ids)
    if allocated_block_ids:
        allocation_high_water = max(allocated_block_ids)
        state.allocation_high_water = max(
            state.allocation_high_water, allocation_high_water
        )
        if os.getenv("DSV4_MOE_KV_TAIL_LOG_ALLOCATIONS", "0").lower() in {
            "1",
            "true",
        }:
            bucket = allocation_high_water // 128
            if bucket > state.logged_allocation_bucket:
                state.logged_allocation_bucket = bucket
                logger.info(
                    "KV-tail scheduler allocation high-water block=%d, "
                    "fence=%d",
                    state.allocation_high_water,
                    state.fence_block,
                )
    if not allocated_block_ids:
        return

    allocation_max = max(allocated_block_ids)
    crossed_tiers = [
        tier
        for tier in state.tiers
        if tier.active and allocation_max >= tier.fence_block
    ]
    if not crossed_tiers:
        return

    for tier in crossed_tiers:
        for provider in state.providers:
            provider.disable_borrowed_buffers(tier.bank_index)
        tier.active = False
        logger.warning(
            "KV allocation reached block %d (%s-tier fence %d); "
            "permanently disabled borrowed expert bank %d while retaining "
            "the full KV allocation.",
            allocation_max,
            tier.name,
            tier.fence_block,
            tier.bank_index,
        )
    state.active = any(tier.active for tier in state.tiers)
    active_fences = [tier.fence_block for tier in state.tiers if tier.active]
    if active_fences:
        state.fence_block = min(active_fences)
