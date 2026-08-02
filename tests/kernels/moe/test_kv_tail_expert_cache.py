# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
)
from vllm.model_executor.layers.fused_moe.kv_tail_expert_cache import (
    attach_kv_tail_expert_cache,
    maybe_disable_kv_tail_expert_cache,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


class _ProviderHolder(torch.nn.Module):
    def __init__(self, provider: CachedWeightProvider) -> None:
        super().__init__()
        self.expert_weight_provider = provider


def test_kv_tail_attach_and_fence(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_CACHE_SIZE", "2")
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_CACHE_MIN_TOKENS", "1")
    monkeypatch.delenv("VLLM_MOE_KV_TAIL_CACHE_LAYER_LIMIT", raising=False)

    w13 = torch.randn(8, 64, 16, dtype=torch.bfloat16)
    w2 = torch.randn(8, 16, 32, dtype=torch.bfloat16)
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=w13,
        w2_weight=w2,
        split="expert",
        name="model.layers.0.ffn.experts",
    )
    model = torch.nn.Module()
    model.holder = _ProviderHolder(provider)

    block_stride = 32 * 1024
    num_blocks = 8
    backing = torch.empty(
        num_blocks * block_stride, dtype=torch.int8, device="cuda"
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing.numel(),
                shared_by=["model.layers.0.self_attn"],
                offset=0,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[],
    )

    state = attach_kv_tail_expert_cache(model, backing, config)
    assert state is not None and state.active
    assert state.fence_block > 1
    assert provider._borrowed_w13 is not None
    assert provider._borrowed_w13.data_ptr() >= (
        backing.data_ptr() + state.fence_block * block_stride
    )

    def output_with_blocks(block_ids: list[int]):
        return SimpleNamespace(
            new_block_ids_to_zero=block_ids,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
        )

    maybe_disable_kv_tail_expert_cache(
        state, output_with_blocks([state.fence_block - 1])
    )
    assert state.active
    maybe_disable_kv_tail_expert_cache(
        state, output_with_blocks([state.fence_block])
    )
    assert not state.active
    assert not provider._borrowed_enabled


def test_kv_tail_burst_tier_revokes_before_base(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_CACHE_SIZE", "2")
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_BURST_CACHE_SIZE", "1")
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_CACHE_MIN_TOKENS", "1")

    w13 = torch.randn(8, 64, 16, dtype=torch.bfloat16)
    w2 = torch.randn(8, 16, 32, dtype=torch.bfloat16)
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=w13,
        w2_weight=w2,
        split="expert",
        name="model.layers.0.ffn.experts",
    )
    model = torch.nn.Module()
    model.holder = _ProviderHolder(provider)

    block_stride = 32 * 1024
    num_blocks = 8
    backing = torch.empty(
        num_blocks * block_stride, dtype=torch.int8, device="cuda"
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing.numel(),
                shared_by=["model.layers.0.self_attn"],
                offset=0,
                block_stride=block_stride,
            )
        ],
        kv_cache_groups=[],
    )
    state = attach_kv_tail_expert_cache(model, backing, config)
    assert state is not None and len(state.tiers) == 2
    base = next(tier for tier in state.tiers if tier.name == "base")
    burst = next(tier for tier in state.tiers if tier.name == "burst")
    assert burst.fence_block < base.fence_block

    def output_with_blocks(block_ids: list[int]):
        return SimpleNamespace(
            new_block_ids_to_zero=block_ids,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
        )

    maybe_disable_kv_tail_expert_cache(
        state, output_with_blocks([burst.fence_block])
    )
    assert state.active and base.active and not burst.active
    assert provider._borrowed_enabled
    assert not provider._borrowed_banks[burst.bank_index].enabled

    maybe_disable_kv_tail_expert_cache(
        state, output_with_blocks([base.fence_block])
    )
    assert not state.active and not base.active
    assert not provider._borrowed_enabled
