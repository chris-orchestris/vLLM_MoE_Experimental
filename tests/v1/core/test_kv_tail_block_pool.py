# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def test_kv_tail_fence_recycles_low_cached_blocks(monkeypatch):
    monkeypatch.setenv("DSV4_MOE_KV_TAIL_CACHE_FENCE_BLOCK", "5")
    pool = BlockPool(
        num_gpu_blocks=10,
        enable_caching=True,
        hash_block_size=16,
    )
    blocks = pool.get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]
    for block in blocks:
        block.set_block_hash(
            make_block_hash_with_group_id(
                BlockHash(f"block-{block.block_id}".encode()), 0
            ),
            num_tokens=16,
        )

    pool.free_blocks(blocks)
    recycled = pool.free_block_queue.popleft_n(2)
    assert [block.block_id for block in recycled] == [1, 2]


def test_default_free_order_remains_unchanged(monkeypatch):
    monkeypatch.delenv("DSV4_MOE_KV_TAIL_CACHE_FENCE_BLOCK", raising=False)
    monkeypatch.delenv("DSV4_MOE_KV_TAIL_BASE_FENCE_BLOCK", raising=False)
    pool = BlockPool(
        num_gpu_blocks=10,
        enable_caching=True,
        hash_block_size=16,
    )
    blocks = pool.get_new_blocks(2)
    for block in blocks:
        block.set_block_hash(
            make_block_hash_with_group_id(
                BlockHash(f"block-{block.block_id}".encode()), 0
            ),
            num_tokens=16,
        )

    pool.free_blocks(blocks)
    untouched = pool.free_block_queue.popleft_n(2)
    assert [block.block_id for block in untouched] == [3, 4]


def test_kv_tail_tier_fence_advances_with_allocations(monkeypatch):
    monkeypatch.setenv("DSV4_MOE_KV_TAIL_CACHE_FENCE_BLOCK", "5")
    monkeypatch.setenv("DSV4_MOE_KV_TAIL_BASE_FENCE_BLOCK", "8")
    pool = BlockPool(
        num_gpu_blocks=10,
        enable_caching=True,
        hash_block_size=16,
    )

    first = pool.get_new_blocks(5)
    assert [block.block_id for block in first] == [1, 2, 3, 4, 5]
    assert pool.kv_tail_expert_cache_fence == 8

    pool.free_blocks(first)
    second = pool.get_new_blocks(8)
    assert max(block.block_id for block in second) >= 8
    assert pool.kv_tail_expert_cache_fence == 0
