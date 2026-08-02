# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CachedWeightProvider (LFRU expert cache)."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
    run_with_expert_cache,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

NUM_EXPERTS = [8, 64]
DTYPES = [torch.bfloat16, torch.float16]
CAPACITIES = [1, 4]
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(num_experts: int, dtype: torch.dtype):
    # Generate in BF16 then cast - torch.randn doesn't support FP8 dtypes
    w13 = torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    return w13.to(dtype), w2.to(dtype)


def _make_scales(num_experts: int):
    w13_s = torch.rand(num_experts, 1, dtype=torch.float32)
    w2_s = torch.rand(num_experts, 1, dtype=torch.float32)
    return w13_s, w2_s


def _make_provider(
    num_experts: int = 8,
    capacity: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    with_scales: bool = False,
    split: str = "token",
    async_staging: bool | None = None,
    name: str = "unnamed",
):
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    kwargs: dict = dict(
        capacity=capacity,
        w13_weight=w13,
        w2_weight=w2,
        split=split,
        async_staging=async_staging,
        name=name,
    )
    scales = None
    if with_scales:
        w13_s, w2_s = _make_scales(num_experts)
        kwargs.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    return CachedWeightProvider(**kwargs), w13, w2, scales


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


def _attach_borrowed_bank(
    provider: CachedWeightProvider, capacity: int, min_tokens: int = 1
) -> int:
    buffers = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for dtype, shape in provider.borrowed_tensor_specs(capacity)
    ]
    return provider.attach_borrowed_buffers(buffers, min_tokens=min_tokens)


# -- Core cache behavior --


def test_borrowed_bank_adds_disjoint_residency():
    provider, w13, w2, _ = _make_provider(
        num_experts=8, capacity=4, split="expert"
    )
    _attach_borrowed_bank(provider, capacity=2)

    topk_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5]], dtype=torch.int32, device="cuda"
    )
    unique_ids = topk_ids.unique().tolist()
    borrowed = provider.plan_borrowed_group(topk_ids, unique_ids)
    primary_groups = provider.plan_expert_groups(
        topk_ids, unique_ids=unique_ids, excluded_ids=set(borrowed)
    )

    assert borrowed == [0, 1]
    assert primary_groups == [[2, 3, 4, 5]]
    borrowed_result = provider.prepare_borrowed_group(borrowed)
    for expert_id in borrowed:
        slot = provider._borrowed_slots[expert_id]
        torch.testing.assert_close(borrowed_result.w1[slot].cpu(), w13[expert_id])
        torch.testing.assert_close(borrowed_result.w2[slot].cpu(), w2[expert_id])

    provider.prepare(topk_ids, primary_groups[0])
    assert set(provider._lru).isdisjoint(provider._borrowed_slots)
    assert len(provider._lru) + len(provider._borrowed_slots) == 6


def test_borrowed_bank_is_one_way_disabled():
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    _attach_borrowed_bank(provider, capacity=2)
    topk_ids = _topk([0, 1, 2, 3, 4, 5])
    unique_ids = topk_ids.unique().tolist()
    assert provider.plan_borrowed_group(topk_ids, unique_ids)

    provider.disable_borrowed_buffers()
    assert provider.plan_borrowed_group(topk_ids, unique_ids) == []


def test_borrowed_tiers_are_disjoint_and_independently_disabled():
    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    base_index = _attach_borrowed_bank(provider, capacity=2)
    burst_index = _attach_borrowed_bank(provider, capacity=2)
    assert (base_index, burst_index) == (0, 1)

    topk_ids = _topk([0, 1, 2, 3, 4, 5, 6])
    unique_ids = topk_ids.unique().tolist()
    groups = provider.plan_borrowed_groups(topk_ids, unique_ids)
    assert groups == [(0, [0, 1]), (1, [2, 3])]
    assert set(groups[0][1]).isdisjoint(groups[1][1])

    provider.disable_borrowed_buffers(burst_index)
    groups = provider.plan_borrowed_groups(topk_ids, unique_ids)
    assert groups == [(0, [0, 1])]
    assert provider._borrowed_enabled


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cold_miss_and_warm_hit(num_experts: int, capacity: int, dtype: torch.dtype):
    """Cold access misses, repeat access hits. GPU buffer matches source."""
    provider, w13, w2, _ = _make_provider(num_experts, capacity, dtype)
    expert_ids = list(range(min(capacity, num_experts)))

    # Cold miss
    result = provider.prepare(_topk(expert_ids))
    assert provider.misses == len(expert_ids)
    assert provider.hits == 0
    assert isinstance(result, ExpertWeightResult)
    assert result.w1 is provider.buf_w13
    assert result.w2 is provider.buf_w2
    assert result.expert_map.shape == (num_experts,)

    # Verify GPU buffer contents match source weights
    for eid in expert_ids:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        torch.testing.assert_close(result.w2[slot].cpu(), w2[eid])

    # Warm hit
    prev_misses = provider.misses
    provider.prepare(_topk(expert_ids))
    assert provider.hits == len(expert_ids)
    assert provider.misses == prev_misses


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_full_equals_num_experts(num_experts: int, dtype: torch.dtype):
    """When capacity == num_experts, all fit with zero evictions."""
    provider, _, _, _ = _make_provider(num_experts, capacity=num_experts, dtype=dtype)
    all_ids = list(range(num_experts))
    provider.prepare(_topk(all_ids))
    assert provider.misses == num_experts
    assert len(provider._free_slots) == 0

    provider.prepare(_topk(all_ids))
    assert provider.hits == num_experts


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_expert_map_points_at_slots(capacity: int):
    """expert_map sends resident experts to their slot and the rest to -1."""
    provider, _, _, _ = _make_provider(capacity=capacity)
    ids = list(range(min(capacity, 8)))
    result = provider.prepare(_topk(ids))

    mapping = result.expert_map.tolist()
    for eid in ids:
        assert mapping[eid] == provider._lru[eid][0]
    for eid in range(len(mapping)):
        if eid not in provider._lru:
            assert mapping[eid] == -1


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_accepts_either_topk_dtype(dtype: torch.dtype):
    """topk_ids may be int32 or int64; the map is always int32."""
    provider, *_ = _make_provider()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    result = provider.prepare(ids)
    assert result.expert_map.dtype == torch.int32


# -- LFRU eviction semantics --


def test_lfru_prefers_evicting_low_frequency():
    """LFRU evicts the expert with lowest freq/age score, not pure LRU.
    A accessed 5x, B accessed 1x. When C arrives, B is evicted, not A.
    """
    provider, w13, _, _ = _make_provider(capacity=2)
    provider.prepare(_topk([0, 1]))
    for _ in range(4):
        provider.prepare(_topk([0]))  # A freq=5
    provider.prepare(_topk([1]))  # touch B for recency parity

    provider.prepare(_topk([2]))  # evicts B (lower freq/age score)
    assert 0 in provider._lru, "High-frequency expert A should survive"
    assert 2 in provider._lru, "New expert C should be cached"
    assert 1 not in provider._lru, "Low-frequency expert B should be evicted"
    slot_c = provider._lru[2][0]
    torch.testing.assert_close(provider.buf_w13[slot_c].cpu(), w13[2])


def test_lfru_evicts_stale_high_freq_expert():
    """High historical freq but old last-access loses to recent low-freq.
    Distinguishes LFRU (score=freq/age) from pure frequency-based caching.
    """
    provider, _, _, _ = _make_provider(capacity=2)

    # Expert 0: accessed 11x early, then becomes stale
    provider.prepare(_topk([0]))
    for _ in range(10):
        provider.prepare(_topk([0]))
    # Expert 1: loaded later, accessed 51x (0 becomes very stale)
    provider.prepare(_topk([1]))
    for _ in range(50):
        provider.prepare(_topk([1]))

    # Expert 0: freq=11, age~62 -> score~0.18. Expert 1: freq=51, age=1 -> 51
    provider.prepare(_topk([2]))
    assert 1 in provider._lru, "Recent high-freq expert should survive"
    assert 0 not in provider._lru, "Stale expert should be evicted"


def test_capacity_one_always_evicts():
    """With capacity=1, every new expert evicts the previous."""
    provider, *_ = _make_provider(capacity=1)
    for eid in range(5):
        provider.prepare(_topk([eid]))
    assert provider.misses == 5
    assert provider.hits == 0
    assert len(provider._lru) == 1
    assert 4 in provider._lru


# -- GPU buffer correctness under eviction --


def test_gpu_buffer_correct_after_eviction():
    """After eviction, the reused slot contains the new expert's weights."""
    provider, w13, w2, _ = _make_provider(capacity=4)
    provider.prepare(_topk([0, 1, 2, 3]))

    # Make 0 the eviction candidate (least recently used, lowest freq)
    provider.prepare(_topk([1, 2, 3]))
    slot_for_0 = provider._lru[0][0]

    provider.prepare(_topk([7]))
    assert provider._lru[7][0] == slot_for_0
    torch.testing.assert_close(provider.buf_w13[slot_for_0].cpu(), w13[7])
    torch.testing.assert_close(provider.buf_w2[slot_for_0].cpu(), w2[7])


def test_batch_experts_survive_intra_batch_eviction():
    """Every expert in a batch maps to a slot holding its own weights.

    A miss loads the expert with freq=1, the lowest possible LFRU score, so a
    later miss in the same batch would evict it and reuse its slot while
    _mapping still points there — feeding the kernel another expert's weights
    with no error raised.
    """
    provider, w13, w2, _ = _make_provider(num_experts=8, capacity=4)

    # Warm every slot with high-frequency experts so the buffer is full and
    # each resident entry outscores a freshly loaded one.
    for _ in range(50):
        provider.prepare(_topk([0, 1, 2, 3]))

    batch = [4, 5, 6, 7]
    result = provider.prepare(_topk(batch))

    for expert_id in batch:
        slot = provider._lru[expert_id][0]
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[expert_id])
        torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[expert_id])

    # Every batch expert must be resident, not mapped away.
    assert all(result.expert_map[e].item() >= 0 for e in batch)


# -- Scale buffer handling --


def test_scale_lifecycle():
    """Scales are allocated, copied on load, and updated on eviction."""
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 requires CUDA capability >= 89")

    provider, _, _, scales = _make_provider(
        capacity=4, dtype=torch.float8_e4m3fn, with_scales=True
    )
    w13_s, w2_s = scales

    # Buffers allocated on GPU
    assert provider.buf_w13_scale is not None
    assert provider.buf_w2_scale is not None
    assert provider.buf_w13_scale.device.type == "cuda"

    # Scales copied correctly on load
    result = provider.prepare(_topk([3, 6]))
    for eid in [3, 6]:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(result.w2_scale[slot].cpu(), w2_s[eid])

    # Fill cache and evict: scales must be updated in evicted slot
    provider.prepare(_topk([0, 1]))  # cache now full: [3, 6, 0, 1]
    provider.prepare(_topk([3, 6, 0]))  # boost freq on 3,6,0; expert 1 stale

    result = provider.prepare(_topk([7]))  # evicts 1
    assert 1 not in provider._lru
    slot_7 = provider._lru[7][0]
    torch.testing.assert_close(provider.buf_w13_scale[slot_7].cpu(), w13_s[7])
    torch.testing.assert_close(provider.buf_w2_scale[slot_7].cpu(), w2_s[7])


def test_no_scales_when_not_provided():
    """Without scale inputs, scale buffers remain None."""
    provider, *_ = _make_provider()
    assert provider.buf_w13_scale is None
    assert provider.buf_w2_scale is None
    result = provider.prepare(_topk([0]))
    assert result.w1_scale is None
    assert result.w2_scale is None


# -- Invalidation --


def test_invalidate_frees_slot():
    """invalidate() removes an expert and returns its slot to the free list."""
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    old_slot = provider._lru[2][0]
    provider.invalidate(2)
    assert 2 not in provider._lru
    assert old_slot in provider._free_slots


def test_invalidate_noop_when_absent():
    """invalidate() on an uncached expert is a no-op."""
    provider, *_ = _make_provider()
    provider.invalidate(99)  # must not raise


# -- Overflow (unique experts > capacity) --


def test_overflow_raises():
    """When unique experts exceed capacity, raise RuntimeError immediately."""
    provider, *_ = _make_provider(capacity=2)
    with pytest.raises(RuntimeError, match="unique experts"):
        provider.prepare(_topk([0, 1, 2, 3]))


# -- CPU pinned memory --


def test_cpu_backing_is_pinned():
    """CPU weight tensors must be pinned for async H2D copies."""
    provider, *_ = _make_provider()
    assert provider._cpu_w13.is_pinned()
    assert provider._cpu_w2.is_pinned()


# -- Expert-group execution --


def _rows(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def test_single_group_when_experts_fit():
    """A forward within capacity runs as one group."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    groups = provider.plan_expert_groups(_rows([[0, 1], [1, 2], [2, 3]]))
    assert groups == [[0, 1, 2, 3]]


def test_groups_partition_all_experts():
    """Every expert appears in exactly one group, and no group overflows."""
    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    groups = provider.plan_expert_groups(topk)

    assert len(groups) > 1
    flat = [e for g in groups for e in g]
    assert sorted(flat) == sorted(topk.unique().tolist())
    assert len(flat) == len(set(flat)), "an expert landed in two groups"
    for g in groups:
        assert len(g) <= provider.capacity


def test_groups_schedule_resident_experts_first():
    """Chunked prefills reuse the group retained by the prior chunk."""
    provider, *_ = _make_provider(num_experts=8, capacity=2, split="expert")
    provider.prepare(_rows([[6, 7]]))

    groups = provider.plan_expert_groups(
        _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    )

    assert groups[0] == [6, 7]
    assert sorted(expert for group in groups for expert in group) == list(range(8))


def test_async_staging_uses_layer_cache_for_resident_group():
    """The zero-copy resident group stays out of the shared staging banks."""
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    provider.prepare(_rows([[6, 7]]))
    initial_misses = provider.misses
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    observed_buffers: list[torch.Tensor] = []

    def run(result, rows, include_shared):
        observed_buffers.append(result.w1)
        return torch.zeros(topk.size(0), 4, device="cuda")

    run_with_expert_cache(provider, topk, run)

    assert observed_buffers[0] is provider.buf_w13
    assert all(buffer is not provider.buf_w13 for buffer in observed_buffers[1:])
    assert provider.misses - initial_misses == 6
    assert provider.hits == 2
    assert set(provider._lru) == {6, 7}


def test_async_staging_waits_for_one_normal_primary_cache_fill():
    """The first forward populates the LRU instead of bypassing it forever."""
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    first_buffers: list[torch.Tensor] = []

    def first_run(result, rows, include_shared):
        first_buffers.append(result.w1)
        return torch.zeros(topk.size(0), 4, device="cuda")

    assert not provider.async_staging_ready
    run_with_expert_cache(provider, topk, first_run)
    assert provider.async_staging_ready
    assert len(provider._lru) == provider.capacity
    assert all(buffer is provider.buf_w13 for buffer in first_buffers)

    second_buffers: list[torch.Tensor] = []

    def second_run(result, rows, include_shared):
        second_buffers.append(result.w1)
        return torch.zeros(topk.size(0), 4, device="cuda")

    run_with_expert_cache(provider, topk, second_run)
    assert second_buffers[0] is provider.buf_w13
    assert all(buffer is not provider.buf_w13 for buffer in second_buffers[1:])


def test_async_staging_capacity_is_independent_of_primary_cache(monkeypatch):
    """A smaller shared bank bounds memory without shrinking decode residency."""
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY", "2")
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
    )
    provider.prepare(_rows([[4, 5, 6, 7]]))

    groups = provider.plan_expert_groups(
        _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    )

    assert groups == [[4, 5, 6, 7], [0, 1], [2, 3]]
    assert provider.capacity == 4
    assert provider._staging_capacity == 2
    assert provider._staging_pool is not None
    assert provider._staging_pool.capacity == 2
    assert all(result.w1.size(0) == 2 for result in provider._staging_pool.banks)


def test_async_staging_index_excludes_borrowed_and_resident_groups(monkeypatch):
    """Only shared-bank groups advance the bounded staging-bank ordinal."""
    provider, *_ = _make_provider(
        num_experts=6,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    _attach_borrowed_bank(provider, capacity=1)
    _attach_borrowed_bank(provider, capacity=1)
    provider.prepare(_rows([[4, 5]]))
    topk = _rows([[0, 1], [2, 3], [4, 5]])
    observed_indices: list[int] = []
    original_prepare = provider.prepare_staged_group

    def record_prepare(expert_ids, group_index, *, wait_for_ready=True):
        observed_indices.append(group_index)
        return original_prepare(
            expert_ids,
            group_index,
            wait_for_ready=wait_for_ready,
        )

    monkeypatch.setattr(provider, "prepare_staged_group", record_prepare)

    def run(result, rows, include_shared):
        return torch.zeros(topk.size(0), 4, device="cuda")

    run_with_expert_cache(provider, topk, run)
    assert observed_indices == [0]


def test_async_staging_launches_first_miss_before_resident_compute(monkeypatch):
    """The first copy gets resident-kernel lead time before its event wait."""
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    provider.prepare(_rows([[6, 7]]))
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    events: list[str] = []
    original_prepare = provider.prepare_staged_group

    def record_prepare(expert_ids, group_index, *, wait_for_ready=True):
        events.append(f"stage:{group_index}:{wait_for_ready}")
        return original_prepare(
            expert_ids,
            group_index,
            wait_for_ready=wait_for_ready,
        )

    monkeypatch.setattr(provider, "prepare_staged_group", record_prepare)

    def run(result, rows, include_shared):
        events.append(
            "run:resident" if result.w1 is provider.buf_w13 else "run:staged"
        )
        return torch.zeros(topk.size(0), 4, device="cuda")

    run_with_expert_cache(provider, topk, run)

    assert events[0] == "stage:0:False"
    assert events[1] == "run:resident"
    assert events[2] == "run:staged"


def test_interlayer_prefetch_uses_third_shared_bank(monkeypatch):
    """A predecessor hides one predicted successor upload in a bounded bank."""
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY", "2")
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH", "1")
    monkeypatch.setenv(
        "VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH_MIN_TOKENS", "1"
    )
    first, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
        name="model.layers.0.ffn.experts",
    )
    second, second_w13, second_w2, _ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
        name="model.layers.1.ffn.experts",
    )
    second.prepare(_rows([[6, 7]]))
    second.remember_prefill_route(
        _rows([[0, 0], [0, 1], [1, 2], [2, 3]])
    )

    first.prefetch_successor(num_tokens=4)

    assert second._staging_pool is first._staging_pool
    assert second._staging_pool is not None
    assert len(second._staging_pool.banks) == 3
    assert second._lookahead_pending_ids == [0, 1]
    lookahead = second.take_lookahead_group([0, 1, 4, 5])
    assert lookahead is not None
    expert_ids, result = lookahead
    assert expert_ids == [0, 1]
    second.wait_lookahead_group()
    torch.testing.assert_close(result.w1[:2].cpu(), second_w13[[0, 1]])
    torch.testing.assert_close(result.w2[:2].cpu(), second_w2[[0, 1]])
    second.release_lookahead_group()
    assert second._lookahead_pending_ids == []


def test_borrowed_planning_honors_lookahead_exclusions():
    """A predicted group cannot also be admitted to a KV-backed bank."""
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
    )
    _attach_borrowed_bank(provider, capacity=2)

    groups = provider.plan_borrowed_groups(
        _rows([[0, 1], [2, 3]]),
        [0, 1, 2, 3],
        excluded_ids={0, 1},
    )

    assert groups == [(0, [2, 3])]


def test_interlayer_prefetch_partitions_forward_exactly_once(monkeypatch):
    """Lookahead, resident, and staged groups remain disjoint and complete."""
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY", "2")
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH", "1")
    monkeypatch.setenv(
        "VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH_MIN_TOKENS", "1"
    )
    first, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
        name="partition.layers.0.ffn.experts",
    )
    second, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
        name="partition.layers.1.ffn.experts",
    )
    second.prepare(_rows([[6, 7]]))
    second.remember_prefill_route(
        _rows([[0, 0], [0, 1], [1, 2], [2, 3]])
    )
    first.prefetch_successor(num_tokens=4)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    exposed_groups: list[list[int]] = []

    def run(result, rows, include_shared):
        mapping = result.expert_map.cpu().tolist()
        exposed_groups.append(
            [expert_id for expert_id, slot in enumerate(mapping) if slot >= 0]
        )
        return torch.zeros(topk.size(0), 4, device="cuda")

    run_with_expert_cache(second, topk, run)

    flattened = [expert_id for group in exposed_groups for expert_id in group]
    assert sorted(flattened) == list(range(8))
    assert len(flattened) == len(set(flattened))
    assert second._lookahead_pending_ids == []


def test_async_staging_survives_borrowed_bank_duplicate_invalidation():
    """A decode-inserted borrowed expert cannot collapse the warm policy."""
    provider, *_ = _make_provider(
        num_experts=8,
        capacity=4,
        split="expert",
        async_staging=True,
    )
    _attach_borrowed_bank(provider, capacity=2)
    provider.prepare(_rows([[0, 4, 5, 6]]))
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    unique_ids = topk.unique().tolist()

    borrowed = provider.plan_borrowed_groups(topk, unique_ids)
    assert borrowed == [(0, [0, 1])]
    assert set(provider._lru) == {4, 5, 6}
    assert provider.async_staging_ready

    groups = provider.plan_expert_groups(
        topk,
        unique_ids=unique_ids,
        excluded_ids={0, 1},
    )
    assert groups == [[4, 5, 6], [2, 3, 7]]


def test_bulk_prefill_uses_static_contiguous_groups(monkeypatch):
    """Large prefills avoid inspecting routes and upload contiguous slices."""
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_BULK_PREFILL_MIN_TOKENS", "4")
    provider, w13, w2, scales = _make_provider(
        num_experts=8,
        capacity=2,
        with_scales=True,
        split="expert",
    )
    # The routes intentionally use only two experts. Once the threshold is
    # reached, planning is static and therefore does not need a device sync.
    topk = _rows([[0, 1], [0, 1], [0, 1], [0, 1]])
    assert provider.uses_bulk_prefill(topk)
    assert provider.plan_expert_groups(topk) == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]

    assert scales is not None
    for group_index, expert_ids in enumerate(provider._bulk_prefill_groups):
        result = provider.prepare_bulk_prefill_group(group_index)
        torch.testing.assert_close(result.w1.cpu(), w13[expert_ids])
        torch.testing.assert_close(result.w2.cpu(), w2[expert_ids])
        assert result.w1_scale is not None
        assert result.w2_scale is not None
        torch.testing.assert_close(result.w1_scale.cpu(), scales[0][expert_ids])
        torch.testing.assert_close(result.w2_scale.cpu(), scales[1][expert_ids])
        mapping = result.expert_map.cpu().tolist()
        assert {
            expert_id for expert_id, slot in enumerate(mapping) if slot >= 0
        } == set(expert_ids)
        for slot, expert_id in enumerate(expert_ids):
            assert mapping[expert_id] == slot

    assert set(provider._lru) == {6, 7}
    assert provider.misses == 8
    assert provider.hits == 0


def test_bulk_prefill_leaves_short_forward_adaptive(monkeypatch):
    """Decode and small prefills keep the route-aware LFRU behavior."""
    monkeypatch.setenv("VLLM_MOE_EXPERT_CACHE_BULK_PREFILL_MIN_TOKENS", "4")
    provider, *_ = _make_provider(num_experts=8, capacity=2, split="expert")
    topk = _rows([[3, 5]])
    assert not provider.uses_bulk_prefill(topk)
    assert provider.plan_expert_groups(topk) == [[3, 5]]


def test_async_staging_double_buffers_groups_without_touching_decode_cache():
    """Staging overlaps reusable banks while preserving every group's data."""
    provider, w13, w2, scales = _make_provider(
        num_experts=8,
        capacity=2,
        with_scales=True,
        split="expert",
        async_staging=True,
    )
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    groups = provider.plan_expert_groups(topk)
    assert groups == [[0, 1], [2, 3], [4, 5], [6, 7]]

    staged_results = []
    mapping_snapshots = []
    group_sums = []
    for group_index, expert_ids in enumerate(groups):
        result = provider.prepare_staged_group(expert_ids, group_index)
        staged_results.append(result)
        mapping_snapshots.append(result.expert_map.clone())
        assert result.w1_scale is not None
        assert result.w2_scale is not None
        group_sums.append(
            result.w1[: len(expert_ids)].float().sum()
            + result.w2[: len(expert_ids)].float().sum()
            + result.w1_scale[: len(expert_ids)].float().sum()
            + result.w2_scale[: len(expert_ids)].float().sum()
        )
        provider.release_staged_group(group_index)

    observed = torch.stack(group_sums).sum().cpu()
    assert scales is not None
    expected = (
        w13.float().sum()
        + w2.float().sum()
        + scales[0].float().sum()
        + scales[1].float().sum()
    )
    torch.testing.assert_close(observed, expected)

    for expert_ids, mapping in zip(groups, mapping_snapshots):
        mapping = mapping.cpu().tolist()
        for slot, expert_id in enumerate(expert_ids):
            assert mapping[expert_id] == slot
        assert all(
            mapping[expert_id] == -1
            for expert_id in range(8)
            if expert_id not in expert_ids
        )

    # Groups 0/2 and 1/3 reuse their respective banks only after an event.
    assert staged_results[0].w1 is staged_results[2].w1
    assert staged_results[1].w1 is staged_results[3].w1
    assert staged_results[0].w1 is not staged_results[1].w1
    assert provider.misses == 8
    assert provider.hits == 0
    assert provider._lru == {}, "prefill staging must preserve the decode cache"


def test_async_staging_waits_before_bank_zero_reuse_across_layers():
    """A following layer cannot overwrite bank zero while it is consumed."""
    first, first_w13, _, _ = _make_provider(
        num_experts=8,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    second, second_w13, _, _ = _make_provider(
        num_experts=8,
        capacity=2,
        split="expert",
        async_staging=True,
    )
    assert first._staging_pool is second._staging_pool

    first_result = first.prepare_staged_group([0, 1], 0)
    # Keep bank zero live long enough for an unsafe next-layer copy to race it.
    torch.cuda._sleep(50_000_000)
    first_sum = first_result.w1[:2].float().sum()
    first.release_staged_group(0)

    # The next layer starts its group numbering at zero and therefore reuses
    # the same shared bank. Its copy must wait for first_sum and the release
    # event above, despite group_index being zero.
    second_result = second.prepare_staged_group([2, 3], 0)
    second_sum = second_result.w1[:2].float().sum()
    second.release_staged_group(0)

    torch.testing.assert_close(first_sum.cpu(), first_w13[:2].float().sum())
    torch.testing.assert_close(second_sum.cpu(), second_w13[2:4].float().sum())


def test_group_count_is_independent_of_token_count():
    """Group count follows the expert set, not the batch size."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    few = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    many = _rows([[0, 1], [2, 3], [4, 5], [6, 7]] * 50)
    assert len(provider.plan_expert_groups(few)) == len(
        provider.plan_expert_groups(many)
    )


def test_expert_map_hides_other_groups():
    """Within a group only that group's experts are resident."""
    provider, w13, _, _ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    for group in provider.plan_expert_groups(topk):
        result = provider.prepare(topk, group)
        mapping = result.expert_map.tolist()
        for eid in group:
            slot = mapping[eid]
            assert slot >= 0
            torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        resident = {e for e, slot in enumerate(mapping) if slot >= 0}
        assert resident == set(group)


def test_group_execution_covers_every_pair():
    """Summing over groups touches each (token, expert) pair exactly once."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: list[int] = []
    firsts: list[bool] = []

    def run(result, rows, include_shared):
        firsts.append(include_shared)
        seen.extend(e for e, slot in enumerate(result.expert_map.tolist()) if slot >= 0)
        return torch.zeros(topk.size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert sorted(seen) == sorted(topk.unique().tolist())
    assert firsts[0] is True and not any(firsts[1:]), "first flag must fire once"
    assert out.shape == (topk.size(0), 4)


# -- Token-split execution --


def test_token_split_single_chunk_when_batch_fits():
    """A batch within capacity is evaluated in one piece."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    plan = provider.plan_chunks(_rows([[0, 1], [1, 2], [2, 3]]))
    assert [rows for rows, _ in plan] == [slice(0, 3)]
    assert plan[0][1] == [0, 1, 2, 3]


def test_token_split_covers_every_row_once():
    """Chunks tile the batch, and each fits the cache."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    plan = provider.plan_chunks(topk)
    chunks = [rows for rows, _ in plan]

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].stop == topk.size(0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.stop == nxt.start
    for rows, unique_ids in plan:
        assert topk[rows].unique().numel() <= provider.capacity
        assert unique_ids == sorted(topk[rows].unique().tolist())


def test_token_split_rejects_single_row_over_capacity():
    """No split helps when one token alone exceeds capacity."""
    provider, *_ = _make_provider(num_experts=8, capacity=2)
    with pytest.raises(RuntimeError, match="one token routes to"):
        provider.plan_chunks(_rows([[0, 1, 2, 3]]))


@pytest.mark.parametrize("split", ["token", "expert"])
def test_both_splits_reach_every_expert(split: str):
    """Whatever the split, every expert the batch needs gets loaded."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split=split)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: set[int] = set()

    def run(result, rows, include_shared):
        seen.update(e for e, s in enumerate(result.expert_map.tolist()) if s >= 0)
        return torch.zeros(topk[rows].size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert seen == set(topk.unique().tolist())
    assert out.shape == (topk.size(0), 4)
