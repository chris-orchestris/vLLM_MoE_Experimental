# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import expert_weight_provider
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    run_with_expert_cache,
)
from vllm.model_executor.layers.fused_moe.gpu_resident_expert_cache import (
    GpuResidentExpertCache,
)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_run_with_gpu_resident_cache_chunks_complete_token_rows() -> None:
    class Provider:
        capacity = 12
        gpu_resident_cache_enabled = True
        gpu_resident_max_tokens = 24
        split = "expert"

        def __init__(self) -> None:
            self.routes: list[torch.Tensor] = []
            self.finished: list[torch.Tensor] = []

        def prepare_gpu_resident(self, route: torch.Tensor):
            self.routes.append(route.clone())
            return SimpleNamespace(expert_map=torch.empty(0))

        def release_mapped_scale_bank(self, result) -> None:
            return None

        def remember_prefill_route(self, route: torch.Tensor) -> None:
            self.finished.append(route)

        def prefetch_successor(self, num_tokens: int) -> None:
            return None

    provider = Provider()
    topk = torch.arange(30, dtype=torch.int32).view(5, 6)
    include_shared: list[bool] = []
    rows_seen: list[slice] = []

    def run(result, rows: slice, shared: bool) -> torch.Tensor:
        include_shared.append(shared)
        rows_seen.append(rows)
        return torch.arange(5, dtype=torch.float32)[rows, None]

    output = run_with_expert_cache(provider, topk, run)

    assert [route.shape for route in provider.routes] == [(2, 6), (2, 6), (1, 6)]
    assert rows_seen == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert include_shared == [True, True, True]
    torch.testing.assert_close(output[:, 0], torch.arange(5, dtype=torch.float32))
    assert provider.finished == [topk]


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_gpu_resident_fallback_acquires_host_state_before_group_planning() -> None:
    class Provider:
        capacity = 4
        gpu_resident_cache_enabled = True
        gpu_resident_max_tokens = 1
        split = "expert"
        async_staging_ready = False
        async_expert_map_enabled = False

        def __init__(self) -> None:
            self.host_authoritative = False

        def acquire_host_cache_state(self) -> None:
            self.host_authoritative = True

        def take_lookahead_group(self, unique_ids):
            assert self.host_authoritative
            return None

        def plan_borrowed_groups(self, *args, **kwargs):
            assert self.host_authoritative
            return []

        def uses_bulk_prefill(self, route: torch.Tensor) -> bool:
            assert self.host_authoritative
            return False

        def plan_expert_groups(self, *args, **kwargs):
            assert self.host_authoritative
            return [[0, 1, 2, 3]]

        def prepare(self, *args, **kwargs):
            assert self.host_authoritative
            return SimpleNamespace(expert_map=torch.empty(0))

        def release_mapped_scale_bank(self, result) -> None:
            return None

        def remember_prefill_route(self, route: torch.Tensor) -> None:
            return None

        def prefetch_successor(self, num_tokens: int) -> None:
            return None

    provider = Provider()
    topk = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)

    output = run_with_expert_cache(
        provider,
        topk,
        lambda result, rows, include_shared: torch.zeros(2, 4),
    )

    assert provider.host_authoritative
    assert output.shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_resident_cache_plans_gathers_and_transfers_ownership() -> None:
    num_experts = 16
    capacity = 8
    host_weight = torch.arange(num_experts * 96, dtype=torch.int32).reshape(
        num_experts, 96
    )
    device_weight = torch.empty(capacity, 96, dtype=torch.int32, device="cuda")
    mapping = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache = GpuResidentExpertCache(
        num_experts=num_experts,
        capacity=capacity,
        host_tensors=[host_weight.pin_memory()],
        device_tensors=[device_weight],
        mapping=mapping,
        # Kernel correctness does not depend on PCIe mapping. Keep this unit
        # test independent of the optional native mapped-tensor extension.
        mapped_alias=lambda tensor: tensor.to(device="cuda"),
    )
    cache.upload_state({}, 0)

    routes = [
        torch.tensor(route, dtype=torch.int32, device="cuda")
        for route in (
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [5, 6, 7, 8],
            [7, 8, 9, 10],
            [1, 8, 10, 15],
        )
    ]
    expected_clock = 0
    for route in routes:
        result_mapping = cache.prepare(route)
        torch.accelerator.synchronize()
        expected = set(route.tolist())
        actual_mapping = result_mapping.cpu().tolist()
        assert {
            expert for expert, slot in enumerate(actual_mapping) if slot >= 0
        } == expected
        for expert in expected:
            slot = actual_mapping[expert]
            torch.testing.assert_close(device_weight[slot].cpu(), host_weight[expert])
        expected_clock += len(expected)

    lru, clock, free_slots = cache.download_state()
    assert clock == expected_clock
    assert len(lru) == capacity
    assert free_slots == []
    assert set(lru) == {
        expert
        for expert, slot in enumerate(cache.host_expert_to_slot.tolist())
        if slot >= 0
    }

    # A CPU-planned forward can take ownership, then return the exact state to
    # the device without invalidating resident weights.
    cache.upload_state(lru, clock)
    final_route = torch.tensor([1, 10, 11, 12], dtype=torch.int32, device="cuda")
    final_mapping = cache.prepare(final_route)
    torch.accelerator.synchronize()
    assert {
        expert for expert, slot in enumerate(final_mapping.cpu().tolist()) if slot >= 0
    } == {1, 10, 11, 12}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_resident_cache_rejects_route_wider_than_capacity() -> None:
    host = torch.zeros(8, 8, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(4, 8, dtype=torch.uint8, device="cuda")
    mapping = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    cache = GpuResidentExpertCache(
        num_experts=8,
        capacity=4,
        host_tensors=[host],
        device_tensors=[device],
        mapping=mapping,
        mapped_alias=lambda tensor: tensor.to(device="cuda"),
    )
    cache.upload_state({}, 0)
    with pytest.raises(ValueError, match="route exceeds cache capacity"):
        cache.prepare(torch.arange(5, dtype=torch.int32, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_resident_cache_promotes_and_reuses_copy_only_victim_rows() -> None:
    num_experts = 12
    capacity = 4
    victim_capacity = 2
    host_weight = torch.arange(num_experts * 96, dtype=torch.int32).reshape(
        num_experts, 96
    )
    device_weight = torch.empty(capacity, 96, dtype=torch.int32, device="cuda")
    victim_weight = torch.empty(victim_capacity, 96, dtype=torch.int32, device="cuda")
    mapping = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    cache = GpuResidentExpertCache(
        num_experts=num_experts,
        capacity=capacity,
        host_tensors=[host_weight.pin_memory()],
        device_tensors=[device_weight],
        mapping=mapping,
        mapped_alias=lambda tensor: tensor.to(device="cuda"),
    )
    cache.attach_victim_tensors([victim_weight])

    primary_lru = {
        expert: [slot, expert + 1, expert + 1]
        for slot, expert in enumerate((0, 1, 2, 3))
    }
    victim_lru = {
        4: [0, 5, 5],
        5: [1, 4, 6],
    }
    for expert, (slot, _, _) in primary_lru.items():
        device_weight[slot].copy_(host_weight[expert])
    for expert, (slot, _, _) in victim_lru.items():
        victim_weight[slot].copy_(host_weight[expert])
    cache.upload_state(primary_lru, 6, victim_lru)

    # Expert 4 promotes from the victim tier while expert 6 is a host miss.
    # Its promotion source slot is protected until the fused row gather reads
    # it; the host miss may preserve a different displaced primary resident.
    route = torch.tensor([1, 4, 6], dtype=torch.int32, device="cuda")
    result_mapping = cache.prepare(route)
    torch.accelerator.synchronize()
    actual_mapping = result_mapping.cpu().tolist()
    assert {expert for expert, slot in enumerate(actual_mapping) if slot >= 0} == {
        1,
        4,
        6,
    }
    for expert in (1, 4, 6):
        torch.testing.assert_close(
            device_weight[actual_mapping[expert]].cpu(), host_weight[expert]
        )

    primary_lru, clock, free_slots = cache.download_state()
    downloaded_victim, victim_free_slots = cache.downloaded_victim_state()
    assert free_slots == []
    assert clock == 9
    assert 4 in primary_lru
    assert 4 not in downloaded_victim
    assert set(primary_lru).isdisjoint(downloaded_victim)
    assert len(downloaded_victim) + len(victim_free_slots) == victim_capacity
    for expert, (slot, _, _) in downloaded_victim.items():
        torch.testing.assert_close(victim_weight[slot].cpu(), host_weight[expert])

    # Hand the two-tier state back to the device and promote the row which the
    # prior host miss stashed. This catches stale victim maps/storage aliases.
    promoted = next(iter(downloaded_victim))
    cache.upload_state(primary_lru, clock, downloaded_victim)
    second_route = torch.tensor([promoted, 7], dtype=torch.int32, device="cuda")
    second_mapping = cache.prepare(second_route)
    torch.accelerator.synchronize()
    actual_mapping = second_mapping.cpu().tolist()
    for expert in (promoted, 7):
        torch.testing.assert_close(
            device_weight[actual_mapping[expert]].cpu(), host_weight[expert]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cached_provider_hands_state_between_gpu_and_host_planners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE", "1")
    monkeypatch.delenv("VLLM_MOE_KV_TAIL_DECODE_VICTIM", raising=False)
    monkeypatch.setattr(
        expert_weight_provider,
        "_mapped_cuda_alias",
        lambda tensor: tensor.to(device="cuda"),
    )
    weights = torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64)
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=weights,
        w2_weight=weights + 1000,
        split="expert",
    )

    first = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    first_result = provider.prepare_gpu_resident(first)
    torch.accelerator.synchronize()
    assert {
        expert
        for expert, slot in enumerate(first_result.expert_map.cpu().tolist())
        if slot >= 0
    } == {0, 1, 2}

    # The ordinary path downloads device policy state before mutating it.
    host_route = torch.tensor([[2, 3]], dtype=torch.int32, device="cuda")
    host_result = provider.prepare(host_route, unique_ids=[2, 3])
    assert provider._gpu_resident_cache is not None
    assert not provider._gpu_resident_cache.device_authoritative
    assert {2, 3}.issubset(provider._lru)
    assert {
        expert
        for expert, slot in enumerate(host_result.expert_map.cpu().tolist())
        if slot >= 0
    } == {2, 3}

    final_route = torch.tensor([[3, 4]], dtype=torch.int32, device="cuda")
    final_result = provider.prepare_gpu_resident(final_route)
    torch.accelerator.synchronize()
    assert {
        expert
        for expert, slot in enumerate(final_result.expert_map.cpu().tolist())
        if slot >= 0
    } == {3, 4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cached_provider_hands_victim_state_and_revokes_kv_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE", "1")
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_DECODE_VICTIM", "1")
    monkeypatch.setattr(
        expert_weight_provider,
        "_mapped_cuda_alias",
        lambda tensor: tensor.to(device="cuda"),
    )
    weights = torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64)
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=weights,
        w2_weight=weights + 1000,
        split="expert",
    )
    borrowed = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for dtype, shape in provider.borrowed_tensor_specs(2)
    ]
    provider.attach_borrowed_buffers(borrowed, min_tokens=1)
    gpu_cache = provider._gpu_resident_cache
    assert gpu_cache is not None
    assert gpu_cache.victim_capacity == 2

    provider.prepare_gpu_resident(
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
    )
    provider.prepare_gpu_resident(
        torch.tensor([[4, 5]], dtype=torch.int32, device="cuda")
    )
    torch.accelerator.synchronize()

    provider._download_gpu_cache_state()
    bank = provider._borrowed_banks[0]
    assert bank.slots
    assert set(provider._lru).isdisjoint(bank.slots)
    promoted = next(iter(bank.slots))
    result = provider.prepare_gpu_resident(
        torch.tensor([[promoted, 6]], dtype=torch.int32, device="cuda")
    )
    torch.accelerator.synchronize()
    result_mapping = result.expert_map.cpu().tolist()
    for expert in (promoted, 6):
        slot = result_mapping[expert]
        torch.testing.assert_close(result.w1[slot].cpu(), weights[expert])
        torch.testing.assert_close(result.w2[slot].cpu(), weights[expert] + 1000)

    # The scheduler may reclaim this KV-backed storage at its fence. State is
    # downloaded first, then all device aliases are dropped before the bank is
    # marked unavailable; subsequent GPU planning cannot touch revoked pages.
    provider.disable_borrowed_buffers()
    assert gpu_cache.victim_capacity == 0
    assert not bank.enabled
    final = provider.prepare_gpu_resident(
        torch.tensor([[promoted, 7]], dtype=torch.int32, device="cuda")
    )
    torch.accelerator.synchronize()
    final_mapping = final.expert_map.cpu().tolist()
    for expert in (promoted, 7):
        slot = final_mapping[expert]
        torch.testing.assert_close(final.w1[slot].cpu(), weights[expert])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cached_provider_retires_redundant_victim_before_gpu_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary copy wins when host planning temporarily overlaps tiers."""
    monkeypatch.setenv("VLLM_MOE_GPU_RESIDENT_EXPERT_CACHE", "1")
    monkeypatch.setenv("VLLM_MOE_KV_TAIL_DECODE_VICTIM", "1")
    monkeypatch.setattr(
        expert_weight_provider,
        "_mapped_cuda_alias",
        lambda tensor: tensor.to(device="cuda"),
    )
    weights = torch.arange(8 * 64, dtype=torch.int32).reshape(8, 64)
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=weights,
        w2_weight=weights + 1000,
        split="expert",
    )
    borrowed = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for dtype, shape in provider.borrowed_tensor_specs(2)
    ]
    provider.attach_borrowed_buffers(borrowed, min_tokens=1)

    # Reproduce the warm-up handoff: a host-planned decode has made expert 0
    # primary while a preceding prefill still records a redundant victim copy.
    provider._buf_w13[0].copy_(weights[0])
    provider._buf_w2[0].copy_(weights[0] + 1000)
    provider._lru = {0: [0, 2, 3]}
    provider._clock = 3
    bank = provider._borrowed_banks[0]
    bank.w13[0].copy_(weights[0])
    bank.w2[0].copy_(weights[0] + 1000)
    bank.slots[0] = 0
    bank.lfru[0] = [1, 2]
    bank.free_slots.remove(0)

    result = provider.prepare_gpu_resident(
        torch.tensor([[0]], dtype=torch.int32, device="cuda")
    )
    torch.accelerator.synchronize()

    assert 0 not in bank.slots
    assert 0 not in bank.lfru
    assert sorted(bank.free_slots) == [0, 1]
    slot = result.expert_map.cpu().tolist()[0]
    torch.testing.assert_close(result.w1[slot].cpu(), weights[0])
    torch.testing.assert_close(result.w2[slot].cpu(), weights[0] + 1000)
