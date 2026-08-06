# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental host-sync-free expert-cache planning and mapped gathers."""

from __future__ import annotations

from collections.abc import Callable

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _analyze_route_kernel(
    route,
    expert_to_slot,
    victim_expert_to_slot,
    frequency,
    last_access,
    clock_ptr,
    mapping,
    needed_out,
    miss_experts,
    miss_victim_slots,
    miss_is_victim,
    miss_count_ptr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ROUTE_IDS: tl.constexpr,
):
    experts = tl.arange(0, EXPERT_BLOCK)
    valid_expert = experts < NUM_EXPERTS
    needed = experts < 0
    for route_index in tl.static_range(0, ROUTE_IDS):
        needed = needed | (experts == tl.load(route + route_index))
    needed = needed & valid_expert

    old_slots = tl.load(expert_to_slot + experts, mask=valid_expert, other=-1)
    hits = needed & (old_slots >= 0)
    old_victim_slots = tl.load(
        victim_expert_to_slot + experts,
        mask=valid_expert,
        other=-1,
    )
    victim_hits = needed & ~hits & (old_victim_slots >= 0)
    misses = needed & ~hits
    host_misses = misses & ~victim_hits
    miss_ordinals = tl.cumsum(misses.to(tl.int32), axis=0) - 1
    needed_ordinals = tl.cumsum(needed.to(tl.int32), axis=0)
    miss_count = tl.sum(misses.to(tl.int32), axis=0)
    needed_count = tl.sum(needed.to(tl.int32), axis=0)
    old_clock = tl.load(clock_ptr)

    tl.store(
        mapping + experts,
        tl.where(hits, old_slots, -1),
        mask=valid_expert,
    )
    tl.store(needed_out + experts, needed.to(tl.uint8), mask=valid_expert)
    tl.store(miss_experts + miss_ordinals, experts, mask=misses)
    tl.store(
        miss_victim_slots + miss_ordinals,
        old_victim_slots,
        mask=misses,
    )
    tl.store(
        miss_is_victim + miss_ordinals,
        victim_hits.to(tl.uint8),
        mask=misses,
    )
    old_frequency = tl.load(frequency + experts, mask=valid_expert, other=0)
    tl.store(
        frequency + experts,
        tl.where(host_misses, 1, old_frequency + 1),
        mask=needed,
    )
    tl.store(
        last_access + experts,
        old_clock + needed_ordinals,
        mask=needed,
    )
    tl.store(clock_ptr, old_clock + needed_count)
    tl.store(miss_count_ptr, miss_count)


@triton.jit
def _rank_victim_slots_kernel(
    slot_to_expert,
    frequency,
    last_access,
    clock_ptr,
    needed,
    slot_ranks,
    CAPACITY: tl.constexpr,
    SLOT_BLOCK: tl.constexpr,
):
    slots = tl.arange(0, SLOT_BLOCK)
    valid = slots < CAPACITY
    residents = tl.load(slot_to_expert + slots, mask=valid, other=-1)
    resident_needed = tl.load(
        needed + residents, mask=valid & (residents >= 0), other=0
    ).to(tl.int1)
    candidates = valid & ~resident_needed
    is_free = residents < 0
    resident_frequency = tl.load(
        frequency + residents, mask=valid & ~is_free, other=0
    ).to(tl.int64)
    resident_last = tl.load(last_access + residents, mask=valid & ~is_free, other=0)
    age = tl.load(clock_ptr) - resident_last + 1

    self_candidate = candidates[:, None]
    other_candidate = candidates[None, :]
    self_free = is_free[:, None]
    other_free = is_free[None, :]
    self_slot = slots[:, None]
    other_slot = slots[None, :]
    self_frequency = resident_frequency[:, None]
    other_frequency = resident_frequency[None, :]
    self_age = age[:, None]
    other_age = age[None, :]
    other_score = other_frequency * self_age
    self_score = self_frequency * other_age
    other_lower_occupied = (
        (~other_free)
        & (~self_free)
        & (
            (other_score < self_score)
            | ((other_score == self_score) & (other_slot < self_slot))
        )
    )
    other_lower = (
        other_candidate
        & self_candidate
        & (
            (other_free & ~self_free)
            | (other_free & self_free & (other_slot < self_slot))
            | other_lower_occupied
        )
    )
    ranks = tl.sum(other_lower.to(tl.int32), axis=1)
    tl.store(slot_ranks + slots, ranks, mask=candidates)
    tl.store(slot_ranks + slots, SLOT_BLOCK, mask=valid & ~candidates)


@triton.jit
def _apply_primary_slots_kernel(
    expert_to_slot,
    slot_to_expert,
    victim_expert_to_slot,
    mapping,
    miss_experts,
    miss_slots,
    miss_victim_slots,
    miss_is_victim,
    displaced_experts,
    victim_promotion_slots,
    stash_victim_slots,
    miss_count_ptr,
    slot_ranks,
    CAPACITY: tl.constexpr,
    SLOT_BLOCK: tl.constexpr,
):
    slots = tl.arange(0, SLOT_BLOCK)
    valid = slots < CAPACITY
    ranks = tl.load(slot_ranks + slots, mask=valid, other=SLOT_BLOCK)
    miss_count = tl.load(miss_count_ptr)
    replace = valid & (ranks < miss_count)
    new_experts = tl.load(miss_experts + ranks, mask=replace, other=0)
    old_experts = tl.load(slot_to_expert + slots, mask=replace, other=-1)
    is_victim = tl.load(miss_is_victim + ranks, mask=replace, other=0).to(tl.int1)
    old_victim_slots = tl.load(miss_victim_slots + ranks, mask=replace, other=-1)
    tl.store(expert_to_slot + old_experts, -1, mask=replace & (old_experts >= 0))
    tl.store(slot_to_expert + slots, new_experts, mask=replace)
    tl.store(expert_to_slot + new_experts, slots, mask=replace)
    tl.store(mapping + new_experts, slots, mask=replace)
    tl.store(miss_slots + ranks, slots, mask=replace)
    tl.store(displaced_experts + ranks, old_experts, mask=replace)
    tl.store(stash_victim_slots + ranks, -1, mask=replace)
    tl.store(
        victim_expert_to_slot + new_experts,
        -1,
        mask=replace & is_victim,
    )
    tl.store(
        victim_promotion_slots + old_victim_slots,
        1,
        mask=replace & is_victim & (old_victim_slots >= 0),
    )


@triton.jit
def _plan_victim_stashes_kernel(
    victim_expert_to_slot,
    victim_slot_to_expert,
    frequency,
    last_access,
    clock_ptr,
    miss_slots,
    miss_is_victim,
    displaced_experts,
    stash_victim_slots,
    miss_count_ptr,
    victim_promotion_slots,
    CAPACITY: tl.constexpr,
    VICTIM_CAPACITY: tl.constexpr,
    VICTIM_BLOCK: tl.constexpr,
):
    """Keep the final LFRU victim set and record only necessary D2D stashes."""
    slots = tl.arange(0, VICTIM_BLOCK)
    valid = slots < VICTIM_CAPACITY
    residents = tl.load(victim_slot_to_expert + slots, mask=valid, other=-1)
    promoted = tl.load(victim_promotion_slots + slots, mask=valid, other=0).to(tl.int1)
    # Which admission owns the final new row for each physical victim slot.
    copy_admissions = tl.full((VICTIM_BLOCK,), -1, tl.int32)
    clock = tl.load(clock_ptr)
    miss_count = tl.load(miss_count_ptr)

    for admission in tl.static_range(0, CAPACITY):
        is_victim = tl.load(
            miss_is_victim + admission,
            mask=admission < miss_count,
            other=0,
        ).to(tl.int1)
        incoming = tl.load(
            displaced_experts + admission,
            mask=admission < miss_count,
            other=-1,
        )
        eligible = (admission < miss_count) & ~is_victim & (incoming >= 0)
        candidates = valid & ~promoted
        free = candidates & (residents < 0)
        free_slot = tl.min(tl.where(free, slots, VICTIM_BLOCK), axis=0)

        resident_frequency = tl.load(
            frequency + residents,
            mask=candidates & (residents >= 0),
            other=0,
        ).to(tl.float32)
        resident_last = tl.load(
            last_access + residents,
            mask=candidates & (residents >= 0),
            other=0,
        )
        age = (clock - resident_last + 1).to(tl.float32)
        scores = tl.where(
            candidates & (residents >= 0),
            resident_frequency / age,
            float("inf"),
        )
        minimum_score = tl.min(scores, axis=0)
        eviction_slot = tl.min(
            tl.where(
                candidates & (scores == minimum_score),
                slots,
                VICTIM_BLOCK,
            ),
            axis=0,
        )
        chosen = tl.where(free_slot < VICTIM_CAPACITY, free_slot, eviction_slot)
        chosen = tl.where(eligible & (chosen < VICTIM_CAPACITY), chosen, -1)
        selected = valid & (slots == chosen)
        old_resident = tl.sum(tl.where(selected, residents + 1, 0), axis=0) - 1

        tl.store(
            victim_expert_to_slot + old_resident,
            -1,
            mask=(chosen >= 0) & (old_resident >= 0),
        )
        tl.store(
            victim_expert_to_slot + incoming,
            chosen,
            mask=(chosen >= 0) & (incoming >= 0),
        )
        residents = tl.where(selected, incoming, residents)
        copy_admissions = tl.where(selected, admission, copy_admissions)

    # Promotion source rows are released only after stash planning. They stay
    # physically untouched until the fused admission gather reads them.
    residents = tl.where(promoted, -1, residents)
    copy_admissions = tl.where(promoted, -1, copy_admissions)
    tl.store(victim_slot_to_expert + slots, residents, mask=valid)
    tl.store(victim_promotion_slots + slots, 0, mask=valid)
    tl.store(
        stash_victim_slots + copy_admissions,
        slots,
        mask=valid & (copy_admissions >= 0),
    )


@triton.jit
def _admission_gather_kernel(
    host_source,
    victim_source,
    destination,
    victim_destination,
    miss_experts,
    miss_slots,
    miss_victim_slots,
    miss_is_victim,
    stash_victim_slots,
    miss_count_ptr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    miss_index = tl.program_id(0)
    block_index = tl.program_id(1)
    miss_count = tl.load(miss_count_ptr)
    active_miss = miss_index < miss_count
    source_row = tl.load(miss_experts + miss_index, mask=active_miss, other=0)
    destination_row = tl.load(miss_slots + miss_index, mask=active_miss, other=0)
    source_victim_row = tl.load(
        miss_victim_slots + miss_index, mask=active_miss, other=0
    )
    is_victim = tl.load(miss_is_victim + miss_index, mask=active_miss, other=0).to(
        tl.int1
    )
    stash_victim_row = tl.load(
        stash_victim_slots + miss_index, mask=active_miss, other=-1
    )
    for offset_base in tl.static_range(0, ROW_BYTES, NUM_TILES * BLOCK):
        offsets = offset_base + block_index * BLOCK + tl.arange(0, BLOCK)
        mask = active_miss & (offsets < ROW_BYTES)
        stash_mask = mask & (stash_victim_row >= 0)
        displaced_values = tl.load(
            destination + destination_row * ROW_BYTES + offsets,
            mask=stash_mask,
            other=0,
        )
        tl.store(
            victim_destination + stash_victim_row * ROW_BYTES + offsets,
            displaced_values,
            mask=stash_mask,
        )
        # Keep the old primary row live until its optional victim stash has
        # been issued by this same program.
        tl.debug_barrier()
        host_values = tl.load(
            host_source + source_row * ROW_BYTES + offsets,
            mask=mask & ~is_victim,
            other=0,
        )
        victim_values = tl.load(
            victim_source + source_victim_row * ROW_BYTES + offsets,
            mask=mask & is_victim,
            other=0,
        )
        values = tl.where(is_victim, victim_values, host_values)
        tl.store(
            destination + destination_row * ROW_BYTES + offsets,
            values,
            mask=mask,
        )


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


class GpuResidentExpertCache:
    """Own cache policy on the GPU and gather misses from mapped host rows.

    Device state remains authoritative through consecutive small decode calls.
    ``download_state`` is used only when a larger CPU-planned forward needs to
    take ownership again; ``upload_state`` restores device ownership afterward.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        capacity: int,
        host_tensors: list[torch.Tensor],
        device_tensors: list[torch.Tensor],
        mapping: torch.Tensor,
        mapped_alias: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if not 0 < capacity <= num_experts:
            raise ValueError("GPU-resident cache capacity is invalid")
        if capacity > 64:
            raise ValueError("GPU-resident cache currently supports capacity <= 64")
        if len(host_tensors) != len(device_tensors) or not host_tensors:
            raise ValueError("GPU-resident cache tensor groups differ")

        self.num_experts = num_experts
        self.capacity = capacity
        self.expert_block = _next_power_of_two(num_experts)
        self.slot_block = _next_power_of_two(capacity)
        self.mapping = mapping
        self.device_authoritative = False
        self.victim_capacity = 0

        self.sources: list[torch.Tensor] = []
        self.destinations: list[torch.Tensor] = []
        self.row_bytes: list[int] = []
        for host, device in zip(host_tensors, device_tensors):
            if (
                not host.is_contiguous()
                or not device.is_contiguous()
                or host.size(0) != num_experts
                or device.size(0) < capacity
            ):
                raise ValueError(
                    "GPU-resident cache requires contiguous per-expert tensors"
                )
            source = mapped_alias(host).view(torch.uint8).reshape(num_experts, -1)
            destination = device.view(torch.uint8).reshape(device.size(0), -1)
            if source.size(1) != destination.size(1):
                raise ValueError("GPU-resident cache row sizes differ")
            self.sources.append(source)
            self.destinations.append(destination)
            self.row_bytes.append(source.size(1))

        device = mapping.device
        self.expert_to_slot = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=device
        )
        self.slot_to_expert = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )
        self.frequency = torch.zeros(num_experts, dtype=torch.int64, device=device)
        self.last_access = torch.zeros(num_experts, dtype=torch.int64, device=device)
        self.clock = torch.zeros(1, dtype=torch.int64, device=device)
        self.needed = torch.zeros(num_experts, dtype=torch.uint8, device=device)
        self.miss_experts = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.miss_slots = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.slot_ranks = torch.zeros(self.slot_block, dtype=torch.int32, device=device)
        self.miss_victim_slots = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )
        self.miss_is_victim = torch.zeros(capacity, dtype=torch.uint8, device=device)
        self.displaced_experts = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )
        self.stash_victim_slots = torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        )
        self.victim_expert_to_slot = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=device
        )
        self.victim_slot_to_expert: torch.Tensor | None = None
        self.victim_promotion_slots: torch.Tensor | None = None
        self.victim_sources: list[torch.Tensor] = []
        self.victim_destinations: list[torch.Tensor] = []
        self.host_victim_expert_to_slot: torch.Tensor | None = None

        self.host_expert_to_slot = torch.full(
            (num_experts,), -1, dtype=torch.int32, pin_memory=True
        )
        self.host_frequency = torch.zeros(
            num_experts, dtype=torch.int64, pin_memory=True
        )
        self.host_last_access = torch.zeros(
            num_experts, dtype=torch.int64, pin_memory=True
        )
        self.host_clock = torch.zeros(1, dtype=torch.int64, pin_memory=True)

    def attach_victim_tensors(self, victim_tensors: list[torch.Tensor]) -> None:
        if self.device_authoritative:
            raise RuntimeError("cannot attach a victim tier during device ownership")
        if self.victim_capacity:
            raise RuntimeError("GPU-resident victim tier is already attached")
        if len(victim_tensors) != len(self.destinations):
            raise ValueError("GPU-resident victim tensor groups differ")
        capacity = victim_tensors[0].size(0)
        if not 0 < capacity <= 32:
            raise ValueError("GPU-resident victim capacity must be in [1, 32]")
        for primary, victim in zip(self.destinations, victim_tensors):
            victim_bytes = victim.view(torch.uint8).reshape(capacity, -1)
            if (
                victim.device != primary.device
                or not victim.is_contiguous()
                or victim_bytes.size(1) != primary.size(1)
            ):
                raise ValueError("GPU-resident victim tensor rows differ")
            self.victim_sources.append(victim_bytes)
            self.victim_destinations.append(victim_bytes)
        self.victim_capacity = capacity
        self.victim_slot_to_expert = torch.full(
            (capacity,), -1, dtype=torch.int32, device=self.mapping.device
        )
        self.victim_promotion_slots = torch.zeros(
            capacity, dtype=torch.uint8, device=self.mapping.device
        )
        self.host_victim_expert_to_slot = torch.full(
            (self.num_experts,), -1, dtype=torch.int32, pin_memory=True
        )

    def detach_victim_tensors(self) -> None:
        if self.device_authoritative:
            raise RuntimeError("download GPU cache state before victim revocation")
        self.victim_capacity = 0
        self.victim_sources.clear()
        self.victim_destinations.clear()
        self.victim_slot_to_expert = None
        self.victim_promotion_slots = None
        self.host_victim_expert_to_slot = None
        self.victim_expert_to_slot.fill_(-1)

    def upload_state(
        self,
        lru: dict[int, list[int]],
        clock: int,
        victim_lru: dict[int, list[int]] | None = None,
    ) -> None:
        self.host_expert_to_slot.fill_(-1)
        self.host_frequency.zero_()
        self.host_last_access.zero_()
        for expert, (slot, frequency, last_access) in lru.items():
            self.host_expert_to_slot[expert] = slot
            self.host_frequency[expert] = frequency
            self.host_last_access[expert] = last_access
        victim_lru = victim_lru or {}
        if victim_lru and not self.victim_capacity:
            raise RuntimeError("victim state was provided without attached tensors")
        for expert, (_, frequency, last_access) in victim_lru.items():
            if expert in lru:
                raise ValueError("an expert cannot reside in both cache tiers")
            self.host_frequency[expert] = frequency
            self.host_last_access[expert] = last_access
        self.host_clock[0] = clock
        self.expert_to_slot.copy_(self.host_expert_to_slot, non_blocking=True)
        self.frequency.copy_(self.host_frequency, non_blocking=True)
        self.last_access.copy_(self.host_last_access, non_blocking=True)
        self.clock.copy_(self.host_clock, non_blocking=True)
        self.slot_to_expert.fill_(-1)
        if lru:
            experts = torch.tensor(
                list(lru), dtype=torch.int32, device=self.mapping.device
            )
            slots = self.expert_to_slot[experts.to(torch.int64)]
            self.slot_to_expert.scatter_(0, slots.to(torch.int64), experts)
        self.victim_expert_to_slot.fill_(-1)
        if self.victim_capacity:
            assert self.victim_slot_to_expert is not None
            assert self.host_victim_expert_to_slot is not None
            self.host_victim_expert_to_slot.fill_(-1)
            self.victim_slot_to_expert.fill_(-1)
            if victim_lru:
                for expert, (slot, _, _) in victim_lru.items():
                    self.host_victim_expert_to_slot[expert] = slot
                self.victim_expert_to_slot.copy_(
                    self.host_victim_expert_to_slot, non_blocking=True
                )
                experts = torch.tensor(
                    list(victim_lru), dtype=torch.int32, device=self.mapping.device
                )
                slots = self.victim_expert_to_slot[experts.to(torch.int64)]
                self.victim_slot_to_expert.scatter_(0, slots.to(torch.int64), experts)
        # All state uploads and the first plan share the current stream.
        self.device_authoritative = True

    def download_state(self) -> tuple[dict[int, list[int]], int, list[int]]:
        self.host_expert_to_slot.copy_(self.expert_to_slot, non_blocking=True)
        self.host_frequency.copy_(self.frequency, non_blocking=True)
        self.host_last_access.copy_(self.last_access, non_blocking=True)
        self.host_clock.copy_(self.clock, non_blocking=True)
        if self.victim_capacity:
            assert self.host_victim_expert_to_slot is not None
            self.host_victim_expert_to_slot.copy_(
                self.victim_expert_to_slot, non_blocking=True
            )
        torch.cuda.current_stream(device=self.mapping.device).synchronize()
        slots = self.host_expert_to_slot.tolist()
        frequencies = self.host_frequency.tolist()
        last_accesses = self.host_last_access.tolist()
        lru = {
            expert: [slot, frequencies[expert], last_accesses[expert]]
            for expert, slot in enumerate(slots)
            if slot >= 0
        }
        used_slots = {entry[0] for entry in lru.values()}
        free_slots = [slot for slot in range(self.capacity) if slot not in used_slots]
        self.device_authoritative = False
        return lru, int(self.host_clock[0]), free_slots

    def downloaded_victim_state(
        self,
    ) -> tuple[dict[int, list[int]], list[int]]:
        if not self.victim_capacity:
            return {}, []
        if self.device_authoritative:
            raise RuntimeError("download primary state before reading victim state")
        assert self.host_victim_expert_to_slot is not None
        slots = self.host_victim_expert_to_slot.tolist()
        frequencies = self.host_frequency.tolist()
        last_accesses = self.host_last_access.tolist()
        lru = {
            expert: [slot, frequencies[expert], last_accesses[expert]]
            for expert, slot in enumerate(slots)
            if slot >= 0
        }
        used_slots = {entry[0] for entry in lru.values()}
        free_slots = [
            slot for slot in range(self.victim_capacity) if slot not in used_slots
        ]
        return lru, free_slots

    def prepare(self, route: torch.Tensor) -> torch.Tensor:
        if not self.device_authoritative:
            raise RuntimeError("GPU-resident expert-cache state was not uploaded")
        route = route.reshape(-1)
        route_ids = route.numel()
        if route_ids > self.capacity:
            raise ValueError("GPU-resident route exceeds cache capacity")

        _analyze_route_kernel[(1,)](
            route,
            self.expert_to_slot,
            self.victim_expert_to_slot,
            self.frequency,
            self.last_access,
            self.clock,
            self.mapping,
            self.needed,
            self.miss_experts,
            self.miss_victim_slots,
            self.miss_is_victim,
            self.miss_count,
            NUM_EXPERTS=self.num_experts,
            EXPERT_BLOCK=self.expert_block,
            ROUTE_IDS=route_ids,
            num_warps=max(1, min(8, self.expert_block // 32)),
        )
        _rank_victim_slots_kernel[(1,)](
            self.slot_to_expert,
            self.frequency,
            self.last_access,
            self.clock,
            self.needed,
            self.slot_ranks,
            CAPACITY=self.capacity,
            SLOT_BLOCK=self.slot_block,
            num_warps=max(1, min(4, self.slot_block // 16)),
        )
        victim_promotion_slots = self.victim_promotion_slots
        if victim_promotion_slots is None:
            # A one-byte inert target keeps the same compiled planner shape
            # while no KV-backed victim bank is attached.
            victim_promotion_slots = self.miss_is_victim[:1]
        _apply_primary_slots_kernel[(1,)](
            self.expert_to_slot,
            self.slot_to_expert,
            self.victim_expert_to_slot,
            self.mapping,
            self.miss_experts,
            self.miss_slots,
            self.miss_victim_slots,
            self.miss_is_victim,
            self.displaced_experts,
            victim_promotion_slots,
            self.stash_victim_slots,
            self.miss_count,
            self.slot_ranks,
            CAPACITY=self.capacity,
            SLOT_BLOCK=self.slot_block,
            num_warps=max(1, min(4, self.slot_block // 16)),
        )
        if self.victim_capacity:
            assert self.victim_slot_to_expert is not None
            assert self.victim_promotion_slots is not None
            victim_block = _next_power_of_two(self.victim_capacity)
            _plan_victim_stashes_kernel[(1,)](
                self.victim_expert_to_slot,
                self.victim_slot_to_expert,
                self.frequency,
                self.last_access,
                self.clock,
                self.miss_slots,
                self.miss_is_victim,
                self.displaced_experts,
                self.stash_victim_slots,
                self.miss_count,
                self.victim_promotion_slots,
                CAPACITY=self.capacity,
                VICTIM_CAPACITY=self.victim_capacity,
                VICTIM_BLOCK=victim_block,
                num_warps=max(1, min(4, victim_block // 16)),
            )
            victim_sources = self.victim_sources
            victim_destinations = self.victim_destinations
        else:
            # No admission can be a victim hit, and no stash slot is assigned.
            # Reuse primary pointers so the inactive kernel operands are valid.
            victim_sources = self.destinations
            victim_destinations = self.destinations
        tensor_groups = zip(
            self.row_bytes,
            self.sources,
            victim_sources,
            self.destinations,
            victim_destinations,
        )
        for (
            row_bytes,
            host_source,
            victim_source,
            destination,
            victim_destination,
        ) in tensor_groups:
            _admission_gather_kernel[(self.capacity, 64)](
                host_source,
                victim_source,
                destination,
                victim_destination,
                self.miss_experts,
                self.miss_slots,
                self.miss_victim_slots,
                self.miss_is_victim,
                self.stash_victim_slots,
                self.miss_count,
                ROW_BYTES=row_bytes,
                BLOCK=1024,
                NUM_TILES=64,
                num_warps=4,
            )
        return self.mapping
