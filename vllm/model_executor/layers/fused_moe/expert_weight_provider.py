# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger

logger = init_logger(__name__)

_STATS_LOG_INTERVAL = 64


def _pinned_cpu_copy(src: torch.Tensor) -> torch.Tensor:
    """Pinned CPU copy of *src* with exactly one cross-device transfer.

    ``src.cpu().pin_memory()`` stages GPU tensors through pageable memory,
    copying twice; copying straight into a pinned allocation halves init
    time and transient host RAM for multi-GB expert tensors.
    """
    if src.device.type == "cpu":
        return src if src.is_pinned() else src.pin_memory()
    dst = torch.empty_like(src, device="cpu", pin_memory=True)
    dst.copy_(src)
    return dst


# How a forward too wide for the cache is broken up.
#   "token"  -- split the rows; every token's full sum still happens inside one
#               kernel call, so output matches the uncached path exactly. Costs
#               one launch per chunk, which approaches one per token as
#               capacity approaches top_k.
#   "expert" -- split the experts; one launch per ceil(experts/capacity)
#               regardless of batch size, and each expert is fetched at most
#               once per forward. Each group's partial sum is rounded to the
#               model dtype before being accumulated, so results differ from
#               the uncached path at rounding level.
MoECacheSplit = Literal["token", "expert"]

# Every row: what the expert split passes, since it never cuts the batch.
_ALL_ROWS = slice(None)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption.

    ``expert_map`` follows the expert-parallel convention the kernels already
    understand: global expert id to buffer slot, or -1 for experts that are not
    resident. Pairs routed to -1 are dropped during alignment, which is what
    lets one forward be evaluated a group of experts at a time.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    expert_map: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


class _SharedExpertStagingPool:
    """Two GPU banks shared by every compatible MoE layer on one device.

    Long prefills touch nearly every expert in every layer, so preserving the
    contents of the per-layer decode cache has little value for those forwards.
    A pair of device-wide staging banks lets the next expert group transfer on
    a copy stream while the current group executes, without doubling all of
    the persistent per-layer caches.
    """

    def __init__(
        self,
        capacity: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
        expert_map: torch.Tensor,
        w13_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
    ) -> None:
        self.copy_stream = torch.cuda.Stream(device=w13.device)
        self.banks: list[ExpertWeightResult] = []
        for _ in range(2):
            self.banks.append(
                ExpertWeightResult(
                    w1=torch.empty_like(w13, device=w13.device),
                    w2=torch.empty_like(w2, device=w2.device),
                    expert_map=torch.empty_like(expert_map, device=expert_map.device),
                    w1_scale=(
                        torch.empty_like(w13_scale, device=w13_scale.device)
                        if w13_scale is not None
                        else None
                    ),
                    w2_scale=(
                        torch.empty_like(w2_scale, device=w2_scale.device)
                        if w2_scale is not None
                        else None
                    ),
                )
            )
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.consumed = [torch.cuda.Event() for _ in range(2)]
        self.capacity = capacity


_SHARED_STAGING_POOLS: dict[tuple[object, ...], _SharedExpertStagingPool] = {}


def _staging_pool_key(
    capacity: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_map: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
) -> tuple[object, ...]:
    def tensor_key(tensor: torch.Tensor | None) -> tuple[object, ...] | None:
        if tensor is None:
            return None
        return (tuple(tensor.shape), tensor.dtype, tensor.device)

    return (
        capacity,
        tensor_key(w13),
        tensor_key(w2),
        tensor_key(expert_map),
        tensor_key(w13_scale),
        tensor_key(w2_scale),
    )


def _get_shared_staging_pool(
    capacity: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_map: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
) -> _SharedExpertStagingPool:
    key = _staging_pool_key(capacity, w13, w2, expert_map, w13_scale, w2_scale)
    pool = _SHARED_STAGING_POOLS.get(key)
    if pool is None:
        pool = _SharedExpertStagingPool(
            capacity, w13, w2, expert_map, w13_scale, w2_scale
        )
        _SHARED_STAGING_POOLS[key] = pool
    return pool


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    prepare() copies any missing experts from CPU to GPU, evicting the
    lowest-scored resident entry when the buffer is full, and returns an
    ExpertWeightResult whose expert_map selects them. Forwards needing more
    experts than fit are handled by run_with_expert_cache(), which splits
    them according to ``split``.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        split: MoECacheSplit = "token",
        name: str = "unnamed",
        async_staging: bool | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self.name = name
        self.split: MoECacheSplit = split
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self._prepare_calls = 0

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
        else:
            cuda_device = w13_weight.device
        self._cpu_w13: torch.Tensor = _pinned_cpu_copy(w13_weight)
        self._cpu_w2: torch.Tensor = _pinned_cpu_copy(w2_weight)

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            # Pinned for the same reason the weights are: these are copied on
            # every miss, and pageable source memory forces a staging copy.
            self._cpu_w13_scale: torch.Tensor | None = _pinned_cpu_copy(w13_scale)
            self._cpu_w2_scale: torch.Tensor | None = _pinned_cpu_copy(w2_scale)
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))

        # Expert map handed to the kernel: expert id to slot for the group
        # being evaluated, -1 for everything else. Rebuilt each prepare() --
        # it must expose exactly the requested group, not whatever else
        # happens to still be resident, or experts already summed in an
        # earlier group would be counted twice. Residency itself lives in
        # _lru; this is only the view the kernel gets. Built in a pinned host
        # mirror and uploaded once, rather than a transfer per entry.
        self._mapping: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=cuda_device
        )
        self._mapping_host: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32
        ).pin_memory()

        if async_staging is None:
            async_staging = os.getenv(
                "VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING", "0"
            ).lower() in {"1", "true"}
        self._async_staging = async_staging and split == "expert"
        self._staging_pool: _SharedExpertStagingPool | None = None
        self._staging_mapping_hosts: list[torch.Tensor] = []
        if self._async_staging:
            self._staging_pool = _get_shared_staging_pool(
                capacity,
                self._buf_w13,
                self._buf_w2,
                self._mapping,
                self._buf_w13_scale,
                self._buf_w2_scale,
            )
            num_groups = (num_experts + capacity - 1) // capacity
            self._staging_mapping_hosts = [
                torch.full((num_experts,), -1, dtype=torch.int32).pin_memory()
                for _ in range(num_groups)
            ]

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])

    @property
    def async_staging_enabled(self) -> bool:
        return self._staging_pool is not None

    def _record_prepare_stats(self, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses
        self._prepare_calls += 1
        if self._prepare_calls % _STATS_LOG_INTERVAL == 0:
            total = self.hits + self.misses
            if total > 0:
                logger.info(
                    "Expert cache %s: %d hits, %d misses (%.1f%% hit rate)",
                    self.name,
                    self.hits,
                    self.misses,
                    100.0 * self.hits / total,
                )

    @eager_break_during_capture
    @torch.compiler.disable
    def prepare_staged_group(
        self, unique_ids: list[int], group_index: int
    ) -> ExpertWeightResult:
        """Stage one prefill expert group on a device-wide double buffer.

        The copy stream waits before reusing a bank, while the compute stream
        waits only for the new group to become ready. Each group owns a pinned
        mapping mirror for the duration of the forward, so the CPU never
        rewrites memory that an asynchronous H2D copy may still be reading.
        """
        pool = self._staging_pool
        if pool is None:
            raise RuntimeError("asynchronous expert staging is not enabled")
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but staging capacity is {self.capacity}."
            )
        if group_index >= len(self._staging_mapping_hosts):
            raise RuntimeError(
                f"expert staging group index {group_index} exceeds the "
                f"{len(self._staging_mapping_hosts)} available mappings"
            )

        bank_index = group_index % 2
        result = pool.banks[bank_index]
        mapping_host = self._staging_mapping_hosts[group_index]
        mapping_host.fill_(-1)
        for slot, expert_id in enumerate(unique_ids):
            mapping_host[expert_id] = slot

        compute_stream = torch.cuda.current_stream(device=result.w1.device)
        with torch.cuda.stream(pool.copy_stream):
            if group_index >= 2:
                pool.copy_stream.wait_event(pool.consumed[bank_index])
            for slot, expert_id in enumerate(unique_ids):
                result.w1[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
                result.w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
                if result.w1_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert result.w2_scale is not None
                    result.w1_scale[slot].copy_(
                        self._cpu_w13_scale[expert_id], non_blocking=True
                    )
                    result.w2_scale[slot].copy_(
                        self._cpu_w2_scale[expert_id], non_blocking=True
                    )
            result.expert_map.copy_(mapping_host, non_blocking=True)
            pool.ready[bank_index].record(pool.copy_stream)

        compute_stream.wait_event(pool.ready[bank_index])
        self._record_prepare_stats(hits=0, misses=len(unique_ids))
        return result

    @eager_break_during_capture
    @torch.compiler.disable
    def release_staged_group(self, group_index: int) -> None:
        """Mark a staged bank reusable after its MoE kernel is enqueued."""
        pool = self._staging_pool
        if pool is None:
            raise RuntimeError("asynchronous expert staging is not enabled")
        bank_index = group_index % 2
        pool.consumed[bank_index].record(
            torch.cuda.current_stream(device=pool.banks[bank_index].w1.device)
        )

    @torch.compiler.disable
    def plan_chunks(self, topk_ids: torch.Tensor) -> list[tuple[slice, list[int]]]:
        """Row ranges whose combined unique expert count fits the cache.

        Routing is per token, so evaluating a subset of rows and concatenating
        the results is equivalent to evaluating the whole batch -- and every
        token's sum still happens in a single kernel call, which is why this
        split reproduces the uncached output exactly.

        Returns a single full-width slice when the batch already fits, which is
        the common case, so callers pay nothing extra for it. Each slice comes
        with the expert ids it needs, computed from host data this method
        already has, so ``prepare()`` need not synchronize again per chunk.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ``(row slice, unique expert ids)`` pairs covering ``topk_ids``.

        Raises:
            RuntimeError: if one token alone routes to more experts than the
                cache can hold, which no amount of splitting can fix.
        """
        num_rows = topk_ids.size(0)
        if num_rows == 0:
            return [(slice(0, 0), [])]

        # The common case only needs the distinct ids, which the device can
        # reduce far more cheaply than transferring every row.
        unique = topk_ids.unique()
        if unique.numel() <= self.capacity:
            return [(slice(0, num_rows), unique.tolist())]

        rows = topk_ids.tolist()
        chunks: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        for i, row in enumerate(rows):
            row_ids = set(row)
            if len(seen | row_ids) > self.capacity:
                if i == start:
                    raise RuntimeError(
                        f"CachedWeightProvider: one token routes to "
                        f"{len(row_ids)} experts but "
                        f"--moe-expert-cache-size={self.capacity}. "
                        f"Set --moe-expert-cache-size >= {len(row_ids)}."
                    )
                chunks.append((slice(start, i), sorted(seen)))
                start = i
                seen = row_ids
            else:
                seen |= row_ids
        chunks.append((slice(start, num_rows), sorted(seen)))
        return chunks

    @torch.compiler.disable
    def plan_expert_groups(self, topk_ids: torch.Tensor) -> list[list[int]]:
        """Split the forward's experts into groups the cache can hold at once.

        A token's output is the weighted sum of its experts' outputs, so the
        sum can be taken a few experts at a time and accumulated: run the
        kernel once per group with an ``expert_map`` that hides the others,
        then add the results. Every (token, expert) pair is still computed
        exactly once, in whichever group owns its expert.

        Cost is ``ceil(experts_used / capacity)`` kernel launches, independent
        of how many tokens are in the batch, and each expert is fetched at most
        once per forward. Splitting the token axis instead costs one launch per
        chunk -- one per token once capacity approaches ``top_k`` -- and
        refetches experts as the cache thrashes.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            Groups of global expert ids, each no larger than ``capacity``. A
            single group when everything already fits, which is the common
            case.
        """
        unique = topk_ids.unique().tolist()
        if len(unique) <= self.capacity:
            return [unique]
        return [
            unique[i : i + self.capacity] for i in range(0, len(unique), self.capacity)
        ]

    @eager_break_during_capture
    @torch.compiler.disable
    def prepare(
        self, topk_ids: torch.Tensor, unique_ids: list[int] | None = None
    ) -> ExpertWeightResult:
        """Make a set of experts resident and return the map selecting them.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs. Only
                read when ``unique_ids`` is not supplied.
            unique_ids: The experts to make resident. Both planners already
                know this, and passing it avoids a device synchronization --
                which matters, because a split forward calls this once per
                piece.

        Returns:
            ExpertWeightResult holding the GPU buffers and an ``expert_map``
            exposing exactly these experts, everything else -1.

        Raises:
            RuntimeError: if more experts are requested than the cache holds.
        """
        if unique_ids is None:
            unique_ids = topk_ids.unique().tolist()
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but --moe-expert-cache-size={self.capacity}. "
                f"Set --moe-expert-cache-size >= {len(unique_ids)}."
            )

        # Experts requested here must never be evicted to make room for
        # another one in the same call -- their slot would be handed to a
        # different expert while they are still expected to be resident. A
        # freshly loaded expert has freq=1, exactly the lowest LFRU score, so
        # it is the first eviction candidate. The map built at the end reads
        # every requested expert back out of _lru, so violating this raises
        # rather than corrupting silently, but it must not happen at all.
        needed = set(unique_ids)

        for expert_id in unique_ids:
            if expert_id in self._lru:
                # Cache hit: update frequency and recency
                self._clock += 1
                entry = self._lru[expert_id]
                entry[1] += 1  # freq
                entry[2] = self._clock  # last access
                self.hits += 1
            else:
                # Cache miss: need to load expert
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict entry with lowest freq/age score
                    best_key = None
                    best_score = float("inf")
                    for k, (s, freq, last) in self._lru.items():
                        if k in needed:
                            continue
                        age = self._clock - last + 1
                        score = freq / age
                        if score < best_score:
                            best_score = score
                            best_key = k
                    # len(unique_ids) <= capacity is enforced above, so at least
                    # one cached expert is outside `needed` whenever the buffer
                    # is full and a miss remains to be served.
                    assert best_key is not None
                    slot = self._lru.pop(best_key)[0]

                # Copy expert weights from CPU to GPU slot
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
                if self._buf_w13_scale is not None:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    assert self._buf_w2_scale is not None
                    self._buf_w13_scale[slot].copy_(
                        self._cpu_w13_scale[expert_id], non_blocking=True
                    )
                    self._buf_w2_scale[slot].copy_(
                        self._cpu_w2_scale[expert_id], non_blocking=True
                    )

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self.misses += 1

        # Hits and misses were updated in the loop because LFRU needs those
        # counters immediately; only advance/log the prepare-call interval.
        self._record_prepare_stats(hits=0, misses=0)

        # Expose exactly this group. Blocking on purpose: the host mirror is
        # rewritten by the next group, so an async copy could still be reading
        # it when that happens.
        self._mapping_host.fill_(-1)
        for expert_id in unique_ids:
            self._mapping_host[expert_id] = self._lru[expert_id][0]
        self._mapping.copy_(self._mapping_host)

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            expert_map=self._mapping,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )


def run_with_expert_cache(
    provider: CachedWeightProvider,
    topk_ids: torch.Tensor,
    run: Callable[[ExpertWeightResult, slice, bool], torch.Tensor],
) -> torch.Tensor:
    """Evaluate a MoE forward through the expert cache.

    ``run`` receives the resident weights with the ``expert_map`` selecting
    them, the rows it should evaluate, and whether it should include work that
    belongs to the forward as a whole rather than to this call -- shared
    experts, most importantly. It sees the original ``topk_ids``; the map is
    what restricts each call to the resident experts.

    The two splits differ only in what is cut. ``"token"`` cuts rows and
    concatenates, so each token's sum stays inside one kernel call and the
    result matches the uncached path exactly. ``"expert"`` cuts the expert set
    and sums, which costs far fewer launches when capacity is small but rounds
    each group's partial sum to the model dtype.

    When everything fits -- the common case for both splits -- ``run`` is
    called exactly once and its result returned untouched.
    """
    # Breakable CUDA graphs can safely replay dynamic cache preparation only
    # when the captured graph has a fixed number of MoE kernel calls. If the
    # total number of routed pairs fits in the cache, the unique expert count
    # necessarily fits too, so prepare() updates its fixed GPU buffers in one
    # eager break and the single MoE kernel remains in the following graph
    # segment. This covers the captured decode sizes (1 and 2) without freezing
    # their expert ids into the graph.
    if topk_ids.numel() <= provider.capacity:
        return run(provider.prepare(topk_ids), _ALL_ROWS, True)

    if provider.split == "expert":
        groups = provider.plan_expert_groups(topk_ids)
        if len(groups) == 1:
            return run(provider.prepare(topk_ids, groups[0]), _ALL_ROWS, True)

        # Accumulate in fp32. It does not recover what each group already lost
        # rounding to the model dtype, but it keeps the sum from losing more.
        accumulator: torch.Tensor | None = None
        out_dtype: torch.dtype | None = None
        for i, expert_ids in enumerate(groups):
            staged = provider.async_staging_enabled
            result = (
                provider.prepare_staged_group(expert_ids, i)
                if staged
                else provider.prepare(topk_ids, expert_ids)
            )
            part = run(result, _ALL_ROWS, i == 0)
            if staged:
                provider.release_staged_group(i)
            if accumulator is None:
                accumulator, out_dtype = part.float(), part.dtype
            else:
                accumulator += part.float()
        assert accumulator is not None and out_dtype is not None
        return accumulator.to(out_dtype)

    plan = provider.plan_chunks(topk_ids)
    if len(plan) == 1:
        rows, unique_ids = plan[0]
        return run(provider.prepare(topk_ids, unique_ids), rows, True)
    return torch.cat(
        [
            run(provider.prepare(topk_ids[rows], unique_ids), rows, True)
            for rows, unique_ids in plan
        ],
        dim=0,
    )
