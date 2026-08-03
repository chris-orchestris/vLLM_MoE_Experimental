# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import importlib.util
import math
import mmap
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_STATS_LOG_INTERVAL = 64

_BATCH_COPY_FUNCTIONS: dict[str, Callable[..., int]] = {}
_EXCLUSIVE_SWAP_FUNCTIONS: dict[str, Callable[..., int]] = {}
_PACKED_SCALE_GATHER_FUNCTIONS: dict[str, Callable[..., int]] = {}
_REGISTERED_HOST_ALLOCATIONS: list[tuple[mmap.mmap, int, int]] = []
_WRITE_COMBINED_HOST_ALLOCATIONS: list[tuple[ctypes.c_void_p, object]] = []
_MAPPED_TENSOR_EXTENSION: ModuleType | None = None
_MAPPED_TENSOR_CUDART: ctypes.CDLL | None = None


@dataclass
class _PackedScaleTransferScratch:
    packed_rows: torch.Tensor
    packed_host_rows: torch.Tensor
    source_hosts: torch.Tensor
    slot_hosts: torch.Tensor
    slot_devices: torch.Tensor
    events: list[torch.Event | None]
    cursor: int = 0


@dataclass
class _PackedScaleHostArena:
    backing: torch.Tensor
    next_layer: int = 0


_PACKED_SCALE_SCRATCH: dict[tuple[int, int, int, int], _PackedScaleTransferScratch] = {}
_PACKED_SCALE_HOST_ARENAS: dict[tuple[int, int, int], _PackedScaleHostArena] = {}


@triton.jit
def _unpack_expert_scales_kernel(
    packed_rows,
    destination_slots,
    w13_scale,
    w2_scale,
    packed_stride,
    w13_stride,
    w2_stride,
    w13_values: tl.constexpr,
    w2_values: tl.constexpr,
    w13_payload_bytes: tl.constexpr,
    block: tl.constexpr,
):
    """Expand lossless base-plus-3-bit scale rows on the active CUDA stream."""
    source_row = tl.program_id(1)
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    total_values = w13_values + w2_values
    valid = offsets < total_values
    is_w13 = offsets < w13_values
    local_offsets = tl.where(is_w13, offsets, offsets - w13_values)

    packed_row = packed_rows + source_row * packed_stride
    payload_offsets = tl.where(is_w13, 2, 2 + w13_payload_bytes)
    bit_offsets = local_offsets * 3
    byte_offsets = bit_offsets // 8
    shifts = bit_offsets % 8
    low = tl.load(
        packed_row + payload_offsets + byte_offsets,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    high = tl.load(
        packed_row + payload_offsets + byte_offsets + 1,
        mask=valid,
        other=0,
    ).to(tl.uint32)
    base = tl.where(
        is_w13,
        tl.load(packed_row),
        tl.load(packed_row + 1),
    ).to(tl.uint32)
    values = (((low | (high << 8)) >> shifts) & 7) + base
    destination_slot = tl.load(destination_slots + source_row)
    tl.store(
        w13_scale + destination_slot * w13_stride + offsets,
        values.to(tl.uint8),
        mask=valid & is_w13,
    )
    tl.store(
        w2_scale + destination_slot * w2_stride + local_offsets,
        values.to(tl.uint8),
        mask=valid & ~is_w13,
    )


def _mapped_cuda_alias(host_tensor: torch.Tensor) -> torch.Tensor:
    """Create a non-owning CUDA tensor over an existing pinned CPU tensor."""
    global _MAPPED_TENSOR_EXTENSION, _MAPPED_TENSOR_CUDART

    if host_tensor.device.type != "cpu" or not host_tensor.is_pinned():
        raise ValueError("mapped expert scales require a pinned CPU tensor")
    if not host_tensor.is_contiguous():
        raise ValueError("mapped expert scales require contiguous storage")

    if _MAPPED_TENSOR_EXTENSION is None:
        extension_path = os.environ.get("VLLM_MOE_MAPPED_TENSOR_LIBRARY")
        if not extension_path:
            raise RuntimeError(
                "VLLM_MOE_MAPPED_TENSOR_LIBRARY must point to the built "
                "mapped-host tensor extension when mapped expert scales are enabled"
            )
        spec = importlib.util.spec_from_file_location(
            "mapped_host_tensor_typed_ext", extension_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load mapped-host tensor extension {extension_path}"
            )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        _MAPPED_TENSOR_EXTENSION = extension

    if _MAPPED_TENSOR_CUDART is None:
        cudart = ctypes.CDLL("libcudart.so.13")
        cudart.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        cudart.cudaHostGetDevicePointer.restype = ctypes.c_int
        _MAPPED_TENSOR_CUDART = cudart

    device_pointer = ctypes.c_void_p()
    error = _MAPPED_TENSOR_CUDART.cudaHostGetDevicePointer(
        ctypes.byref(device_pointer), host_tensor.data_ptr(), 0
    )
    if error != 0 or device_pointer.value is None:
        raise RuntimeError(
            f"cudaHostGetDevicePointer for expert scales failed with CUDA error {error}"
        )

    extension = _MAPPED_TENSOR_EXTENSION
    dtype_name = {
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.float32: "float32",
    }.get(host_tensor.dtype, "uint8")
    mapped = extension.mapped_cuda_tensor_typed(
        device_pointer.value,
        list(host_tensor.shape),
        torch.accelerator.current_device_index(),
        dtype_name,
    )
    if mapped.dtype != host_tensor.dtype:
        mapped = mapped.view(host_tensor.dtype)
    return mapped


def _load_batch_copy_function(library_path: str) -> Callable[..., int]:
    function = _BATCH_COPY_FUNCTIONS.get(library_path)
    if function is not None:
        return function

    library = ctypes.CDLL(library_path)
    function = library.vllm_cuda_batch_copy_host_to_device
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    _BATCH_COPY_FUNCTIONS[library_path] = function
    return function


def _load_exclusive_swap_function(library_path: str) -> Callable[..., int]:
    function = _EXCLUSIVE_SWAP_FUNCTIONS.get(library_path)
    if function is not None:
        return function

    library = ctypes.CDLL(library_path)
    function = library.vllm_cuda_exclusive_expert_swap_wave_async
    pointer_array = ctypes.POINTER(ctypes.c_uint64)
    function.argtypes = [
        pointer_array,
        pointer_array,
        pointer_array,
        pointer_array,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    _EXCLUSIVE_SWAP_FUNCTIONS[library_path] = function
    return function


def _load_packed_scale_gather_function(library_path: str) -> Callable[..., int]:
    function = _PACKED_SCALE_GATHER_FUNCTIONS.get(library_path)
    if function is not None:
        return function

    library = ctypes.CDLL(library_path)
    function = library.vllm_cpu_gather_u8_rows
    function.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    _PACKED_SCALE_GATHER_FUNCTIONS[library_path] = function
    return function


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


def _pack_three_bit_scale_matrix(
    source: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one uint8 base and a lossless 3-bit payload for every row."""
    if source.element_size() != 1:
        raise ValueError("packed expert scales require one-byte E8M0 scale tensors")
    host = source.detach()
    if host.device.type != "cpu":
        host = host.cpu()
    host_bytes = host.contiguous().view(torch.uint8)
    values = np.asarray(host_bytes.numpy()).reshape(host.size(0), -1)
    bases = values.min(axis=1).astype(np.uint8, copy=False)
    spans = values.max(axis=1).astype(np.int16) - bases.astype(np.int16)
    worst_span = int(spans.max())
    if worst_span > 7:
        raise ValueError(
            "packed expert scales require every row to span at most eight "
            f"consecutive uint8 values; observed span {worst_span}"
        )
    deltas = values - bases[:, None]
    value_count = values.shape[1]
    padded_count = (value_count + 7) // 8 * 8
    if padded_count != value_count:
        padded = np.zeros((values.shape[0], padded_count), dtype=np.uint8)
        padded[:, :value_count] = deltas
        deltas = padded
    groups = deltas.reshape(values.shape[0], -1, 8)
    packed = np.empty((values.shape[0], groups.shape[1], 3), dtype=np.uint8)
    packed[:, :, 0] = (
        groups[:, :, 0] | (groups[:, :, 1] << 3) | ((groups[:, :, 2] & 3) << 6)
    )
    packed[:, :, 1] = (
        (groups[:, :, 2] >> 2)
        | (groups[:, :, 3] << 1)
        | (groups[:, :, 4] << 4)
        | ((groups[:, :, 5] & 1) << 7)
    )
    packed[:, :, 2] = (
        (groups[:, :, 5] >> 1) | (groups[:, :, 6] << 2) | (groups[:, :, 7] << 5)
    )
    payload_bytes = (value_count * 3 + 7) // 8
    return bases, packed.reshape(values.shape[0], -1)[:, :payload_bytes]


def registered_host_empty(
    *shape: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate a 2 MiB-aligned, CUDA-registered CPU tensor.

    The expert cache streams already-repacked FP4 weights from host memory.
    On the measured PCIe-only Ampere host, aligned anonymous mappings
    registered with CUDA sustained about 4% more H2D bandwidth than the
    default PyTorch pinned allocator for the exact four-field expert-copy
    pattern. Allocations are intentionally process-lifetime: these tensors are
    the persistent backing store for every cached MoE layer.
    """
    elements = math.prod(shape)
    tensor_bytes = elements * dtype.itemsize
    huge_page_bytes = 2 << 20
    registered_bytes = (
        (tensor_bytes + huge_page_bytes - 1) // huge_page_bytes * huge_page_bytes
    )
    mapping = mmap.mmap(
        -1,
        registered_bytes + huge_page_bytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    mapping_address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    address = (
        (mapping_address + huge_page_bytes - 1) // huge_page_bytes * huge_page_bytes
    )
    mapping_offset = address - mapping_address

    cudart = ctypes.CDLL("libcudart.so.13")
    cudart.cudaHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cudart.cudaHostRegister.restype = ctypes.c_int
    error = cudart.cudaHostRegister(address, registered_bytes, 1)
    if error != 0:
        mapping.close()
        raise RuntimeError(
            "cudaHostRegister for aligned expert backing failed with CUDA "
            f"error {error}"
        )

    storage = torch.frombuffer(
        mapping,
        dtype=torch.uint8,
        count=tensor_bytes,
        offset=mapping_offset,
    )
    tensor = storage.view(dtype).view(*shape)
    _REGISTERED_HOST_ALLOCATIONS.append((mapping, address, registered_bytes))
    logger.info_once(
        "Using 2 MiB-aligned CUDA-registered host backing for repacked expert weights.",
        scope="global",
    )
    return tensor


def file_backed_host_empty(
    *shape: int,
    dtype: torch.dtype,
    staging_dir: str | None = None,
) -> torch.Tensor:
    """Allocate a reclaimable temporary tensor on the configured filesystem."""
    staging_dir = staging_dir or os.environ.get("VLLM_MOE_EXPERT_CACHE_STAGING_DIR")
    if not staging_dir:
        raise RuntimeError(
            "exclusive expert caching requires "
            "VLLM_MOE_EXPERT_CACHE_STAGING_DIR on a disk-backed filesystem"
        )
    os.makedirs(staging_dir, exist_ok=True)
    elements = math.prod(shape)
    tensor_bytes = elements * dtype.itemsize
    descriptor, path = tempfile.mkstemp(
        prefix="vllm-exclusive-experts-",
        dir=staging_dir,
    )
    try:
        os.posix_fallocate(descriptor, 0, tensor_bytes)
        os.close(descriptor)
        descriptor = -1
        tensor = torch.from_file(
            path,
            shared=True,
            size=elements,
            dtype=dtype,
        ).view(*shape)
        # The mapping remains valid and reclaimable after unlink. Its blocks
        # are released automatically when the tensor is replaced or on exit.
        os.unlink(path)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.exists(path):
            os.unlink(path)
        raise
    logger.info_once(
        "Using reclaimable filesystem-backed staging for exclusive expert "
        "cache initialization.",
        scope="global",
    )
    return tensor


def write_combined_host_empty(
    *shape: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate 2 MiB-aligned write-combined pinned host storage.

    Repacked expert weights are written once by the CPU and thereafter only
    read by GPU DMA. ``cudaHostAllocWriteCombined`` avoids polluting CPU caches
    and can remove PCIe snoop traffic for exactly that unidirectional access
    pattern. The base allocation and ctypes buffer stay alive for the process;
    these tensors are the persistent backing store of the expert cache.
    """
    elements = math.prod(shape)
    tensor_bytes = elements * dtype.itemsize
    alignment = 2 << 20
    allocation_bytes = tensor_bytes + alignment

    cudart = ctypes.CDLL("libcudart.so.13")
    cudart.cudaHostAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cudart.cudaHostAlloc.restype = ctypes.c_int
    base_pointer = ctypes.c_void_p()
    # cudaHostAllocPortable | cudaHostAllocWriteCombined.
    error = cudart.cudaHostAlloc(ctypes.byref(base_pointer), allocation_bytes, 1 | 4)
    if error != 0 or base_pointer.value is None:
        raise RuntimeError(
            "cudaHostAlloc for write-combined expert backing failed with "
            f"CUDA error {error}"
        )

    address = (base_pointer.value + alignment - 1) // alignment * alignment
    owner = (ctypes.c_uint8 * tensor_bytes).from_address(address)
    storage = torch.frombuffer(owner, dtype=torch.uint8, count=tensor_bytes)
    tensor = storage.view(dtype).view(*shape)
    _WRITE_COMBINED_HOST_ALLOCATIONS.append((base_pointer, owner))
    logger.info_once(
        "Using 2 MiB-aligned CUDA write-combined host backing for repacked "
        "expert weights.",
        scope="global",
    )
    return tensor


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
    mapped_scale_bank: int | None = None


@dataclass
class _BorrowedExpertBank:
    """One independently revocable bank backed by lent KV-cache pages."""

    capacity: int
    min_tokens: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    mapping: torch.Tensor
    mapping_host: torch.Tensor
    slots: dict[int, int]
    free_slots: list[int]
    enabled: bool = True


class _SharedExpertStagingPool:
    """GPU banks shared by every compatible MoE layer on one device.

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
        num_banks: int,
    ) -> None:
        self.copy_stream = torch.cuda.Stream(device=w13.device)
        self.banks: list[ExpertWeightResult] = []
        for _ in range(num_banks):
            self.banks.append(
                ExpertWeightResult(
                    w1=torch.empty(
                        capacity,
                        *w13.shape[1:],
                        dtype=w13.dtype,
                        device=w13.device,
                    ),
                    w2=torch.empty(
                        capacity,
                        *w2.shape[1:],
                        dtype=w2.dtype,
                        device=w2.device,
                    ),
                    expert_map=torch.empty_like(expert_map, device=expert_map.device),
                    w1_scale=(
                        torch.empty(
                            capacity,
                            *w13_scale.shape[1:],
                            dtype=w13_scale.dtype,
                            device=w13_scale.device,
                        )
                        if w13_scale is not None
                        else None
                    ),
                    w2_scale=(
                        torch.empty(
                            capacity,
                            *w2_scale.shape[1:],
                            dtype=w2_scale.dtype,
                            device=w2_scale.device,
                        )
                        if w2_scale is not None
                        else None
                    ),
                )
            )
        self.ready = [torch.cuda.Event() for _ in range(num_banks)]
        self.consumed = [torch.cuda.Event() for _ in range(num_banks)]
        # Every layer with this weight shape shares these banks. Seed the
        # reusable events on the owning device so the copy stream can always
        # wait before overwriting a bank, including group 0 of the next layer.
        # Restricting the wait to group_index >= 2 protects reuse within one
        # layer but races when the following layer wraps back to bank 0.
        initial_stream = torch.cuda.current_stream(device=w13.device)
        for event in self.consumed:
            event.record(initial_stream)
        self.capacity = capacity


_SHARED_STAGING_POOLS: dict[tuple[object, ...], _SharedExpertStagingPool] = {}


def _staging_pool_key(
    capacity: int,
    num_banks: int,
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
        num_banks,
        tensor_key(w13),
        tensor_key(w2),
        tensor_key(expert_map),
        tensor_key(w13_scale),
        tensor_key(w2_scale),
    )


def _get_shared_staging_pool(
    capacity: int,
    num_banks: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_map: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
) -> _SharedExpertStagingPool:
    key = _staging_pool_key(
        capacity,
        num_banks,
        w13,
        w2,
        expert_map,
        w13_scale,
        w2_scale,
    )
    pool = _SHARED_STAGING_POOLS.get(key)
    if pool is None:
        pool = _SharedExpertStagingPool(
            capacity,
            w13,
            w2,
            expert_map,
            w13_scale,
            w2_scale,
            num_banks,
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
        exclusive: bool = False,
    ) -> None:
        num_experts = w13_weight.size(0)

        self.capacity = capacity
        self.name = name
        self.split: MoECacheSplit = split
        self._num_experts = num_experts
        self.exclusive = exclusive
        if exclusive and capacity >= num_experts:
            raise ValueError("exclusive expert caching requires capacity < num_experts")
        self.hits = 0
        self.misses = 0
        self._prepare_calls = 0
        self._route_stats_min_tokens = int(
            os.getenv("VLLM_MOE_ROUTE_STATS_MIN_TOKENS", "0")
        )
        if self._route_stats_min_tokens < 0:
            raise ValueError("VLLM_MOE_ROUTE_STATS_MIN_TOKENS must be >= 0")
        self._route_stats_arm_file = os.getenv("VLLM_MOE_ROUTE_STATS_ARM_FILE", "")
        self._route_stats_logged = False
        self._borrowed_banks: list[_BorrowedExpertBank] = []

        # Long prefills route enough token/expert pairs that they touch every
        # expert in practice. Discovering that set with unique().tolist()
        # synchronizes the GPU once per MoE layer, then the generic cache path
        # submits four H2D copies per individual cache miss. The opt-in bulk
        # path instead visits fixed, contiguous expert groups and uploads each
        # weight/scale tensor with one large copy. It deliberately applies
        # only to expert-split forwards; decode and short prefills retain the
        # adaptive LFRU path.
        bulk_prefill_min_tokens = int(
            os.getenv("VLLM_MOE_EXPERT_CACHE_BULK_PREFILL_MIN_TOKENS", "0")
        )
        if bulk_prefill_min_tokens < 0:
            raise ValueError(
                "VLLM_MOE_EXPERT_CACHE_BULK_PREFILL_MIN_TOKENS must be >= 0"
            )
        self._bulk_prefill_min_tokens = (
            bulk_prefill_min_tokens if split == "expert" else 0
        )
        bulk_copy_chunk_experts = int(
            os.getenv(
                "VLLM_MOE_EXPERT_CACHE_BULK_COPY_CHUNK_EXPERTS",
                str(capacity),
            )
        )
        if bulk_copy_chunk_experts <= 0:
            raise ValueError(
                "VLLM_MOE_EXPERT_CACHE_BULK_COPY_CHUNK_EXPERTS must be > 0"
            )
        self._bulk_copy_chunk_experts = min(bulk_copy_chunk_experts, capacity)

        self._batched_copy_min_experts = int(
            os.getenv("VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_MIN_EXPERTS", "0")
        )
        if self._batched_copy_min_experts < 0:
            raise ValueError(
                "VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_MIN_EXPERTS must be >= 0"
            )
        self._batched_copy_min_tokens = int(
            os.getenv("VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_MIN_TOKENS", "512")
        )
        if self._batched_copy_min_tokens < 0:
            raise ValueError(
                "VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_MIN_TOKENS must be >= 0"
            )
        self._mapped_scales = os.getenv(
            "VLLM_MOE_EXPERT_CACHE_MAPPED_SCALES", "0"
        ).lower() in {"1", "true"}
        packed_scales_requested = exclusive or os.getenv(
            "VLLM_MOE_EXPERT_CACHE_PACKED_SCALES", "0"
        ).lower() in {"1", "true"}
        self._packed_scales = (
            w13_scale is not None and w2_scale is not None and packed_scales_requested
        )
        if exclusive and self._mapped_scales:
            raise ValueError(
                "exclusive expert caching is incompatible with mapped scales"
            )
        if self._packed_scales and self._mapped_scales:
            raise ValueError("packed expert scales are incompatible with mapped scales")
        if exclusive and self._bulk_prefill_min_tokens > 0:
            raise ValueError(
                "exclusive expert caching is incompatible with the bulk-prefill "
                "cache path; use asynchronous staging instead"
            )
        if exclusive and self._batched_copy_min_experts > 0:
            raise ValueError(
                "exclusive expert caching has its own bidirectional batched-copy "
                "path; disable VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_MIN_EXPERTS"
            )

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
        else:
            cuda_device = w13_weight.device
        # Exclusive initialization copies directly from the layer parameters
        # into the compact backing. Materializing a full pinned copy first is
        # wasteful and, more importantly, the pinned allocator may retain each
        # released full-size allocation until all layers have been converted.
        # That defeats the RAM-saving mode during load and can force swapping.
        self._cpu_w13: torch.Tensor = (
            w13_weight if exclusive else _pinned_cpu_copy(w13_weight)
        )
        self._cpu_w2: torch.Tensor = (
            w2_weight if exclusive else _pinned_cpu_copy(w2_weight)
        )

        device_capacity = capacity + 1 if exclusive else capacity

        self._buf_w13: torch.Tensor = torch.empty(
            device_capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            device_capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        self._packed_scale_backing: torch.Tensor | None = None
        self._packed_scale_w13_values = 0
        self._packed_scale_w2_values = 0
        self._packed_scale_w13_payload_bytes = 0
        self._packed_scale_row_bytes = 0
        self._packed_scale_gather_function: Callable[..., int] | None = None

        if w13_scale is not None and w2_scale is not None:
            # Pinned for the same reason the weights are: these are copied on
            # every miss, and pageable source memory forces a staging copy.
            if self._packed_scales:
                self._cpu_w13_scale = None
                self._cpu_w2_scale = None
                self._initialize_packed_scale_backing(
                    w13_scale,
                    w2_scale,
                    cuda_device,
                )
            else:
                self._cpu_w13_scale = (
                    w13_scale if exclusive else _pinned_cpu_copy(w13_scale)
                )
                self._cpu_w2_scale = (
                    w2_scale if exclusive else _pinned_cpu_copy(w2_scale)
                )
            if self._mapped_scales:
                default_banks = math.ceil(num_experts / capacity)
                mapped_scale_banks = int(
                    os.getenv("VLLM_MOE_MAPPED_SCALE_BANKS", str(default_banks))
                )
                if mapped_scale_banks <= 0:
                    raise ValueError("VLLM_MOE_MAPPED_SCALE_BANKS must be positive")
                self._mapped_scale_hosts: list[tuple[torch.Tensor, torch.Tensor]] = []
                self._mapped_scale_aliases: list[tuple[torch.Tensor, torch.Tensor]] = []
                for _ in range(mapped_scale_banks):
                    w13_host = torch.empty(
                        capacity,
                        *w13_scale.shape[1:],
                        dtype=w13_scale.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    w2_host = torch.empty(
                        capacity,
                        *w2_scale.shape[1:],
                        dtype=w2_scale.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self._mapped_scale_hosts.append((w13_host, w2_host))
                    self._mapped_scale_aliases.append(
                        (_mapped_cuda_alias(w13_host), _mapped_cuda_alias(w2_host))
                    )
                self._mapped_scale_events = [
                    torch.cuda.Event() for _ in range(mapped_scale_banks)
                ]
                self._mapped_scale_in_use = [False] * mapped_scale_banks
                self._mapped_scale_cursor = 0
                self.mapped_scale_waits = 0
                self._buf_w13_scale, self._buf_w2_scale = self._mapped_scale_aliases[0]
                logger.info_once(
                    "Using CUDA-mapped host scales for the expert cache; "
                    "cache groups rotate over %d host scale banks.",
                    mapped_scale_banks,
                    scope="global",
                )
            else:
                self._buf_w13_scale = torch.empty(
                    device_capacity,
                    *w13_scale.shape[1:],
                    dtype=w13_scale.dtype,
                    device=cuda_device,
                )
                self._buf_w2_scale = torch.empty(
                    device_capacity,
                    *w2_scale.shape[1:],
                    dtype=w2_scale.dtype,
                    device=cuda_device,
                )
                self._mapped_scale_hosts = []
                self._mapped_scale_aliases = []
                self._mapped_scale_events = []
                self._mapped_scale_in_use = []
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None
            self._mapped_scale_hosts = []
            self._mapped_scale_aliases = []
            self._mapped_scale_events = []
            self._mapped_scale_in_use = []

        self._exclusive_swap_function: Callable[..., int] | None = None
        self._exclusive_h2d_stream: torch.Stream | None = None
        self._exclusive_d2h_stream: torch.Stream | None = None
        self._exclusive_h2d_done: torch.Event | None = None
        self._exclusive_d2h_done: torch.Event | None = None
        self._exclusive_swap_started = False
        self._exclusive_host_read_events: dict[int, torch.Event] = {}
        self._exclusive_host_slots: list[int] = []
        self._exclusive_host_experts: list[int] = []
        self._exclusive_host_spare = -1
        self._exclusive_gpu_spare = -1
        self._exclusive_gpu_row_pointers: list[list[int]] = []
        self._exclusive_host_row_pointers: list[list[int]] = []
        self._exclusive_copy_sizes: list[int] = []
        self._exclusive_gpu_destinations: ctypes.Array[ctypes.c_uint64] | None = None
        self._exclusive_host_sources: ctypes.Array[ctypes.c_uint64] | None = None
        self._exclusive_host_destinations: ctypes.Array[ctypes.c_uint64] | None = None
        self._exclusive_gpu_sources: ctypes.Array[ctypes.c_uint64] | None = None
        self._exclusive_sizes: ctypes.Array[ctypes.c_size_t] | None = None
        if exclusive:
            self._initialize_exclusive_backing(cuda_device)

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        if exclusive:
            self._lru = {
                expert_id: [expert_id, 1, expert_id + 1]
                for expert_id in range(capacity)
            }
            self._clock = capacity
            self._free_slots = []
        else:
            self._lru = {}
            self._clock = 0
            self._free_slots = list(range(capacity))

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

        self._batched_copy_function: Callable[..., int] | None = None
        self._batched_copy_stream: torch.cuda.Stream | None = None
        self._batched_copy_ready: torch.cuda.Event | None = None
        if self._batched_copy_min_experts > 0:
            library_path = os.getenv("VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_LIBRARY")
            if not library_path:
                raise RuntimeError(
                    "VLLM_MOE_EXPERT_CACHE_BATCHED_MEMCPY_LIBRARY must point "
                    "to the built CUDA batch-copy shim when batched expert "
                    "copies are enabled"
                )
            self._batched_copy_function = _load_batch_copy_function(library_path)
            self._batched_copy_stream = torch.cuda.Stream(device=cuda_device)
            self._batched_copy_ready = torch.cuda.Event()
            logger.info_once(
                "Enabled CUDA batched expert-cache H2D copies at >=%d misses "
                "and >=%d tokens",
                self._batched_copy_min_experts,
                self._batched_copy_min_tokens,
            )

        self._bulk_prefill_groups: list[list[int]] = []
        self._bulk_prefill_mappings: list[torch.Tensor] = []
        if self._bulk_prefill_min_tokens > 0:
            for start in range(0, num_experts, capacity):
                end = min(start + capacity, num_experts)
                self._bulk_prefill_groups.append(list(range(start, end)))
                mapping = torch.full(
                    (num_experts,), -1, dtype=torch.int32, device=cuda_device
                )
                mapping[start:end] = torch.arange(
                    end - start, dtype=torch.int32, device=cuda_device
                )
                self._bulk_prefill_mappings.append(mapping)
            logger.info_once(
                "Enabled contiguous expert-cache bulk prefill path at >=%d tokens "
                "with %d-expert weight-copy chunks",
                self._bulk_prefill_min_tokens,
                self._bulk_copy_chunk_experts,
            )

        if async_staging is None:
            async_staging = os.getenv(
                "VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING", "0"
            ).lower() in {"1", "true"}
        if self._mapped_scales and async_staging:
            raise ValueError("mapped expert scales do not support asynchronous staging")
        self._async_staging = async_staging and split == "expert"
        self._interlayer_prefetch = self._async_staging and os.getenv(
            "VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH", "0"
        ).lower() in {"1", "true"}
        self._interlayer_prefetch_min_tokens = int(
            os.getenv(
                "VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH_MIN_TOKENS",
                "512",
            )
        )
        if self._interlayer_prefetch_min_tokens < 0:
            raise ValueError(
                "VLLM_MOE_EXPERT_CACHE_INTERLAYER_PREFETCH_MIN_TOKENS must be >= 0"
            )
        self._staging_pool: _SharedExpertStagingPool | None = None
        self._staging_capacity = capacity
        self._staging_mapping_hosts: list[torch.Tensor] = []
        self._lookahead_prediction: list[int] = []
        self._lookahead_pending_ids: list[int] = []
        self._lookahead_mapping_host: torch.Tensor | None = None
        self._lookahead_registry_key: tuple[int, str, int] | None = None
        if self._async_staging:
            self._staging_capacity = int(
                os.getenv(
                    "VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY",
                    str(capacity),
                )
            )
            if not 0 < self._staging_capacity <= capacity:
                raise ValueError(
                    "VLLM_MOE_EXPERT_CACHE_ASYNC_STAGING_CAPACITY must be "
                    f"in [1, {capacity}]"
                )
            self._staging_pool = _get_shared_staging_pool(
                self._staging_capacity,
                3 if self._interlayer_prefetch else 2,
                self._buf_w13,
                self._buf_w2,
                self._mapping,
                self._buf_w13_scale,
                self._buf_w2_scale,
            )
            num_groups = (
                num_experts + self._staging_capacity - 1
            ) // self._staging_capacity
            self._staging_mapping_hosts = [
                torch.full((num_experts,), -1, dtype=torch.int32).pin_memory()
                for _ in range(num_groups)
            ]
            if self._interlayer_prefetch:
                self._lookahead_mapping_host = torch.full(
                    (num_experts,), -1, dtype=torch.int32
                ).pin_memory()
                self._lookahead_registry_key = self._parse_layer_registry_key(
                    name,
                    cuda_device,
                )
                if self._lookahead_registry_key is not None:
                    _INTERLAYER_PROVIDERS[self._lookahead_registry_key] = self

    def _initialize_packed_scale_backing(
        self,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        cuda_device: torch.device,
    ) -> None:
        """Replace full uint8 host scales with lossless 3-bit pageable rows."""
        w13_bases, w13_payloads = _pack_three_bit_scale_matrix(w13_scale)
        w2_bases, w2_payloads = _pack_three_bit_scale_matrix(w2_scale)
        if w13_payloads.shape[0] != self._num_experts:
            raise ValueError("packed W13 scale expert count is inconsistent")
        if w2_payloads.shape[0] != self._num_experts:
            raise ValueError("packed W2 scale expert count is inconsistent")

        self._packed_scale_w13_values = w13_scale[0].numel()
        self._packed_scale_w2_values = w2_scale[0].numel()
        self._packed_scale_w13_payload_bytes = w13_payloads.shape[1]
        # One final guard byte makes the kernel's cross-byte uint16 load safe.
        self._packed_scale_row_bytes = (
            2 + w13_payloads.shape[1] + w2_payloads.shape[1] + 1
        )
        default_arena_layers = "46" if self.exclusive else "0"
        arena_layers = int(
            os.getenv("VLLM_MOE_PACKED_SCALE_ARENA_LAYERS", default_arena_layers)
        )
        if arena_layers < 0:
            raise ValueError("VLLM_MOE_PACKED_SCALE_ARENA_LAYERS must be >= 0")
        if arena_layers:
            device_index = cuda_device.index
            if device_index is None:
                device_index = torch.accelerator.current_device_index()
            arena_key = (
                device_index,
                self._num_experts,
                self._packed_scale_row_bytes,
            )
            arena = _PACKED_SCALE_HOST_ARENAS.get(arena_key)
            if arena is None:
                arena = _PackedScaleHostArena(
                    backing=torch.empty(
                        arena_layers,
                        self._num_experts,
                        self._packed_scale_row_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                    )
                )
                _PACKED_SCALE_HOST_ARENAS[arena_key] = arena
                logger.info_once(
                    "Preallocated one %.2f GiB pageable arena for %d packed "
                    "expert-scale layers per rank.",
                    arena.backing.numel() / (1 << 30),
                    arena_layers,
                    scope="global",
                )
            elif arena.backing.size(0) != arena_layers:
                raise ValueError(
                    "packed expert-scale arena capacity changed within one process"
                )
            if arena.next_layer >= arena.backing.size(0):
                raise RuntimeError(
                    "packed expert-scale arena is full; increase "
                    "VLLM_MOE_PACKED_SCALE_ARENA_LAYERS"
                )
            backing = arena.backing[arena.next_layer]
            arena.next_layer += 1
            backing.zero_()
        else:
            backing = torch.zeros(
                self._num_experts,
                self._packed_scale_row_bytes,
                dtype=torch.uint8,
                device="cpu",
            )
        backing[:, 0].copy_(torch.from_numpy(w13_bases))
        backing[:, 1].copy_(torch.from_numpy(w2_bases))
        w13_end = 2 + w13_payloads.shape[1]
        backing[:, 2:w13_end].copy_(torch.from_numpy(w13_payloads))
        backing[:, w13_end : w13_end + w2_payloads.shape[1]].copy_(
            torch.from_numpy(w2_payloads)
        )
        self._packed_scale_backing = backing

        original_bytes = self._num_experts * (
            self._packed_scale_w13_values + self._packed_scale_w2_values
        )
        logger.info_once(
            "Using lossless 3-bit expert-scale backing: %.2f MiB replaces "
            "%.2f MiB per layer/rank (%.1f%% smaller).",
            backing.numel() / (1 << 20),
            original_bytes / (1 << 20),
            100.0 * (1.0 - backing.numel() / original_bytes),
            scope="global",
        )

    def _copy_packed_scales(
        self,
        destination_w13: torch.Tensor,
        destination_w2: torch.Tensor,
        copies: list[tuple[int, int]],
    ) -> None:
        """Upload and expand several packed scale rows with one GPU kernel."""
        if not copies:
            return
        backing = self._packed_scale_backing
        if backing is None:
            raise RuntimeError("packed expert-scale backing is not initialized")
        if len(copies) > self.capacity:
            raise RuntimeError("packed expert-scale batch exceeds cache capacity")
        if destination_w13.element_size() != 1 or destination_w2.element_size() != 1:
            raise RuntimeError(
                "packed expert-scale destinations must have one-byte elements"
            )

        destination_w13_bytes = destination_w13.view(torch.uint8)
        destination_w2_bytes = destination_w2.view(torch.uint8)

        stream = current_platform.current_stream(destination_w13_bytes.device)
        device_index = destination_w13_bytes.device.index
        assert device_index is not None
        ring_count = int(os.getenv("VLLM_MOE_PACKED_SCALE_TRANSFER_SLOTS", "4"))
        if ring_count <= 0:
            raise ValueError("VLLM_MOE_PACKED_SCALE_TRANSFER_SLOTS must be > 0")
        # A scratch allocation per CUDA stream is unexpectedly expensive here:
        # exclusive mode owns one D2H stream per MoE layer, so 43 streams would
        # strand roughly 145 MiB of device staging and more than 1 GiB of pinned
        # host staging after the first cache sweep.  A small event-protected
        # device-wide ring preserves cross-stream overlap without consuming the
        # execution headroom needed by concurrent Ampere prefill.
        scratch_key = (
            device_index,
            self.capacity,
            self._packed_scale_row_bytes,
            ring_count,
        )
        transfer = _PACKED_SCALE_SCRATCH.get(scratch_key)
        if transfer is None:
            transfer = _PackedScaleTransferScratch(
                packed_rows=torch.empty(
                    ring_count,
                    self.capacity,
                    self._packed_scale_row_bytes,
                    dtype=torch.uint8,
                    device=destination_w13_bytes.device,
                ),
                packed_host_rows=torch.empty(
                    ring_count,
                    self.capacity,
                    self._packed_scale_row_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                ),
                source_hosts=torch.empty(
                    ring_count,
                    self.capacity,
                    dtype=torch.int64,
                    device="cpu",
                ),
                slot_hosts=torch.empty(
                    ring_count,
                    self.capacity,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                ),
                slot_devices=torch.empty(
                    ring_count,
                    self.capacity,
                    dtype=torch.int32,
                    device=destination_w13_bytes.device,
                ),
                events=[None] * ring_count,
            )
            _PACKED_SCALE_SCRATCH[scratch_key] = transfer

        ring_index = transfer.cursor % transfer.slot_hosts.size(0)
        transfer.cursor += 1
        previous = transfer.events[ring_index]
        if previous is not None:
            previous.synchronize()
        packed_rows = transfer.packed_rows[ring_index]
        slot_host = transfer.slot_hosts[ring_index]
        slot_device = transfer.slot_devices[ring_index]
        packed_host_rows = transfer.packed_host_rows[ring_index]
        source_host = transfer.source_hosts[ring_index]
        for index, (destination_slot, expert_id) in enumerate(copies):
            slot_host[index] = destination_slot
            source_host[index] = expert_id
        gather_function = self._packed_scale_gather_function
        if gather_function is None:
            np.take(
                backing.numpy(),
                source_host[: len(copies)].numpy(),
                axis=0,
                out=packed_host_rows[: len(copies)].numpy(),
            )
        else:
            error = gather_function(
                packed_host_rows.data_ptr(),
                backing.data_ptr(),
                source_host.data_ptr(),
                self._packed_scale_row_bytes,
                len(copies),
            )
            if error != 0:
                raise RuntimeError("native packed-scale row gather failed")
        packed_rows[: len(copies)].copy_(
            packed_host_rows[: len(copies)], non_blocking=True
        )
        slot_device[: len(copies)].copy_(slot_host[: len(copies)], non_blocking=True)
        total_values = self._packed_scale_w13_values + self._packed_scale_w2_values
        grid = (triton.cdiv(total_values, 256), len(copies))
        _unpack_expert_scales_kernel[grid](
            packed_rows,
            slot_device,
            destination_w13_bytes,
            destination_w2_bytes,
            packed_rows.stride(0),
            destination_w13_bytes.stride(0),
            destination_w2_bytes.stride(0),
            w13_values=self._packed_scale_w13_values,
            w2_values=self._packed_scale_w2_values,
            w13_payload_bytes=self._packed_scale_w13_payload_bytes,
            block=256,
            num_warps=4,
        )
        ready = previous if previous is not None else current_platform.Event()
        ready.record(stream)
        transfer.events[ring_index] = ready

    def _allocate_exclusive_host_tensor(
        self, source: torch.Tensor, host_rows: int
    ) -> torch.Tensor:
        shape = (host_rows, *source.shape[1:])
        registered = os.getenv(
            "VLLM_MOE_EXPERT_CACHE_EXCLUSIVE_REGISTERED_BACKING", "1"
        ).lower() in {"1", "true"}
        if registered:
            return registered_host_empty(*shape, dtype=source.dtype)
        return torch.empty(
            *shape,
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )

    def _initialize_exclusive_backing(self, cuda_device: torch.device) -> None:
        """Split one full expert tensor set into compact CPU and GPU ownership.

        One spare row is retained on each side. A miss exchanges the incoming
        CPU row with a GPU victim through those distinct spares, allowing H2D
        and D2H to overlap without either DMA engine touching the same memory.
        """
        host_experts = self._num_experts - self.capacity
        host_rows = host_experts + 1
        sources = [self._cpu_w13, self._cpu_w2]
        # Packed scales are immutable and retain a compact row for every
        # expert. They do not participate in exclusive ownership rotation:
        # an incoming row is expanded directly into the new GPU spare, while
        # an evicted row needs no scale D2H copy. This removes two transfers
        # from every native swap and avoids a GPU->CPU repack dependency.
        if self._cpu_w13_scale is not None and not self._packed_scales:
            assert self._cpu_w2_scale is not None
            sources.extend([self._cpu_w13_scale, self._cpu_w2_scale])

        gpu_tensors = [self._buf_w13, self._buf_w2]
        if self._buf_w13_scale is not None and not self._packed_scales:
            assert self._buf_w2_scale is not None
            gpu_tensors.extend([self._buf_w13_scale, self._buf_w2_scale])

        current_stream = current_platform.current_stream(cuda_device)
        for gpu_tensor, source in zip(gpu_tensors, sources):
            gpu_tensor[: self.capacity].copy_(
                source[: self.capacity], non_blocking=True
            )

        if self._packed_scales:
            assert self._buf_w13_scale is not None
            assert self._buf_w2_scale is not None
            self._copy_packed_scales(
                self._buf_w13_scale,
                self._buf_w2_scale,
                [(expert_id, expert_id) for expert_id in range(self.capacity)],
            )

        compact = []
        for source in sources:
            target = self._allocate_exclusive_host_tensor(source, host_rows)
            target[:host_experts].copy_(source[self.capacity :])
            compact.append(target)

        # The full source parameters are released as soon as this provider is
        # installed. Ensure their initial resident uploads have consumed them.
        current_stream.synchronize()
        self._cpu_w13, self._cpu_w2 = compact[:2]
        if len(compact) == 4:
            self._cpu_w13_scale, self._cpu_w2_scale = compact[2:]

        self._exclusive_host_slots = [-1] * self.capacity + list(range(host_experts))
        self._exclusive_host_experts = list(range(self.capacity, self._num_experts)) + [
            -1
        ]
        self._exclusive_host_spare = host_experts
        self._exclusive_gpu_spare = self.capacity

        bundled_library = Path(__file__).with_name("cuda_expert_cache_helper.so")
        library_path = os.getenv(
            "VLLM_MOE_EXPERT_CACHE_EXCLUSIVE_SWAP_LIBRARY",
            str(bundled_library),
        )
        if not Path(library_path).is_file():
            raise RuntimeError(
                "exclusive expert caching requires the native helper at "
                f"{library_path}; run tools/install_moe_experimental.sh or set "
                "VLLM_MOE_EXPERT_CACHE_EXCLUSIVE_SWAP_LIBRARY"
            )
        self._exclusive_swap_function = _load_exclusive_swap_function(library_path)
        if self._packed_scales:
            self._packed_scale_gather_function = _load_packed_scale_gather_function(
                library_path
            )
        self._exclusive_h2d_stream = current_platform.Stream(device=cuda_device)
        self._exclusive_d2h_stream = current_platform.Stream(device=cuda_device)
        self._exclusive_h2d_done = current_platform.Event()
        self._exclusive_d2h_done = current_platform.Event()
        # Materialize reusable cudaEvent_t handles once. The native wave shim
        # records these after every swap and leaves the final record visible
        # to the compute stream and all later host-row readers.
        self._exclusive_h2d_done.record(self._exclusive_h2d_stream)
        self._exclusive_d2h_done.record(self._exclusive_d2h_stream)

        self._exclusive_gpu_row_pointers = []
        self._exclusive_host_row_pointers = []
        self._exclusive_copy_sizes = []
        for gpu_tensor, host_tensor in zip(gpu_tensors, compact):
            if not gpu_tensor[0].is_contiguous() or not host_tensor[0].is_contiguous():
                raise RuntimeError(
                    "exclusive expert swaps require contiguous per-expert tensors"
                )
            gpu_stride = gpu_tensor.stride(0) * gpu_tensor.element_size()
            host_stride = host_tensor.stride(0) * host_tensor.element_size()
            self._exclusive_gpu_row_pointers.append(
                [
                    gpu_tensor.data_ptr() + row * gpu_stride
                    for row in range(gpu_tensor.size(0))
                ]
            )
            self._exclusive_host_row_pointers.append(
                [
                    host_tensor.data_ptr() + row * host_stride
                    for row in range(host_tensor.size(0))
                ]
            )
            self._exclusive_copy_sizes.append(
                host_tensor[0].numel() * host_tensor.element_size()
            )

        copies_per_swap = len(self._exclusive_copy_sizes)
        max_wave_copies = self.capacity * copies_per_swap
        pointer_array = ctypes.c_uint64 * max_wave_copies
        self._exclusive_gpu_destinations = pointer_array()
        self._exclusive_host_sources = pointer_array()
        self._exclusive_host_destinations = pointer_array()
        self._exclusive_gpu_sources = pointer_array()
        self._exclusive_sizes = (ctypes.c_size_t * max_wave_copies)()
        for swap_index in range(self.capacity):
            base = swap_index * copies_per_swap
            for copy_index, copy_size in enumerate(self._exclusive_copy_sizes):
                self._exclusive_sizes[base + copy_index] = copy_size

        source_bytes = sum(source.numel() * source.element_size() for source in sources)
        compact_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in compact
        )
        logger.info_once(
            "Exclusive CPU/GPU expert cache with packed scales enabled: "
            "%d GPU residents plus "
            "one GPU/CPU rotating spare; compact host backing saves %.2f GiB "
            "per compatible provider set before allocator overhead.",
            self.capacity,
            (source_bytes - compact_bytes) / (1 << 30),
            scope="global",
        )

    def _cpu_expert_row(self, tensor: torch.Tensor, expert_id: int) -> torch.Tensor:
        if not self.exclusive:
            return tensor[expert_id]
        host_slot = self._exclusive_host_slots[expert_id]
        if host_slot < 0:
            raise RuntimeError(
                f"expert {expert_id} is GPU-owned and has no CPU backing row"
            )
        return tensor[host_slot]

    def _wait_for_exclusive_host_updates(self, stream: torch.Stream) -> None:
        """Keep a host-row reader behind the most recent D2H ownership move."""
        if self.exclusive and self._exclusive_swap_started:
            event = self._exclusive_d2h_done
            assert event is not None
            stream.wait_event(event)

    def _record_exclusive_host_reads(self, stream: torch.Stream) -> None:
        """Protect host rows until all H2D readers on *stream* are finished."""
        if not self.exclusive:
            return
        stream_handle = stream.cuda_stream
        event = self._exclusive_host_read_events.get(stream_handle)
        if event is None:
            event = current_platform.Event()
            self._exclusive_host_read_events[stream_handle] = event
        event.record(stream)

    def _begin_exclusive_swaps(self, compute_stream: torch.Stream) -> None:
        """Protect GPU victims and host rows before the first swap in a wave."""
        d2h_stream = self._exclusive_d2h_stream
        if d2h_stream is None:
            raise RuntimeError("exclusive expert-cache D2H stream is unavailable")
        d2h_stream.wait_stream(compute_stream)
        for event in self._exclusive_host_read_events.values():
            d2h_stream.wait_event(event)

    def _plan_exclusive_swap(
        self,
        incoming_id: int,
        victim_id: int,
        victim_slot: int,
    ) -> tuple[int, tuple[int, int, int, int]]:
        """Rotate ownership metadata and return one native-wave slot plan."""
        incoming_host_slot = self._exclusive_host_slots[incoming_id]
        host_spare = self._exclusive_host_spare
        gpu_spare = self._exclusive_gpu_spare
        if incoming_host_slot < 0:
            raise RuntimeError(f"expert {incoming_id} is not CPU-owned")
        if self._exclusive_host_experts[incoming_host_slot] != incoming_id:
            raise RuntimeError("exclusive expert-cache host ownership is corrupt")
        if self._exclusive_host_experts[host_spare] != -1:
            raise RuntimeError("exclusive expert-cache host spare is occupied")

        self._exclusive_host_slots[incoming_id] = -1
        self._exclusive_host_slots[victim_id] = host_spare
        self._exclusive_host_experts[host_spare] = victim_id
        self._exclusive_host_experts[incoming_host_slot] = -1
        self._exclusive_host_spare = incoming_host_slot
        self._exclusive_gpu_spare = victim_slot
        slots = (gpu_spare, incoming_host_slot, host_spare, victim_slot)
        return gpu_spare, slots

    def _queue_exclusive_swap_wave(
        self,
        swaps: list[tuple[int, int, int, int]],
    ) -> None:
        """Submit all misses from one prepare() through one native call."""
        if not swaps:
            return
        if len(swaps) > self.capacity:
            raise RuntimeError("exclusive expert-cache swap wave exceeds capacity")

        function = self._exclusive_swap_function
        h2d_stream = self._exclusive_h2d_stream
        d2h_stream = self._exclusive_d2h_stream
        h2d_done = self._exclusive_h2d_done
        d2h_done = self._exclusive_d2h_done
        gpu_destinations = self._exclusive_gpu_destinations
        host_sources = self._exclusive_host_sources
        host_destinations = self._exclusive_host_destinations
        gpu_sources = self._exclusive_gpu_sources
        sizes = self._exclusive_sizes
        if (
            function is None
            or h2d_stream is None
            or d2h_stream is None
            or h2d_done is None
            or d2h_done is None
            or gpu_destinations is None
            or host_sources is None
            or host_destinations is None
            or gpu_sources is None
            or sizes is None
        ):
            raise RuntimeError("exclusive expert-cache swap path is unavailable")

        self._begin_exclusive_swaps(
            current_platform.current_stream(self._buf_w13.device)
        )
        if self._exclusive_swap_started:
            # The prior wave may still own both rotating spares. Capture its
            # final event records before the native shim starts reusing them.
            h2d_stream.wait_event(d2h_done)
            d2h_stream.wait_event(h2d_done)

        copies_per_swap = len(self._exclusive_copy_sizes)
        for swap_index, (
            gpu_destination_slot,
            host_source_slot,
            host_destination_slot,
            gpu_source_slot,
        ) in enumerate(swaps):
            base = swap_index * copies_per_swap
            for copy_index in range(copies_per_swap):
                index = base + copy_index
                gpu_destinations[index] = self._exclusive_gpu_row_pointers[copy_index][
                    gpu_destination_slot
                ]
                host_sources[index] = self._exclusive_host_row_pointers[copy_index][
                    host_source_slot
                ]
                host_destinations[index] = self._exclusive_host_row_pointers[
                    copy_index
                ][host_destination_slot]
                gpu_sources[index] = self._exclusive_gpu_row_pointers[copy_index][
                    gpu_source_slot
                ]

        device_index = self._buf_w13.device.index
        assert device_index is not None
        error = function(
            gpu_destinations,
            host_sources,
            host_destinations,
            gpu_sources,
            sizes,
            len(swaps),
            copies_per_swap,
            h2d_stream.cuda_stream,
            d2h_stream.cuda_stream,
            h2d_done.cuda_event,
            d2h_done.cuda_event,
            device_index,
        )
        if error != 0:
            raise RuntimeError(
                "exclusive cudaMemcpyBatchAsync swap wave failed with "
                f"CUDA error {error}"
            )
        self._exclusive_swap_started = True

    @staticmethod
    def _parse_layer_registry_key(
        name: str, device: torch.device
    ) -> tuple[int, str, int] | None:
        marker = ".layers."
        if marker not in name:
            return None
        prefix, suffix = name.split(marker, 1)
        layer_text = suffix.split(".", 1)[0]
        try:
            layer_index = int(layer_text)
        except ValueError:
            return None
        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        return device_index, prefix, layer_index

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

    def _base_borrowed_bank(self) -> _BorrowedExpertBank | None:
        return self._borrowed_banks[0] if self._borrowed_banks else None

    # Keep the original one-bank diagnostics/tests source-compatible while the
    # runtime can carry multiple independently revocable tiers.
    @property
    def _borrowed_capacity(self) -> int:
        bank = self._base_borrowed_bank()
        return bank.capacity if bank is not None else 0

    @property
    def _borrowed_min_tokens(self) -> int:
        bank = self._base_borrowed_bank()
        return bank.min_tokens if bank is not None else 0

    @property
    def _borrowed_enabled(self) -> bool:
        bank = self._base_borrowed_bank()
        return bank.enabled if bank is not None else False

    @property
    def _borrowed_w13(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.w13 if bank is not None else None

    @property
    def _borrowed_w2(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.w2 if bank is not None else None

    @property
    def _borrowed_w13_scale(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.w13_scale if bank is not None else None

    @property
    def _borrowed_w2_scale(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.w2_scale if bank is not None else None

    @property
    def _borrowed_mapping(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.mapping if bank is not None else None

    @property
    def _borrowed_mapping_host(self) -> torch.Tensor | None:
        bank = self._base_borrowed_bank()
        return bank.mapping_host if bank is not None else None

    @property
    def _borrowed_slots(self) -> dict[int, int]:
        bank = self._base_borrowed_bank()
        return bank.slots if bank is not None else {}

    @property
    def _borrowed_free_slots(self) -> list[int]:
        bank = self._base_borrowed_bank()
        return bank.free_slots if bank is not None else []

    def borrowed_tensor_specs(
        self, capacity: int
    ) -> list[tuple[torch.dtype, tuple[int, ...]]]:
        """Shapes needed for an auxiliary expert bank.

        The runner uses this to carve typed views out of never-yet-allocated
        high KV block slabs.  Returning only metadata keeps the provider
        independent of the KV allocator.
        """
        if capacity <= 0:
            raise ValueError("borrowed expert capacity must be positive")

        tensors = [self._buf_w13, self._buf_w2]
        if self._buf_w13_scale is not None:
            assert self._buf_w2_scale is not None
            tensors.extend([self._buf_w13_scale, self._buf_w2_scale])
        return [(tensor.dtype, (capacity, *tensor.shape[1:])) for tensor in tensors]

    def attach_borrowed_buffers(
        self,
        buffers: list[torch.Tensor],
        min_tokens: int,
    ) -> int:
        """Attach an independently revocable cache bank from lent KV pages.

        Banks are ordered from most durable to most opportunistic.  The
        returned index is carried by the runner's KV-tail tier state so a
        scheduler fence can disable exactly the bank whose pages it reached.
        """
        expected = 4 if self._buf_w13_scale is not None else 2
        if len(buffers) != expected:
            raise ValueError(
                f"borrowed expert bank expected {expected} tensors, "
                f"received {len(buffers)}"
            )
        capacity = buffers[0].size(0)
        specs = self.borrowed_tensor_specs(capacity)
        for tensor, (dtype, shape) in zip(buffers, specs):
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    "borrowed expert tensor mismatch: "
                    f"got {tensor.dtype} {tuple(tensor.shape)}, "
                    f"expected {dtype} {shape}"
                )
            if tensor.device != self._buf_w13.device or not tensor.is_contiguous():
                raise ValueError(
                    "borrowed expert tensors must be contiguous on the cache GPU"
                )

        bank = _BorrowedExpertBank(
            capacity=capacity,
            min_tokens=min_tokens,
            w13=buffers[0],
            w2=buffers[1],
            w13_scale=buffers[2] if expected == 4 else None,
            w2_scale=buffers[3] if expected == 4 else None,
            mapping=torch.full(
                (self._num_experts,),
                -1,
                dtype=torch.int32,
                device=self._buf_w13.device,
            ),
            mapping_host=torch.full(
                (self._num_experts,), -1, dtype=torch.int32
            ).pin_memory(),
            slots={},
            free_slots=list(range(capacity - 1, -1, -1)),
        )
        self._borrowed_banks.append(bank)
        return len(self._borrowed_banks) - 1

    def disable_borrowed_buffers(self, bank_index: int = 0) -> None:
        """Permanently stop touching one KV-backed bank in this process."""
        if not 0 <= bank_index < len(self._borrowed_banks):
            raise IndexError(f"borrowed expert bank index {bank_index} is invalid")
        bank = self._borrowed_banks[bank_index]
        if not bank.enabled:
            return
        bank.enabled = False
        bank.slots.clear()
        bank.free_slots.clear()

    def _plan_one_borrowed_group(
        self,
        bank_index: int,
        topk_ids: torch.Tensor,
        valid_unique_ids: list[int],
        excluded_ids: set[int],
    ) -> list[int]:
        bank = self._borrowed_banks[bank_index]
        if not bank.enabled or topk_ids.size(0) < bank.min_tokens:
            return []

        if self.exclusive:
            # Decode can promote an expert which a prior prefill left in a
            # borrowed bank. The primary copy is then authoritative, so retire
            # the duplicate before selecting the next borrowed group.
            for expert_id in list(bank.slots):
                if expert_id in self._lru:
                    bank.free_slots.append(bank.slots.pop(expert_id))

        used = set(valid_unique_ids)
        group = [
            expert_id
            for expert_id in bank.slots
            if expert_id in used and expert_id not in excluded_ids
        ]
        vacancies = bank.capacity - len(bank.slots)
        if vacancies > 0:
            candidates = [
                expert_id
                for expert_id in valid_unique_ids
                if expert_id not in bank.slots
                and expert_id not in excluded_ids
                and (not self.exclusive or expert_id not in self._lru)
            ]
            group.extend(candidates[:vacancies])

        if not self.exclusive:
            for expert_id in group:
                self.invalidate(expert_id)
        return group

    def plan_borrowed_groups(
        self,
        topk_ids: torch.Tensor,
        unique_ids: list[int],
        excluded_ids: set[int] | None = None,
    ) -> list[tuple[int, list[int]]]:
        """Choose disjoint expert groups for every still-active KV tier."""
        valid_unique_ids = [
            expert_id for expert_id in unique_ids if 0 <= expert_id < self._num_experts
        ]
        excluded = set() if excluded_ids is None else set(excluded_ids)
        groups: list[tuple[int, list[int]]] = []
        for bank_index in range(len(self._borrowed_banks)):
            group = self._plan_one_borrowed_group(
                bank_index, topk_ids, valid_unique_ids, excluded
            )
            if group:
                groups.append((bank_index, group))
                excluded.update(group)
        return groups

    def plan_borrowed_group(
        self, topk_ids: torch.Tensor, unique_ids: list[int]
    ) -> list[int]:
        """Compatibility helper returning the most durable bank's group."""
        if not self._borrowed_banks:
            return []
        valid_unique_ids = [
            expert_id for expert_id in unique_ids if 0 <= expert_id < self._num_experts
        ]
        return self._plan_one_borrowed_group(0, topk_ids, valid_unique_ids, set())

    @eager_break_during_capture
    @torch.compiler.disable
    def prepare_borrowed_group(
        self, expert_ids: list[int], bank_index: int = 0
    ) -> ExpertWeightResult:
        """Populate/select one group in an active KV-tail expert bank."""
        if not 0 <= bank_index < len(self._borrowed_banks):
            raise IndexError(f"borrowed expert bank index {bank_index} is invalid")
        bank = self._borrowed_banks[bank_index]
        if not bank.enabled:
            raise RuntimeError("borrowed expert bank is not active")
        if len(expert_ids) > bank.capacity:
            raise RuntimeError("borrowed expert group exceeds its capacity")

        copy_stream = current_platform.current_stream(bank.w13.device)
        self._wait_for_exclusive_host_updates(copy_stream)
        scale_copies: list[tuple[int, int]] = []
        for expert_id in expert_ids:
            if expert_id in bank.slots:
                continue
            if not bank.free_slots:
                raise RuntimeError("borrowed expert bank has no free slot")
            slot = bank.free_slots.pop()
            bank.slots[expert_id] = slot
            bank.w13[slot].copy_(
                self._cpu_expert_row(self._cpu_w13, expert_id),
                non_blocking=True,
            )
            bank.w2[slot].copy_(
                self._cpu_expert_row(self._cpu_w2, expert_id),
                non_blocking=True,
            )
            if bank.w13_scale is not None:
                assert bank.w2_scale is not None
                if self._packed_scales:
                    scale_copies.append((slot, expert_id))
                else:
                    assert self._cpu_w13_scale is not None
                    assert self._cpu_w2_scale is not None
                    bank.w13_scale[slot].copy_(
                        self._cpu_expert_row(self._cpu_w13_scale, expert_id),
                        non_blocking=True,
                    )
                    bank.w2_scale[slot].copy_(
                        self._cpu_expert_row(self._cpu_w2_scale, expert_id),
                        non_blocking=True,
                    )

        if scale_copies:
            assert bank.w13_scale is not None
            assert bank.w2_scale is not None
            self._copy_packed_scales(
                bank.w13_scale,
                bank.w2_scale,
                scale_copies,
            )

        bank.mapping_host.fill_(-1)
        for expert_id in expert_ids:
            bank.mapping_host[expert_id] = bank.slots[expert_id]
        bank.mapping.copy_(bank.mapping_host)
        self._record_exclusive_host_reads(copy_stream)
        return ExpertWeightResult(
            w1=bank.w13,
            w2=bank.w2,
            expert_map=bank.mapping,
            w1_scale=bank.w13_scale,
            w2_scale=bank.w2_scale,
        )

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            if self.exclusive:
                raise RuntimeError(
                    "exclusive expert-cache residents must be exchanged, not "
                    "invalidated"
                )
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])

    @property
    def async_staging_enabled(self) -> bool:
        return self._staging_pool is not None

    @property
    def async_staging_ready(self) -> bool:
        """Whether miss-only staging can preserve a warm primary cache.

        The original staging experiment bypassed ``prepare()`` from the first
        forward onward.  That left the layer LRU empty, so every later prompt
        recopied even experts which should have been hits.  Run one normal
        fill before staging misses; afterward the resident group keeps the
        primary cache hot while the shared banks carry only true misses.
        """
        # Decode may briefly admit an expert already held by a borrowed
        # prefill bank. The next long-prefill plan invalidates that duplicate,
        # so requiring an exactly full LRU would disable staging even though
        # nearly every primary slot remains useful. Conversely, a handful of
        # decode entries is not enough to justify bypassing the normal fill.
        minimum_residency = max(1, (self.capacity + 1) // 2)
        return self._staging_pool is not None and len(self._lru) >= minimum_residency

    def group_is_resident(self, expert_ids: list[int]) -> bool:
        """Whether every expert in a planned group is in the layer cache."""
        return all(expert_id in self._lru for expert_id in expert_ids)

    def remember_prefill_route(self, topk_ids: torch.Tensor) -> None:
        """Remember the prior prefill's busiest experts for lookahead."""
        if (
            not self._interlayer_prefetch
            or topk_ids.size(0) < self._interlayer_prefetch_min_tokens
        ):
            return
        valid = topk_ids[(topk_ids >= 0) & (topk_ids < self._num_experts)]
        counts = torch.bincount(
            valid.to(torch.int64), minlength=self._num_experts
        ).tolist()
        self._lookahead_prediction = sorted(
            (
                expert_id
                for expert_id in range(self._num_experts)
                if counts[expert_id] > 0
            ),
            key=lambda expert_id: (-counts[expert_id], expert_id),
        )

    def prefetch_successor(self, num_tokens: int) -> None:
        """Upload one predicted group for the next MoE layer."""
        key = self._lookahead_registry_key
        if key is None or num_tokens < self._interlayer_prefetch_min_tokens:
            return
        device_index, prefix, layer_index = key
        successor = _INTERLAYER_PROVIDERS.get((device_index, prefix, layer_index + 1))
        if successor is not None:
            successor._schedule_lookahead()

    def _schedule_lookahead(self) -> None:
        pool = self._staging_pool
        if (
            not self._interlayer_prefetch
            or pool is None
            or len(pool.banks) < 3
            or not self._lookahead_prediction
            or self._lookahead_pending_ids
        ):
            return

        resident = set(self._lru)
        for bank in self._borrowed_banks:
            if bank.enabled:
                resident.update(bank.slots)
        expert_ids = [
            expert_id
            for expert_id in self._lookahead_prediction
            if expert_id not in resident
        ][: self._staging_capacity]
        if not expert_ids:
            return

        bank_index = 2
        result = pool.banks[bank_index]
        mapping_host = self._lookahead_mapping_host
        assert mapping_host is not None
        mapping_host.fill_(-1)
        for slot, expert_id in enumerate(expert_ids):
            mapping_host[expert_id] = slot

        with torch.cuda.stream(pool.copy_stream):
            self._wait_for_exclusive_host_updates(pool.copy_stream)
            pool.copy_stream.wait_event(pool.consumed[bank_index])
            scale_copies: list[tuple[int, int]] = []
            for slot, expert_id in enumerate(expert_ids):
                result.w1[slot].copy_(
                    self._cpu_expert_row(self._cpu_w13, expert_id),
                    non_blocking=True,
                )
                result.w2[slot].copy_(
                    self._cpu_expert_row(self._cpu_w2, expert_id),
                    non_blocking=True,
                )
                if result.w1_scale is not None:
                    assert result.w2_scale is not None
                    if self._packed_scales:
                        scale_copies.append((slot, expert_id))
                    else:
                        assert self._cpu_w13_scale is not None
                        assert self._cpu_w2_scale is not None
                        result.w1_scale[slot].copy_(
                            self._cpu_expert_row(self._cpu_w13_scale, expert_id),
                            non_blocking=True,
                        )
                        result.w2_scale[slot].copy_(
                            self._cpu_expert_row(self._cpu_w2_scale, expert_id),
                            non_blocking=True,
                        )
            if scale_copies:
                assert result.w1_scale is not None
                assert result.w2_scale is not None
                self._copy_packed_scales(
                    result.w1_scale,
                    result.w2_scale,
                    scale_copies,
                )
            result.expert_map.copy_(mapping_host, non_blocking=True)
            self._record_exclusive_host_reads(pool.copy_stream)
            pool.ready[bank_index].record(pool.copy_stream)
        self._lookahead_pending_ids = expert_ids

    def take_lookahead_group(
        self, unique_ids: list[int]
    ) -> tuple[list[int], ExpertWeightResult] | None:
        """Return the predicted experts actually used by this forward."""
        pool = self._staging_pool
        if pool is None or len(pool.banks) < 3 or not self._lookahead_pending_ids:
            return None
        used = set(unique_ids)
        actual = [
            expert_id for expert_id in self._lookahead_pending_ids if expert_id in used
        ]
        if not actual:
            self._lookahead_pending_ids = []
            return None
        return actual, pool.banks[2]

    def wait_lookahead_group(self) -> None:
        pool = self._staging_pool
        if pool is None or len(pool.banks) < 3:
            raise RuntimeError("inter-layer expert prefetch is not enabled")
        torch.cuda.current_stream(device=pool.banks[2].w1.device).wait_event(
            pool.ready[2]
        )

    def release_lookahead_group(self) -> None:
        pool = self._staging_pool
        if pool is None or len(pool.banks) < 3:
            raise RuntimeError("inter-layer expert prefetch is not enabled")
        pool.consumed[2].record(
            torch.cuda.current_stream(device=pool.banks[2].w1.device)
        )
        self._lookahead_pending_ids = []

    def _prepare_mapped_scale_bank(
        self, expert_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Populate the next host scale bank in the current weight-slot order."""
        if not self._mapped_scale_aliases:
            raise RuntimeError("mapped expert scales are not enabled")
        assert self._cpu_w13_scale is not None
        assert self._cpu_w2_scale is not None

        bank_index = self._mapped_scale_cursor
        self._mapped_scale_cursor = (bank_index + 1) % len(self._mapped_scale_aliases)
        if self._mapped_scale_in_use[bank_index]:
            event = self._mapped_scale_events[bank_index]
            if not event.query():
                self.mapped_scale_waits += 1
                event.synchronize()

        w13_host, w2_host = self._mapped_scale_hosts[bank_index]
        for expert_id in expert_ids:
            slot = self._lru[expert_id][0]
            w13_host[slot].copy_(self._cpu_w13_scale[expert_id])
            w2_host[slot].copy_(self._cpu_w2_scale[expert_id])
        w13_alias, w2_alias = self._mapped_scale_aliases[bank_index]
        return w13_alias, w2_alias, bank_index

    def release_mapped_scale_bank(self, result: ExpertWeightResult) -> None:
        """Protect a mapped scale bank until its queued Marlin work completes."""
        bank_index = result.mapped_scale_bank
        if bank_index is None:
            return
        self._mapped_scale_events[bank_index].record(
            torch.cuda.current_stream(device=self._buf_w13.device)
        )
        self._mapped_scale_in_use[bank_index] = True

    def uses_bulk_prefill(self, topk_ids: torch.Tensor) -> bool:
        """Whether this forward can use fixed contiguous expert groups.

        This test reads shape metadata only. In particular, it must not inspect
        tensor values or synchronize the device: avoiding that synchronization
        is one of the fast path's primary purposes.
        """
        return (
            self._bulk_prefill_min_tokens > 0
            and topk_ids.size(0) >= self._bulk_prefill_min_tokens
            and len(self._bulk_prefill_groups) > 1
        )

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

    def _copy_experts_batched(self, misses: list[tuple[int, int]]) -> None:
        """Submit all weight and scale misses as one CUDA batch."""
        function = self._batched_copy_function
        copy_stream = self._batched_copy_stream
        ready = self._batched_copy_ready
        if function is None or copy_stream is None or ready is None:
            raise RuntimeError("CUDA batched expert copy is not initialized")

        destination_pointers: list[int] = []
        source_pointers: list[int] = []
        sizes: list[int] = []

        def append_tensor_copies(
            destination: torch.Tensor, source: torch.Tensor
        ) -> None:
            if not destination[0].is_contiguous() or not source[0].is_contiguous():
                raise RuntimeError(
                    "CUDA batched expert copy requires contiguous per-expert tensors"
                )
            destination_stride = destination.stride(0) * destination.element_size()
            source_stride = source.stride(0) * source.element_size()
            copy_size = source[0].numel() * source.element_size()
            for slot, expert_id in misses:
                destination_pointers.append(
                    destination.data_ptr() + slot * destination_stride
                )
                source_pointers.append(source.data_ptr() + expert_id * source_stride)
                sizes.append(copy_size)

        append_tensor_copies(self._buf_w13, self._cpu_w13)
        append_tensor_copies(self._buf_w2, self._cpu_w2)
        if (
            self._buf_w13_scale is not None
            and not self._mapped_scales
            and not self._packed_scales
        ):
            assert self._cpu_w13_scale is not None
            assert self._cpu_w2_scale is not None
            assert self._buf_w2_scale is not None
            append_tensor_copies(self._buf_w13_scale, self._cpu_w13_scale)
            append_tensor_copies(self._buf_w2_scale, self._cpu_w2_scale)

        count = len(destination_pointers)
        destinations_array = (ctypes.c_uint64 * count)(*destination_pointers)
        sources_array = (ctypes.c_uint64 * count)(*source_pointers)
        sizes_array = (ctypes.c_size_t * count)(*sizes)

        compute_stream = torch.cuda.current_stream(device=self._buf_w13.device)
        copy_stream.wait_stream(compute_stream)
        device_index = self._buf_w13.device.index
        assert device_index is not None
        error = function(
            destinations_array,
            sources_array,
            sizes_array,
            count,
            copy_stream.cuda_stream,
            device_index,
        )
        if error != 0:
            raise RuntimeError(f"cudaMemcpyBatchAsync failed with CUDA error {error}")
        ready.record(copy_stream)
        compute_stream.wait_event(ready)

    @eager_break_during_capture
    @torch.compiler.disable
    def prepare_staged_group(
        self,
        unique_ids: list[int],
        group_index: int,
        *,
        wait_for_ready: bool = True,
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
        if len(unique_ids) > pool.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but staging capacity is {pool.capacity}."
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
            # Pools are shared across layers, so bank lifetime is not scoped
            # to this provider's group indices. Always honor the last consumer
            # event; the events are initialized as reusable when the pool is
            # created.
            pool.copy_stream.wait_event(pool.consumed[bank_index])
            self._wait_for_exclusive_host_updates(pool.copy_stream)
            scale_copies: list[tuple[int, int]] = []
            for slot, expert_id in enumerate(unique_ids):
                result.w1[slot].copy_(
                    self._cpu_expert_row(self._cpu_w13, expert_id),
                    non_blocking=True,
                )
                result.w2[slot].copy_(
                    self._cpu_expert_row(self._cpu_w2, expert_id),
                    non_blocking=True,
                )
                if result.w1_scale is not None:
                    assert result.w2_scale is not None
                    if self._packed_scales:
                        scale_copies.append((slot, expert_id))
                    else:
                        assert self._cpu_w13_scale is not None
                        assert self._cpu_w2_scale is not None
                        result.w1_scale[slot].copy_(
                            self._cpu_expert_row(self._cpu_w13_scale, expert_id),
                            non_blocking=True,
                        )
                        result.w2_scale[slot].copy_(
                            self._cpu_expert_row(self._cpu_w2_scale, expert_id),
                            non_blocking=True,
                        )
            if scale_copies:
                assert result.w1_scale is not None
                assert result.w2_scale is not None
                self._copy_packed_scales(
                    result.w1_scale,
                    result.w2_scale,
                    scale_copies,
                )
            result.expert_map.copy_(mapping_host, non_blocking=True)
            self._record_exclusive_host_reads(pool.copy_stream)
            pool.ready[bank_index].record(pool.copy_stream)

        if wait_for_ready:
            compute_stream.wait_event(pool.ready[bank_index])
        self._record_prepare_stats(hits=0, misses=len(unique_ids))
        return result

    @eager_break_during_capture
    @torch.compiler.disable
    def wait_staged_group(self, group_index: int) -> None:
        """Make the compute stream wait for an earlier speculative upload."""
        pool = self._staging_pool
        if pool is None:
            raise RuntimeError("asynchronous expert staging is not enabled")
        bank_index = group_index % 2
        current_platform.current_stream(pool.banks[bank_index].w1.device).wait_event(
            pool.ready[bank_index]
        )

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

    @eager_break_during_capture
    @torch.compiler.disable
    def prepare_bulk_prefill_group(self, group_index: int) -> ExpertWeightResult:
        """Upload one fixed contiguous expert group with bulk H2D copies.

        The existing per-layer cache buffers are reused, so this adds no large
        device allocation. Copies and the following MoE kernel share the
        current stream; overwriting the buffer for the next group is therefore
        ordered after the current group's kernel without a host synchronize.
        """
        if group_index >= len(self._bulk_prefill_groups):
            raise RuntimeError(
                f"bulk prefill group index {group_index} exceeds the "
                f"{len(self._bulk_prefill_groups)} available groups"
            )

        expert_ids = self._bulk_prefill_groups[group_index]
        start = expert_ids[0]
        end = expert_ids[-1] + 1
        count = end - start

        # A single 64-expert W13 transfer is roughly 128 MiB for DSV4 TP4.
        # Four workers issuing transfers that large at once produce unfair,
        # bursty PCIe service. Chunk only the large weight tensors to preserve
        # inter-rank fairness while still removing most copy submissions. The
        # much smaller scale tensors remain one transfer per group.
        for source_start in range(start, end, self._bulk_copy_chunk_experts):
            source_end = min(source_start + self._bulk_copy_chunk_experts, end)
            target_start = source_start - start
            target_end = source_end - start
            self._buf_w13[target_start:target_end].copy_(
                self._cpu_w13[source_start:source_end], non_blocking=True
            )
            self._buf_w2[target_start:target_end].copy_(
                self._cpu_w2[source_start:source_end], non_blocking=True
            )
        if self._buf_w13_scale is not None and not self._mapped_scales:
            assert self._buf_w2_scale is not None
            if self._packed_scales:
                self._copy_packed_scales(
                    self._buf_w13_scale,
                    self._buf_w2_scale,
                    [(slot, expert_id) for slot, expert_id in enumerate(expert_ids)],
                )
            else:
                assert self._cpu_w13_scale is not None
                assert self._cpu_w2_scale is not None
                self._buf_w13_scale[:count].copy_(
                    self._cpu_w13_scale[start:end], non_blocking=True
                )
                self._buf_w2_scale[:count].copy_(
                    self._cpu_w2_scale[start:end], non_blocking=True
                )

        # The bulk upload overwrites every occupied slot. Keep the adaptive
        # cache metadata truthful so the next decode/short-prefill prepare()
        # can reuse or evict the final group's weights normally.
        self._clock += 1
        self._lru = {
            expert_id: [slot, 1, self._clock]
            for slot, expert_id in enumerate(expert_ids)
        }
        self._free_slots = list(range(count, self.capacity))
        self._record_prepare_stats(hits=0, misses=count)

        result_w13_scale = self._buf_w13_scale
        result_w2_scale = self._buf_w2_scale
        mapped_scale_bank = None
        if self._mapped_scales:
            result_w13_scale, result_w2_scale, mapped_scale_bank = (
                self._prepare_mapped_scale_bank(expert_ids)
            )

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            expert_map=self._bulk_prefill_mappings[group_index],
            w1_scale=result_w13_scale,
            w2_scale=result_w2_scale,
            mapped_scale_bank=mapped_scale_bank,
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
        valid_ids = topk_ids[(topk_ids >= 0) & (topk_ids < self._num_experts)]
        unique = valid_ids.unique()
        if unique.numel() <= self.capacity:
            return [(slice(0, num_rows), unique.tolist())]

        rows = topk_ids.tolist()
        chunks: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        for i, row in enumerate(rows):
            row_ids = {
                expert_id for expert_id in row if 0 <= expert_id < self._num_experts
            }
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
    def plan_expert_groups(
        self,
        topk_ids: torch.Tensor,
        unique_ids: list[int] | None = None,
        excluded_ids: set[int] | None = None,
    ) -> list[list[int]]:
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
        if (
            not self._route_stats_logged
            and self._route_stats_min_tokens > 0
            and topk_ids.size(0) >= self._route_stats_min_tokens
            and (
                not self._route_stats_arm_file
                or os.path.exists(self._route_stats_arm_file)
            )
        ):
            # One-shot diagnostic for choosing a principled CPU/GPU or
            # multi-device split.  Only 256 int64 counters cross the PCIe bus;
            # subsequent forwards pay no instrumentation cost.
            flat_ids = topk_ids.reshape(-1)
            # Profiling/capture inputs can contain the -1 sentinel. Although
            # the arm file keeps those warmups out of this diagnostic, retain
            # the same validity rule as the MoE router so instrumentation can
            # never make an otherwise-valid forward fail.
            valid_ids = flat_ids[(flat_ids >= 0) & (flat_ids < self._num_experts)]
            counts = torch.bincount(
                valid_ids.to(torch.int64), minlength=self._num_experts
            ).cpu()
            used = [
                (expert_id, int(count))
                for expert_id, count in enumerate(counts.tolist())
                if count > 0
            ]
            used.sort(key=lambda item: (-item[1], item[0]))
            total_pairs = sum(count for _, count in used)
            coverage = {}
            for resident_count in (8, 16, 32, 48, 64, 96, 128):
                covered = sum(count for _, count in used[:resident_count])
                coverage[resident_count] = covered / total_pairs if total_pairs else 0.0
            rare = {
                threshold: sum(1 for _, count in used if count <= threshold)
                for threshold in (1, 2, 4, 8, 16)
            }
            logger.info(
                "MoE route histogram %s: tokens=%d pairs=%d unique=%d "
                "coverage=%s rare_experts=%s counts=%s",
                self.name,
                topk_ids.size(0),
                total_pairs,
                len(used),
                coverage,
                rare,
                used,
            )
            self._route_stats_logged = True

        if self.uses_bulk_prefill(topk_ids) and not excluded_ids:
            return self._bulk_prefill_groups

        unique = topk_ids.unique().tolist() if unique_ids is None else unique_ids
        unique = [
            expert_id for expert_id in unique if 0 <= expert_id < self._num_experts
        ]
        if excluded_ids:
            unique = [
                expert_id for expert_id in unique if expert_id not in excluded_ids
            ]
        if len(unique) <= self.capacity:
            return [unique]
        # A chunked prefill commonly touches all experts in every chunk. The
        # prior implementation began from unique() order, immediately evicted
        # the group left resident by the preceding chunk, and therefore copied
        # every expert again. Consume residents first so up to one full group
        # is evaluated without H2D traffic before cold groups reuse its slots.
        resident = [expert_id for expert_id in unique if expert_id in self._lru]
        missing = [expert_id for expert_id in unique if expert_id not in self._lru]
        if self.async_staging_ready:
            # Do not pad the resident group with misses. A mixed group would
            # be staged in its entirety, redundantly uploading every cache hit
            # and recreating the failure mode of the first async prototype.
            groups = [resident] if resident else []
            groups.extend(
                missing[i : i + self._staging_capacity]
                for i in range(0, len(missing), self._staging_capacity)
            )
            return groups
        unique = resident + missing
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
        unique_ids = [
            expert_id for expert_id in unique_ids if 0 <= expert_id < self._num_experts
        ]
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

        misses_to_copy: list[tuple[int, int]] = []
        exclusive_swaps: list[tuple[int, int, int, int]] = []
        exclusive_scale_copies: list[tuple[int, int]] = []
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
                    if self.exclusive:
                        raise RuntimeError(
                            "exclusive expert cache unexpectedly has a free slot"
                        )
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
                    victim_id = best_key
                    victim_slot = self._lru.pop(best_key)[0]
                    if self.exclusive:
                        slot, swap = self._plan_exclusive_swap(
                            expert_id, victim_id, victim_slot
                        )
                        exclusive_swaps.append(swap)
                        if self._packed_scales:
                            exclusive_scale_copies.append((slot, expert_id))
                    else:
                        slot = victim_slot

                self._clock += 1
                self._lru[expert_id] = [slot, 1, self._clock]
                self.misses += 1
                if not self.exclusive:
                    misses_to_copy.append((slot, expert_id))

        result_w13_scale = self._buf_w13_scale
        result_w2_scale = self._buf_w2_scale
        mapped_scale_bank = None
        if self._mapped_scales:
            result_w13_scale, result_w2_scale, mapped_scale_bank = (
                self._prepare_mapped_scale_bank(unique_ids)
            )

        if exclusive_swaps:
            self._queue_exclusive_swap_wave(exclusive_swaps)
            h2d_done = self._exclusive_h2d_done
            d2h_done = self._exclusive_d2h_done
            d2h_stream = self._exclusive_d2h_stream
            assert h2d_done is not None
            assert d2h_done is not None
            assert d2h_stream is not None
            compute_stream = current_platform.current_stream(self._buf_w13.device)
            if exclusive_scale_copies:
                assert self._buf_w13_scale is not None
                assert self._buf_w2_scale is not None
                # The final D2H record means every destination slot is no
                # longer read by its evicted expert. Expand immutable packed
                # scales on that stream so the kernel can overlap the final
                # large weight H2D on the opposite copy engine.
                with torch.cuda.stream(d2h_stream):
                    self._copy_packed_scales(
                        self._buf_w13_scale,
                        self._buf_w2_scale,
                        exclusive_scale_copies,
                    )
                    d2h_done.record(d2h_stream)
                compute_stream.wait_event(d2h_done)
            compute_stream.wait_event(h2d_done)

        if (
            self._batched_copy_function is not None
            and len(misses_to_copy) >= self._batched_copy_min_experts
            and topk_ids.size(0) >= self._batched_copy_min_tokens
        ):
            self._copy_experts_batched(misses_to_copy)
        else:
            for slot, expert_id in misses_to_copy:
                self._buf_w13[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
                self._buf_w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
                if self._buf_w13_scale is not None and not self._mapped_scales:
                    assert self._buf_w2_scale is not None
                    if not self._packed_scales:
                        assert self._cpu_w13_scale is not None
                        assert self._cpu_w2_scale is not None
                        self._buf_w13_scale[slot].copy_(
                            self._cpu_w13_scale[expert_id], non_blocking=True
                        )
                        self._buf_w2_scale[slot].copy_(
                            self._cpu_w2_scale[expert_id], non_blocking=True
                        )

        packed_scale_copies = [] if self.exclusive else misses_to_copy
        if self._packed_scale_backing is not None and packed_scale_copies:
            assert self._buf_w13_scale is not None
            assert self._buf_w2_scale is not None
            self._copy_packed_scales(
                self._buf_w13_scale,
                self._buf_w2_scale,
                packed_scale_copies,
            )

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
            w1_scale=result_w13_scale,
            w2_scale=result_w2_scale,
            mapped_scale_bank=mapped_scale_bank,
        )


_INTERLAYER_PROVIDERS: dict[tuple[int, str, int], CachedWeightProvider] = {}


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

    def run_and_release(
        result: ExpertWeightResult, rows: slice, include_shared: bool
    ) -> torch.Tensor:
        part = run(result, rows, include_shared)
        provider.release_mapped_scale_bank(result)
        return part

    def finish_forward(output: torch.Tensor) -> torch.Tensor:
        provider.remember_prefill_route(topk_ids)
        provider.prefetch_successor(topk_ids.size(0))
        return output

    # Breakable CUDA graphs can safely replay dynamic cache preparation only
    # when the captured graph has a fixed number of MoE kernel calls. If the
    # total number of routed pairs fits in the cache, the unique expert count
    # necessarily fits too, so prepare() updates its fixed GPU buffers in one
    # eager break and the single MoE kernel remains in the following graph
    # segment. This covers the captured decode sizes (1 and 2) without freezing
    # their expert ids into the graph.
    if topk_ids.numel() <= provider.capacity:
        return finish_forward(
            run_and_release(provider.prepare(topk_ids), _ALL_ROWS, True)
        )

    if provider.split == "expert":
        unique_ids = topk_ids.unique().tolist()
        lookahead = provider.take_lookahead_group(unique_ids)
        lookahead_ids = lookahead[0] if lookahead is not None else []
        borrowed_groups = provider.plan_borrowed_groups(
            topk_ids,
            unique_ids,
            excluded_ids=set(lookahead_ids),
        )
        borrowed_ids = {
            expert_id for _, group in borrowed_groups for expert_id in group
        }
        # Keep the policy fixed for the whole forward. Borrowed-bank planning
        # intentionally invalidates duplicate primary entries, so snapshot
        # readiness afterward: the group planner and executor must observe the
        # same post-invalidation cache state.
        async_staging_ready = provider.async_staging_ready
        bulk_prefill = (
            provider.uses_bulk_prefill(topk_ids)
            and not borrowed_groups
            and lookahead is None
        )
        groups = provider.plan_expert_groups(
            topk_ids,
            unique_ids=unique_ids,
            excluded_ids=borrowed_ids | set(lookahead_ids),
        )
        if not borrowed_groups and lookahead is None and len(groups) == 1:
            return finish_forward(
                run_and_release(
                    provider.prepare(topk_ids, groups[0]),
                    _ALL_ROWS,
                    True,
                )
            )

        execution_groups: list[tuple[str, int | None, list[int]]] = [
            ("borrowed", bank_index, group) for bank_index, group in borrowed_groups
        ]
        if lookahead is not None:
            execution_groups.append(("lookahead", None, lookahead_ids))
        execution_groups.extend(("primary", None, group) for group in groups if group)

        # Launch the first true-miss upload before enqueueing the resident
        # groups. Deferring only its ready-event wait gives the copy stream the
        # full resident-compute lead time without allocating another bank.
        prestaged_execution_index: int | None = None
        prestaged_result: ExpertWeightResult | None = None
        if async_staging_ready and not bulk_prefill:
            for i, (kind, _, expert_ids) in enumerate(execution_groups):
                if kind == "primary" and not provider.group_is_resident(expert_ids):
                    prestaged_execution_index = i
                    prestaged_result = provider.prepare_staged_group(
                        expert_ids,
                        0,
                        wait_for_ready=False,
                    )
                    break

        # Accumulate in fp32. It does not recover what each group already lost
        # rounding to the model dtype, but it keeps the sum from losing more.
        accumulator: torch.Tensor | None = None
        out_dtype: torch.dtype | None = None
        staged_group_index = 0
        for i, (kind, borrowed_bank, expert_ids) in enumerate(execution_groups):
            resident = kind != "primary" or provider.group_is_resident(expert_ids)
            staged = async_staging_ready and not bulk_prefill and not resident
            result = (
                provider.prepare_borrowed_group(expert_ids, borrowed_bank)
                if kind == "borrowed" and borrowed_bank is not None
                else (
                    lookahead[1]
                    if kind == "lookahead" and lookahead is not None
                    else (
                        provider.prepare_bulk_prefill_group(i)
                        if bulk_prefill
                        else (
                            (
                                prestaged_result
                                if i == prestaged_execution_index
                                else provider.prepare_staged_group(
                                    expert_ids, staged_group_index
                                )
                            )
                            if staged
                            else provider.prepare(topk_ids, expert_ids)
                        )
                    )
                )
            )
            if kind == "lookahead":
                provider.wait_lookahead_group()
            if staged and i == prestaged_execution_index:
                assert prestaged_result is not None
                provider.wait_staged_group(staged_group_index)
            if result is None:
                raise RuntimeError("pre-staged expert weights were not prepared")
            part = run_and_release(result, _ALL_ROWS, i == 0)
            if kind == "lookahead":
                provider.release_lookahead_group()
            if staged:
                provider.release_staged_group(staged_group_index)
                staged_group_index += 1
            if accumulator is None:
                accumulator, out_dtype = part.float(), part.dtype
            else:
                accumulator += part.float()
        assert accumulator is not None and out_dtype is not None
        return finish_forward(accumulator.to(out_dtype))

    plan = provider.plan_chunks(topk_ids)
    if len(plan) == 1:
        rows, unique_ids = plan[0]
        return finish_forward(
            run_and_release(provider.prepare(topk_ids, unique_ids), rows, True)
        )
    return finish_forward(
        torch.cat(
            [
                run_and_release(
                    provider.prepare(topk_ids[rows], unique_ids), rows, True
                )
                for rows, unique_ids in plan
            ],
            dim=0,
        )
    )
