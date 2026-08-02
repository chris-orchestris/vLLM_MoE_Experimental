# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned-host all-reduce for PCIe-only tensor-parallel CUDA groups.

This backend is deliberately narrow and opt-in.  It targets large BF16
prefill tensors on PCIe GPUs when direct GPU peer-to-peer transfers are
unavailable.  Each rank copies into a shared, CUDA-registered host mapping,
reduces a disjoint slice of the tensor on the CPU, and copies the result back.
Small/decode tensors and CUDA-graph capture continue to use NCCL.
"""

from __future__ import annotations

import ctypes
import importlib.util
import mmap
import os
import queue
import threading
import uuid

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_HEADER_BYTES = 4096
_CUDA_HOST_REGISTER_PORTABLE = 1


def _env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class HostAllReduce:
    """Cooperative pinned-host all-reduce for PCIe tensor parallelism."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        rank_in_group: int,
        world_size: int,
        global_ranks: list[int],
    ) -> None:
        self.disabled = True
        self._mapping: mmap.mmap | None = None
        self._shared: torch.Tensor | None = None
        self._mapped_output: torch.Tensor | None = None
        self._registered_address: int | None = None
        self._pipeline_worker: threading.Thread | None = None

        if os.environ.get("VLLM_HOST_ALLREDUCE", "0") != "1":
            return
        if world_size < 2:
            logger.warning_once(
                "VLLM_HOST_ALLREDUCE requires tensor parallel size >= 2; got %d.",
                world_size,
            )
            return
        if device.type != "cuda":
            logger.warning_once("VLLM_HOST_ALLREDUCE requires CUDA devices.")
            return

        self.cpu_group = cpu_group
        self.device = device
        self.rank = rank_in_group
        self.world_size = world_size
        self.min_bytes = _env_int("VLLM_HOST_ALLREDUCE_MIN_BYTES", 8 << 20)
        self.max_bytes = _env_int("VLLM_HOST_ALLREDUCE_MAX_BYTES", 64 << 20)
        self.threads_per_rank = _env_int(
            "VLLM_HOST_ALLREDUCE_THREADS_PER_RANK", 14
        )
        self.use_mapped_output = (
            os.environ.get("VLLM_HOST_ALLREDUCE_MAPPED_OUTPUT", "0") == "1"
        )
        self.pipeline_chunks = int(
            os.environ.get("VLLM_HOST_ALLREDUCE_PIPELINE_CHUNKS", "0")
        )
        if self.pipeline_chunks < 0 or self.pipeline_chunks > 32:
            raise ValueError(
                "VLLM_HOST_ALLREDUCE_PIPELINE_CHUNKS must be between 0 and 32"
            )
        if self.pipeline_chunks and not self.use_mapped_output:
            raise ValueError(
                "VLLM_HOST_ALLREDUCE_PIPELINE_CHUNKS requires mapped output"
            )
        if self.min_bytes > self.max_bytes:
            raise ValueError(
                "VLLM_HOST_ALLREDUCE_MIN_BYTES must not exceed "
                "VLLM_HOST_ALLREDUCE_MAX_BYTES"
            )
        # BF16 is the only supported dtype, so keep the shared capacity aligned
        # to its two-byte element size.
        self.max_bytes -= self.max_bytes % torch.bfloat16.itemsize
        max_elements = self.max_bytes // torch.bfloat16.itemsize
        data_bytes = (self.world_size + 1) * self.max_bytes
        mapping_bytes = _HEADER_BYTES + data_bytes

        path_holder: list[str | None] = [None]
        if self.rank == 0:
            path_holder[0] = f"/dev/shm/vllm-host-ar-{uuid.uuid4().hex}"
            descriptor = os.open(
                path_holder[0], os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            try:
                os.ftruncate(descriptor, mapping_bytes)
            finally:
                os.close(descriptor)
        dist.broadcast_object_list(
            path_holder, src=global_ranks[0], group=self.cpu_group
        )
        shared_path = path_holder[0]
        assert shared_path is not None

        descriptor = os.open(shared_path, os.O_RDWR)
        try:
            self._mapping = mmap.mmap(descriptor, mapping_bytes)
        finally:
            os.close(descriptor)

        header_address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        library_path = os.environ.get("VLLM_HOST_ALLREDUCE_LIBRARY")
        if not library_path:
            raise RuntimeError(
                "VLLM_HOST_ALLREDUCE_LIBRARY must point to the built host "
                "barrier library when pinned-host all-reduce is enabled"
            )
        self._barrier_library = ctypes.CDLL(library_path)
        self._barrier_library.vllm_host_barriers_init.argtypes = [ctypes.c_void_p]
        self._barrier_library.vllm_host_barriers_init.restype = None
        self._barrier_library.vllm_host_barrier.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        self._barrier_library.vllm_host_barrier.restype = None
        self._header_address = header_address
        if self.rank == 0:
            self._barrier_library.vllm_host_barriers_init(header_address)
        dist.barrier(group=self.cpu_group)

        self._shared = torch.frombuffer(
            self._mapping,
            dtype=torch.bfloat16,
            count=(self.world_size + 1) * max_elements,
            offset=_HEADER_BYTES,
        ).view(self.world_size + 1, max_elements)
        self._cudart = ctypes.CDLL("libcudart.so.13")
        self._cudart.cudaHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._cudart.cudaHostRegister.restype = ctypes.c_int
        self._cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self._cudart.cudaHostUnregister.restype = ctypes.c_int
        self._registered_address = self._shared.data_ptr()
        register_flags = _CUDA_HOST_REGISTER_PORTABLE
        if self.use_mapped_output:
            # cudaHostRegisterMapped makes the shared result directly
            # addressable by SMs. The next MHC kernel can consume it without
            # first materializing a duplicate in device memory.
            register_flags |= 2
        error = self._cudart.cudaHostRegister(
            self._registered_address, data_bytes, register_flags
        )
        if error != 0:
            raise RuntimeError(
                "cudaHostRegister for VLLM_HOST_ALLREDUCE failed with CUDA "
                f"error {error}"
            )
        if self.use_mapped_output:
            self._mapped_output = self._make_mapped_output(max_elements)
        if self.pipeline_chunks:
            self._initialize_pipeline()
        dist.barrier(group=self.cpu_group)
        if self.rank == 0:
            os.unlink(shared_path)
        dist.barrier(group=self.cpu_group)

        # Symmetric workers each reduce a disjoint output slice. The caller
        # sizes each rank's CPU pool for the host's physical-core budget.
        torch.set_num_threads(self.threads_per_rank)
        self.disabled = False
        logger.info_once(
            "Enabled pinned-host TP all-reduce for BF16 tensors in [%d, %d] "
            "bytes with %d CPU threads per rank.",
            self.min_bytes,
            self.max_bytes,
            self.threads_per_rank,
            scope="global",
        )

    def _make_mapped_output(self, max_elements: int) -> torch.Tensor:
        self._cudart.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._cudart.cudaHostGetDevicePointer.restype = ctypes.c_int
        assert self._shared is not None
        host_output = self._shared[self.world_size]
        device_pointer = ctypes.c_void_p()
        error = self._cudart.cudaHostGetDevicePointer(
            ctypes.byref(device_pointer), host_output.data_ptr(), 0
        )
        if error != 0 or device_pointer.value is None:
            raise RuntimeError(
                "cudaHostGetDevicePointer for VLLM_HOST_ALLREDUCE failed "
                f"with CUDA error {error}"
            )

        extension_path = os.environ.get(
            "VLLM_HOST_ALLREDUCE_MAPPED_TENSOR_LIBRARY"
        )
        if not extension_path:
            raise RuntimeError(
                "VLLM_HOST_ALLREDUCE_MAPPED_TENSOR_LIBRARY must point to the "
                "built mapped-host tensor extension when mapped output is enabled"
            )
        spec = importlib.util.spec_from_file_location(
            "mapped_host_tensor_ext", extension_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load mapped-host tensor extension {extension_path}"
            )
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        mapped = extension.mapped_cuda_tensor(
            device_pointer.value,
            [max_elements],
            torch.cuda.current_device(),
        )
        logger.info_once(
            "VLLM_HOST_ALLREDUCE will return a mapped-host result directly to "
            "the next CUDA consumer.",
            scope="global",
        )
        return mapped

    def _initialize_pipeline(self) -> None:
        self._pipeline_stream = torch.cuda.Stream(device=self.device)
        self._pipeline_events = [
            torch.cuda.Event() for _ in range(self.pipeline_chunks)
        ]
        self._pipeline_ready = [
            threading.Event() for _ in range(self.pipeline_chunks)
        ]
        self._pipeline_jobs: queue.SimpleQueue[bool | None] = queue.SimpleQueue()
        self._pipeline_worker = threading.Thread(
            target=self._pipeline_copy_worker,
            name=f"host-ar-copy-{self.rank}",
            daemon=True,
        )
        self._pipeline_worker.start()
        logger.info_once(
            "VLLM_HOST_ALLREDUCE will pipeline D2H and cooperative CPU "
            "reduction in %d chunks.",
            self.pipeline_chunks,
            scope="global",
        )

    def _pipeline_copy_worker(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            job = self._pipeline_jobs.get()
            if job is None:
                return
            for index, event in enumerate(self._pipeline_events):
                event.synchronize()
                self._barrier(index)
                self._pipeline_ready[index].set()

    def should_use(self, input_: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if input_.dtype is not torch.bfloat16 or not input_.is_contiguous():
            return False
        if input_.device.type != "cuda":
            return False
        size_bytes = input_.numel() * input_.element_size()
        if size_bytes < self.min_bytes or size_bytes > self.max_bytes:
            return False
        # Host synchronization is neither capture-safe nor useful for the
        # small decode tensors handled by the CUDA graph.
        return not torch.cuda.is_current_stream_capturing()

    def _reduce_host_slice(self, start: int, end: int) -> None:
        """Sum one disjoint output slice across every participating rank."""
        assert self._shared is not None
        output_slice = self._shared[self.world_size, start:end]
        torch.add(
            self._shared[0, start:end],
            self._shared[1, start:end],
            out=output_slice,
        )
        for source_rank in range(2, self.world_size):
            output_slice.add_(self._shared[source_rank, start:end])

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        assert self._shared is not None
        elements = input_.numel()
        flat_input = input_.view(-1)
        if self.pipeline_chunks:
            return self._all_reduce_pipelined(input_, flat_input)
        with torch.profiler.record_function("vllm::host_ar_d2h_and_sync"):
            self._shared[self.rank, :elements].copy_(flat_input, non_blocking=True)

            # Synchronize only the calling stream: this waits for the producer
            # and D2H copy without imposing a device-wide barrier.
            stream = torch.cuda.current_stream(input_.device)
            stream.synchronize()
        with torch.profiler.record_function("vllm::host_ar_input_barrier"):
            self._barrier(0)

        with torch.profiler.record_function("vllm::host_ar_cpu_reduce"):
            start = self.rank * elements // self.world_size
            end = (self.rank + 1) * elements // self.world_size
            self._reduce_host_slice(start, end)
        with torch.profiler.record_function("vllm::host_ar_output_barrier"):
            self._barrier(1)

        if self._mapped_output is not None:
            output = self._mapped_output[:elements].view_as(input_)
        else:
            with torch.profiler.record_function("vllm::host_ar_h2d_submit"):
                output = torch.empty_like(input_)
                output.view(-1).copy_(
                    self._shared[self.world_size, :elements], non_blocking=True
                )
        return output

    def _all_reduce_pipelined(
        self, input_: torch.Tensor, flat_input: torch.Tensor
    ) -> torch.Tensor:
        """Overlap chunk D2H with cooperative CPU reduction.

        A persistent helper waits for each copy-stream event and enters that
        chunk's cross-rank barrier. Rank ``i % world_size`` reduces chunk i as
        soon as every GPU's D2H for that chunk is visible. This pipelines the
        later D2H chunks behind reductions of the earlier chunks.
        """
        assert self._shared is not None
        assert self._mapped_output is not None
        elements = flat_input.numel()
        for ready in self._pipeline_ready:
            ready.clear()

        compute_stream = torch.cuda.current_stream(input_.device)
        self._pipeline_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self._pipeline_stream):
            for index, event in enumerate(self._pipeline_events):
                start = index * elements // self.pipeline_chunks
                end = (index + 1) * elements // self.pipeline_chunks
                self._shared[self.rank, start:end].copy_(
                    flat_input[start:end], non_blocking=True
                )
                event.record(self._pipeline_stream)
        self._pipeline_jobs.put(True)

        for index in range(self.rank, self.pipeline_chunks, self.world_size):
            self._pipeline_ready[index].wait()
            start = index * elements // self.pipeline_chunks
            end = (index + 1) * elements // self.pipeline_chunks
            self._reduce_host_slice(start, end)

        # Every output chunk has exactly one owner. The final barrier ensures
        # all owners have published their slices before any GPU reads the
        # mapped result.
        self._barrier(self.pipeline_chunks)
        return self._mapped_output[:elements].view_as(input_)

    def _barrier(self, index: int) -> None:
        self._barrier_library.vllm_host_barrier(
            self._header_address, index, self.world_size
        )

    def destroy(self) -> None:
        if self.disabled:
            return
        torch.cuda.synchronize(self.device)
        if self._pipeline_worker is not None:
            self._pipeline_jobs.put(None)
            self._pipeline_worker.join()
            self._pipeline_worker = None
        if self._registered_address is not None:
            self._mapped_output = None
            error = self._cudart.cudaHostUnregister(self._registered_address)
            if error != 0:
                logger.warning(
                    "cudaHostUnregister for VLLM_HOST_ALLREDUCE failed with "
                    "CUDA error %d.",
                    error,
                )
            self._registered_address = None
        self._shared = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self.disabled = True
