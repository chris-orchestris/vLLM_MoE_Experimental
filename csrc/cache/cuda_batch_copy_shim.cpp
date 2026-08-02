// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// A C ABI keeps the Python side independent of CUDA's compound attribute
// layout. This shim needs only a C++ compiler and libcudart; it contains no
// device code and therefore does not require nvcc.
extern "C" int vllm_cuda_batch_copy_host_to_device(
    std::uintptr_t* destinations, std::uintptr_t* sources, std::size_t* sizes,
    std::size_t count, std::uintptr_t stream_handle, int device) {
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }

  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attributes.srcLocHint = {cudaMemLocationTypeHost, 0};
  attributes.dstLocHint = {cudaMemLocationTypeDevice, device};
  std::size_t attribute_index = 0;
  return static_cast<int>(cudaMemcpyBatchAsync(
      reinterpret_cast<void* const*>(destinations),
      reinterpret_cast<const void* const*>(sources), sizes, count, &attributes,
      &attribute_index, 1, reinterpret_cast<cudaStream_t>(stream_handle)));
}

// Queue a complete full-duplex expert-swap wave with one Python-to-C
// transition. Each swap contains ``copies_per_swap`` tensor pairs (weights
// and, when present, scales). H2D and D2H use distinct rotating spare slots,
// allowing Ampere's independent copy engines to overlap the two directions.
//
// Consecutive swaps reuse the preceding victim GPU slot and incoming host
// slot. The crossed event waits preserve those dependencies inside this shim
// instead of returning to Python for every expert. The caller installs the
// corresponding crossed wait against the preceding *wave* before entry.
extern "C" int vllm_cuda_exclusive_expert_swap_wave_async(
    std::uintptr_t* gpu_destinations, std::uintptr_t* host_sources,
    std::uintptr_t* host_destinations, std::uintptr_t* gpu_sources,
    std::size_t* sizes, std::size_t swap_count, std::size_t copies_per_swap,
    std::uintptr_t h2d_stream_handle, std::uintptr_t d2h_stream_handle,
    std::uintptr_t h2d_event_handle, std::uintptr_t d2h_event_handle,
    int device) {
  if (swap_count == 0 || copies_per_swap == 0 || h2d_event_handle == 0 ||
      d2h_event_handle == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    return static_cast<int>(error);
  }

  auto h2d_stream = reinterpret_cast<cudaStream_t>(h2d_stream_handle);
  auto d2h_stream = reinterpret_cast<cudaStream_t>(d2h_stream_handle);
  auto h2d_event = reinterpret_cast<cudaEvent_t>(h2d_event_handle);
  auto d2h_event = reinterpret_cast<cudaEvent_t>(d2h_event_handle);

  cudaMemcpyAttributes h2d_attributes{};
  h2d_attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  h2d_attributes.srcLocHint = {cudaMemLocationTypeHost, 0};
  h2d_attributes.dstLocHint = {cudaMemLocationTypeDevice, device};
  std::size_t h2d_attribute_index = 0;

  cudaMemcpyAttributes d2h_attributes{};
  d2h_attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  d2h_attributes.srcLocHint = {cudaMemLocationTypeDevice, device};
  d2h_attributes.dstLocHint = {cudaMemLocationTypeHost, 0};
  std::size_t d2h_attribute_index = 0;

  for (std::size_t swap = 0; swap < swap_count; ++swap) {
    const std::size_t offset = swap * copies_per_swap;
    error = cudaMemcpyBatchAsync(
        reinterpret_cast<void* const*>(gpu_destinations + offset),
        reinterpret_cast<const void* const*>(host_sources + offset),
        sizes + offset, copies_per_swap, &h2d_attributes, &h2d_attribute_index,
        1, h2d_stream);
    if (error != cudaSuccess) {
      return static_cast<int>(error);
    }

    error = cudaMemcpyBatchAsync(
        reinterpret_cast<void* const*>(host_destinations + offset),
        reinterpret_cast<const void* const*>(gpu_sources + offset),
        sizes + offset, copies_per_swap, &d2h_attributes, &d2h_attribute_index,
        1, d2h_stream);
    if (error != cudaSuccess) {
      return static_cast<int>(error);
    }

    error = cudaEventRecord(h2d_event, h2d_stream);
    if (error != cudaSuccess) {
      return static_cast<int>(error);
    }
    error = cudaEventRecord(d2h_event, d2h_stream);
    if (error != cudaSuccess) {
      return static_cast<int>(error);
    }

    if (swap + 1 < swap_count) {
      error = cudaStreamWaitEvent(h2d_stream, d2h_event, 0);
      if (error != cudaSuccess) {
        return static_cast<int>(error);
      }
      error = cudaStreamWaitEvent(d2h_stream, h2d_event, 0);
      if (error != cudaSuccess) {
        return static_cast<int>(error);
      }
    }
  }

  return static_cast<int>(cudaSuccess);
}
