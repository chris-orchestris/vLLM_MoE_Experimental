// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// A C ABI keeps the Python side independent of CUDA's compound attribute
// layout. This shim needs only a C++ compiler and libcudart; it contains no
// device code and therefore does not require nvcc.
extern "C" int vllm_cuda_batch_copy_host_to_device(
    std::uintptr_t* destinations,
    std::uintptr_t* sources,
    std::size_t* sizes,
    std::size_t count,
    std::uintptr_t stream_handle,
    int device) {
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
      reinterpret_cast<const void* const*>(sources),
      sizes,
      count,
      &attributes,
      &attribute_index,
      1,
      reinterpret_cast<cudaStream_t>(stream_handle)));
}
