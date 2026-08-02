#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor mapped_cuda_tensor(
    std::uint64_t pointer,
    const std::vector<std::int64_t>& sizes,
    int device_index) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(torch::Device(torch::kCUDA, device_index));
  return torch::from_blob(
      reinterpret_cast<void*>(pointer), sizes, [](void*) {}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mapped_cuda_tensor", &mapped_cuda_tensor);
}
