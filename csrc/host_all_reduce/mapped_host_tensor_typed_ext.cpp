#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

torch::Tensor mapped_cuda_tensor_typed(
    std::uint64_t pointer,
    const std::vector<std::int64_t>& sizes,
    int device_index,
    const std::string& dtype_name) {
  torch::Dtype dtype;
  if (dtype_name == "bfloat16") {
    dtype = torch::kBFloat16;
  } else if (dtype_name == "int32") {
    dtype = torch::kInt32;
  } else if (dtype_name == "uint8") {
    dtype = torch::kUInt8;
  } else if (dtype_name == "float32") {
    dtype = torch::kFloat32;
  } else {
    throw std::invalid_argument("unsupported mapped tensor dtype " + dtype_name);
  }
  const auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .device(torch::Device(torch::kCUDA, device_index));
  return torch::from_blob(
      reinterpret_cast<void*>(pointer), sizes, [](void*) {}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mapped_cuda_tensor_typed", &mapped_cuda_tensor_typed);
}
