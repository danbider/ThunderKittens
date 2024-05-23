#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>


extern void hedgehog_based_tk(torch::Tensor q, torch::Tensor k, torch::Tensor qf, torch::Tensor kf, torch::Tensor v, torch::Tensor o, torch::Tensor kv);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("hedgehog_based_tk", hedgehog_based_tk);
}
 
