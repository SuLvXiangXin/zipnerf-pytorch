#include <torch/extension.h>

#include "gridencoder.h"
#include "pdf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
    m.def("grad_total_variation", &grad_total_variation, "grad_total_variation (CUDA)");
    m.def("sample_intervals", static_cast<at::Tensor (*)(const bool, const at::Tensor, const at::Tensor, const int64_t, const bool)>(&sample_intervals), "sample_intervals (CUDA, with default parameters)");
    m.def("sample_intervals", static_cast<at::Tensor (*)(const bool, const at::Tensor, const at::Tensor, const int64_t, const bool, at::Tensor, at::Tensor)>(&sample_intervals), "sample_intervals (CUDA)");
}
