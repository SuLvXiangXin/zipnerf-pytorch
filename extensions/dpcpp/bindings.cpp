#include <torch/extension.h>

#include "include/gridencoder.h"
#include "include/rays.h"

const char* module_name = "_dpcpp_backend";

PYBIND11_MODULE(module_name, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (DPCPP)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (DPCPP)");
    m.def("grad_total_variation", &grad_total_variation, "grad_total_variation (DPCPP)");
    m.def("cast_rays_dpcpp", &cast_rays_dpcpp, "cast_rays_dpcpp (DPCPP)");
    m.def("get_gpu_vendor", &get_gpu_vendor, "gpu_vendor (DPCPP)");
}

PyMODINIT_FUNC PyInit__dpcpp_backend(void) {
    pybind11::module m("_dpcpp_backend");  // Declare and initialize the module object
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (DPCPP)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (DPCPP)");
    m.def("grad_total_variation", &grad_total_variation, "grad_total_variation (DPCPP)");
    m.def("cast_rays_dpcpp", &cast_rays_dpcpp, "cast_rays_dpcpp (DPCPP)");
    m.def("get_gpu_vendor", &get_gpu_vendor, "gpu_vendor (DPCPP)");

    return m.ptr();  // Return the module object pointer
}

