#ifndef _MATRIX_MULTIPLICATION_H_ENCODE_H
#define _MATRIX_MULTIPLICATION_H_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

void cast_rays_dpcpp(const at::Tensor& tdist, const at::Tensor& origins, const at::Tensor& directions, const at::Tensor& cam_dirs, const at::Tensor& radii,
                     const int seed1, const int seed2, const int seed3, const bool rand, const int n, const int m, const float std_scale, 
                     at::Tensor& means, at::Tensor& stds);
#endif
