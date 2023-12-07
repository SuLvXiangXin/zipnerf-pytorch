#ifndef _PDF_H
#define _PDF_H

#include <stdint.h>
#include <torch/torch.h>

at::Tensor sample_intervals(
    const bool rand,
    const at::Tensor sdist,
    const at::Tensor cdfs,
    const int64_t num_samples,
    const bool single_jitter);


at::Tensor sample_intervals(
    const bool rand,
    const at::Tensor sdist,
    const at::Tensor cdfs,
    const int64_t num_samples,
    const bool single_jitter,
    at::Tensor intervals,
    at::Tensor samples);

#endif
