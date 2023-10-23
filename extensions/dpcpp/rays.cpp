// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <oneapi/dpl/random>
#include <dpct/dpct.hpp>

#include <ATen/ATen.h>

#include <random>
#include <algorithm>
#include <iostream>
using namespace sycl;

inline float random(float f) {
    //float fractional = sycl::modf(sycl::sin(f) * 43758.5453123f, &integral);
    float number = sycl::sin(f) * 43758.5453123f;
    float fractional = number - ((int)number);

    return (float)fractional;
}

template <typename scalar_t>
void kernel_cast_rays(
    const scalar_t * __restrict__ tdist,
    const scalar_t * __restrict__ origins,
    const scalar_t * __restrict__ directions,
    const scalar_t * __restrict__ cam_dirs,
    const scalar_t * __restrict__ radii,
    const uint32_t seed1,
    const uint32_t seed2,
    const uint32_t seed3,
    const bool rand,
    const int n,
    const int m,
    const float std_scale,
    local_accessor<float> rands_local,
    scalar_t * __restrict__ means,
    scalar_t * __restrict__ stds,
    const sycl::nd_item<3> &item_ct1) {
    constexpr scalar_t pi = 3.141592653589793;
    constexpr scalar_t sqrt2 = 1.4142135623730951;
    constexpr scalar_t sqrt7 = 2.6457513110645907;
    constexpr scalar_t deg_init[6] = {0, 2 * pi / 3, 4 * pi / 3, pi, 5 * pi / 3, pi / 3};

    const uint32_t segment_count = item_ct1.get_local_range(1);
    const uint32_t sample = item_ct1.get_group(0);
    const uint32_t segment = item_ct1.get_local_id(1);
    const uint32_t angle = item_ct1.get_local_id(2);

    const uint32_t segment_idx = sample * item_ct1.get_local_range(1) + segment;
    const uint32_t angle_idx = segment_idx * 6 + angle;
    const uint32_t sample_mul3 = sample * 3;

    if (angle > 5) {
        return;
    }

    if (angle == 0) {
        rands_local[segment] = random(seed1 * segment_idx);
        rands_local[segment + segment_count] = random(seed2 * segment_idx);
    
        if (segment < 3) {
            rands_local[(segment_count << 1) + segment] = random(seed3 * angle_idx);
        }
    }

    tdist += sample * (item_ct1.get_local_range(1) + 1) + segment;

    scalar_t t0 = tdist[0];
    scalar_t t1 = tdist[1];
    scalar_t t_m = (t0 + t1) / 2;
    scalar_t t_d = (t1 - t0) / 2;
    scalar_t t0_2 = t0 * t0;
    scalar_t t1_2 = t1 * t1;
    scalar_t t_d2 = t_d * t_d;
    scalar_t t_m2 = t_m * t_m;

    scalar_t t = t0 + t_d / (t_d2 + 3.f * t_m2) * (t1_2 + 2.f * t_m2 + 3.f / sqrt7 * (0.4f * angle - 1.f) * sycl::sqrt((t_d2 - t_m2) * (t_d2 - t_m2) + 4.f * t_m2 * t_m2));
    scalar_t deg = deg_init[angle];

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (rand) {
        //rand_vec1 += sample * item_ct1.get_local_range(1) + segment;
        deg = deg + 2.f * pi * rands_local[segment];

        if (rands_local[segment + item_ct1.get_local_range(1)] > 0.5f)
        {
            deg = pi * 5.f / 3.f - deg;
        }
    }

    radii += sample;
    scalar_t mean_basis = radii[0] * t / sqrt2;
    scalar_t mean1 = mean_basis * sycl::cos(deg);
    scalar_t mean2 = mean_basis * sycl::sin(deg);

    stds += angle_idx;
    stds[0] = std_scale * radii[0] * t / sqrt2;

    cam_dirs += sample_mul3;
    //rand_vec2 += sample_mul3;
    float3 cam_dir{cam_dirs[0], cam_dirs[1], cam_dirs[2]};

    float3 rand_dir{rands_local[segment_count << 1], rands_local[(segment_count << 1) + 1], rands_local[(segment_count << 1) + 2]};
    float3 ortho1 = normalize(cross(cam_dir, rand_dir));
    float3 ortho2 = normalize(cross(cam_dir, ortho1));

    directions += sample_mul3;
    scalar_t m1 = mean1 * ortho1[0] + mean2 * ortho2[0] + t * directions[0];
    scalar_t m2 = mean1 * ortho1[1] + mean2 * ortho2[1] + t * directions[1];
    scalar_t m3 = mean1 * ortho1[2] + mean2 * ortho2[2] + t * directions[2];

    origins += sample_mul3;
    m1 += origins[0];
    m2 += origins[1];
    m3 += origins[2];

    means += angle_idx * 3;
    means[0] = m1;
    means[1] = m2;
    means[2] = m3;
}

template <typename scalar_t>
void sycl_cast_rays(const scalar_t *tdist, const scalar_t *origins, const scalar_t *directions, const scalar_t *cam_dirs, const scalar_t *radii,
                    const int seed1, const int seed2, const int seed3, const bool rand, const int n, const int m, const float std_scale, 
                    scalar_t *means, scalar_t *stds, const int sample_num, const int segment_num)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> distr;

    q_ct1.submit([&](sycl::handler& h) {
        local_accessor<float> rands_local{segment_num * 2 + 8, h};

        q_ct1.parallel_for(
            sycl::nd_range<3>(sycl::range<3>((uint32_t)sample_num, (uint32_t)segment_num, 8),
                              sycl::range<3>(1, (uint32_t)segment_num, 8)),
            [=](sycl::nd_item<3> item_ct1)  {
                kernel_cast_rays<scalar_t>(tdist, origins, directions, cam_dirs, radii, seed1, seed2, seed3, rand, n, m, std_scale, rands_local, means, stds, item_ct1);
        });
    });

}

void cast_rays_dpcpp(const at::Tensor& tdist, const at::Tensor& origins, const at::Tensor& directions, const at::Tensor& cam_dirs, const at::Tensor& radii,
                     const int seed1, const int seed2, const int seed3, const bool rand, const int n, const int m, const float std_scale, 
                     at::Tensor& means, at::Tensor& stds)
{
    const int dim0 = tdist.size(0);
    const int last_dim = tdist.size(tdist.dim() - 1);
    for (int i = 1; i < tdist.dim() - 1; i++)
    {
        assert(tdist.size(i) == 1);
    }

    if (tdist.scalar_type() == at::kFloat) {
        sycl_cast_rays<float>(tdist.data_ptr<float>(), origins.data_ptr<float>(), directions.data_ptr<float>(), cam_dirs.data_ptr<float>(), 
                              radii.data_ptr<float>(), seed1, seed2, seed3, rand, n, m, std_scale, means.data_ptr<float>(), stds.data_ptr<float>(), dim0, last_dim - 1);
    }
}
