#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>
#include <iostream>

using namespace sycl;

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

static bool select_device = false;

// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
inline at::Half atomicAdd(at::Half *address, at::Half val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, never use it.
  //return atomicAdd(reinterpret_cast<__half*>(address), val);
    return  (at::Half)0.0;
}

bool select_custom_device() {
    std::cout << "Select_custom_device " << std::endl;

    // Figure out how many devices exist
    dpct::device_info prop;
    int dev = 0;
    int n_dev = dpct::dev_mgr::instance().device_count();
    std::cout << "query device count: " << n_dev << std::endl;

    for (int i = 0; i < n_dev; i++) {
	    dpct::dev_mgr::instance().get_device(i).get_device_info(prop);
        std::string name = prop.get_name();
	    bool is_gpu = dpct::dev_mgr::instance().get_device(i).is_gpu();
	    bool is_cpu = dpct::dev_mgr::instance().get_device(i).is_cpu();

    #if INTEL_GPU == 1
        if (name.find("Intel(R)") != std::string::npos) {
            dev = i;
            break;
        }
    #elif NVIDIA_GPU == 1
        if (name.find("NVIDIA") != std::string::npos) {
	        dev = i;
	        break;
        }
    #endif
	}
    dpct::dev_mgr::instance().select_device(dev);
    dpct::dev_mgr::instance().get_device(dev).get_device_info(prop);
    std::cout << "Running on " << prop.get_name() << std::endl;

    return true;
}

template <typename T>
inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T, typename T2>
inline T clamp(const T v, const T2 lo, const T2 hi) {
    return min(max(v, lo), hi);
}

template <typename T>
inline T smoothstep(T val) {
    return val*val*(3.0f - 2.0f * val);
}

template <typename T>
inline T smoothstep_derivative(T val) {
    return 6*val*(1.0f - val);
}


template <uint32_t D>
uint32_t fast_hash(const uint32_t pos_grid[D]) {

    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}


template <uint32_t D, uint32_t C>
uint32_t get_grid_index(const uint32_t gridtype, const bool align_corners, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
    uint32_t stride = 1;
    uint32_t index = 0;

    //#pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        stride *= align_corners ? resolution: (resolution + 1);
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

    return (index % hashmap_size) * C + ch;
}


template <typename scalar_t, uint32_t D, uint32_t C>
void kernel_grid(
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    scalar_t * __restrict__ dy_dx,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp,
    const sycl::nd_item<3> &item_ct1) {

    const uint32_t b = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

    if (b >= B) return;

    const uint32_t level = item_ct1.get_group(1);

    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0;
        }
        if (dy_dx) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0;
                }       
            }
        }

        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = sycl::exp2(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)sycl::ceil((float)scale) + 1;

    // calculate coordinate (always use float for precision!)
    float pos[D];
    float pos_deriv[D]; 
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = sycl::floor(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos_deriv[d] = smoothstep_derivative(pos[d]);
            pos[d] = smoothstep(pos[d]);
        } else {
            pos_deriv[d] = 1.0f; // linear deriv is default to 1
        }

    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // interpolate
    scalar_t results[C] = {0}; // temp results in register

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f\n", b, level, idx, index, w, grid[index]);
    }    

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch];
    }

    // prepare dy_dx
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (dy_dx) {

        dy_dx += b * D * L * C + level * D * C; // B L D C

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            scalar_t results_grad[C] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid_local);

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_deriv[gd];
                }

            }

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }

}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const float * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const uint32_t gridtype,
    const bool align_corners,
    const uint32_t interp,
    local_accessor<int> indices,
    local_accessor<float> wgrid,
    local_accessor<bool> issue,
    const sycl::nd_item<3> &item_ct1) {

    const uint32_t b = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2)) *
                       N_C / C;
    uint32_t id = item_ct1.get_local_linear_id();
    issue[id] = false;
    if (b >= B) return;

    const uint32_t level = item_ct1.get_group(1);
    const uint32_t ch = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2)) *
                            N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = sycl::exp2(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)sycl::ceil((float)scale) + 1;

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = sycl::floor(pos[d]);
        pos[d] -= (float)pos_grid[d];
        // smoothstep instead of linear
        if (interp == 1) {
            pos[d] = smoothstep(pos[d]);
        }
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(gridtype, align_corners, ch, hashmap_size, resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                sycl::half2 v = {(sycl::half)(w * grad_cur[c]),
                                 (sycl::half)(w * grad_cur[c + 1])};

                //DPCT1007:8: Migration of half version of atomicAdd is not
                //supported.

                //atomicAdd((sycl::half2 *)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                wgrid[id] = w * grad_cur[c];
                indices[id] = index;
                issue[id] = true;

                #if INTEL_GPU == 1
                for (uint32_t exp = 0; exp < 2; exp++) {
                    uint32_t ngbr = id + (1 << exp);	//neighbour id

                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    if ((id % (1 << (exp + 1))) == 0 && indices[id] == indices[ngbr] && issue[id] && issue[ngbr]) {
                        wgrid[id] += wgrid[ngbr];
                        issue[ngbr] = false;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }
                #endif

                if (issue[id]) {
                    dpct::atomic_fetch_add(&grad_grid[index + c], wgrid[id]);
                }
            }
        }
     }

}


template <typename scalar_t, uint32_t D, uint32_t C>
void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,  
    scalar_t * __restrict__ grad_inputs, 
    uint32_t B, uint32_t L,
    const sycl::nd_item<3> &item_ct1) {
    const uint32_t t = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    dy_dx += b * L * D * C;

    scalar_t result = 0;
    
    //# pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}

template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const float *inputs, const scalar_t *embeddings,
                         const int *offsets, scalar_t *outputs,
                         const uint32_t B, const uint32_t C, const uint32_t L,
                         const float S, const uint32_t H, scalar_t *dy_dx,
                         const uint32_t gridtype, const bool align_corners,
                         const uint32_t interp) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    //static constexpr uint32_t N_THREAD = 512;
    static constexpr uint32_t N_THREAD = 512;

    const sycl::range<3> blocks_hashgrid = {1, L, div_round_up(B, N_THREAD)};
    const sycl::range<3> global_hashgrid = {1, L, div_round_up(B, N_THREAD)};
    const sycl::range<3> local_hashgrid = {1, 1, N_THREAD};

    switch (C) {
        case 1:
            q_ct1.submit([&](sycl::handler& h) {
                sycl::stream out(1024, 256, h);
                h.parallel_for(
                    sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD), local_hashgrid),
                    [=](sycl::nd_item<3> item_ct1) {
                    kernel_grid<scalar_t, D, 1>(
                        inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx,
                        gridtype, align_corners, interp, item_ct1);
                });
            });
            break;
        case 2: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grid<scalar_t, D, 2>(
                    inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx,
                    gridtype, align_corners, interp, item_ct1);
            });
            break;
        case 4:
            q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grid<scalar_t, D, 4>(
                    inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx,
                    gridtype, align_corners, interp, item_ct1);
            });
            break;
        case 8: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grid<scalar_t, D, 8>(
                    inputs, embeddings, offsets, outputs, B, L, S, H, dy_dx,
                    gridtype, align_corners, interp, item_ct1);
            });
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }

}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [B, L * D * C]

template <typename scalar_t>
void grid_encode_forward_dpcpp(const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {

    //printf("B batch size: {%d}, D coord dim: {%d}, C embedding dim for each level: {%d}, L levels: {%d}, S resolution multiplier at each level: {%f}, H base resolution: {%d}\n", B, D, C, L, S, H);

    switch (D) {
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 4: kernel_grid_wrapper<scalar_t, 4>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        case 5: kernel_grid_wrapper<scalar_t, 5>(inputs, embeddings, offsets, outputs, B, C, L, S, H, dy_dx, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(
    const scalar_t *grad, const float *inputs, const scalar_t *embeddings,
    const int *offsets, scalar_t *grad_embeddings, const uint32_t B,
    const uint32_t C, const uint32_t L, const float S, const uint32_t H,
    scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype,
    const bool align_corners, const uint32_t interp) {

    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    static constexpr uint32_t N_THREAD = 512;

    const uint32_t N_C = std::min(2u, C); // n_features_per_thread
    const sycl::range<3> blocks_hashgrid = {
        1, L, div_round_up(B * C / N_C, N_THREAD)};
    switch (C) {
        case 1:
        q_ct1.submit([&](sycl::handler& h) {
            local_accessor<int> indices{N_THREAD, h};
            local_accessor<float> wgrid{N_THREAD, h};
            local_accessor<bool> issue{N_THREAD, h};

            q_ct1.parallel_for(
                sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                                  sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1)  {
                    kernel_grid_backward<scalar_t, D, 1, 1>(
                        grad, inputs, embeddings, offsets, grad_embeddings, B, L, S,
                        H, gridtype, align_corners, interp, indices, wgrid, issue, item_ct1);
                });
            });
            if (dy_dx) q_ct1.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, div_round_up(B * D, N_THREAD)) *
                        sycl::range<3>(1, 1, N_THREAD),
                    sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_input_backward<scalar_t, D, 1>(
                        grad, dy_dx, grad_inputs, B, L, item_ct1);
                });
            break;
        case 2:
        q_ct1.submit([&](sycl::handler& h) {
            local_accessor<int> indices{N_THREAD, h};
            local_accessor<float> wgrid{N_THREAD, h};
            local_accessor<bool> issue{N_THREAD, h};

            q_ct1.parallel_for(
                sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                                  sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_grid_backward<scalar_t, D, 2, 2>(
                        grad, inputs, embeddings, offsets, grad_embeddings, B, L, S,
                        H, gridtype, align_corners, interp, indices, wgrid, issue, item_ct1);
                });
            });
            if (dy_dx) q_ct1.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, div_round_up(B * D, N_THREAD)) *
                        sycl::range<3>(1, 1, N_THREAD),
                    sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_input_backward<scalar_t, D, 2>(
                        grad, dy_dx, grad_inputs, B, L, item_ct1);
                });
            break;
        case 4:
        q_ct1.submit([&](sycl::handler& h) {
            local_accessor<int> indices{N_THREAD, h};
            local_accessor<float> wgrid{N_THREAD, h};
            local_accessor<bool> issue{N_THREAD, h};

            q_ct1.parallel_for(
                sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                                  sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_grid_backward<scalar_t, D, 4, 2>(
                        grad, inputs, embeddings, offsets, grad_embeddings, B, L, S,
                        H, gridtype, align_corners, interp, indices, wgrid, issue, item_ct1);
                });
            });
            if (dy_dx) q_ct1.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, div_round_up(B * D, N_THREAD)) *
                        sycl::range<3>(1, 1, N_THREAD),
                    sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_input_backward<scalar_t, D, 4>(
                        grad, dy_dx, grad_inputs, B, L, item_ct1);
                });
            break;
        case 8:
        q_ct1.submit([&](sycl::handler& h) {
            local_accessor<int> indices{N_THREAD, h};
            local_accessor<float> wgrid{N_THREAD, h};
            local_accessor<bool> issue{N_THREAD, h};

            q_ct1.parallel_for(
                sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                                  sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1)  {
                    kernel_grid_backward<scalar_t, D, 8, 2>(
                        grad, inputs, embeddings, offsets, grad_embeddings, B, L, S,
                        H, gridtype, align_corners, interp, indices, wgrid, issue, item_ct1);
                });
            });
            if (dy_dx) q_ct1.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, div_round_up(B * D, N_THREAD)) *
                        sycl::range<3>(1, 1, N_THREAD),
                    sycl::range<3>(1, 1, N_THREAD)),
                [=](sycl::nd_item<3> item_ct1) {
                    kernel_input_backward<scalar_t, D, 8>(
                        grad, dy_dx, grad_inputs, B, L, item_ct1);
                });
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution

template <typename scalar_t>
void grid_encode_backward_dpcpp(const scalar_t *grad, const float *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, scalar_t *dy_dx, scalar_t *grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 4: kernel_grid_backward_wrapper<scalar_t, 4>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        case 5: kernel_grid_backward_wrapper<scalar_t, 5>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interp); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {

    if (!select_device) {
        printf("select device ...");
        select_device = select_custom_device();
    }

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    // CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    // CHECK_IS_FLOATING(dy_dx);

    //printf("B batch size: {%d}, D coord dim: {%d}, C embedding dim for each level: {%d}, L levels: {%d}, S resolution multiplier at each level: {%f}, H base resolution: {%d}\n", B, D, C, L, S, H);

    if (embeddings.scalar_type() == at::kFloat) {
        grid_encode_forward_dpcpp(inputs.data_ptr<float>(), embeddings.data_ptr<float>(), offsets.data_ptr<int>(), outputs.data_ptr<float>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<float>() : nullptr, gridtype, align_corners, interp);
    } else if (embeddings.scalar_type() == at::kHalf) {
        printf("!!!!!!!!!!!!!!!! dpct::atomic_fetch_add does NOT support half float \n");
        // sycl/atomic_ref.hpp:116:3: error: static assertion failed due to requirement 'detail::IsValidAtomicRefType<c10::Half>::value': Invalid atomic type.
        // Valid types are int, unsigned int, long, unsigned long, long long, unsigned long long,
        // float, double and pointer types

        //grid_encode_forward_dpcpp(inputs.data_ptr<float>(), embeddings.data_ptr<c10::Half>(), offsets.data_ptr<int>(), outputs.data_ptr<c10::Half>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<c10::Half>() : nullptr, gridtype, align_corners, interp);
    }
}

void grid_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const at::optional<at::Tensor> dy_dx, at::optional<at::Tensor> grad_inputs, const uint32_t gridtype, const bool align_corners, const uint32_t interp) {
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    // CHECK_CONTIGUOUS(dy_dx);
    // CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    // CHECK_IS_FLOATING(dy_dx);
    // CHECK_IS_FLOATING(grad_inputs);

    if (grad.scalar_type() == at::kFloat) {
        grid_encode_backward_dpcpp(grad.data_ptr<float>(), inputs.data_ptr<float>(), embeddings.data_ptr<float>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<float>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<float>() : nullptr, grad_inputs.has_value() ? grad_inputs.value().data_ptr<float>() : nullptr, gridtype, align_corners, interp);
    } else if (grad.scalar_type() == at::kHalf) {
        printf("!!!!!!!!!!!!!!!! dpct::atomic_fetch_add does NOT support half float \n");
        // sycl/atomic_ref.hpp:116:3: error: static assertion failed due to requirement 'detail::IsValidAtomicRefType<c10::Half>::value': Invalid atomic type.
        // Valid types are int, unsigned int, long, unsigned long, long long, unsigned long long,
        // float, double and pointer types
        //grid_encode_backward_dpcpp(grad.data_ptr<c10::Half>(), inputs.data_ptr<float>(), embeddings.data_ptr<c10::Half>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<c10::Half>(), B, D, C, L, S, H, dy_dx.has_value() ? dy_dx.value().data_ptr<c10::Half>() : nullptr, grad_inputs.has_value() ? grad_inputs.value().data_ptr<c10::Half>() : nullptr, gridtype, align_corners, interp);
    }

}


template <typename scalar_t, uint32_t D, uint32_t C>
void kernel_grad_tv(
    const scalar_t * __restrict__ inputs,
    const scalar_t * __restrict__ grid, 
    scalar_t * __restrict__ grad, 
    const int * __restrict__ offsets, 
    const float weight,
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const uint32_t gridtype,
    const bool align_corners,
    const sycl::nd_item<3> &item_ct1) {

    const uint32_t b = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

    if (b >= B) return;

    const uint32_t level = item_ct1.get_group(1);

    // locate
    inputs += b * D;
    grid += (uint32_t)offsets[level] * C;
    grad += (uint32_t)offsets[level] * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }

    // if input out of bound, do nothing
    if (flag_oob) return;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = sycl::exp2(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)sycl::ceil((float)scale) + 1;

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D]; // [0, resolution]

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = inputs[d] * scale + (align_corners ? 0.0f : 0.5f);
        pos_grid[d] = sycl::floor(pos[d]);
        // pos[d] -= (float)pos_grid[d]; // not used
    }

    //printf("[b=%d, l=%d] pos=(%f, %f)+(%d, %d)\n", b, level, pos[0], pos[1], pos_grid[0], pos_grid[1]);

    // total variation on pos_grid
    scalar_t results[C] = {0}; // temp results in register
    scalar_t idelta[C] = {0};

    uint32_t index = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

    scalar_t w = weight / (2 * D);

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {

        uint32_t cur_d = pos_grid[d];
        scalar_t grad_val;

        // right side
        if (cur_d < resolution) {
            pos_grid[d] = cur_d + 1;
            uint32_t index_right = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_right + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_right + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // left side
        if (cur_d > 0) {
            pos_grid[d] = cur_d - 1;
            uint32_t index_left = get_grid_index<D, C>(gridtype, align_corners, 0, hashmap_size, resolution, pos_grid);

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                // results[ch] += w * clamp(grid[index + ch] - grid[index_left + ch], -1.0f, 1.0f);
                grad_val = (grid[index + ch] - grid[index_left + ch]);
                results[ch] += grad_val;
                idelta[ch] += grad_val * grad_val;
            }
        }

        // reset
        pos_grid[d] = cur_d;
    }

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        // index may collide, so use atomic!

        //DPCT1064:9: Migrated rsqrtf call is used in a macro definition and is
        //not valid for all macro uses. Adjust the code.

        dpct::atomic_fetch_add(&grad[index + ch],
                               w * results[ch] * sycl::rsqrt((float)(idelta[ch] + 1e-9f)));
    }
}

template <typename scalar_t, uint32_t D>
void kernel_grad_tv_wrapper(const scalar_t *inputs, const scalar_t *embeddings,
                            scalar_t *grad, const int *offsets,
                            const float weight, const uint32_t B,
                            const uint32_t C, const uint32_t L, const float S,
                            const uint32_t H, const uint32_t gridtype,
                            const bool align_corners) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    //static constexpr uint32_t N_THREAD = 512;
    static constexpr uint32_t N_THREAD = 512;

    const sycl::range<3> blocks_hashgrid = {1, L, div_round_up(B, N_THREAD)};
    switch (C) {
        /*
        DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        case 1: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grad_tv<scalar_t, D, 1>(
                    inputs, embeddings, grad, offsets, weight, B, L, S, H,
                    gridtype, align_corners, item_ct1);
            });
            break;
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        case 2: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grad_tv<scalar_t, D, 2>(
                    inputs, embeddings, grad, offsets, weight, B, L, S, H,
                    gridtype, align_corners, item_ct1);
            });
            break;
        /*
        DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        case 4: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grad_tv<scalar_t, D, 4>(
                    inputs, embeddings, grad, offsets, weight, B, L, S, H,
                    gridtype, align_corners, item_ct1);
            });
            break;
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        case 8: q_ct1.parallel_for(
            sycl::nd_range<3>(blocks_hashgrid * sycl::range<3>(1, 1, N_THREAD),
                              sycl::range<3>(1, 1, N_THREAD)),
            [=](sycl::nd_item<3> item_ct1) {
                kernel_grad_tv<scalar_t, D, 8>(
                    inputs, embeddings, grad, offsets, weight, B, L, S, H,
                    gridtype, align_corners, item_ct1);
            });
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


template <typename scalar_t>
void grad_total_variation_dpcpp(const scalar_t *inputs, const scalar_t *embeddings, scalar_t *grad, const int *offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {
    switch (D) {
        case 2: kernel_grad_tv_wrapper<scalar_t, 2>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 3: kernel_grad_tv_wrapper<scalar_t, 3>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 4: kernel_grad_tv_wrapper<scalar_t, 4>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        case 5: kernel_grad_tv_wrapper<scalar_t, 5>(inputs, embeddings, grad, offsets, weight, B, C, L, S, H, gridtype, align_corners); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }   
}

void grad_total_variation(const at::Tensor inputs, const at::Tensor embeddings, at::Tensor grad, const at::Tensor offsets, const float weight, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const uint32_t gridtype, const bool align_corners) {

    if (grad.scalar_type() == at::kFloat) {
        grad_total_variation_dpcpp(inputs.data_ptr<float>(), embeddings.data_ptr<float>(), grad.data_ptr<float>(), offsets.data_ptr<int>(), weight, B, D, C, L, S, H, gridtype, align_corners);
    } else if (grad.scalar_type() == at::kHalf) {
        printf("!!!!!!!!!!!!!!!!!! dpct::atomic_fetch_add does NOT support half float \n");
        // sycl/atomic_ref.hpp:116:3: error: static assertion failed due to requirement 'detail::IsValidAtomicRefType<c10::Half>::value': Invalid atomic type.
        // Valid types are int, unsigned int, long, unsigned long, long long, unsigned long long,
        // float, double and pointer types
        //grad_total_variation_dpcpp(inputs.data_ptr<c10::Half>(), embeddings.data_ptr<c10::Half>(), grad.data_ptr<c10::Half>(), offsets.data_ptr<int>(), weight, B, D, C, L, S, H, gridtype, align_corners);
    }
}
