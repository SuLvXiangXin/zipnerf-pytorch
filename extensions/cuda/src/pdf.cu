#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <curand.h>
#include <curand_kernel.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CUDA_GET_THREAD_ID(tid, Q)                         \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q)                                          \
    return

#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

__device__ const float eps = 1e-8;

template <typename scalar_t>
inline __device__ __host__ scalar_t ceil_div(scalar_t a, scalar_t b)
{
  return (a + b - 1) / b;
}


struct RaySegments {
    RaySegments(const at::Tensor& t) :
        vals(t.defined() ? t.data_ptr<float>() : nullptr),
        n_edges(t.defined() ? t.numel() : 0),
        n_edges_per_ray(t.defined() ? t.size(-1) : 0)
    { }

    float* vals;

    int64_t n_edges;
    int32_t n_edges_per_ray;
};



// Taken from:
// https://github.com/pytorch/pytorch/blob/8f1c3c68d3aba5c8898bfb3144988aab6776d549/aten/src/ATen/native/cuda/Bucketization.cu
template<typename input_t>
__device__ int64_t lower_bound(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    }
    else {
      end = mid;
    }
  }
  return start;
}

template <typename scalar_t>
__device__ int64_t upper_bound(const scalar_t *data_ss, int64_t start, int64_t end, const scalar_t val, const int64_t *data_sort)
{
  // sorter gives relative ordering for ND tensors, so we need to save and add the non-updated start as an offset
  // i.e. the second row of a 3x3 tensors starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end)
  {
    const int64_t mid = start + ((end - start) >> 1);
    const scalar_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val))
    {
      start = mid + 1;
    }
    else
    {
      end = mid;
    }
  }
  return start;
}


__global__ void sample_intervals_kernel(
    const bool rand,
    const RaySegments ray_segments,
    const float *cdfs,
    const bool single_jitter,
    const bool deterministic_center,
    at::PhiloxCudaState philox_args,
    RaySegments samples
    )
{
    // parallelize over outputs
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < samples.n_edges; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id = tid / samples.n_edges_per_ray;
        int64_t n_samples = samples.n_edges_per_ray;
        int64_t sid = tid - ray_id * samples.n_edges_per_ray;

        int64_t base = ray_id * ray_segments.n_edges_per_ray;
        int64_t last = base + ray_segments.n_edges_per_ray - 1;

        float u = 0.0f;
        float bias = 0.5f;

        if (rand) {
            auto seeds = at::cuda::philox::unpack(philox_args);
            curandStatePhilox4_32_10_t state;
            curand_init(std::get<0>(seeds), ray_id, std::get<1>(seeds), &state);
            bias = curand_uniform(&state);

            float u_max = eps + (1.0 - eps) / n_samples;
            float max_jitter = (1.0 - u_max) / (n_samples - 1) - 1e-8;
            u = (1.0 - u_max) * sid / (n_samples - 1) + bias * max_jitter;
        } else {
            float pad = 1.0 / (2 * n_samples);
            float start = deterministic_center ? pad : 0.0;
            float end = deterministic_center ? 1.0 - pad : 1.0;
            u = start + (end - start) * sid / (n_samples - 1);
        }

        // searchsorted with "right" option:
        // i.e. cdfs[p - 1] <= u < cdfs[p]
        int64_t p = upper_bound<float>(cdfs, base, last, u, nullptr);
        int64_t p0 = max(min(p - 1, last), base);
        int64_t p1 = max(min(p, last), base);

        float u_lower = cdfs[p0];
        float u_upper = cdfs[p1];
        float t_lower = ray_segments.vals[p0];
        float t_upper = ray_segments.vals[p1];

        float t;
        if (u_upper - u_lower < 1e-10f) {
            t = (t_lower + t_upper) * 0.5f;
        } else {
            float scaling = (t_upper - t_lower) / (u_upper - u_lower);
            t = (u - u_lower) * scaling + t_lower;
        }
        samples.vals[tid] = t;
    }
}

__global__ void compute_intervels_kernel(
    const RaySegments ray_segments,
    RaySegments samples,
    RaySegments intervals
    )
{
    // parallelize over samples
    for (int64_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < samples.n_edges; tid += blockDim.x * gridDim.x)
    {
        int32_t ray_id = tid / samples.n_edges_per_ray;
        int64_t n_samples = samples.n_edges_per_ray;
        int64_t sid = tid - ray_id * samples.n_edges_per_ray;

        int64_t base = ray_id * ray_segments.n_edges_per_ray;
        int64_t last = base + ray_segments.n_edges_per_ray - 1;

        int64_t base_out = ray_id * intervals.n_edges_per_ray;

        float t_min = ray_segments.vals[base];
        float t_max = ray_segments.vals[last];

        if (sid == 0) {
            float t = samples.vals[tid];
            float t_next = samples.vals[tid + 1]; // FIXME: out of bounds?
            float half_width = (t_next - t) * 0.5f;
            intervals.vals[base_out] = fmaxf(t - half_width, t_min);

        } else {
            float t = samples.vals[tid];
            float t_prev = samples.vals[tid - 1];
            float t_edge = (t + t_prev) * 0.5f;
            int64_t idx = base_out + sid;
            intervals.vals[idx] = t_edge;

            if (sid == n_samples - 1) {
                float half_width = (t - t_prev) * 0.5f;
                intervals.vals[idx + 1] = fminf(t + half_width, t_max);
            }
        }
    }
}


std::vector<torch::Tensor> kernel_sample_intervals(
    const bool rand,
    const torch::Tensor sdist,  // [..., n_edges_per_ray]
    const torch::Tensor cdfs,   // [..., n_edges_per_ray]
    const int64_t num_samples,
    const bool single_jitter,
    const bool deterministic_center,
    torch::Tensor intervals,
    torch::Tensor samples
    )
{
    DEVICE_GUARD(cdfs);
    CHECK_INPUT(cdfs);
    TORCH_CHECK(cdfs.numel() == sdist.numel());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // For jittering
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState rng_engine_inputs;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      rng_engine_inputs = gen->philox_cuda_state(4);
    }

    //int64_t max_threads = 512;
    int64_t max_threads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    int64_t max_blocks = 65535;
    int64_t n_samples = samples.numel();

    dim3 threads = dim3(min(max_threads, n_samples));
    dim3 blocks = dim3(min(max_blocks, ceil_div<int64_t>(n_samples, threads.x)));

    // step 1. compute samples
    sample_intervals_kernel<<<blocks, threads, 0, stream>>>(
        rand,
        RaySegments(sdist),
        cdfs.data_ptr<float>(),
        single_jitter,
        deterministic_center,
        rng_engine_inputs,
        RaySegments(samples));

    // step 2. compute the intervals.
    compute_intervels_kernel<<<blocks, threads, 0, stream>>>(
        RaySegments(sdist),
        RaySegments(samples),
        RaySegments(intervals));  // output intervals

    return {intervals, samples};
}


at::Tensor sample_intervals(
    const bool rand,
    const at::Tensor sdist,
    const at::Tensor cdfs,
    const int64_t num_samples,
    const bool single_jitter)
{
    if (num_samples <= 1)
        return at::Tensor();

    std::vector<at::Tensor> sample_list;
    const bool deterministic_center = true;

    auto data_size = sdist.sizes().vec();
    data_size.back() = num_samples;
    at::Tensor samples = torch::empty(data_size, cdfs.options());
    data_size.back() = num_samples + 1;
    at::Tensor intervals = torch::empty(data_size, cdfs.options());

    sample_list = kernel_sample_intervals(rand, sdist, cdfs, num_samples, single_jitter, deterministic_center, intervals, samples);

    return sample_list[0];
}


at::Tensor sample_intervals(
    const bool rand,
    const at::Tensor sdist,
    const at::Tensor cdfs,
    const int64_t num_samples,
    const bool single_jitter,
    at::Tensor intervals,
    at::Tensor samples)
{
    if (num_samples <= 1)
        return at::Tensor();

    std::vector<at::Tensor> sample_list;
    const bool deterministic_center = true;

    sample_list = kernel_sample_intervals(rand, sdist, cdfs, num_samples, single_jitter, deterministic_center, intervals, samples);

    return sample_list[0];
}

