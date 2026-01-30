#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/BFloat16.h>

namespace pufferlib {

template<typename T>
__host__ __device__ void puff_advantage_row_cuda(const T* values, const T* rewards, const T* dones,
        const T* importance, T* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - float(dones[t_next]);
        float imp = float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float delta = rho_t*(float(rewards[t_next]) + gamma*float(values[t_next])*nextnonterminal - float(values[t]));
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = T(lastpufferlam);
    }
}

template<typename T>
void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon, torch::ScalarType expected_dtype) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == expected_dtype, "All tensors must have matching dtype");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

template<typename T>
__global__ void puff_advantage_kernel(const T* values, const T* rewards,
        const T* dones, const T* importance, T* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_cuda<T>(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

template<typename T>
void compute_puff_advantage_cuda_impl(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip, torch::ScalarType dtype) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda<T>(values, rewards, dones, importance, advantages, num_steps, horizon, dtype);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<T><<<blocks, threads_per_block>>>(
        values.data_ptr<T>(),
        rewards.data_ptr<T>(),
        dones.data_ptr<T>(),
        importance.data_ptr<T>(),
        advantages.data_ptr<T>(),
        static_cast<float>(gamma),
        static_cast<float>(lambda),
        static_cast<float>(rho_clip),
        static_cast<float>(c_clip),
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    auto dtype = values.dtype();
    if (dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_impl<float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip, torch::kFloat32);
    } else if (dtype == torch::kBFloat16) {
        compute_puff_advantage_cuda_impl<at::BFloat16>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip, torch::kBFloat16);
    } else {
        TORCH_CHECK(false, "Only float32 and bfloat16 supported for advantage computation");
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
