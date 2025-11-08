#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "ops.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>

#define BLOCK_SIZE 256
inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// If you can get this to work, go ahead. I tried.
// NVCC won't parse templated types in kernel launches
/*
template <template <class> class KernelFn, typename... Args>
void dispatch_and_launch(const at::Tensor& example_tensor, Args... args) {
    const int64_t N = example_tensor.numel();
    const int64_t block = LAUNCH_BLOCK_SIZE;
    const int64_t grid = (N + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAGuard device_guard(example_tensor.device());

    at::ScalarType dtype = example_tensor.scalar_type();
    if (dtype == at::ScalarType::Float) {
        KernelFn<float><<<grid, block, 0, stream>>>(args..., N);
    } else if (dtype == at::ScalarType::Half) {
        KernelFn<__half><<<grid, block, 0, stream>>>(args..., N);
    } else if (dtype == at::ScalarType::BFloat16) {
        KernelFn<__nv_bfloat16><<<grid, block, 0, stream>>>(args..., N);
    } else {
        AT_ERROR("Unsupported dtype: ", dtype);
    }
}
*/

template<typename T>
__global__ void log_coeffs_and_values_kernel(
    T* log_coeffs,
    T* log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = float(gate[idx]);
    float h = float(hidden[idx]);

    log_coeffs[idx] = -softplus_fwd(g);
    float log_z = -softplus_fwd(-g);
    float log_tilde_h;
    if (h >= 0.0f) {
        float relu_h = relu(h);
        log_tilde_h = logf(relu_h + 0.5f);
    } else {
        log_tilde_h = -softplus_fwd(-h);
    }
    log_values[idx] = log_z + log_tilde_h;
}

template<typename T>
__global__ void log_coeffs_and_values_backward_kernel(
    T* grad_gate,
    T* grad_hidden,
    const T* grad_log_coeffs,
    const T* grad_log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = float(gate[idx]);
    float h = float(hidden[idx]);

    float grad_lc = float(grad_log_coeffs[idx]);
    float grad_lv = float(grad_log_values[idx]);
    float grad_g_from_lc = -softplus_bwd(grad_lc, g);
    float grad_g_from_lz = -softplus_bwd(grad_lv, -g);
    float grad_g_total = grad_g_from_lc + grad_lv * (-grad_g_from_lz);
    grad_gate[idx] = T(grad_g_total);
    float log_tilde_h;
    float grad_h_from_lt;
    if (h >= 0.0f) {
        float relu_h = relu(h);
        log_tilde_h = logf(relu_h + 0.5f);
        float inner_grad = 1.0f / (relu_h + 0.5f);
        grad_h_from_lt = relu_backward(h, inner_grad * grad_lv);
    } else {
        log_tilde_h = -softplus_fwd(-h);
        grad_h_from_lt = -softplus_bwd(-grad_lv, -h);
    }
    grad_hidden[idx] = T(grad_h_from_lt);
}

template<typename T>
void launch_log_coeffs_and_values(
    T* log_coeffs,
    T* log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int grid = grid_size(N);
    log_coeffs_and_values_kernel<T><<<grid, BLOCK_SIZE>>>(
        log_coeffs,
        log_values,
        gate,
        hidden,
        N
    );

    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_log_coeffs_and_values_backward(
    T* grad_gate,
    T* grad_hidden,
    const T* grad_log_coeffs,
    const T* grad_log_values,
    const T* gate,
    const T* hidden,
    int N
) {
    int grid = grid_size(N);
    log_coeffs_and_values_backward_kernel<T><<<grid, BLOCK_SIZE>>>(
        grad_gate,
        grad_hidden,
        grad_log_coeffs,
        grad_log_values,
        gate,
        hidden,
        N
    );

    // Optional: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
