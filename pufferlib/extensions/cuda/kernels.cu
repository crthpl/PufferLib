#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "ops.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>

#define SEQ_SIZE 1
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
__global__ void mingru_gate_inference_kernel(
    T* out,
    const T* gate_in,
    const T* hidden_in,
    const T* state_in,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float gate = float(gate_in[idx]);
    float hidden = float(hidden_in[idx]);
    float state = float(state_in[idx]);
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = tilde_relu_fwd(hidden);
    float out_val = lerp(state, hidden_tilde, gate_sigmoid);
    out[idx] = T(out_val);
}

template<typename T>
void launch_mingru_gate_inference(
    T* out,
    const T* gate_in,
    const T* hidden_in,
    const T* state_in,
    int N
) {
    int grid = grid_size(N);
    mingru_gate_inference_kernel<T><<<grid, BLOCK_SIZE>>>(
        out,
        gate_in,
        hidden_in,
        state_in,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


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

template<typename T>
__global__ void fused_scan_forward_kernel(
    const T* __restrict__ log_coeffs,   // (B, T, H)
    const T* __restrict__ log_values,   // (B, T+1, H)
    T* __restrict__ out,                // (B, T, H)
    int T_seq,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    // TODO: Preallocate from torch
    __shared__ float coeffs[1024];
    __shared__ float values[1025];
    float a_star[1024];
    float z[1025];
    float s[1025];

    // Load log_coeffs[t] -> coeffs[t]
    for (int t = 0; t < T_seq; t++) {
        coeffs[t] = float(log_coeffs[b * T_seq * H + t * H + h]);
    }

    for (int t = 0; t < T_seq; t++) {
        values[t] = float(log_values[b * (T_seq + 1) * H + t * H + h]);
    }

    cumsum_forward(coeffs, a_star, T_seq);
    for (int t = 0; t < T_seq; t++) {
        z[t] = values[t] - a_star[t];
    }

    z[T_seq] = values[T_seq] - a_star[T_seq - 1];  // last value
    logcumsumexp_forward(z, s, T_seq + 1);
    for (int t = 0; t < T_seq; t++) {
        float log_h = a_star[t] + s[t];
        out[b * T_seq * H + t * H + h] = T(exp_safe(log_h));
    }
}

template<typename T>
__global__ void fused_scan_backward_kernel(
    const T* __restrict__ grad_out,
    const T* __restrict__ log_coeffs,
    const T* __restrict__ log_values,
    const T* __restrict__ out,
    T* __restrict__ grad_log_coeffs,
    T* __restrict__ grad_log_values,
    int T_seq,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    float coeffs[1024];
    float values[1025];
    float a_star[1024];
    float z[1025];
    float s[1025];
    float grad_lc[1024];
    float grad_lv[1025];

    // Load inputs
    for (int t = 0; t < T_seq; t++) {
        coeffs[t] = float(log_coeffs[b * T_seq * H + t * H + h]);
    }
    for (int t = 0; t < T_seq; t++) {
        values[t] = float(log_values[b * (T_seq + 1) * H + t * H + h]);
    }

    // Recompute a_star, z, s
    cumsum_forward(coeffs, a_star, T_seq);
    for (int t = 0; t < T_seq; t++) {
        z[t] = values[t] - a_star[t];
    }
    z[T_seq] = values[T_seq] - a_star[T_seq - 1];
    logcumsumexp_forward(z, s, T_seq + 1);

    // grad_log_h[t] = grad_out[t] * out[t]
    float grad_log_h[1024];
    for (int t = 0; t < T_seq; t++) {
        grad_log_h[t] = float(grad_out[b * T_seq * H + t * H + h]) *
                        float(out[b * T_seq * H + t * H + h]);
    }

    // grad_z = logcumsumexp_backward(grad_log_h padded)
    float grad_s[1025];
    for (int t = 0; t < T_seq; t++) grad_s[t] = grad_log_h[t];
    grad_s[T_seq] = 0.0f;

    logcumsumexp_backward(z, s, grad_s, grad_lv, T_seq + 1);

    // grad_a_star[t] = grad_log_h[t] (t < T) - grad_lv[t] (t <= T)
    float grad_a_star[1025] = {0};
    for (int t = 0; t < T_seq; t++) grad_a_star[t] += grad_log_h[t];
    for (int t = 0; t < T_seq; t++) {
        if (t < T_seq) grad_a_star[t] -= grad_lv[t];
        else           grad_a_star[T_seq - 1] -= grad_lv[t];
    }

    // grad_log_coeffs[t] = cumsum_backward(grad_a_star[t+1])
    for (int t = 0; t < T_seq; t++) {
        grad_lc[t] = grad_a_star[t + 1];
    }
    cumsum_backward(grad_lc, grad_lc, T_seq);

    // Write back
    for (int t = 0; t < T_seq; t++) {
        grad_log_coeffs[b * T_seq * H + t * H + h] = T(grad_lc[t]);
    }
    for (int t = 0; t < T_seq; t++) {
        grad_log_values[b * (T_seq + 1) * H + t * H + h] = T(grad_lv[t]);
    }
}

template<typename T>
void launch_fused_scan_forward(
    T* out,
    const T* log_coeffs,
    const T* log_values,
    int T_seq,
    int H,
    int B
) {
    int total = B * H;
    int grid = grid_size(total);

    fused_scan_forward_kernel<T><<<grid, BLOCK_SIZE>>>(
        log_coeffs,
        log_values,
        out,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in forward: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
void launch_fused_scan_backward(
    T* grad_log_coeffs,
    T* grad_log_values,
    const T* grad_out,
    const T* log_coeffs,
    const T* log_values,
    const T* out,
    int T_seq,
    int H,
    int B
) {
    int total = B * H;
    int grid = grid_size(total);

    fused_scan_backward_kernel<T><<<grid, SEQ_SIZE>>>(
        grad_out,
        log_coeffs,
        log_values,
        out,
        grad_log_coeffs,
        grad_log_values,
        T_seq,
        H,
        B
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error in backward: %s\n", cudaGetErrorString(err));
    }
}

