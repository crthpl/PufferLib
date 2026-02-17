#ifndef PUFFERLIB_MODULES_CU
#define PUFFERLIB_MODULES_CU

#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../modules.h"
#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>

using std::tuple;
using std::vector;
namespace nn = torch::nn;
typedef torch::Tensor Tensor;
using tensor_list = torch::autograd::tensor_list;
using AutogradCtx = torch::autograd::AutogradContext;

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

// cuBLAS matmul: out(M,N) = a(M,K) @ b(N,K)^T, all row-major PufTensors
// Uses bf16 inputs with f32 compute.
void puf_mm(PufTensor& a, PufTensor& b, PufTensor& out) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[0];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    // Row-major C(M,N) = A(M,K) @ B(N,K)^T
    // cuBLAS col-major: C^T(N,M) = B @ A^T
    // transa=T on B(N,K), transb=N on A(M,K)
    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data, CUDA_R_16BF, K,   // weight (N,K) row-major
        a.data, CUDA_R_16BF, K,   // input  (M,K) row-major
        &beta,
        out.data, CUDA_R_16BF, N, // output (M,N) row-major
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// cuBLAS matmul: out(M,N) = a(K,M)^T @ b(K,N), all row-major PufTensors
void puf_mm_tn(PufTensor& a, PufTensor& b, PufTensor& out) {
    int K = a.shape[0];
    int M = a.shape[1];
    int N = b.shape[1];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    // Row-major C(M,N) = A(K,M)^T @ B(K,N)
    // cuBLAS col-major: C^T(N,M) = op(A_cub)(N,K) @ op(B_cub)(K,M)
    // A_cub = b_ptr: row(K,N)→col(N,K), transa=N, lda=N
    // B_cub = a_ptr: row(K,M)→col(M,K), transb=T → (K,M), ldb=M
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        b.data, CUDA_R_16BF, N,   // b row(K,N) → col(N,K), lda=N
        a.data, CUDA_R_16BF, M,   // a row(K,M) → col(M,K), ldb=M
        &beta,
        out.data, CUDA_R_16BF, N, // out row(M,N) → col(N,M), ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// cuBLAS matmul: out(M,N) = a(M,K) @ b(K,N), all row-major PufTensors
void puf_mm_nn(PufTensor& a, PufTensor& b, PufTensor& out) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    // Row-major C(M,N) = A(M,K) @ B(K,N)
    // cuBLAS col-major: C^T(N,M) = B^T(N,K) @ A^T(K,M)
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data, CUDA_R_16BF, N,   // B(K,N) row-major, ldb=N
        a.data, CUDA_R_16BF, K,   // A(M,K) row-major, lda=K
        &beta,
        out.data, CUDA_R_16BF, N, // C(M,N) row-major, ldc=N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// out = alpha * a @ b^T + beta * out (bf16, f32 compute)
void puf_addmm(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[0];

    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data, CUDA_R_16BF, K,
        a.data, CUDA_R_16BF, K,
        &beta,
        out.data, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// out = alpha * a @ b + beta * out (bf16, f32 compute, no transpose)
void puf_addmm_nn(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];

    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data, CUDA_R_16BF, N,
        a.data, CUDA_R_16BF, K,
        &beta,
        out.data, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Cast+copy kernel: bf16 → f32
__global__ void cast_bf16_to_f32_kernel(
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __bfloat162float(src[idx]);
}

void puf_cast_bf16_to_f32(PufTensor& dst, const PufTensor& src) {
    assert(dst.numel == src.numel && "puf_cast_bf16_to_f32: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cast_bf16_to_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const __nv_bfloat16*)src.data, dst.numel);
}

// Frobenius norm of bf16 tensor
__global__ void norm_bf16_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

void puf_norm(const PufTensor& src, float* out_ptr) {
    assert(src.dtype_size == 2 && "puf_norm: expected bf16");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(out_ptr, 0, sizeof(float), stream);
    int blocks = std::min((int)grid_size(src.numel), 256);
    norm_bf16_kernel<<<blocks, 256, 0, stream>>>(
        out_ptr, (const __nv_bfloat16*)src.data, src.numel);
    // out_ptr now holds sum of squares; caller uses puf_normalize to apply sqrt + divide
}

// dst(bf16) *= 1/max(sqrt(*norm_ptr), eps)
__global__ void normalize_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ norm_ptr,
    float eps, int n
) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(__bfloat162float(dst[idx]) * inv_norm);
    }
}

void puf_normalize(PufTensor& dst, const float* norm_ptr, float eps) {
    assert(dst.dtype_size == 2 && "puf_normalize: expected bf16");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    normalize_bf16_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.data, norm_ptr, eps, dst.numel);
}

// Cast+copy kernel: f32 → bf16
__global__ void cast_f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src) {
    assert(dst.numel == src.numel && "puf_cast_f32_to_bf16: size mismatch");
    assert(dst.dtype_size == 2 && src.dtype_size == 4);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cast_f32_to_bf16_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.data, (const float*)src.data, dst.numel);
}

// Cast f32(R,C) → bf16(C,R) with transpose
__global__ void cast_f32_to_bf16_transpose_kernel(
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ src,
    int R, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    int r = idx / C;
    int c = idx % C;
    dst[c * R + r] = __float2bfloat16(src[r * C + c]);
}

void puf_cast_f32_to_bf16_transpose(PufTensor& dst, const PufTensor& src) {
    assert(src.dtype_size == 4 && dst.dtype_size == 2);
    int R = src.shape[0], C = src.shape[1];
    assert(dst.shape[0] == C && dst.shape[1] == R);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cast_f32_to_bf16_transpose_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.data, (const float*)src.data, R, C);
}

// Transpose f32(R,C) → f32(C,R)
__global__ void transpose_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int R, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    int r = idx / C;
    int c = idx % C;
    dst[c * R + r] = src[r * C + c];
}

void puf_transpose_f32(PufTensor& dst, const PufTensor& src) {
    assert(src.dtype_size == 4 && dst.dtype_size == 4);
    int R = src.shape[0], C = src.shape[1];
    assert(dst.shape[0] == C && dst.shape[1] == R);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    transpose_f32_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, R, C);
}

// Copy from torch::Tensor into PufTensor with dtype conversion
void puf_copy_from_torch(PufTensor& dst, Tensor src) {
    assert(dst.numel == src.numel() && "puf_copy_from_torch: size mismatch");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (src.scalar_type() == torch::kFloat32 && dst.dtype_size == 2) {
        // f32 → bf16
        cast_f32_to_bf16_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)dst.data,
            (const float*)src.data_ptr(),
            dst.numel);
    } else if (src.element_size() == dst.dtype_size) {
        // Same dtype: raw memcpy
        cudaMemcpyAsync(dst.data, src.data_ptr(), dst.nbytes(),
            cudaMemcpyDeviceToDevice, stream);
    } else {
        assert(false && "puf_copy_from_torch: unsupported dtype conversion");
    }
}

// PufTensor→PufTensor memcpy (same dtype, same size)
void puf_copy(PufTensor& dst, const PufTensor& src) {
    assert(dst.numel == src.numel && "puf_copy: size mismatch");
    assert(dst.dtype_size == src.dtype_size && "puf_copy: dtype mismatch");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(dst.data, src.data, dst.nbytes(),
        cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PufTensor& dst) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemsetAsync(dst.data, 0, dst.nbytes(), stream);
}

__global__ void scale_f32_kernel(float* __restrict__ dst, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

void puf_scale(PufTensor& dst, float alpha) {
    assert(dst.dtype_size == 4 && "puf_scale: expected f32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, alpha, dst.numel);
}

__global__ void axpy_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

void puf_axpy(PufTensor& dst, const PufTensor& src, float alpha) {
    assert(dst.numel == src.numel && "puf_axpy: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy: expected f32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    axpy_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, alpha, dst.numel);
}

// Device-pointer variants: read scalar from device memory (graph-safe)
__global__ void scale_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

void puf_scale_dev(PufTensor& dst, const float* alpha_ptr) {
    assert(dst.dtype_size == 4 && "puf_scale_dev: expected f32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    scale_f32_dev_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, alpha_ptr, dst.numel);
}

__global__ void axpy_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                     const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

void puf_axpy_dev(PufTensor& dst, const PufTensor& src, const float* alpha_ptr) {
    assert(dst.numel == src.numel && "puf_axpy_dev: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy_dev: expected f32");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    axpy_f32_dev_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, alpha_ptr, dst.numel);
}

__global__ void compute_lr_scalars_kernel(const float* __restrict__ lr, float wd,
                                           float* __restrict__ neg_lr, float* __restrict__ wd_scale) {
    *neg_lr = -(*lr);
    *wd_scale = 1.0f - (*lr) * wd;
}

void compute_lr_scalars(const float* lr_ptr, float weight_decay, float* neg_lr_ptr, float* wd_scale_ptr) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    compute_lr_scalars_kernel<<<1, 1, 0, stream>>>(lr_ptr, weight_decay, neg_lr_ptr, wd_scale_ptr);
}

// dst(fp32) += src(bf16) kernel
__global__ void add_bf16_to_f32_kernel(
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += __bfloat162float(src[idx]);
    }
}

void puf_add(PufTensor& dst, const PufTensor& src) {
    assert(dst.numel == src.numel && "puf_add: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2 && "puf_add: expected fp32 += bf16");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    add_bf16_to_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const __nv_bfloat16*)src.data, dst.numel);
}

// Column-wise sum reduction: dst(1, C) += src(R, C).sum(dim=0), both f32
__global__ void sum_rows_add_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int R, int C
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) return;
    float sum = 0.0f;
    for (int r = 0; r < R; r++) {
        sum += src[r * C + col];
    }
    dst[col] += sum;
}

void puf_sum_rows_add(PufTensor& dst, PufTensor& src) {
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_sum_rows_add: expected f32");
    int R = src.numel / src.shape[src.ndim - 1];
    int C = src.shape[src.ndim - 1];
    assert(dst.numel == C && "puf_sum_rows_add: dst must have C elements");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    sum_rows_add_kernel<<<grid_size(C), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, R, C);
}

// Assemble fused decoder grad from separate logits/value grads with f32→bf16 cast
__global__ void assemble_decoder_grad_kernel(
    __nv_bfloat16* __restrict__ dst,
    const float* __restrict__ grad_logits,
    const float* __restrict__ grad_value,
    int B_TT, int od, int od_plus_1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) return;
    int row = idx / od_plus_1;
    int col = idx % od_plus_1;
    float val = (col < od) ? grad_logits[row * od + col] : grad_value[row];
    dst[idx] = __float2bfloat16(val);
}

void puf_assemble_decoder_grad(PufTensor& dst, PufTensor& grad_logits, PufTensor& grad_value) {
    int B_TT = dst.size(0);
    int od_plus_1 = dst.size(1);
    int od = od_plus_1 - 1;

    int total = B_TT * od_plus_1;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    assemble_decoder_grad_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.data,
        (const float*)grad_logits.data,
        (const float*)grad_value.data,
        B_TT, od, od_plus_1);
}

// Fused: chunk + mingru_gate + sigmoid(proj) * out
// combined is (B, 3*H) = [hidden, gate, proj]
// state is (B, H)
// returns {out, next_state} where:
//   out (B, H) = sigmoid(proj) * mingru_out
//   next_state (B, H) = mingru_out (for recurrence)
void mingru_gate(Tensor state, Tensor combined, Tensor out, Tensor next_state) {
    TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
    TORCH_CHECK(combined.is_cuda(), "combined must be on CUDA");
    TORCH_CHECK(state.dtype() == combined.dtype(), "dtypes must match");
    TORCH_CHECK(state.dim() == 2 && combined.dim() == 2, "must be 2D tensors");
    TORCH_CHECK(combined.size(1) == 3 * state.size(1), "combined must be 3*H");
    TORCH_CHECK(state.size(0) == combined.size(0), "batch size must match");
    TORCH_CHECK(state.is_contiguous() && combined.is_contiguous(), "must be contiguous");

    int B = static_cast<int>(state.size(0));
    int H = static_cast<int>(state.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mingru_gate_inference_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)out.data_ptr(),
        (precision_t*)next_state.data_ptr(),
        (const precision_t*)combined.data_ptr(),
        (const precision_t*)state.data_ptr(),
        H, B);
}

// PufTensor overload: all PufTensors
void mingru_gate(PufTensor& state, PufTensor& combined, PufTensor& out, PufTensor& next_state) {
    int B = static_cast<int>(state.size(0));
    int H = static_cast<int>(state.size(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mingru_gate_inference_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)out.data,
        (precision_t*)next_state.data,
        (const precision_t*)combined.data,
        (const precision_t*)state.data,
        H, B);
}

// Prefix scan forward — writes into pre-allocated bufs
void prefix_scan_forward(PufTensor& combined, PufTensor& state, PrefixScan& bufs) {
    assert(combined.ndim == 3 && "combined must be (B, T, 3*H)");
    assert(state.ndim == 3 && "state must be (B, 1, H)");
    assert(state.size(0) == combined.size(0) && "B must match");
    assert(state.size(1) == 1 && "state T dim must be 1");
    assert(combined.size(2) == 3 * state.size(2) && "combined must be 3*H");

    int B = static_cast<int>(combined.size(0));
    int T = static_cast<int>(combined.size(1));
    int H = static_cast<int>(state.size(2));

    // Save raw pointers + dims for backward
    bufs.combined_ptr = combined.data;
    bufs.state_ptr = state.data;
    bufs.B = B;
    bufs.T = T;
    bufs.H = H;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_scan_forward_kernel_checkpointed<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)bufs.out.data,
        (precision_t*)bufs.next_state.data,
        (float*)bufs.a_star.data,
        (float*)bufs.s_vals.data,
        (float*)bufs.log_values_buf.data,
        (const precision_t*)combined.data,
        (const precision_t*)state.data,
        T, H, B);
}

// Prefix scan backward — writes into pre-allocated bufs
void prefix_scan_backward(
    PufTensor& grad_out, PufTensor& grad_next_state, PrefixScan& bufs) {
    int B = bufs.B;
    int T = bufs.T;
    int H = bufs.H;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_scan_backward_kernel_checkpointed<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)bufs.grad_combined.data,
        (precision_t*)bufs.grad_state.data,
        (const precision_t*)grad_out.data,
        (const precision_t*)grad_next_state.data,
        (const precision_t*)bufs.combined_ptr,
        (const precision_t*)bufs.state_ptr,
        (float*)bufs.a_star.data,
        (float*)bufs.s_vals.data,
        (float*)bufs.log_values_buf.data,
        T, H, B);
}

// LogCumsumExp x = (B, T, H)
tensor_list LogCumsumExp::forward(AutogradCtx* ctx, Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    auto device = x.device();
    int B = static_cast<int>(x.size(0));
    int T = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));

    auto out = torch::empty({B, T, H}, x.options());
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto s_buf = torch::empty({B, T, H}, options_double);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    logcumsumexp_forward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)out.data_ptr(),
        s_buf.data_ptr<double>(),
        (const precision_t*)x.data_ptr(),
        T, H, B);

    ctx->save_for_backward({x, out, s_buf});
    return {out};
}

tensor_list LogCumsumExp::backward(AutogradCtx* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0].contiguous();
    auto s_buf = saved[2];

    auto grad_out = grad_outputs[0].contiguous();
    int B = static_cast<int>(x.size(0));
    int T = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));

    auto grad_x = torch::empty_like(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    logcumsumexp_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)grad_x.data_ptr(),
        (const precision_t*)grad_out.data_ptr(),
        (const precision_t*)x.data_ptr(),
        s_buf.data_ptr<double>(),
        T, H, B);

    return {grad_x};
}

// PPOLoss: fused PPO clipped loss with value loss
tensor_list PPOLoss::forward(
    AutogradCtx* ctx,
    Tensor logits,
    Tensor logstd,
    Tensor values_pred,
    Tensor actions,
    Tensor old_logprobs,
    Tensor advantages,
    Tensor prio,
    Tensor values,
    Tensor returns,
    Tensor ratio_out,
    Tensor newvalue_out,
    Tensor act_sizes,
    Tensor losses_acc,
    double clip_coef,
    double vf_clip_coef,
    double vf_coef,
    double ent_coef
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(act_sizes.is_cuda(), "act_sizes must be on CUDA");
    TORCH_CHECK(act_sizes.dtype() == torch::kInt32, "act_sizes must be int32");
    auto dtype = logits.dtype();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kBFloat16,
                "Only float32 and bfloat16 supported");

    bool is_continuous = logstd.defined() && logstd.numel() > 0;

    // Create empty CUDA tensor for logstd if discrete (undefined tensors can't be saved)
    auto logstd_to_save = is_continuous ? logstd : torch::empty({0}, logits.options());

    auto device = logits.device();
    int N = static_cast<int>(logits.size(0));
    int T = static_cast<int>(logits.size(1));
    int A_total = static_cast<int>(logits.size(2));
    int num_atns = static_cast<int>(act_sizes.size(0));

    // Compute advantage mean/var internally (rolled in from caller)
    auto [adv_var, adv_mean] = torch::var_mean(advantages);

    // Reshape actions from (N, T, num_atns) to (N*T, num_atns) for kernel
    auto actions_flat = actions.reshape({N * T, num_atns}).contiguous();

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto loss_output = torch::empty({1}, options_float);

    // saved_for_backward not used by optimized backward, but kernel still writes to it
    auto saved_for_backward = torch::zeros({N * T, 5}, options_double);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // strides for non-contiguous tensor support
    auto logits_strides = logits.strides();
    auto values_strides = values_pred.strides();
    int logits_stride_n = logits_strides[0];
    int logits_stride_t = logits_strides[1];
    int logits_stride_a = logits_strides[2];
    int values_stride_n = values_strides[0];
    int values_stride_t = values_strides[1];

    int total = N * T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    cudaMemsetAsync(loss_output.data_ptr<float>(), 0, sizeof(float), stream);
    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        loss_output.data_ptr<float>(),
        losses_acc.data_ptr<float>(),
        saved_for_backward.data_ptr<double>(),
        (precision_t*)ratio_out.data_ptr(),
        (precision_t*)newvalue_out.data_ptr(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)values_pred.data_ptr(),
        actions_flat.data_ptr<double>(),
        (const precision_t*)old_logprobs.data_ptr(),
        advantages.data_ptr<float>(),
        (const precision_t*)prio.data_ptr(),
        (const precision_t*)values.data_ptr(),
        (const precision_t*)returns.data_ptr(),
        adv_mean.data_ptr<float>(),
        adv_var.data_ptr<float>(),
        act_sizes.data_ptr<int>(),
        num_atns,
        static_cast<float>(clip_coef),
        static_cast<float>(vf_clip_coef),
        static_cast<float>(vf_coef),
        static_cast<float>(ent_coef),
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);

    // Accumulate LOSS_N counter (rolled in from caller)
    losses_acc.select(0, LOSS_N).add_(1.0);

    ctx->saved_data["clip_coef"] = clip_coef;
    ctx->saved_data["vf_clip_coef"] = vf_clip_coef;
    ctx->saved_data["vf_coef"] = vf_coef;
    ctx->saved_data["ent_coef"] = ent_coef;
    ctx->saved_data["is_continuous"] = is_continuous;

    ctx->save_for_backward({logits, logstd_to_save, values_pred, actions_flat, old_logprobs, advantages,
                            prio, values, returns, adv_mean, adv_var, act_sizes});

    return {loss_output};
}

tensor_list PPOLoss::backward(
    AutogradCtx* ctx,
    tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto logits = saved[0].contiguous();
    auto logstd = saved[1];  // may be empty for discrete
    auto values_pred = saved[2].contiguous();
    auto actions_flat = saved[3].contiguous();  // already (N*T, num_atns) from forward
    auto old_logprobs = saved[4].contiguous();
    auto advantages = saved[5].contiguous();
    auto prio = saved[6].contiguous();
    auto values = saved[7].contiguous();
    auto returns = saved[8].contiguous();
    auto adv_mean = saved[9].contiguous();
    auto adv_var = saved[10].contiguous();
    auto act_sizes = saved[11];  // already on CUDA and contiguous

    int N = static_cast<int>(logits.size(0));
    int T = static_cast<int>(logits.size(1));
    int A_total = static_cast<int>(logits.size(2));
    int num_atns = static_cast<int>(act_sizes.size(0));

    float clip_coef = ctx->saved_data["clip_coef"].to<double>();
    float vf_clip_coef = ctx->saved_data["vf_clip_coef"].to<double>();
    float vf_coef = ctx->saved_data["vf_coef"].to<double>();
    float ent_coef = ctx->saved_data["ent_coef"].to<double>();
    bool is_continuous = ctx->saved_data["is_continuous"].to<bool>();

    auto grad_loss = grad_outputs[0].to(torch::kFloat32).contiguous();

    // keep gradients in fp32 for precision / training stability issues
    auto grad_logits = torch::empty(logits.sizes(), logits.options().dtype(torch::kFloat32));
    auto grad_values_pred = torch::empty(values_pred.sizes(), values_pred.options().dtype(torch::kFloat32));
    // For continuous: grad_logstd has same shape as logstd
    Tensor grad_logstd;
    if (is_continuous) {
        logstd = logstd.contiguous();
        grad_logstd = torch::empty(logstd.sizes(), logstd.options().dtype(torch::kFloat32));
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto logits_strides = logits.strides();
    auto values_strides = values_pred.strides();
    int logits_stride_n = logits_strides[0];
    int logits_stride_t = logits_strides[1];
    int logits_stride_a = logits_strides[2];
    int values_stride_n = values_strides[0];
    int values_stride_t = values_strides[1];

    int total = N * T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        grad_logits.data_ptr<float>(),
        is_continuous ? grad_logstd.data_ptr<float>() : nullptr,
        grad_values_pred.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)values_pred.data_ptr(),
        actions_flat.data_ptr<double>(),
        (const precision_t*)old_logprobs.data_ptr(),
        advantages.data_ptr<float>(),
        (const precision_t*)prio.data_ptr(),
        (const precision_t*)values.data_ptr(),
        (const precision_t*)returns.data_ptr(),
        adv_mean.data_ptr<float>(),
        adv_var.data_ptr<float>(),
        act_sizes.data_ptr<int>(),
        num_atns,
        clip_coef, vf_clip_coef,
        vf_coef, ent_coef,
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);

    return {
        grad_logits,
        is_continuous ? grad_logstd : Tensor(),  // grad_logstd
        grad_values_pred,
        {}, {}, {}, {}, {}, {},  // actions, old_logprobs, advantages, prio, values, returns
        {}, {},                   // ratio_out, newvalue_out (no grad needed)
        {},                       // act_sizes (no grad needed)
        {},                       // losses_acc (no grad needed)
        {}, {}, {}, {}           // clip_coef, vf_clip_coef, vf_coef, ent_coef
    };
}


// Fused PPO loss forward + backward (no autograd)
// Runs forward kernel then backward kernel, writes gradients into pre-allocated PPOBuffers
void ppo_loss_fwd_bwd(
    Tensor logits, Tensor logstd, Tensor values_pred,
    Tensor actions, Tensor old_logprobs, Tensor advantages,
    Tensor prio, Tensor values, Tensor returns,
    Tensor ratio_out, Tensor newvalue_out,
    Tensor act_sizes, Tensor losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
    PPOBuffers& bufs
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(act_sizes.is_cuda() && act_sizes.dtype() == torch::kInt32,
                "act_sizes must be int32 on CUDA");

    // logits/values_pred may be non-contiguous (fused decoder output) — kernel handles via strides
    // Grad outputs use contiguous layout (nt * A_total indexing)
    TORCH_CHECK(old_logprobs.is_contiguous(), "old_logprobs must be contiguous");
    TORCH_CHECK(advantages.is_contiguous(), "advantages must be contiguous");
    TORCH_CHECK(prio.is_contiguous(), "prio must be contiguous");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(returns.is_contiguous(), "returns must be contiguous");

    bool is_continuous = logstd.defined() && logstd.numel() > 0;
    // TODO: pre-allocate contiguous logstd buffer to remove this alloc
    if (is_continuous) logstd = logstd.contiguous();

    int N = static_cast<int>(logits.size(0));
    int T = static_cast<int>(logits.size(1));
    int A_total = static_cast<int>(logits.size(2));
    int num_atns = static_cast<int>(act_sizes.size(0));
    int total = N * T;

    auto [adv_var, adv_mean] = torch::var_mean(advantages);
    auto actions_flat = actions.reshape({total, num_atns});
    TORCH_CHECK(actions_flat.is_contiguous(), "actions must be contiguous");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int logits_stride_n = logits.stride(0);
    int logits_stride_t = logits.stride(1);
    int logits_stride_a = logits.stride(2);
    int values_stride_n = values_pred.stride(0);
    int values_stride_t = values_pred.stride(1);

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    // Forward kernel — writes into pre-allocated bufs.loss_output, bufs.saved_for_bwd
    cudaMemsetAsync(bufs.loss_output.data_ptr<float>(), 0, sizeof(float), stream);
    bufs.saved_for_bwd.zero_();
    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        bufs.loss_output.data_ptr<float>(),
        losses_acc.data_ptr<float>(),
        bufs.saved_for_bwd.data_ptr<double>(),
        (precision_t*)ratio_out.data_ptr(),
        (precision_t*)newvalue_out.data_ptr(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)values_pred.data_ptr(),
        actions_flat.data_ptr<double>(),
        (const precision_t*)old_logprobs.data_ptr(),
        advantages.data_ptr<float>(),
        (const precision_t*)prio.data_ptr(),
        (const precision_t*)values.data_ptr(),
        (const precision_t*)returns.data_ptr(),
        adv_mean.data_ptr<float>(),
        adv_var.data_ptr<float>(),
        act_sizes.data_ptr<int>(),
        num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef,
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);

    losses_acc.select(0, LOSS_N).add_(1.0);

    // Backward kernel — writes into pre-allocated bufs.grad_logits, bufs.grad_values, bufs.grad_logstd
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        bufs.grad_logits.data_ptr<float>(),
        is_continuous ? bufs.grad_logstd.data_ptr<float>() : nullptr,
        bufs.grad_values.data_ptr<float>(),
        bufs.grad_loss.data_ptr<float>(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)values_pred.data_ptr(),
        actions_flat.data_ptr<double>(),
        (const precision_t*)old_logprobs.data_ptr(),
        advantages.data_ptr<float>(),
        (const precision_t*)prio.data_ptr(),
        (const precision_t*)values.data_ptr(),
        (const precision_t*)returns.data_ptr(),
        adv_mean.data_ptr<float>(),
        adv_var.data_ptr<float>(),
        act_sizes.data_ptr<int>(),
        num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef,
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);
}

// Fused sample_logits: handles both discrete and continuous action sampling
void sample_logits(
    Tensor logits,
    Tensor logstd,  // Empty tensor for discrete, defined for continuous
    Tensor value,
    Tensor actions_out,
    Tensor logprobs_out,
    Tensor value_out,
    Tensor act_sizes,
    uint64_t seed,
    Tensor offset
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D (B, num_atns or sum(act_sizes))");
    TORCH_CHECK(logits.stride(1) == 1, "logits must be contiguous in last dim");
    TORCH_CHECK(actions_out.is_contiguous(), "actions_out must be contiguous");
    TORCH_CHECK(logprobs_out.is_contiguous(), "logprobs_out must be contiguous");
    TORCH_CHECK(value_out.is_contiguous(), "value_out must be contiguous");
    TORCH_CHECK(actions_out.dtype() == torch::kFloat64, "actions_out must be float64");
    TORCH_CHECK(offset.dtype() == torch::kInt64, "offset must be int64");
    TORCH_CHECK(offset.is_cuda(), "offset must be on CUDA");
    TORCH_CHECK(act_sizes.dtype() == torch::kInt32, "act_sizes must be int32");
    TORCH_CHECK(act_sizes.is_cuda(), "act_sizes must be on CUDA");

    bool is_continuous = logstd.defined() && logstd.numel() > 0;

    int B = static_cast<int>(logits.size(0));
    int num_atns = static_cast<int>(act_sizes.size(0));
    int logits_stride = static_cast<int>(logits.stride(0));
    int logstd_stride = is_continuous ? static_cast<int>(logstd.stride(0)) : 0;
    int value_stride = static_cast<int>(value.stride(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sample_logits_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        actions_out.data_ptr<double>(),
        (precision_t*)logprobs_out.data_ptr(),
        (precision_t*)value_out.data_ptr(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)value.data_ptr(),
        act_sizes.data_ptr<int>(),
        seed,
        offset.data_ptr<int64_t>(),
        num_atns, B, logits_stride, logstd_stride, value_stride,
        is_continuous);
}

// FCMax: fused FC -> Max
tensor_list FCMax::forward(
    AutogradCtx* ctx,
    Tensor x,      // (B, N, D_in)
    Tensor W,      // (D_out, D_in)
    Tensor b       // (D_out)
) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(W.is_cuda(), "W must be on CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    int B = x.size(0);
    int N = x.size(1);
    int D_in = x.size(2);
    int D_out = W.size(0);

    auto out = torch::empty({B, D_out}, x.options());
    auto argmax = torch::empty({B, D_out}, torch::dtype(torch::kInt32).device(x.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto W_f32 = W.dtype() == torch::kFloat32 ? W : W.to(torch::kFloat32);
    auto b_f32 = b.dtype() == torch::kFloat32 ? b : b.to(torch::kFloat32);
    fc_max_forward_kernel<<<grid_size(B * D_out), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)out.data_ptr(),
        argmax.data_ptr<int>(),
        (const precision_t*)x.data_ptr(),
        W_f32.data_ptr<float>(),
        b_f32.data_ptr<float>(),
        B, N, D_in, D_out);

    ctx->save_for_backward({x, W, argmax});
    ctx->saved_data["B"] = B;
    ctx->saved_data["N"] = N;
    ctx->saved_data["D_in"] = D_in;
    ctx->saved_data["D_out"] = D_out;

    return {out, argmax};
}

tensor_list FCMax::backward(AutogradCtx* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto W = saved[1];
    auto argmax = saved[2];
    auto grad_out = grad_outputs[0].contiguous();

    auto dtype = x.dtype();
    int B = ctx->saved_data["B"].toInt();
    int N = ctx->saved_data["N"].toInt();
    int D_in = ctx->saved_data["D_in"].toInt();
    int D_out = ctx->saved_data["D_out"].toInt();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Accumulate in fp32 (atomicAdd requires fp32), W is always fp32
    auto opts_f32 = x.options().dtype(torch::kFloat32);
    auto grad_x_f32 = torch::zeros({B, N, D_in}, opts_f32);
    auto grad_W_f32 = torch::zeros({D_out, D_in}, opts_f32);
    auto grad_b_f32 = torch::zeros({D_out}, opts_f32);
    auto W_f32 = W.dtype() == torch::kFloat32 ? W : W.to(torch::kFloat32);

    fc_max_backward_kernel<<<grid_size(B * D_out), BLOCK_SIZE, 0, stream>>>(
        grad_x_f32.data_ptr<float>(),
        grad_W_f32.data_ptr<float>(),
        grad_b_f32.data_ptr<float>(),
        (const precision_t*)grad_out.data_ptr(),
        (const precision_t*)x.data_ptr(),
        W_f32.data_ptr<float>(),
        argmax.data_ptr<int>(),
        B, N, D_in, D_out);

    auto grad_x = (dtype == torch::kBFloat16) ? grad_x_f32.to(torch::kBFloat16) : grad_x_f32;
    return {grad_x, grad_W_f32, grad_b_f32};
}


void train_select_and_copy_cuda(
    Tensor observations, Tensor actions,
    Tensor logprobs, Tensor values, Tensor advantages,
    Tensor idx, Tensor mb_prio,
    Tensor dst_obs, Tensor dst_state,
    Tensor dst_actions, Tensor dst_logprobs,
    Tensor dst_advantages, Tensor dst_prio,
    Tensor dst_values, Tensor dst_returns
) {
    int mb_segs = idx.size(0);
    int horizon = values.size(1);
    int obs_rb = observations.stride(0) * observations.element_size();
    int act_rb = actions.stride(0) * actions.element_size();
    int lp_rb = logprobs.stride(0) * logprobs.element_size();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dst_state.zero_();

    select_copy_kernel<<<dim3(mb_segs, 5), SELECT_COPY_THREADS, 0, stream>>>(
        idx.data_ptr<int64_t>(),
        (const char*)observations.data_ptr(), (char*)dst_obs.data_ptr(), obs_rb,
        (const char*)actions.data_ptr(), (char*)dst_actions.data_ptr(), act_rb,
        (const char*)logprobs.data_ptr(), (char*)dst_logprobs.data_ptr(), lp_rb,
        (const precision_t*)values.data_ptr(), (precision_t*)dst_values.data_ptr(),
        advantages.data_ptr<float>(), dst_advantages.data_ptr<float>(),
        (precision_t*)dst_returns.data_ptr(), horizon,
        mb_prio.data_ptr<float>(), (precision_t*)dst_prio.data_ptr());
}

// Host dispatch: replaces ~9 PyTorch kernel launches with 3 custom + multinomial
tuple<Tensor, Tensor> prio_replay_cuda(
    Tensor advantages,       // (S, T) float32
    float prio_alpha,
    int minibatch_segments,
    int total_agents,
    float anneal_beta
) {
    int S = advantages.size(0);
    int T = advantages.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto prio_probs = torch::empty({S}, advantages.options());

    compute_prio_adv_reduction<<<S, PRIO_WARP_SIZE, 0, stream>>>(
        advantages.data_ptr<float>(), prio_probs.data_ptr<float>(),
        prio_alpha, T);

    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        prio_probs.data_ptr<float>(), S);

    auto idx = at::multinomial(prio_probs, minibatch_segments, true);

    auto mb_prio = torch::empty({minibatch_segments, 1}, advantages.options());
    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        idx.data_ptr<int64_t>(), prio_probs.data_ptr<float>(),
        mb_prio.data_ptr<float>(),
        total_agents, anneal_beta, minibatch_segments);

    return {idx, mb_prio};
}

// =============================================================================
// Puff Advantage CUDA dispatch
// =============================================================================

namespace pufferlib {

void vtrace_check_cuda(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    auto input_dtype = values.dtype();
    for (const Tensor& t : {values, rewards, dones, importance}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == input_dtype, "Input tensors must have matching dtype");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
    // advantages can be different dtype (fp32 for precision)
    TORCH_CHECK(advantages.dim() == 2, "Advantages must be 2D");
    TORCH_CHECK(advantages.device() == device, "Advantages must be on same device");
    TORCH_CHECK(advantages.size(0) == num_steps, "Advantages first dimension must match");
    TORCH_CHECK(advantages.size(1) == horizon, "Advantages second dimension must match");
    if (!advantages.is_contiguous()) {
        advantages.contiguous();
    }
}

template<typename TIn, typename TOut>
void puff_advantage_cuda_impl(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    constexpr int N = 16 / sizeof(TIn);
    auto kernel = (horizon % N == 0 && sizeof(TOut) == 4)
        ? puff_advantage_kernel<TIn, TOut>
        : puff_advantage_kernel_scalar<TIn, TOut>;

    kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<TIn>(), rewards.data_ptr<TIn>(),
        dones.data_ptr<TIn>(), importance.data_ptr<TIn>(),
        advantages.data_ptr<TOut>(),
        static_cast<float>(gamma), static_cast<float>(lambda),
        static_cast<float>(rho_clip), static_cast<float>(c_clip),
        num_steps, horizon);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void puff_advantage_cuda(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    auto input_dtype = values.dtype();
    auto output_dtype = advantages.dtype();

    // Support bf16 inputs with fp32 output for precision
    if (input_dtype == torch::kFloat32 && output_dtype == torch::kFloat32) {
        puff_advantage_cuda_impl<float, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kFloat32) {
        puff_advantage_cuda_impl<at::BFloat16, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kBFloat16) {
        puff_advantage_cuda_impl<at::BFloat16, at::BFloat16>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination: inputs must be float32 or bfloat16, advantages must be float32 or bfloat16");
    }
}

} // namespace pufferlib

#endif // PUFFERLIB_MODULES_CU
