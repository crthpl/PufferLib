#ifndef PUFFERLIB_MODULES_CU
#define PUFFERLIB_MODULES_CU

#ifdef PUFFERLIB_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cassert>

#include "../modules.h"
#include "../legacy_modules.h"
#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>

using std::tuple;
using std::vector;
#ifdef PUFFERLIB_TORCH
typedef torch::Tensor Tensor;
#endif

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

// cuBLAS matmul: out(M,N) = a(M,K) @ b(N,K)^T, all row-major PufTensors
// Uses bf16 inputs with f32 compute.
void puf_mm(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[0];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
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
void puf_mm_tn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int K = a.shape[0];
    int M = a.shape[1];
    int N = b.shape[1];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
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
void puf_mm_nn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
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
void puf_addmm(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta, cudaStream_t stream) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[0];

    cublasHandle_t handle = get_cublas_handle();
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
void puf_addmm_nn(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta, cudaStream_t stream) {
    int M = a.shape[0];
    int K = a.shape[1];
    int N = b.shape[1];

    cublasHandle_t handle = get_cublas_handle();
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

void puf_cast_bf16_to_f32(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_cast_bf16_to_f32: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2);
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

void puf_norm(const PufTensor& src, float* out_ptr, cudaStream_t stream) {
    assert(src.dtype_size == 2 && "puf_norm: expected bf16");
    cudaMemsetAsync(out_ptr, 0, sizeof(float), stream);
    int blocks = std::min((int)grid_size(src.numel), 256);
    norm_bf16_kernel<<<blocks, 256, 0, stream>>>(
        out_ptr, (const __nv_bfloat16*)src.data, src.numel);
    // out_ptr now holds sum of squares; caller uses puf_normalize to apply sqrt + divide
}

// f32 sum-of-squares reduction (same pattern as bf16 but reads float directly)
__global__ void norm_f32_kernel(float* __restrict__ out, const float* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += src[i] * src[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// dst *= min(max_norm / (sqrt(sum_sq) + eps), 1.0)
__global__ void clip_by_norm_f32_kernel(float* __restrict__ dst, const float* __restrict__ sum_sq_ptr,
                                         float max_norm, float eps, int n) {
    float total_norm = sqrtf(*sum_sq_ptr);
    float clip_coef = fminf(max_norm / (total_norm + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] *= clip_coef;
    }
}

void puf_clip_grad_norm(PufTensor& grad, float max_norm, float* scratch, cudaStream_t stream) {
    assert(grad.dtype_size == 4 && "puf_clip_grad_norm: expected f32");
    cudaMemsetAsync(scratch, 0, sizeof(float), stream);
    int blocks = std::min((int)grid_size(grad.numel), 256);
    norm_f32_kernel<<<blocks, 256, 0, stream>>>(scratch, (float*)grad.data, grad.numel);
    clip_by_norm_f32_kernel<<<grid_size(grad.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)grad.data, scratch, max_norm, 1e-6f, grad.numel);
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

void puf_normalize(PufTensor& dst, const float* norm_ptr, float eps, cudaStream_t stream) {
    assert(dst.dtype_size == 2 && "puf_normalize: expected bf16");
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

void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_cast_f32_to_bf16: size mismatch");
    assert(dst.dtype_size == 2 && src.dtype_size == 4);
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

void puf_cast_f32_to_bf16_transpose(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(src.dtype_size == 4 && dst.dtype_size == 2);
    int R = src.shape[0], C = src.shape[1];
    assert(dst.shape[0] == C && dst.shape[1] == R);
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

void puf_transpose_f32(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(src.dtype_size == 4 && dst.dtype_size == 4);
    int R = src.shape[0], C = src.shape[1];
    assert(dst.shape[0] == C && dst.shape[1] == R);
    transpose_f32_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, R, C);
}

// Swap dims 0 and 1: src(A, B, C) → dst(B, A, C).  C=1 for 2D.
// Each thread moves one scalar element.
template <typename T>
__global__ void transpose_01_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int A, int B, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) return;
    int a = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

void puf_transpose_01(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    int A = src.shape[0];
    int B = src.shape[1];
    int C = (src.ndim >= 3) ? src.shape[2] : 1;
    assert(dst.shape[0] == B && dst.shape[1] == A);
    assert(dst.dtype_size == src.dtype_size);
    int n = A * B * C;
    switch (src.dtype_size) {
        case 2: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint16_t*)dst.data, (const uint16_t*)src.data, A, B, C); break;
        case 4: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint32_t*)dst.data, (const uint32_t*)src.data, A, B, C); break;
        case 8: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint64_t*)dst.data, (const uint64_t*)src.data, A, B, C); break;
        default: assert(false && "puf_transpose_01: unsupported dtype_size");
    }
}

// Sign-correct Q columns: Q[:, j] *= sign(R[j, j])
// A is (m, n) column-major, Q is (m, n) column-major (after orgqr, shares same memory)
__global__ void sign_correct_columns_kernel(float* Q, const float* diag_signs, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    int j = idx / m;  // column index (column-major: consecutive elements are same column)
    Q[idx] *= diag_signs[j];
}

// Extract sign(diag(A)) where A is (m, n) column-major. diag[i] = A[i + i*m]
__global__ void extract_diag_sign_kernel(float* signs, const float* A, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = A[i + (int64_t)i * m];
    signs[i] = (val >= 0.0f) ? 1.0f : -1.0f;
}

// Scale and convert column-major Q to row-major dst (with optional dtype cast)
// Q_colmaj is (m, n) column-major, dst is (m, n) row-major
__global__ void colmaj_to_rowmaj_scale_f32_kernel(
        float* dst, const float* Q, float gain, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    int i = idx / n;  // row in row-major
    int j = idx % n;  // col in row-major
    dst[idx] = Q[i + (int64_t)j * m] * gain;  // column-major read
}

__global__ void colmaj_to_rowmaj_scale_bf16_kernel(
        __nv_bfloat16* dst, const float* Q, float gain, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    int i = idx / n;
    int j = idx % n;
    dst[idx] = __float2bfloat16(Q[i + (int64_t)j * m] * gain);
}

// Orthogonal init: QR-based, matches torch.nn.init.orthogonal_
void puf_orthogonal_init(PufTensor& dst, float gain, uint64_t seed, cudaStream_t stream) {
    assert(dst.ndim == 2);
    int64_t rows = dst.shape[0];
    int64_t cols = dst.shape[1];
    assert(rows > 0 && cols > 0);

    bool transposed = rows < cols;
    int m = transposed ? (int)cols : (int)rows;  // tall dim (m >= n)
    int n = transposed ? (int)rows : (int)cols;   // short dim

    // Allocate workspace: random matrix + tau + signs + devInfo
    float *A, *tau, *signs;
    int *devInfo;
    int64_t mn = (int64_t)m * n;
    // curandGenerateNormal requires even count
    int64_t rand_count = (mn % 2 == 0) ? mn : mn + 1;
    cudaMalloc(&A, rand_count * sizeof(float));
    cudaMalloc(&tau, n * sizeof(float));
    cudaMalloc(&signs, n * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    // Fill with random normals
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, A, rand_count, 0.0f, 1.0f);
    curandDestroyGenerator(gen);

    // QR decomposition (column-major)
    cusolverDnHandle_t solver;
    cusolverDnCreate(&solver);

    int lwork;
    cusolverDnSgeqrf_bufferSize(solver, m, n, A, m, &lwork);
    float* work;
    cudaMalloc(&work, lwork * sizeof(float));

    cusolverDnSgeqrf(solver, m, n, A, m, tau, work, lwork, devInfo);

    // Extract sign(diag(R)) before orgqr overwrites R
    extract_diag_sign_kernel<<<grid_size(n), BLOCK_SIZE>>>(signs, A, m, n);

    // Reconstruct explicit Q
    int lwork2;
    cusolverDnSorgqr_bufferSize(solver, m, n, n, A, m, tau, &lwork2);
    if (lwork2 > lwork) {
        cudaFree(work);
        cudaMalloc(&work, lwork2 * sizeof(float));
    }
    cusolverDnSorgqr(solver, m, n, n, A, m, tau, work, lwork2, devInfo);

    // Apply sign correction: Q[:, j] *= sign[j]
    sign_correct_columns_kernel<<<grid_size(mn), BLOCK_SIZE>>>(A, signs, m, n);

    // Convert column-major Q to row-major dst with gain and optional transpose-back
    if (transposed) {
        // Q is (m=cols, n=rows) col-major. Q^T in (rows, cols) row-major has
        // identical flat layout (Q^T_rm[j, i] = Q_cm[i, j] = A[i + j*cols]).
        // Just cast + scale, no reorder needed.
        if (dst.dtype_size == 2) {
            cast_f32_to_bf16_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)dst.data, A, rows * cols);
        } else {
            cudaMemcpyAsync(dst.data, A, rows * cols * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }
        if (gain != 1.0f) {
            // Apply gain via existing puf_scale
            PufTensor tmp = dst;
            puf_scale(tmp, gain, stream);
        }
    } else {
        // Q is (rows, cols) col-major → need to transpose to (rows, cols) row-major
        if (dst.dtype_size == 2) {
            colmaj_to_rowmaj_scale_bf16_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)dst.data, A, gain, rows, cols);
        } else {
            colmaj_to_rowmaj_scale_f32_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (float*)dst.data, A, gain, rows, cols);
        }
    }

    // Cleanup
    cudaFree(A);
    cudaFree(tau);
    cudaFree(signs);
    cudaFree(work);
    cudaFree(devInfo);
    cusolverDnDestroy(solver);
}

#ifdef PUFFERLIB_TORCH
// Copy from torch::Tensor into PufTensor with dtype conversion
void puf_copy_from_torch(PufTensor& dst, Tensor src, cudaStream_t stream) {
    assert(dst.numel == src.numel() && "puf_copy_from_torch: size mismatch");

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
#endif // PUFFERLIB_TORCH

// PufTensor→PufTensor memcpy (same dtype, same size)
void puf_copy(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_copy: size mismatch");
    assert(dst.dtype_size == src.dtype_size && "puf_copy: dtype mismatch");
    cudaMemcpyAsync(dst.data, src.data, dst.nbytes(),
        cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PufTensor& dst, cudaStream_t stream) {
    cudaMemsetAsync(dst.data, 0, dst.nbytes(), stream);
}

// Fill bf16 tensor with a float value
__global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, __nv_bfloat16 val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = val;
}

// Fill f32 tensor with a float value
__global__ void fill_f32_kernel(float* __restrict__ dst, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = val;
}

void puf_fill(PufTensor& dst, float val, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        fill_bf16_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)dst.data, __float2bfloat16(val), dst.numel);
    } else {
        assert(dst.dtype_size == 4 && "puf_fill: expected bf16 or f32");
        fill_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
            (float*)dst.data, val, dst.numel);
    }
}

// Clamp bf16 in-place
__global__ void clamp_bf16_kernel(__nv_bfloat16* __restrict__ dst, float lo, float hi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(dst[idx]);
        dst[idx] = __float2bfloat16(fminf(fmaxf(v, lo), hi));
    }
}

void puf_clamp(PufTensor& dst, float lo, float hi, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        clamp_bf16_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)dst.data, lo, hi, dst.numel);
    } else {
        assert(false && "puf_clamp: only bf16 supported for now");
    }
}

__global__ void scale_f32_kernel(float* __restrict__ dst, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

void puf_scale(PufTensor& dst, float alpha, cudaStream_t stream) {
    assert(dst.dtype_size == 4 && "puf_scale: expected f32");
    scale_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, alpha, dst.numel);
}

__global__ void axpy_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

void puf_axpy(PufTensor& dst, const PufTensor& src, float alpha, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_axpy: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy: expected f32");
    axpy_f32_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, alpha, dst.numel);
}

// Device-pointer variants: read scalar from device memory (graph-safe)
__global__ void scale_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

void puf_scale_dev(PufTensor& dst, const float* alpha_ptr, cudaStream_t stream) {
    assert(dst.dtype_size == 4 && "puf_scale_dev: expected f32");
    scale_f32_dev_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, alpha_ptr, dst.numel);
}

__global__ void axpy_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                     const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

void puf_axpy_dev(PufTensor& dst, const PufTensor& src, const float* alpha_ptr, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_axpy_dev: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy_dev: expected f32");
    axpy_f32_dev_kernel<<<grid_size(dst.numel), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.data, (const float*)src.data, alpha_ptr, dst.numel);
}

__global__ void compute_lr_scalars_kernel(const float* __restrict__ lr, float wd,
                                           float* __restrict__ neg_lr, float* __restrict__ wd_scale) {
    *neg_lr = -(*lr);
    *wd_scale = 1.0f - (*lr) * wd;
}

void compute_lr_scalars(const float* lr_ptr, float weight_decay, float* neg_lr_ptr, float* wd_scale_ptr, cudaStream_t stream) {
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

void puf_add(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel == src.numel && "puf_add: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2 && "puf_add: expected fp32 += bf16");
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

void puf_sum_rows_add(PufTensor& dst, PufTensor& src, cudaStream_t stream) {
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_sum_rows_add: expected f32");
    int R = src.numel / src.shape[src.ndim - 1];
    int C = src.shape[src.ndim - 1];
    assert(dst.numel == C && "puf_sum_rows_add: dst must have C elements");
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

void puf_assemble_decoder_grad(PufTensor& dst, PufTensor& grad_logits, PufTensor& grad_value, cudaStream_t stream) {
    int B_TT = dst.size(0);
    int od_plus_1 = dst.size(1);
    int od = od_plus_1 - 1;

    int total = B_TT * od_plus_1;
    assemble_decoder_grad_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.data,
        (const float*)grad_logits.data,
        (const float*)grad_value.data,
        B_TT, od, od_plus_1);
}

/// var_mean: two-pass (mean then variance) using shared-memory block reduction
__global__ void var_mean_kernel(const float* __restrict__ src, float* __restrict__ var_out,
        float* __restrict__ mean_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    // Pass 1: compute sum for mean
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += src[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    if (tid == 0) *mean_out = mean;
    __syncthreads();

    // Pass 2: compute sum of squared deviations
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = src[i] - mean;
        ss += d * d;
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *var_out = sdata[0] / (float)(n - 1);  // unbiased (Bessel's correction)
}

void puf_var_mean(const PufTensor& src, float* var_out, float* mean_out, cudaStream_t stream) {
    assert(src.dtype_size == 4 && "puf_var_mean: expected f32");
    var_mean_kernel<<<1, 256, 0, stream>>>(
        (const float*)src.data, var_out, mean_out, src.numel);
}

// Add a scalar to a single device float
__global__ void add_scalar_kernel(float* __restrict__ ptr, float val) {
    *ptr += val;
}

void puf_add_scalar(float* ptr, float val, cudaStream_t stream) {
    add_scalar_kernel<<<1, 1, 0, stream>>>(ptr, val);
}

// Scatter rows: dst[idx[i], :] = src[i, :] for each i
__global__ void index_copy_kernel(char* __restrict__ dst, const int64_t* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int64_t dst_row = idx[i];
        memcpy(dst + dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
}

void puf_index_copy(PufTensor& dst, const PufTensor& idx, const PufTensor& src, cudaStream_t stream) {
    int num_idx = idx.numel;
    int row_bytes = src.numel / src.shape[0] * src.dtype_size;
    index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, stream>>>(
        (char*)dst.data, (const int64_t*)idx.data, (const char*)src.data, num_idx, row_bytes);
}

// Cast uint8 → precision_t
__global__ void cast_u8_to_precision_kernel(precision_t* __restrict__ dst,
        const unsigned char* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float((float)src[idx]);
}

void puf_cast_u8_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    cast_u8_to_precision_kernel<<<grid_size(src.numel), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)dst.data, (const unsigned char*)src.data, src.numel);
}

// Cast f32 → precision_t (identity if f32 mode, cast if bf16)
void puf_cast_f32_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        puf_cast_f32_to_bf16(dst, src, stream);
    } else {
        puf_copy(dst, src, stream);
    }
}

// PufTensor overload: all PufTensors
void mingru_gate(PufTensor& state, PufTensor& combined, PufTensor& out, PufTensor& next_state, cudaStream_t stream) {
    int B = static_cast<int>(state.size(0));
    int H = static_cast<int>(state.size(1));

    mingru_gate_inference_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)out.data,
        (precision_t*)next_state.data,
        (const precision_t*)combined.data,
        (const precision_t*)state.data,
        H, B);
}

// Prefix scan forward — writes into pre-allocated bufs
void prefix_scan_forward(PufTensor& combined, PufTensor& state, PrefixScan& bufs, cudaStream_t stream) {
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
    PufTensor& grad_out, PufTensor& grad_next_state, PrefixScan& bufs, cudaStream_t stream) {
    int B = bufs.B;
    int T = bufs.T;
    int H = bufs.H;

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

// =============================================================================
// Legacy torch-dependent functions (autograd, torch::Tensor wrappers)
// Declarations in legacy_modules.h — will be removed as we migrate to PufTensor
// =============================================================================
#ifdef PUFFERLIB_TORCH

using tensor_list = torch::autograd::tensor_list;
using AutogradCtx = torch::autograd::AutogradContext;

// Fused mingru gate inference (torch::Tensor overload)
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
    auto logstd_to_save = is_continuous ? logstd : torch::empty({0}, logits.options());

    auto device = logits.device();
    int N = static_cast<int>(logits.size(0));
    int T = static_cast<int>(logits.size(1));
    int A_total = static_cast<int>(logits.size(2));
    int num_atns = static_cast<int>(act_sizes.size(0));

    auto [adv_var, adv_mean] = torch::var_mean(advantages);
    auto actions_flat = actions.reshape({N * T, num_atns}).contiguous();

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto loss_output = torch::empty({1}, options_float);
    auto saved_for_backward = torch::zeros({N * T, 5}, options_double);
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
    auto logstd = saved[1];
    auto values_pred = saved[2].contiguous();
    auto actions_flat = saved[3].contiguous();
    auto old_logprobs = saved[4].contiguous();
    auto advantages = saved[5].contiguous();
    auto prio = saved[6].contiguous();
    auto values = saved[7].contiguous();
    auto returns = saved[8].contiguous();
    auto adv_mean = saved[9].contiguous();
    auto adv_var = saved[10].contiguous();
    auto act_sizes = saved[11];

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
    auto grad_logits = torch::empty(logits.sizes(), logits.options().dtype(torch::kFloat32));
    auto grad_values_pred = torch::empty(values_pred.sizes(), values_pred.options().dtype(torch::kFloat32));
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
        is_continuous ? grad_logstd : Tensor(),
        grad_values_pred,
        {}, {}, {}, {}, {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}
    };
}

// Fused PPO loss forward + backward (no autograd)
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

    TORCH_CHECK(old_logprobs.is_contiguous(), "old_logprobs must be contiguous");
    TORCH_CHECK(advantages.is_contiguous(), "advantages must be contiguous");
    TORCH_CHECK(prio.is_contiguous(), "prio must be contiguous");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(returns.is_contiguous(), "returns must be contiguous");

    bool is_continuous = logstd.defined() && logstd.numel() > 0;
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
#endif // PUFFERLIB_TORCH (ppo_loss_fwd_bwd Tensor overload)

// PufTensor overload of ppo_loss_fwd_bwd — no torch deps
void ppo_loss_fwd_bwd(
    PufTensor& logits, PufTensor& logstd, PufTensor& values_pred,
    PufTensor& actions, PufTensor& old_logprobs, PufTensor& advantages,
    PufTensor& prio, PufTensor& values, PufTensor& returns,
    PufTensor& ratio_out, PufTensor& newvalue_out,
    PufTensor& act_sizes, PufTensor& losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
    PPOBuffersPuf& bufs, bool is_continuous,
    int logits_stride_n, int logits_stride_t, int logits_stride_a,
    int values_stride_n, int values_stride_t,
    cudaStream_t stream
) {
    int N = logits.shape[0];
    int T = logits.shape[1];
    int A_total = logits.shape[2];
    int num_atns = act_sizes.numel;
    int total = N * T;

    // var_mean on advantages (f32) into scratch
    float* adv_var_ptr = (float*)bufs.adv_scratch.data;
    float* adv_mean_ptr = adv_var_ptr + 1;
    puf_var_mean(advantages, adv_var_ptr, adv_mean_ptr, stream);

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    cudaMemsetAsync((float*)bufs.loss_output.data, 0, sizeof(float), stream);
    puf_zero(bufs.saved_for_bwd, stream);

    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        (float*)bufs.loss_output.data,
        (float*)losses_acc.data,
        (double*)bufs.saved_for_bwd.data,
        (precision_t*)ratio_out.data,
        (precision_t*)newvalue_out.data,
        (const precision_t*)logits.data,
        is_continuous ? (const precision_t*)logstd.data : nullptr,
        (const precision_t*)values_pred.data,
        (double*)actions.data,
        (const precision_t*)old_logprobs.data,
        (float*)advantages.data,
        (const precision_t*)prio.data,
        (const precision_t*)values.data,
        (const precision_t*)returns.data,
        adv_mean_ptr,
        adv_var_ptr,
        (int*)act_sizes.data,
        num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef,
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);

    puf_add_scalar((float*)losses_acc.data + LOSS_N, 1.0f, stream);

    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        (float*)bufs.grad_logits.data,
        is_continuous ? (float*)bufs.grad_logstd.data : nullptr,
        (float*)bufs.grad_values.data,
        (float*)bufs.grad_loss.data,
        (const precision_t*)logits.data,
        is_continuous ? (const precision_t*)logstd.data : nullptr,
        (const precision_t*)values_pred.data,
        (double*)actions.data,
        (const precision_t*)old_logprobs.data,
        (float*)advantages.data,
        (const precision_t*)prio.data,
        (const precision_t*)values.data,
        (const precision_t*)returns.data,
        adv_mean_ptr,
        adv_var_ptr,
        (int*)act_sizes.data,
        num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef,
        T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a,
        values_stride_n, values_stride_t,
        is_continuous);
}

// Fused sample_logits: handles both discrete and continuous action sampling
// PufTensor version — logits may be strided (fused decoder output)
void sample_logits(
    PufTensor& logits, PufTensor& logstd, PufTensor& value,
    PufTensor& actions_out, PufTensor& logprobs_out, PufTensor& value_out,
    PufTensor& act_sizes, uint64_t seed, int64_t* offset_ptr,
    int logits_stride, int logstd_stride, int value_stride,
    cudaStream_t stream
) {
    bool is_continuous = logstd.data != nullptr && logstd.numel > 0;
    int B = actions_out.shape[0];
    int num_atns = act_sizes.numel;

    sample_logits_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        (double*)actions_out.data,
        (precision_t*)logprobs_out.data,
        (precision_t*)value_out.data,
        (const precision_t*)logits.data,
        is_continuous ? (const precision_t*)logstd.data : nullptr,
        (const precision_t*)value.data,
        (int*)act_sizes.data,
        seed,
        offset_ptr,
        num_atns, B, logits_stride, logstd_stride, value_stride,
        is_continuous);
}

#ifdef PUFFERLIB_TORCH
// Tensor overload for Python binding / legacy path
void sample_logits(
    Tensor logits, Tensor logstd, Tensor value,
    Tensor actions_out, Tensor logprobs_out, Tensor value_out,
    Tensor act_sizes, uint64_t seed, Tensor offset
) {
    PufTensor p_logits = PufTensor::from_torch(logits);
    PufTensor p_logstd = (logstd.defined() && logstd.numel() > 0) ? PufTensor::from_torch(logstd) : PufTensor();
    PufTensor p_value = PufTensor::from_torch(value);
    PufTensor p_actions = PufTensor::from_torch(actions_out);
    PufTensor p_logprobs = PufTensor::from_torch(logprobs_out);
    PufTensor p_value_out = PufTensor::from_torch(value_out);
    PufTensor p_act_sizes = PufTensor::from_torch(act_sizes);

    int logits_stride = static_cast<int>(logits.stride(0));
    int logstd_stride = (logstd.defined() && logstd.numel() > 0) ? static_cast<int>(logstd.stride(0)) : 0;
    int value_stride = static_cast<int>(value.stride(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sample_logits(p_logits, p_logstd, p_value, p_actions, p_logprobs, p_value_out,
        p_act_sizes, seed, offset.data_ptr<int64_t>(),
        logits_stride, logstd_stride, value_stride, stream);
}

// FCMax: fused FC -> Max (torch autograd)
tensor_list FCMax::forward(
    AutogradCtx* ctx, Tensor x, Tensor W, Tensor b
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
#endif // PUFFERLIB_TORCH (FCMax, sample_logits Tensor overload)

void train_select_and_copy_cuda(
    PufTensor& observations, PufTensor& actions,
    PufTensor& logprobs, PufTensor& values, PufTensor& advantages,
    PufTensor& idx, PufTensor& mb_prio,
    PufTensor& dst_obs, PufTensor& dst_state,
    PufTensor& dst_actions, PufTensor& dst_logprobs,
    PufTensor& dst_advantages, PufTensor& dst_prio,
    PufTensor& dst_values, PufTensor& dst_returns,
    cudaStream_t stream
) {
    int mb_segs = idx.shape[0];
    int horizon = values.shape[1];
    int obs_rb = (observations.numel / observations.shape[0]) * observations.dtype_size;
    int act_rb = (actions.numel / actions.shape[0]) * actions.dtype_size;
    int lp_rb = (logprobs.numel / logprobs.shape[0]) * logprobs.dtype_size;

    puf_zero(dst_state, stream);

    select_copy_kernel<<<dim3(mb_segs, 5), SELECT_COPY_THREADS, 0, stream>>>(
        (int64_t*)idx.data,
        (const char*)observations.data, (char*)dst_obs.data, obs_rb,
        (const char*)actions.data, (char*)dst_actions.data, act_rb,
        (const char*)logprobs.data, (char*)dst_logprobs.data, lp_rb,
        (const precision_t*)values.data, (precision_t*)dst_values.data,
        (float*)advantages.data, (float*)dst_advantages.data,
        (precision_t*)dst_returns.data, horizon,
        (float*)mb_prio.data, (precision_t*)dst_prio.data);
}

#ifdef PUFFERLIB_TORCH
tuple<Tensor, Tensor> prio_replay_cuda(
    Tensor advantages, float prio_alpha,
    int minibatch_segments, int total_agents, float anneal_beta
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
#endif // PUFFERLIB_TORCH (prio_replay_cuda Tensor overload)

// Multinomial with replacement: sample minibatch_segments indices from probs (S,)
// Single-block: thread 0 builds CDF, then each thread samples one index via binary search
__global__ void multinomial_with_replacement_kernel(
        int64_t* __restrict__ out_idx,
        const float* __restrict__ probs,
        int S, int num_samples,
        uint64_t seed, int64_t* __restrict__ offset_ptr) {
    extern __shared__ float shared_cdf[];
    int tid = threadIdx.x;

    // Thread 0 builds CDF in shared memory
    if (tid == 0) {
        float cum = 0.0f;
        for (int i = 0; i < S; i++) {
            cum += probs[i];
            shared_cdf[i] = cum;
        }
    }
    __syncthreads();

    // Each thread samples one index
    if (tid < num_samples) {
        int64_t off = atomicAdd((unsigned long long*)offset_ptr, 1ULL);
        // Inline philox-style RNG (same as sample_logits_kernel)
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, off, 0, &rng_state);
        float u = curand_uniform(&rng_state);

        // Binary search CDF
        int lo = 0, hi = S - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (shared_cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        out_idx[tid] = lo;
    }
}

// PufTensor overload of prio_replay — no torch deps
void prio_replay_cuda(
    PufTensor& advantages, float prio_alpha,
    int minibatch_segments, int total_agents, float anneal_beta,
    PrioBuffers& bufs, uint64_t seed, int64_t* offset_ptr,
    cudaStream_t stream
) {
    int S = advantages.shape[0];
    int T = advantages.shape[1];

    compute_prio_adv_reduction<<<S, PRIO_WARP_SIZE, 0, stream>>>(
        (float*)advantages.data, (float*)bufs.prio_probs.data,
        prio_alpha, T);

    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        (float*)bufs.prio_probs.data, S);

    // Multinomial with replacement using our own kernel
    int block = ((minibatch_segments + 31) / 32) * 32;  // round up to warp
    if (block < 32) block = 32;
    int smem = S * sizeof(float);
    multinomial_with_replacement_kernel<<<1, block, smem, stream>>>(
        (int64_t*)bufs.idx.data, (float*)bufs.prio_probs.data,
        S, minibatch_segments, seed, offset_ptr);

    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        (int64_t*)bufs.idx.data, (float*)bufs.prio_probs.data,
        (float*)bufs.mb_prio.data,
        total_agents, anneal_beta, minibatch_segments);
}

// =============================================================================
// Puff Advantage CUDA dispatch
// =============================================================================

namespace pufferlib {

void puff_advantage_cuda(PufTensor& values, PufTensor& rewards,
        PufTensor& dones, PufTensor& importance, PufTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip,
        cudaStream_t stream) {
    int num_steps = values.shape[0];
    int horizon = values.shape[1];
    assert(advantages.dtype_size == 4 && "advantages must be f32");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    constexpr int N = 16 / sizeof(precision_t);
    auto kernel = (horizon % N == 0)
        ? puff_advantage_kernel
        : puff_advantage_kernel_scalar;

    kernel<<<blocks, threads_per_block, 0, stream>>>(
        (precision_t*)values.data, (precision_t*)rewards.data,
        (precision_t*)dones.data, (precision_t*)importance.data,
        (float*)advantages.data,
        gamma, lambda, rho_clip, c_clip,
        num_steps, horizon);
}

#ifdef PUFFERLIB_TORCH
// Tensor overload for legacy cpp fallback path
void puff_advantage_cuda(Tensor values, Tensor rewards,
        Tensor dones, Tensor importance, Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    PufTensor v = PufTensor::from_torch(values);
    PufTensor r = PufTensor::from_torch(rewards);
    PufTensor d = PufTensor::from_torch(dones);
    PufTensor imp = PufTensor::from_torch(importance);
    PufTensor adv = PufTensor::from_torch(advantages);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    puff_advantage_cuda(v, r, d, imp, adv,
        (float)gamma, (float)lambda, (float)rho_clip, (float)c_clip, stream);
}
#endif // PUFFERLIB_TORCH (puff_advantage_cuda Tensor overload)

} // namespace pufferlib

#endif // PUFFERLIB_MODULES_CU
