#ifndef PUFFERLIB_KERNELS_CU
#define PUFFERLIB_KERNELS_CU

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>
#include <cublas_v2.h>
#include <curand.h>
#include <cassert>
#include <stdlib.h>

// PufferLib defaults to bf16, but float32 is supported with the --precision compile-time flag
#ifdef PRECISION_FLOAT
typedef float precision_t;
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_32F;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F;
#define NCCL_PRECISION ncclFloat
#define to_float(x) (x)
#define from_float(x) (x)
#else
typedef __nv_bfloat16 precision_t;
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;
static constexpr cudaDataType_t CUBLAS_PRECISION = CUDA_R_16BF;
static constexpr cublasComputeType_t CUBLAS_COMPUTE_PRECISION = CUBLAS_COMPUTE_32F_FAST_16BF;
#define NCCL_PRECISION ncclBfloat16
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif

#include "structs.h"

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

#define PPO_THREADS 256
#define SELECT_COPY_THREADS 256
#define MAX_ATN_HEADS 16


#define BLOCK_SIZE 256
inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

#define SEQ_SIZE 256
inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f
__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    return (x_scaled > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
}

__device__ __forceinline__ float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) {
        return grad_output;
    }
    float exp_beta_x = expf(beta_x);
    return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float relu_backward(float x, float grad_output) {
    return (x > 0.0f) ? grad_output : 0.0f;
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float sigmoid_backward(float x, float grad_output) {
    float sig = sigmoid(x);
    return grad_output * sig * (1.0f - sig);
}

__device__ __inline__ float fast_tanh(float x) {
    float v1 = fminf(fmaxf(x, -9.0f), 9.0f);
    float v2 = v1 * v1;
    float p = v2 * -2.76076847742355e-16f + 2.00018790482477e-13f;
    p = v2 * p + -8.60467152213735e-11f;
    p = v2 * p + 5.12229709037114e-08f;
    p = v2 * p + 1.48572235717979e-05f;
    p = v2 * p + 6.37261928875436e-04f;
    p = v2 * p + 4.89352455891786e-03f;
    p = v1 * p;
    float q = v2 * 1.19825839466702e-06f + 1.18534705686654e-04f;
    q = v2 * q + 2.26843463243900e-03f;
    q = v2 * q + 4.89352518554385e-03f;
    return p / q;
}

__device__ __inline__ float fast_sigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (fast_tanh(x * 0.5f) + 1.0f) * 0.5f));
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

__device__ __forceinline__ float logaddexp(float a, float b) {
    float m = fmaxf(a, b), diff = fminf(a, b) - m;
    return (diff < -88.0f) ? m : m + log1pf(__expf(diff));
}

__device__ __forceinline__ void copy_bytes(
    const char* __restrict__ src, char* __restrict__ dst,
    int src_row, int dst_row, int row_bytes
) {
    const int* soffset = (const int*)(src + (int64_t)src_row * row_bytes);
    int* doffset = (int*)(dst + (int64_t)dst_row * row_bytes);
    for (int i = threadIdx.x; i < row_bytes / 4; i += blockDim.x) {
        doffset[i] = soffset[i];
    }
}

// Transpose dims 0,1: [A, B, C] -> [B, A, C]. For 2D, pass C=1.
__global__ void transpose_102(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C), rem = idx % (B * C), b = rem / C, c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

// This exists for actions (currently fp64)
__global__ void transpose_102(double* __restrict__ dst,
        const double* __restrict__ src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int a = idx / (B * C), rem = idx % (B * C), b = rem / C, c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

__global__ void fill_precision_kernel(precision_t* __restrict__ dst, precision_t val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = val;
    }
}

__global__ void clamp_precision_kernel(precision_t* __restrict__ dst, float lo, float hi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = to_float(dst[idx]);
        dst[idx] = from_float(fminf(fmaxf(v, lo), hi));
    }
}

__global__ void add_kernel(float* __restrict__ dst, const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += to_float(src[idx]);
    }
}

__global__ void add_kernel(precision_t* __restrict__ dst, const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}

// Dense row-major GEMM: C(M,N) = alpha * op_a(A) @ op_b(B) + beta * C
// Strides derived from M, N, K assuming tightly packed row-major storage.
static inline void cublasGemmExDense(
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;

    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        void* workspace = nullptr;
        cudaMalloc(&workspace, 32 * 1024 * 1024);
        cublasSetWorkspace(handle, workspace, 32 * 1024 * 1024);
    }

    cublasSetStream(handle, stream);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE_PRECISION, CUBLAS_GEMM_DEFAULT);
}

// out(...,N) = a(...,K) @ b(N,K)^T  — leading dims folded into M
void puf_mm(PufTensor* a, PufTensor* b, PufTensor* out, cudaStream_t stream) {
    int M = a->batch_size() * a->shape[a->ndim()-2];
    int K = a->shape[a->ndim()-1];
    int N = b->shape[b->ndim()-2];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->bytes, b->bytes, out->bytes, stream);
}

// out(M,N) = a(...,M)^T @ b(...,N)  — leading dims folded into K
void puf_mm_tn(PufTensor* a, PufTensor* b, PufTensor* out, cudaStream_t stream) {
    int M = a->shape[a->ndim()-1];
    int K = a->batch_size() * a->shape[a->ndim()-2];
    int N = b->shape[b->ndim()-1];
    cublasGemmExDense(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->bytes, b->bytes, out->bytes, stream);
}

// out(...,N) = a(...,K) @ b(K,N)  — leading dims folded into M
void puf_mm_nn(PufTensor* a, PufTensor* b, PufTensor* out, cudaStream_t stream) {
    int M = a->batch_size() * a->shape[a->ndim()-2];
    int K = a->shape[a->ndim()-1];
    int N = b->shape[b->ndim()-1];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->bytes, b->bytes, out->bytes, stream);
}

static void puf_addmm_nn(PufTensor* a, PufTensor* b, PufTensor* out,
        float alpha, float beta, cudaStream_t stream) {
    int M = a->batch_size() * a->shape[a->ndim()-2];
    int K = a->shape[a->ndim()-1];
    int N = b->shape[b->ndim()-1];
    cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->bytes, b->bytes, out->bytes, stream, alpha, beta);
}

__global__ void cast_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(src[idx]);
    }
}

__global__ void cast_kernel(float* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

__global__ void cast_kernel(precision_t* __restrict__ dst,
        const unsigned char* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float((float)src[idx]);
    }
}

__global__ void cast_kernel(unsigned char* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = to_float(src[idx]);
    }
}

void puf_copy(PufTensor* dst, const PufTensor* src, cudaStream_t stream) {
    assert(dst->numel() == src->numel() && "puf_copy: size mismatch");
    assert(dst->dtype_size == src->dtype_size && "puf_copy: dtype mismatch");
    cudaMemcpyAsync(dst->bytes, src->bytes, dst->numel() * dst->dtype_size, cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PufTensor* dst, cudaStream_t stream) {
    cudaMemsetAsync(dst->bytes, 0, dst->numel() * dst->dtype_size, stream);
}

void puf_add(PufTensor* dst, const PufTensor* src, cudaStream_t stream) {
    assert(dst->numel() == src->numel() && "puf_add: size mismatch");
    assert(dst->dtype_size == 4 && "puf_add: dst must be f32");
    add_kernel<<<grid_size(dst->numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst->bytes, (const precision_t*)src->bytes, dst->numel());
}

void puf_kaiming_init(PufTensor* dst, float gain, ulong seed, cudaStream_t stream) {
    assert(dst->ndim() == 2);
    long rows = dst->shape[0], cols = dst->shape[1];
    assert(rows > 0 && cols > 0);
    long n = rows * cols;

    float std = gain / std::sqrt((float)cols);  // fan_in = cols for (out, in) layout

    long rand_count = (n % 2 == 0) ? n : n + 1;
    float* buf;
    cudaMalloc(&buf, rand_count * sizeof(float));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, buf, rand_count, 0.0f, std);
    curandDestroyGenerator(gen);

    if (dst->dtype_size == 4) {
        cudaMemcpyAsync(dst->bytes, buf, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>((precision_t*)dst->bytes, buf, n);
    }

    cudaFree(buf);
}

void alloc_register(Allocator* alloc, PufTensor* ptr) {
    alloc->regs = (PufTensor**)realloc(alloc->regs, (alloc->num_regs + 1) * sizeof(PufTensor*));
    alloc->regs[alloc->num_regs++] = ptr;
    alloc->total_elems += ptr->numel();
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += ptr->numel() * ptr->dtype_size;
}

void alloc_create(Allocator* alloc) {
    if (alloc->total_bytes == 0) {
        return;
    }
    cudaMalloc(&alloc->mem, alloc->total_bytes);
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        PufTensor* t = alloc->regs[i];
        offset = (offset + 15) & ~15;
        t->bytes = (char*)alloc->mem + offset;
        offset += t->numel() * t->dtype_size;
    }
}

void alloc_free(Allocator* alloc) {
    if (alloc->mem) { cudaFree(alloc->mem); alloc->mem = nullptr; }
    if (alloc->regs) { free(alloc->regs); alloc->regs = nullptr; }
    alloc->num_regs = 0;
    alloc->total_elems = 0;
}

#endif // PUFFERLIB_KERNELS_CU
