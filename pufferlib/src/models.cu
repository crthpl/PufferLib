#ifndef PUFFERLIB_MODELS_CU
#define PUFFERLIB_MODELS_CU

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cassert>
#include <vector>
#include <string>
#include <cstdint>
#include <nccl.h>

#include <stdio.h>
#include <stdlib.h>

using std::vector;

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;   // bytes per element
#else
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;   // bytes per element
#endif

// ============================================================================
// PufTensor — minimal tensor view (no torch dependency)
// ============================================================================

#define PUF_MAX_DIMS 8

// Minimal tensor: raw pointer + shape, no torch dependency in the struct itself.
// Memory is owned by an Allocator buffer — PufTensor is just a view.
struct PufTensor {
    char* bytes = nullptr;
    int64_t shape[PUF_MAX_DIMS] = {};
    int dtype_size = 0;      // bytes per element (2 for bf16/f16, 4 for f32, 8 for f64)

    __host__ __device__ int ndim() const {
        int n = 0;
        while (n < PUF_MAX_DIMS && shape[n] != 0) {
            n++;
        }
        return n;
    }

    __host__ __device__ int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
            n *= shape[i];
        }
        return n;
    }


    const char* dtype_name() const {
        switch (dtype_size) {
            case 1: return "i8";
            case 2: return "bf16";
            case 4: return "f32";
            case 8: return "f64";
            default: return "?";
        }
    }

    const char* repr() const {
        static char buf[256];
        if (!bytes) { snprintf(buf, sizeof(buf), "PufTensor(empty)"); return buf; }
        int pos = snprintf(buf, sizeof(buf), "PufTensor(%s, [", dtype_name());
        for (int i = 0; i < ndim() && pos < (int)sizeof(buf) - 32; i++) {
            pos += snprintf(buf + pos, sizeof(buf) - pos, "%s%lld", i ? ", " : "", (long long)shape[i]);
        }
        snprintf(buf + pos, sizeof(buf) - pos, "], %lld elems)", (long long)numel());
        return buf;
    }
};

// Loss component indices
enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_N = 7, NUM_LOSSES = 8,
};

// Prefix scan buffers
struct PrefixScan {
    void* combined_ptr;
    void* state_ptr;
    int B, T, H;
    PufTensor a_star, s_vals, log_values_buf;
    PufTensor out, next_state;
    PufTensor grad_combined, grad_state;
    PrefixScan() : combined_ptr(nullptr), state_ptr(nullptr), B(0), T(0), H(0) {}
};

// ============================================================================
// Allocator — single contiguous GPU buffer with PufTensor views
// ============================================================================

struct Allocator {
    struct Reg {
        PufTensor* ptr;
        int64_t size;
        std::vector<int64_t> shape;
        int elem_size;
    };
    std::vector<Reg> regs;
    void* mem = nullptr;
    int64_t total_elems = 0;

    void reg(PufTensor* ptr, std::vector<int64_t> shape, int esz) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        regs.push_back({ptr, size, shape, esz});
    }

    void create() {
        int64_t total_bytes = 0;
        total_elems = 0;
        for (auto& r : regs) {
            total_bytes += r.size * r.elem_size;
            total_elems += r.size;
        }
        if (total_bytes > 0) {
            cudaMalloc(&mem, total_bytes);
            cudaMemset(mem, 0, total_bytes);
            int64_t offset = 0;
            for (auto& r : regs) {
                r.ptr->bytes = (char*)mem + offset;
                r.ptr->dtype_size = r.elem_size;
                for (int i = 0; i < PUF_MAX_DIMS; i++)
                    r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 0;
                offset += r.size * r.elem_size;
            }
        }
    }

    void destroy() {
        if (mem) { cudaFree(mem); mem = nullptr; }
    }
};

// Groups 3 allocators for policy: params, grads, activations
struct AllocSet {
    Allocator params, grads, acts;
    int esz = 0;  // element size for params/grads
    void create() { params.create(); grads.create(); acts.create(); }
    void destroy() { params.destroy(); grads.destroy(); acts.destroy(); }
};

// Pre-allocated buffers for prio_replay
struct PrioBuffers {
    PufTensor prio_probs, cdf, idx, mb_prio;
};

void register_prio_buffers(PrioBuffers& bufs, Allocator& alloc, int S, int minibatch_segments) {
    alloc.reg(&bufs.prio_probs, {S}, sizeof(float));
    alloc.reg(&bufs.cdf, {S}, sizeof(float));
    alloc.reg(&bufs.idx, {minibatch_segments}, sizeof(int64_t));
    alloc.reg(&bufs.mb_prio, {minibatch_segments, 1}, sizeof(float));
}

// Pre-allocated buffers for PPO loss
struct PPOBuffersPuf {
    PufTensor loss_output, saved_for_bwd, grad_loss;
    PufTensor grad_logits, grad_values, grad_logstd, adv_scratch;
};

void register_ppo_buffers(PPOBuffersPuf& bufs, Allocator& alloc, int N, int T, int A_total, bool is_continuous) {
    int64_t total = (int64_t)N * T;
    alloc.reg(&bufs.loss_output, {1}, sizeof(float));
    alloc.reg(&bufs.saved_for_bwd, {total, 5}, sizeof(double));
    alloc.reg(&bufs.grad_loss, {1}, sizeof(float));
    alloc.reg(&bufs.grad_logits, {N, T, A_total}, sizeof(float));
    alloc.reg(&bufs.grad_values, {N, T, 1}, sizeof(float));
    if (is_continuous) alloc.reg(&bufs.grad_logstd, {N, T, A_total}, sizeof(float));
    alloc.reg(&bufs.adv_scratch, {2}, sizeof(float));
}

void post_create_ppo_buffers(PPOBuffersPuf& bufs) {
    float one = 1.0f;
    cudaMemcpy(bufs.grad_loss.bytes, &one, sizeof(float), cudaMemcpyHostToDevice);
}

// ============================================================================
// Native Policy, Muon optimizer, and supporting structs
// ============================================================================

struct ParamShape {
    int64_t numel;
    std::vector<int64_t> shape;
};

// ============================================================================
// Rollout and training graph buffers — used by both kernels and host code
// ============================================================================

struct RolloutBuf {
    PufTensor observations;  // (horizon, segments, input_size) PRECISION
    PufTensor actions;       // (horizon, segments, num_atns) f64
    PufTensor values;        // (horizon, segments) PRECISION
    PufTensor logprobs;      // (horizon, segments) PRECISION
    PufTensor rewards;       // (horizon, segments) PRECISION
    PufTensor terminals;     // (horizon, segments) PRECISION
    PufTensor ratio;         // (horizon, segments) PRECISION
    PufTensor importance;    // (horizon, segments) PRECISION

};

void register_rollout_buffers(RolloutBuf& bufs, Allocator& alloc, int dim0, int dim1, int input_size, int num_atns) {
    int psz = PRECISION_SIZE;
    alloc.reg(&bufs.observations, {dim0, dim1, input_size}, psz);
    alloc.reg(&bufs.actions, {dim0, dim1, num_atns}, sizeof(double));
    alloc.reg(&bufs.values, {dim0, dim1}, psz);
    alloc.reg(&bufs.logprobs, {dim0, dim1}, psz);
    alloc.reg(&bufs.rewards, {dim0, dim1}, psz);
    alloc.reg(&bufs.terminals, {dim0, dim1}, psz);
    alloc.reg(&bufs.ratio, {dim0, dim1}, psz);
    alloc.reg(&bufs.importance, {dim0, dim1}, psz);
}

struct TrainGraph {
    PufTensor mb_obs;         // (S, H, input_size) PRECISION
    PufTensor mb_state;       // (L, S, 1, hidden) PRECISION
    PufTensor mb_actions;     // (S, H, num_atns) f64
    PufTensor mb_logprobs;    // (S, H) PRECISION
    PufTensor mb_advantages;  // (S, H) f32
    PufTensor mb_prio;        // (S, 1) PRECISION
    PufTensor mb_values;      // (S, H) PRECISION
    PufTensor mb_returns;     // (S, H) PRECISION
    PufTensor mb_ratio;       // (S, H) PRECISION
    PufTensor mb_newvalue;    // (S, H, 1) PRECISION

};

void register_train_buffers(TrainGraph& bufs, Allocator& alloc, int S, int H, int input_size,
        int hidden_size, int num_atns, int num_layers) {
    int psz = PRECISION_SIZE;
    alloc.reg(&bufs.mb_obs, {S, H, input_size}, psz);
    alloc.reg(&bufs.mb_state, {num_layers, S, 1, hidden_size}, psz);
    alloc.reg(&bufs.mb_actions, {S, H, num_atns}, sizeof(double));
    alloc.reg(&bufs.mb_logprobs, {S, H}, psz);
    alloc.reg(&bufs.mb_advantages, {S, H}, sizeof(float));
    alloc.reg(&bufs.mb_prio, {S, 1}, psz);
    alloc.reg(&bufs.mb_values, {S, H}, psz);
    alloc.reg(&bufs.mb_returns, {S, H}, psz);
    alloc.reg(&bufs.mb_ratio, {S, H}, psz);
    alloc.reg(&bufs.mb_newvalue, {S, H, 1}, psz);
}

#include "kernels.cu"

// ============================================================================
// cuBLAS matmuls: all row-major PufTensors, bf16 with f32 compute
// ============================================================================

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        static void* workspace = nullptr;
        if (!workspace) cudaMalloc(&workspace, 4096 * 8);
        cublasSetWorkspace(handle, workspace, 4096 * 8);
    }
    return handle;
}

// out(M,N) = a(M,K) @ b(N,K)^T
void puf_mm(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int M = a.shape[0], K = a.shape[1], N = b.shape[0];
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
        b.bytes, CUDA_R_16BF, K, a.bytes, CUDA_R_16BF, K, &beta,
        out.bytes, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// out(M,N) = a(K,M)^T @ b(K,N)
void puf_mm_tn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int K = a.shape[0], M = a.shape[1], N = b.shape[1];
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha,
        b.bytes, CUDA_R_16BF, N, a.bytes, CUDA_R_16BF, M, &beta,
        out.bytes, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// out(M,N) = a(M,K) @ b(K,N)
void puf_mm_nn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream) {
    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        b.bytes, CUDA_R_16BF, N, a.bytes, CUDA_R_16BF, K, &beta,
        out.bytes, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static void puf_addmm_nn(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta, cudaStream_t stream) {
    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        b.bytes, CUDA_R_16BF, N, a.bytes, CUDA_R_16BF, K, &beta,
        out.bytes, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ============================================================================
// Orthogonal init (cuSOLVER + cuRAND)
// ============================================================================

void puf_orthogonal_init(PufTensor& dst, float gain, uint64_t seed, cudaStream_t stream) {
    assert(dst.ndim() == 2);
    int64_t rows = dst.shape[0], cols = dst.shape[1];
    assert(rows > 0 && cols > 0);

    bool transposed = rows < cols;
    int m = transposed ? (int)cols : (int)rows;
    int n = transposed ? (int)rows : (int)cols;

    float *A, *tau, *signs;
    int *devInfo;
    int64_t mn = (int64_t)m * n;
    int64_t rand_count = (mn % 2 == 0) ? mn : mn + 1;
    cudaMalloc(&A, rand_count * sizeof(float));
    cudaMalloc(&tau, n * sizeof(float));
    cudaMalloc(&signs, n * sizeof(float));
    cudaMalloc(&devInfo, sizeof(int));

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, A, rand_count, 0.0f, 1.0f);
    curandDestroyGenerator(gen);

    cusolverDnHandle_t solver;
    cusolverDnCreate(&solver);

    int lwork;
    cusolverDnSgeqrf_bufferSize(solver, m, n, A, m, &lwork);
    float* work;
    cudaMalloc(&work, lwork * sizeof(float));
    cusolverDnSgeqrf(solver, m, n, A, m, tau, work, lwork, devInfo);

    extract_diag_sign_kernel<<<grid_size(n), BLOCK_SIZE>>>(signs, A, m, n);

    int lwork2;
    cusolverDnSorgqr_bufferSize(solver, m, n, n, A, m, tau, &lwork2);
    if (lwork2 > lwork) {
        cudaFree(work);
        cudaMalloc(&work, lwork2 * sizeof(float));
    }
    cusolverDnSorgqr(solver, m, n, n, A, m, tau, work, lwork2, devInfo);

    sign_correct_columns_kernel<<<grid_size(mn), BLOCK_SIZE>>>(A, signs, m, n);

    if (transposed) {
        if (dst.dtype_size == 2) {
            cast_f32_to_bf16_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)dst.bytes, A, rows * cols);
        } else {
            cudaMemcpyAsync(dst.bytes, A, rows * cols * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }
        if (gain != 1.0f) {
            scale_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
                (float*)dst.bytes, gain, dst.numel());
        }
    } else {
        if (dst.dtype_size == 2) {
            colmaj_to_rowmaj_scale_bf16_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)dst.bytes, A, gain, rows, cols);
        } else {
            colmaj_to_rowmaj_scale_f32_kernel<<<grid_size(rows * cols), BLOCK_SIZE, 0, stream>>>(
                (float*)dst.bytes, A, gain, rows, cols);
        }
    }

    cudaFree(A);
    cudaFree(tau);
    cudaFree(signs);
    cudaFree(work);
    cudaFree(devInfo);
    cusolverDnDestroy(solver);
}

// ============================================================================
// PufTensor wrappers that dispatch to kernels
// ============================================================================

// Scratch buffer for partial sums in vector norm kernels (clip_grad_norm + Muon NS)
static float* norm_partials_buf = nullptr;
static void ensure_norm_partials() {
    if (!norm_partials_buf) cudaMalloc(&norm_partials_buf, 256 * sizeof(float));
}


void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_cast_f32_to_bf16: size mismatch");
    assert(dst.dtype_size == 2 && src.dtype_size == 4);
    cast_f32_to_bf16_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.bytes, (const float*)src.bytes, dst.numel());
}

void puf_transpose_01(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    int A = src.shape[0], B = src.shape[1];
    int C = (src.ndim() >= 3) ? src.shape[2] : 1;
    assert(dst.shape[0] == B && dst.shape[1] == A);
    assert(dst.dtype_size == src.dtype_size);
    int n = A * B * C;
    switch (src.dtype_size) {
        case 2: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint16_t*)dst.bytes, (const uint16_t*)src.bytes, A, B, C); break;
        case 4: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint32_t*)dst.bytes, (const uint32_t*)src.bytes, A, B, C); break;
        case 8: transpose_01_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (uint64_t*)dst.bytes, (const uint64_t*)src.bytes, A, B, C); break;
        default: assert(false && "puf_transpose_01: unsupported dtype_size");
    }
}

void puf_copy(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_copy: size mismatch");
    assert(dst.dtype_size == src.dtype_size && "puf_copy: dtype mismatch");
    cudaMemcpyAsync(dst.bytes, src.bytes, dst.numel() * dst.dtype_size, cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PufTensor& dst, cudaStream_t stream) {
    cudaMemsetAsync(dst.bytes, 0, dst.numel() * dst.dtype_size, stream);
}

void puf_add(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_add: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2 && "puf_add: expected fp32 += bf16");
    add_bf16_to_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, (const __nv_bfloat16*)src.bytes, dst.numel());
}

// ============================================================================
// High-level PufTensor orchestration (dispatch to kernels)
// ============================================================================

void ppo_loss_fwd_bwd(
    PufTensor& dec_out,          // (N, T, fused_cols) — fused logits+value from decoder
    PufTensor& logstd,           // continuous logstd or empty
    TrainGraph& graph,
    PufTensor& act_sizes, PufTensor& losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
    PPOBuffersPuf& bufs, bool is_continuous,
    cudaStream_t stream
) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int num_atns = act_sizes.numel();
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    // Strides for fused (N, T, logits|value) layout
    int logits_stride_n = T * fused_cols;
    int logits_stride_t = fused_cols;
    int logits_stride_a = 1;
    int values_stride_n = T * fused_cols;
    int values_stride_t = fused_cols;

    // Pointers into fused decoder output
    const precision_t* logits_ptr = (const precision_t*)dec_out.bytes;
    const precision_t* values_pred_ptr = logits_ptr + A_total;
    const precision_t* logstd_ptr = is_continuous ? (const precision_t*)logstd.bytes : nullptr;

    float* adv_var_ptr = (float*)bufs.adv_scratch.bytes;
    float* adv_mean_ptr = adv_var_ptr + 1;
    var_mean_kernel<<<1, 256, 0, stream>>>(
        (const float*)graph.mb_advantages.bytes, adv_var_ptr, adv_mean_ptr, graph.mb_advantages.numel());

    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    static float* ppo_partials_buf = nullptr;
    static int ppo_partials_capacity = 0;
    int ppo_partials_needed = ppo_grid * (LOSS_N + 1);
    if (!ppo_partials_buf || ppo_partials_needed > ppo_partials_capacity) {
        if (ppo_partials_buf) cudaFree(ppo_partials_buf);
        ppo_partials_capacity = ppo_partials_needed;
        cudaMalloc(&ppo_partials_buf, ppo_partials_capacity * sizeof(float));
    }

    cudaMemsetAsync((float*)bufs.loss_output.bytes, 0, sizeof(float), stream);

    ppo_loss_fwd_bwd_kernel<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        ppo_partials_buf,
        (float*)bufs.grad_logits.bytes, is_continuous ? (float*)bufs.grad_logstd.bytes : nullptr,
        (float*)bufs.grad_values.bytes,
        logits_ptr, logstd_ptr,
        values_pred_ptr, (double*)graph.mb_actions.bytes,
        (const precision_t*)graph.mb_logprobs.bytes, (float*)graph.mb_advantages.bytes,
        (const precision_t*)graph.mb_prio.bytes, (const precision_t*)graph.mb_values.bytes, (const precision_t*)graph.mb_returns.bytes,
        adv_mean_ptr, adv_var_ptr, (int*)act_sizes.bytes, num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef, T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a, values_stride_n, values_stride_t, is_continuous);

    ppo_loss_reduce_kernel<<<1, LOSS_N + 1, 0, stream>>>(
        (float*)bufs.loss_output.bytes, (float*)losses_acc.bytes, ppo_partials_buf, ppo_grid);
}



// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_with_replacement_kernel(
        int64_t* __restrict__ out_idx, const float* __restrict__ probs,
        int S, int num_samples, uint64_t seed, int64_t* __restrict__ offset_ptr) {
    extern __shared__ float shared_cdf[];
    int tid = threadIdx.x;
    if (tid == 0) {
        float cum = 0.0f;
        for (int i = 0; i < S; i++) { cum += probs[i]; shared_cdf[i] = cum; }
    }
    __syncthreads();
    if (tid < num_samples) {
        uint64_t base_off = *offset_ptr;
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, base_off + tid, 0, &rng_state);
        float u = curand_uniform(&rng_state);
        int lo = 0, hi = S - 1;
        while (lo < hi) { int mid = (lo + hi) / 2; if (shared_cdf[mid] < u) lo = mid + 1; else hi = mid; }
        out_idx[tid] = lo;
    }
    if (tid == 0) atomicAdd((unsigned long long*)offset_ptr, (unsigned long long)num_samples);
}

void prio_replay_cuda(
    PufTensor& advantages, float prio_alpha,
    int minibatch_segments, int total_agents, float anneal_beta,
    PrioBuffers& bufs, uint64_t seed, int64_t* offset_ptr, cudaStream_t stream
) {
    int S = advantages.shape[0], T = advantages.shape[1];
    compute_prio_adv_reduction<<<S, PRIO_WARP_SIZE, 0, stream>>>(
        (float*)advantages.bytes, (float*)bufs.prio_probs.bytes, prio_alpha, T);
    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        (float*)bufs.prio_probs.bytes, S);
    int block = ((minibatch_segments + 31) / 32) * 32;
    if (block < 32) block = 32;
    multinomial_with_replacement_kernel<<<1, block, S * (int)sizeof(float), stream>>>(
        (int64_t*)bufs.idx.bytes, (float*)bufs.prio_probs.bytes, S, minibatch_segments, seed, offset_ptr);
    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        (int64_t*)bufs.idx.bytes, (float*)bufs.prio_probs.bytes,
        (float*)bufs.mb_prio.bytes, total_agents, anneal_beta, minibatch_segments);
}

void puff_advantage_cuda(PufTensor& values, PufTensor& rewards,
        PufTensor& dones, PufTensor& importance, PufTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip, cudaStream_t stream) {
    int num_steps = values.shape[0], horizon = values.shape[1];
    assert(advantages.dtype_size == 4 && "advantages must be f32");
    int blocks = (num_steps + 255) / 256;
    constexpr int N = 16 / sizeof(precision_t);
    auto kernel = (horizon % N == 0) ? puff_advantage_kernel : puff_advantage_kernel_scalar;
    kernel<<<blocks, 256, 0, stream>>>(
        (precision_t*)values.bytes, (precision_t*)rewards.bytes,
        (precision_t*)dones.bytes, (precision_t*)importance.bytes,
        (float*)advantages.bytes, gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

// ============================================================================
// Policy, MinGRU, Encoder, Decoder, Muon — defined after kernels/helpers
// so method bodies can use kernel launches and cuBLAS wrappers inline.
// ============================================================================
//
struct EncoderRolloutBuffer {
    PufTensor out;           // (B_inf, out_dim) — inference mm_out dest
};

struct EncoderTrainBuffer {
    PufTensor out;           // (B_TT, out_dim) — mm_out dest, reused as grad buffer in backward
    PufTensor saved_input;   // (B_TT, in_dim) — saved for backward
    PufTensor wgrad_scratch; // (out_dim, in_dim) — bf16 weight grad output
};

// Function pointer types for swappable encoders
struct Encoder;
typedef void (*encoder_forward_fn)(Encoder* e, PufTensor input, PufTensor out, cudaStream_t stream);
typedef void (*encoder_backward_fn)(Encoder* e, PufTensor grad, PufTensor saved_input, PufTensor wgrad, cudaStream_t stream);
typedef void (*encoder_init_weights_fn)(Encoder* e, uint64_t* seed, cudaStream_t stream);
typedef void (*encoder_reg_params_fn)(Encoder* e, Allocator* alloc, int esz);
typedef void (*encoder_reg_train_fn)(Encoder* e, EncoderTrainBuffer* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*encoder_reg_rollout_fn)(Encoder* e, EncoderRolloutBuffer* buf, Allocator* alloc, int B);

struct Encoder {
    PufTensor weight;
    int in_dim, out_dim;
    encoder_forward_fn forward;
    encoder_backward_fn backward;
    encoder_init_weights_fn init_weights;
    encoder_reg_params_fn reg_params;
    encoder_reg_train_fn reg_train;
    encoder_reg_rollout_fn reg_rollout;
};

// --- Linear encoder implementation ---

static void linear_encoder_forward(Encoder* e, PufTensor input, PufTensor out, cudaStream_t stream) {
    puf_mm(input, e->weight, out, stream);
}

static void linear_encoder_backward(Encoder* e, PufTensor grad, PufTensor saved_input, PufTensor wgrad, cudaStream_t stream) {
    puf_mm_tn(grad, saved_input, wgrad, stream);
}

static void linear_encoder_init_weights(Encoder* e, uint64_t* seed, cudaStream_t stream) {
    PufTensor w = {.bytes = e->weight.bytes, .shape = {e->out_dim, e->in_dim}, .dtype_size = e->weight.dtype_size};
    puf_orthogonal_init(w, std::sqrt(2.0f), (*seed)++, stream);
}

static void linear_encoder_reg_params(Encoder* e, Allocator* alloc, int esz) {
    alloc->reg(&e->weight, {e->out_dim, e->in_dim}, esz);
}

static void linear_encoder_reg_train(Encoder* e, EncoderTrainBuffer* buf, Allocator* acts, Allocator* grads, int B_TT) {
    int psz = PRECISION_SIZE;
    acts->reg(&buf->out, {B_TT, e->out_dim}, psz);
    acts->reg(&buf->saved_input, {B_TT, e->in_dim}, psz);
    grads->reg(&buf->wgrad_scratch, {e->out_dim, e->in_dim}, psz);
}

static void linear_encoder_reg_rollout(Encoder* e, EncoderRolloutBuffer* buf, Allocator* alloc, int B) {
    alloc->reg(&buf->out, {B, e->out_dim}, PRECISION_SIZE);
}


struct DecoderRolloutBuffer {
    PufTensor out;           // (B_inf, output+1) — inference mm_out dest
};

struct DecoderTrainBuffer {
    PufTensor out;            // (B_TT, output+1) — mm_out dest
    PufTensor saved_input;    // (B_TT, hidden) — saved for backward
    PufTensor grad_out;       // (B_TT, output+1) — fused grad from PPO (assembled by kernel)
    PufTensor wgrad_scratch;  // (output+1, hidden) — bf16 weight grad output
    PufTensor logstd_scratch; // (1, output) — bf16 logstd grad (continuous only, in grads allocator)
};

// Function pointer types for swappable decoders
struct Decoder;
typedef void (*decoder_forward_fn)(Decoder* d, PufTensor input, PufTensor out, cudaStream_t stream);
typedef void (*decoder_backward_fn)(Decoder* d, PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
    PufTensor saved_input, PufTensor grad_out, PufTensor wgrad, PufTensor logstd_scratch,
    PufTensor grad_input, cudaStream_t stream);
typedef void (*decoder_init_weights_fn)(Decoder* d, uint64_t* seed, cudaStream_t stream);
typedef void (*decoder_reg_params_fn)(Decoder* d, Allocator* alloc, int esz);
typedef void (*decoder_reg_train_fn)(Decoder* d, DecoderTrainBuffer* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*decoder_reg_rollout_fn)(Decoder* d, DecoderRolloutBuffer* buf, Allocator* alloc, int B);

struct Decoder {
    PufTensor weight;
    PufTensor logstd;
    int hidden_dim, output_dim;
    bool continuous;
    decoder_forward_fn forward;
    decoder_backward_fn backward;
    decoder_init_weights_fn init_weights;
    decoder_reg_params_fn reg_params;
    decoder_reg_train_fn reg_train;
    decoder_reg_rollout_fn reg_rollout;
};

// --- Linear decoder implementation ---

static void linear_decoder_forward(Decoder* d, PufTensor input, PufTensor out, cudaStream_t stream) {
    puf_mm(input, d->weight, out, stream);
}

static void linear_decoder_init_weights(Decoder* d, uint64_t* seed, cudaStream_t stream) {
    PufTensor w = {.bytes = d->weight.bytes, .shape = {d->output_dim + 1, d->hidden_dim}, .dtype_size = d->weight.dtype_size};
    puf_orthogonal_init(w, 0.01f, (*seed)++, stream);
}

static void linear_decoder_reg_params(Decoder* d, Allocator* alloc, int esz) {
    alloc->reg(&d->weight, {d->output_dim + 1, d->hidden_dim}, esz);
    if (d->continuous) alloc->reg(&d->logstd, {1, d->output_dim}, esz);
}

static void linear_decoder_reg_train(Decoder* d, DecoderTrainBuffer* buf, Allocator* acts, Allocator* grads, int B_TT) {
    int psz = PRECISION_SIZE;
    acts->reg(&buf->out, {B_TT, d->output_dim + 1}, psz);
    acts->reg(&buf->saved_input, {B_TT, d->hidden_dim}, psz);
    acts->reg(&buf->grad_out, {B_TT, d->output_dim + 1}, psz);
    grads->reg(&buf->wgrad_scratch, {d->output_dim + 1, d->hidden_dim}, psz);
    if (d->continuous) grads->reg(&buf->logstd_scratch, {1, d->output_dim}, psz);
}

static void linear_decoder_reg_rollout(Decoder* d, DecoderRolloutBuffer* buf, Allocator* alloc, int B) {
    alloc->reg(&buf->out, {B, d->output_dim + 1}, PRECISION_SIZE);
}

static void linear_decoder_backward(Decoder* d, PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
    PufTensor saved_input, PufTensor grad_out, PufTensor wgrad, PufTensor logstd_scratch,
    PufTensor grad_input, cudaStream_t stream) {
    int B_TT = saved_input.shape[0];
    int od = d->output_dim, od1 = od + 1;
    assemble_decoder_grad_kernel<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)grad_out.bytes, (const float*)grad_logits.bytes,
        (const float*)grad_value.bytes, B_TT, od, od1);
    puf_mm_tn(grad_out, saved_input, wgrad, stream);
    if (d->continuous && grad_logstd.bytes != nullptr) {
        sum_rows_to_bf16_kernel<<<grid_size(d->output_dim), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)logstd_scratch.bytes, (const float*)grad_logstd.bytes,
            B_TT, d->output_dim);
    }
    puf_mm_nn(grad_out, d->weight, grad_input, stream);
}


struct MinGRURolloutBuffer {
    int num_layers;
    vector<PufTensor> combined;  // per-layer (B_inf, 3*H)
    PufTensor out;               // (B_inf, H)
    PufTensor next_state;        // (B_inf, H)
};

struct MinGRUTrainBuffer {
    int num_layers;
    vector<PufTensor> saved_inputs;    // per-layer (B, TT, H)
    vector<PrefixScan> scan_bufs;      // per-layer scan state
    vector<PufTensor> combined_bufs;   // per-layer (B_TT, 3*H)
    vector<PufTensor> wgrad_scratch;   // per-layer (3*H, H) bf16 weight grad output
    PufTensor grad_input_buf;          // (B_TT, H)
    PufTensor grad_next_state;         // (B, 1, H)
};

struct MinGRU {
    int hidden, num_layers;
    vector<PufTensor> weights;
};

void mingru_init(MinGRU* m, Allocator& params, int hidden, int num_layers, int esz) {
    m->hidden = hidden;
    m->num_layers = num_layers;
    m->weights.resize(num_layers);
    for (int i = 0; i < num_layers; i++) {
        params.reg(&m->weights[i], {3 * hidden, hidden}, esz);
    }
}

void mingru_register_train(MinGRU* m, Allocator& alloc, MinGRUTrainBuffer& buf, int B, int TT) {
    int H = m->hidden, B_TT = B * TT, psz = PRECISION_SIZE;
    buf.num_layers = m->num_layers;
    buf.saved_inputs.resize(m->num_layers);
    buf.scan_bufs.resize(m->num_layers);
    buf.combined_bufs.resize(m->num_layers);
    buf.wgrad_scratch.resize(m->num_layers);
    alloc.reg(&buf.grad_input_buf, {B_TT, H}, psz);
    alloc.reg(&buf.grad_next_state, {B, 1, H}, psz);
    for (int i = 0; i < m->num_layers; i++) {
        alloc.reg(&buf.saved_inputs[i], {B, TT, H}, psz);
        alloc.reg(&buf.combined_bufs[i], {B_TT, 3 * H}, psz);
        alloc.reg(&buf.scan_bufs[i].out, {B, TT, H}, psz);
        alloc.reg(&buf.scan_bufs[i].next_state, {B, 1, H}, psz);
        alloc.reg(&buf.scan_bufs[i].a_star, {B, TT + 1, H}, sizeof(float));
        alloc.reg(&buf.scan_bufs[i].s_vals, {B, TT + 1, H}, sizeof(float));
        alloc.reg(&buf.scan_bufs[i].log_values_buf, {B, TT + 1, H}, sizeof(float));
        alloc.reg(&buf.scan_bufs[i].grad_combined, {B, TT, 3 * H}, psz);
        alloc.reg(&buf.scan_bufs[i].grad_state, {B, 1, H}, psz);
    }
}

void mingru_register_rollout(MinGRU* m, Allocator& alloc, MinGRURolloutBuffer& buf, int B_inf) {
    int H = m->hidden, dsz = PRECISION_SIZE;
    buf.num_layers = m->num_layers;
    buf.combined.resize(m->num_layers);
    for (int i = 0; i < m->num_layers; i++)
        alloc.reg(&buf.combined[i], {B_inf, 3 * H}, dsz);
    alloc.reg(&buf.out, {B_inf, H}, dsz);
    alloc.reg(&buf.next_state, {B_inf, H}, dsz);
}

PufTensor mingru_state_layer(MinGRU* m, PufTensor& state, int i) {
    int64_t B = state.shape[1], H = state.shape[2];
    return {.bytes = state.bytes + i * B * H * state.dtype_size, .shape = {B, H}, .dtype_size = state.dtype_size};
}

void mingru_append_param_shapes(MinGRU* m, vector<ParamShape>& out) {
    for (int i = 0; i < m->num_layers; i++) {
        vector<int64_t> s(m->weights[i].shape, m->weights[i].shape + m->weights[i].ndim());
        out.push_back({m->weights[i].numel(), s});
    }
}

void mingru_init_weights(MinGRU* m, uint64_t& seed, cudaStream_t stream) {
    for (int i = 0; i < m->num_layers; i++) {
        PufTensor w2d = {.bytes = m->weights[i].bytes, .shape = {3 * m->hidden, m->hidden}, .dtype_size = m->weights[i].dtype_size};
        puf_orthogonal_init(w2d, 1.0f, seed++, stream);
    }
}

PufTensor mingru_forward(MinGRU* m, PufTensor x, PufTensor state, MinGRURolloutBuffer& buf, cudaStream_t stream) {
    for (int i = 0; i < m->num_layers; i++) {
        PufTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(x, m->weights[i], buf.combined[i], stream);
        int Bi = static_cast<int>(state_i.shape[0]);
        int Hi = static_cast<int>(state_i.shape[1]);
        mingru_gate_inference_kernel<<<grid_size(Bi * Hi), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)buf.out.bytes, (precision_t*)buf.next_state.bytes,
            (const precision_t*)buf.combined[i].bytes, (const precision_t*)state_i.bytes, Hi, Bi);
        puf_copy(state_i, buf.next_state, stream);
        x = buf.out;
    }
    return x;
}

PufTensor mingru_forward_train(MinGRU* m, PufTensor x, PufTensor state, MinGRUTrainBuffer& buf, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(buf.saved_inputs[i], x, stream);
        PufTensor state_i = mingru_state_layer(m, state, i);
        PufTensor state_3d = {.bytes = state_i.bytes, .shape = {B, 1, m->hidden}, .dtype_size = state_i.dtype_size};
        PufTensor x_flat = {.bytes = x.bytes, .shape = {B * TT, m->hidden}, .dtype_size = x.dtype_size};
        puf_mm(x_flat, m->weights[i], buf.combined_bufs[i], stream);
        PufTensor combined_3d = {.bytes = buf.combined_bufs[i].bytes, .shape = {B, TT, 3 * m->hidden}, .dtype_size = buf.combined_bufs[i].dtype_size};
        buf.scan_bufs[i].combined_ptr = combined_3d.bytes;
        buf.scan_bufs[i].state_ptr = state_3d.bytes;
        buf.scan_bufs[i].B = B; buf.scan_bufs[i].T = TT; buf.scan_bufs[i].H = m->hidden;
        fused_scan_forward_kernel_checkpointed<<<grid_size(B * m->hidden), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)buf.scan_bufs[i].out.bytes, (precision_t*)buf.scan_bufs[i].next_state.bytes,
            (float*)buf.scan_bufs[i].a_star.bytes, (float*)buf.scan_bufs[i].s_vals.bytes,
            (float*)buf.scan_bufs[i].log_values_buf.bytes,
            (const precision_t*)combined_3d.bytes, (const precision_t*)state_3d.bytes, TT, m->hidden, B);
        x = buf.scan_bufs[i].out;
    }
    return x;
}

PufTensor mingru_backward(MinGRU* m, PufTensor grad, MinGRUTrainBuffer& buf, cudaStream_t stream) {
    int B = grad.shape[0], TT = grad.shape[1], H = grad.shape[2];
    for (int i = m->num_layers - 1; i >= 0; i--) {
        fused_scan_backward_kernel_checkpointed<<<grid_size(buf.scan_bufs[i].B * buf.scan_bufs[i].H), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)buf.scan_bufs[i].grad_combined.bytes, (precision_t*)buf.scan_bufs[i].grad_state.bytes,
            (const precision_t*)grad.bytes, (const precision_t*)buf.grad_next_state.bytes,
            (const precision_t*)buf.scan_bufs[i].combined_ptr, (const precision_t*)buf.scan_bufs[i].state_ptr,
            (float*)buf.scan_bufs[i].a_star.bytes, (float*)buf.scan_bufs[i].s_vals.bytes,
            (float*)buf.scan_bufs[i].log_values_buf.bytes,
            buf.scan_bufs[i].T, buf.scan_bufs[i].H, buf.scan_bufs[i].B);
        PufTensor gc_flat = {.bytes = buf.scan_bufs[i].grad_combined.bytes, .shape = {B * TT, 3 * H}, .dtype_size = buf.scan_bufs[i].grad_combined.dtype_size};
        PufTensor inp_flat = {.bytes = buf.saved_inputs[i].bytes, .shape = {B * TT, H}, .dtype_size = buf.saved_inputs[i].dtype_size};
        puf_mm_tn(gc_flat, inp_flat, buf.wgrad_scratch[i], stream);
        puf_mm_nn(gc_flat, m->weights[i], buf.grad_input_buf, stream);
        grad = {.bytes = buf.grad_input_buf.bytes, .shape = {B, TT, H}, .dtype_size = buf.grad_input_buf.dtype_size};
    }
    return grad;
}

struct PolicyRolloutBuffer {
    EncoderRolloutBuffer enc;
    DecoderRolloutBuffer dec;
    MinGRURolloutBuffer rnn;
};

struct PolicyTrainBuffer {
    EncoderTrainBuffer enc;
    DecoderTrainBuffer dec;
    MinGRUTrainBuffer rnn;
};

struct Policy {
    Encoder encoder;
    Decoder decoder;
    MinGRU rnn;
    int num_atns;
    PolicyTrainBuffer train;
};


PufTensor policy_forward(Policy* p, PufTensor obs, PufTensor state, PolicyRolloutBuffer& buf, cudaStream_t stream) {
    p->encoder.forward(&p->encoder, obs, buf.enc.out, stream);
    PufTensor h = mingru_forward(&p->rnn, buf.enc.out, state, buf.rnn, stream);
    p->decoder.forward(&p->decoder, h, buf.dec.out, stream);
    return buf.dec.out;
}

PufTensor policy_forward_train(Policy* p, PufTensor x, PufTensor state, PolicyTrainBuffer& buf, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    PufTensor x_flat = {.bytes = x.bytes, .shape = {B * TT, p->encoder.in_dim}, .dtype_size = x.dtype_size};
    puf_copy(buf.enc.saved_input, x_flat, stream);
    p->encoder.forward(&p->encoder, buf.enc.saved_input, buf.enc.out, stream);
    PufTensor h = {.bytes = buf.enc.out.bytes, .shape = {B, TT, p->encoder.out_dim}, .dtype_size = buf.enc.out.dtype_size};
    h = mingru_forward_train(&p->rnn, h, state, buf.rnn, stream);
    PufTensor flat_h = {.bytes = h.bytes, .shape = {B * TT, p->encoder.out_dim}, .dtype_size = h.dtype_size};
    puf_copy(buf.dec.saved_input, flat_h, stream);
    p->decoder.forward(&p->decoder, flat_h, buf.dec.out, stream);
    PufTensor result = {.bytes = buf.dec.out.bytes, .shape = {B, TT, p->decoder.output_dim + 1}, .dtype_size = buf.dec.out.dtype_size};
    return result;
}

void policy_backward(Policy* p, PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
              PolicyTrainBuffer& buf, cudaStream_t stream) {
    int B_TT = buf.dec.saved_input.shape[0];
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    PufTensor gl_flat = {.bytes = grad_logits.bytes, .shape = {B_TT, p->decoder.output_dim}, .dtype_size = grad_logits.dtype_size};
    PufTensor gv_flat = {.bytes = grad_value.bytes, .shape = {B_TT}, .dtype_size = grad_value.dtype_size};
    p->decoder.backward(&p->decoder, gl_flat, grad_logstd, gv_flat,
        buf.dec.saved_input, buf.dec.grad_out, buf.dec.wgrad_scratch, buf.dec.logstd_scratch,
        buf.enc.out, stream);
    PufTensor grad_h = {.bytes = buf.enc.out.bytes, .shape = {B, TT, p->encoder.out_dim}, .dtype_size = buf.enc.out.dtype_size};
    grad_h = mingru_backward(&p->rnn, grad_h, buf.rnn, stream);
    PufTensor grad_enc = {.bytes = grad_h.bytes, .shape = {B_TT, p->encoder.out_dim}, .dtype_size = grad_h.dtype_size};
    p->encoder.backward(&p->encoder, grad_enc, buf.enc.saved_input, buf.enc.wgrad_scratch, stream);
}


inline float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct NSScratch {
    PufTensor x, A, gram, tmp;
    PufTensor result_f32;
    float* norm_ptr;
    int64_t max_M, max_N;
};

PufTensor ns_slice(PufTensor& buf, int64_t rows, int64_t cols) {
    return {.bytes = buf.bytes, .shape = {rows, cols}, .dtype_size = buf.dtype_size};
}

struct Muon {
    double momentum, weight_decay, eps;
    float lr_val_init;
    float* lr_ptr;
    float* lr_derived_ptr;
    PufTensor lr_puf, lr_derived_puf, ns_norm_puf;
    PufTensor wb_puf, mb_puf, gc_puf, up_puf;
    NSScratch ns;
    std::vector<ParamShape> param_shapes;
    ncclComm_t nccl_comm;
    int world_size;
};

void muon_init(Muon* m, std::vector<ParamShape> param_shapes, PufTensor weight_buffer,
               double lr_val, double momentum, double eps, double weight_decay) {
    m->momentum = momentum;
    m->weight_decay = weight_decay;
    m->eps = eps;
    m->lr_val_init = (float)lr_val;
    m->lr_ptr = nullptr;
    m->lr_derived_ptr = nullptr;
    m->wb_puf = weight_buffer;
    m->param_shapes = std::move(param_shapes);
    m->nccl_comm = nullptr;
    m->world_size = 1;
    m->ns = {};
}

void muon_register_buffers(Muon* m, Allocator& alloc) {
    int64_t n = m->wb_puf.numel();
    alloc.reg(&m->lr_puf, {1}, sizeof(float));
    alloc.reg(&m->lr_derived_puf, {2}, sizeof(float));
    alloc.reg(&m->mb_puf, {n}, sizeof(float));
    alloc.reg(&m->gc_puf, {n}, sizeof(float));
    alloc.reg(&m->up_puf, {n}, sizeof(float));
    int64_t max_M = 0, max_N = 0;
    for (auto& ps : m->param_shapes) {
        if (ps.shape.size() >= 2) {
            int64_t R = ps.shape[0], C = ps.numel / R;
            max_M = std::max(max_M, std::min(R, C));
            max_N = std::max(max_N, std::max(R, C));
        }
    }
    if (max_M > 0) {
        m->ns.max_M = max_M; m->ns.max_N = max_N;
        alloc.reg(&m->ns.x, {max_M, max_N}, 2);
        alloc.reg(&m->ns.A, {max_M, max_M}, 2);
        alloc.reg(&m->ns.gram, {max_M, max_M}, 2);
        alloc.reg(&m->ns.tmp, {max_M, max_N}, 2);
        alloc.reg(&m->ns.result_f32, {max_M, max_N}, sizeof(float));
        alloc.reg(&m->ns_norm_puf, {1}, sizeof(float));
    }
}

void muon_post_create(Muon* m) {
    m->lr_ptr = (float*)m->lr_puf.bytes;
    m->lr_derived_ptr = (float*)m->lr_derived_puf.bytes;
    if (m->ns_norm_puf.bytes) m->ns.norm_ptr = (float*)m->ns_norm_puf.bytes;
    cudaMemcpy(m->lr_ptr, &m->lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(m->lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(m->mb_puf.bytes, 0, m->mb_puf.numel() * sizeof(float));
}

void muon_step(Muon* m, cudaStream_t stream = 0) {
    if (m->wb_puf.bytes == nullptr) return;
    if (m->nccl_comm != nullptr && m->world_size > 1) {
        ncclAllReduce(m->gc_puf.bytes, m->gc_puf.bytes, m->gc_puf.numel(),
                      ncclFloat, ncclAvg, m->nccl_comm, stream);
    }
    nesterov_f32_kernel<<<grid_size(m->mb_puf.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)m->mb_puf.bytes, (float*)m->gc_puf.bytes, (float)m->momentum, m->mb_puf.numel());
    puf_zero(m->up_puf, stream);
    int64_t offset = 0;
    for (auto& ps : m->param_shapes) {
        float* gc_ptr = (float*)m->gc_puf.bytes + offset;
        float* up_ptr = (float*)m->up_puf.bytes + offset;
        if (ps.shape.size() >= 2) {
            int64_t R = ps.shape[0], C = ps.numel / R;
            bool transposed = R > C;
            int64_t M = transposed ? C : R, N = transposed ? R : C;
            PufTensor G_f32 = {.bytes = (char*)gc_ptr, .shape = {R, C}, .dtype_size = 4};
            PufTensor x = ns_slice(m->ns.x, M, N);
            PufTensor A = ns_slice(m->ns.A, M, M);
            PufTensor gram = ns_slice(m->ns.gram, M, M);
            PufTensor tmp = ns_slice(m->ns.tmp, M, N);
            if (transposed) {
                cast_f32_to_bf16_transpose_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
                    (__nv_bfloat16*)x.bytes, (const float*)G_f32.bytes, (int)R, (int)C);
            } else {
                cast_f32_to_bf16_kernel<<<grid_size(x.numel()), BLOCK_SIZE, 0, stream>>>(
                    (__nv_bfloat16*)x.bytes, (const float*)G_f32.bytes, x.numel());
            }
            ensure_norm_partials();
            {
                int nblk = std::min((int)grid_size(x.numel()), 256);
                norm_bf16_kernel<<<nblk, 256, 0, stream>>>(
                    norm_partials_buf, (const __nv_bfloat16*)x.bytes, x.numel());
                norm_reduce_kernel<<<1, 256, 0, stream>>>(m->ns.norm_ptr, norm_partials_buf, nblk);
            }
            normalize_bf16_kernel<<<grid_size(x.numel()), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)x.bytes, m->ns.norm_ptr, 1e-7f, x.numel());
            for (int i = 0; i < 5; ++i) {
                float a = (float)ns_coeffs[i][0], b = (float)ns_coeffs[i][1], c = (float)ns_coeffs[i][2];
                PufTensor& src = (i % 2 == 0) ? x : tmp;
                PufTensor& dst = (i % 2 == 0) ? tmp : x;
                puf_mm(src, src, A, stream);
                puf_copy(gram, A, stream);
                puf_addmm_nn(A, A, gram, c, b, stream);
                puf_copy(dst, src, stream);
                puf_addmm_nn(gram, src, dst, 1.0f, a, stream);
            }
            PufTensor& result_bf16 = tmp;
            float scale = (float)std::sqrt(std::max(1.0, (double)M / (double)N));
            PufTensor out_f32 = {.bytes = (char*)up_ptr, .shape = {R, C}, .dtype_size = 4};
            PufTensor res_f32 = ns_slice(m->ns.result_f32, M, N);
            res_f32.dtype_size = 4;
            cast_bf16_to_f32_kernel<<<grid_size(res_f32.numel()), BLOCK_SIZE, 0, stream>>>(
                (float*)res_f32.bytes, (const __nv_bfloat16*)result_bf16.bytes, res_f32.numel());
            if (scale != 1.0f) scale_f32_kernel<<<grid_size(res_f32.numel()), BLOCK_SIZE, 0, stream>>>(
                (float*)res_f32.bytes, scale, res_f32.numel());
            if (transposed) {
                transpose_f32_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
                    (float*)out_f32.bytes, (const float*)res_f32.bytes, (int)M, (int)N);
            } else {
                puf_copy(out_f32, res_f32, stream);
            }
        } else {
            PufTensor src_puf = {.bytes = (char*)gc_ptr, .shape = {ps.numel}, .dtype_size = 4};
            PufTensor dst_puf = {.bytes = (char*)up_ptr, .shape = {ps.numel}, .dtype_size = 4};
            puf_copy(dst_puf, src_puf, stream);
        }
        offset += ps.numel;
    }
    muon_weight_update_kernel<<<grid_size(m->wb_puf.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)m->wb_puf.bytes, (const float*)m->up_puf.bytes, m->lr_ptr, (float)m->weight_decay, m->wb_puf.numel());
}


#endif // PUFFERLIB_MODELS_CU
