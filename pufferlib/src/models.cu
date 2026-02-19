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
        while (n < PUF_MAX_DIMS && shape[n] != 0) n++;
        return n;
    }

    __host__ __device__ int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i];
        return n;
    }

    __host__ __device__ int64_t size(int dim) const { return shape[dim]; }
    __host__ __device__ int64_t nbytes() const { return numel() * dtype_size; }

    const char* dtype_name() const {
        switch (dtype_size) {
            case 1: return "i8";
            case 2: return "bf16";
            case 4: return "f32";
            case 8: return "f64";
            default: return "?";
        }
    }

    std::string repr() const {
        std::string s = "PufTensor(";
        if (!bytes) return s + "empty)";
        s += dtype_name();
        s += ", [";
        for (int i = 0; i < ndim(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "], ";
        s += std::to_string(numel()) + " elems)";
        return s;
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
    void register_buffers(Allocator& alloc, int S, int minibatch_segments) {
        alloc.reg(&prio_probs, {S}, sizeof(float));
        alloc.reg(&cdf, {S}, sizeof(float));
        alloc.reg(&idx, {minibatch_segments}, sizeof(int64_t));
        alloc.reg(&mb_prio, {minibatch_segments, 1}, sizeof(float));
    }
};

// Pre-allocated buffers for PPO loss
struct PPOBuffersPuf {
    PufTensor loss_output, saved_for_bwd, grad_loss;
    PufTensor grad_logits, grad_values, grad_logstd, adv_scratch;
    void register_buffers(Allocator& alloc, int N, int T, int A_total, bool is_continuous) {
        int64_t total = (int64_t)N * T;
        alloc.reg(&loss_output, {1}, sizeof(float));
        alloc.reg(&saved_for_bwd, {total, 5}, sizeof(double));
        alloc.reg(&grad_loss, {1}, sizeof(float));
        alloc.reg(&grad_logits, {N, T, A_total}, sizeof(float));
        alloc.reg(&grad_values, {N, T, 1}, sizeof(float));
        if (is_continuous) alloc.reg(&grad_logstd, {N, T, A_total}, sizeof(float));
        alloc.reg(&adv_scratch, {2}, sizeof(float));
    }
    void post_create() {
        float one = 1.0f;
        cudaMemcpy(grad_loss.bytes, &one, sizeof(float), cudaMemcpyHostToDevice);
    }
};

// ============================================================================
// Native Policy, Muon optimizer, and supporting structs
// ============================================================================

namespace pufferlib {

using std::vector;

// Param shape info (used by Muon and Policy)
struct ParamShape {
    int64_t numel;
    std::vector<int64_t> shape;
};

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;   // bytes per element
#else
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;   // bytes per element
#endif

// Activation buffers for encoder — separate from weights so multiple copies can exist
struct EncoderActivations {
    PufTensor saved_input;   // (B_TT, in_dim) — saved for backward
    PufTensor out;           // (B_TT, out_dim) — mm_out dest, reused as grad buffer in backward
    PufTensor wgrad_scratch; // (out_dim, in_dim) — scratch for weight grad mm_out
    PufTensor inf_out;       // (B_inf, out_dim) — inference mm_out dest
};

// Activation buffers for decoder — separate from weights so multiple copies can exist
struct DecoderActivations {
    PufTensor saved_input;   // (B_TT, hidden) — saved for backward
    PufTensor out;           // (B_TT, output+1) — mm_out dest
    PufTensor grad_out;      // (B_TT, output+1) — fused grad from PPO (assembled by kernel)
    PufTensor wgrad_scratch; // (output+1, hidden) — scratch for weight grad mm_out
    PufTensor inf_out;       // (B_inf, output+1) — inference mm_out dest
};

} // namespace pufferlib (constants + activation structs)

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

    void register_buffers(Allocator& alloc, int dim0, int dim1, int input_size, int num_atns) {
        int psz = pufferlib::PRECISION_SIZE;
        alloc.reg(&observations, {dim0, dim1, input_size}, psz);
        alloc.reg(&actions, {dim0, dim1, num_atns}, sizeof(double));
        alloc.reg(&values, {dim0, dim1}, psz);
        alloc.reg(&logprobs, {dim0, dim1}, psz);
        alloc.reg(&rewards, {dim0, dim1}, psz);
        alloc.reg(&terminals, {dim0, dim1}, psz);
        alloc.reg(&ratio, {dim0, dim1}, psz);
        alloc.reg(&importance, {dim0, dim1}, psz);
    }
};

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

    void register_buffers(Allocator& alloc, int S, int H, int input_size,
            int hidden_size, int num_atns, int num_layers) {
        int psz = pufferlib::PRECISION_SIZE;
        alloc.reg(&mb_obs, {S, H, input_size}, psz);
        alloc.reg(&mb_state, {num_layers, S, 1, hidden_size}, psz);
        alloc.reg(&mb_actions, {S, H, num_atns}, sizeof(double));
        alloc.reg(&mb_logprobs, {S, H}, psz);
        alloc.reg(&mb_advantages, {S, H}, sizeof(float));
        alloc.reg(&mb_prio, {S, 1}, psz);
        alloc.reg(&mb_values, {S, H}, psz);
        alloc.reg(&mb_returns, {S, H}, psz);
        alloc.reg(&mb_ratio, {S, H}, psz);
        alloc.reg(&mb_newvalue, {S, H, 1}, psz);
    }
};

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
    cudaMemcpyAsync(dst.bytes, src.bytes, dst.nbytes(), cudaMemcpyDeviceToDevice, stream);
}

void puf_zero(PufTensor& dst, cudaStream_t stream) {
    cudaMemsetAsync(dst.bytes, 0, dst.nbytes(), stream);
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

namespace pufferlib {

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

} // namespace pufferlib

// ============================================================================
// Policy, MinGRU, Encoder, Decoder, Muon — defined after kernels/helpers
// so method bodies can use kernel launches and cuBLAS wrappers inline.
// ============================================================================

namespace pufferlib {

using std::vector;

struct NativeEncoder {
    PufTensor weight, weight_grad;
    int in_dim, out_dim;

    NativeEncoder() : in_dim(0), out_dim(0) {}
    NativeEncoder(AllocSet& alloc, int input, int hidden)
        : in_dim(input), out_dim(hidden) {
        alloc.params.reg(&weight, {hidden, input}, alloc.esz);
        alloc.grads.reg(&weight_grad, {hidden, input}, alloc.esz);
    }
    void register_activations(Allocator& alloc, EncoderActivations& act, int B_TT) {
        int psz = PRECISION_SIZE;
        alloc.reg(&act.saved_input, {B_TT, in_dim}, psz);
        alloc.reg(&act.out, {B_TT, out_dim}, psz);
        alloc.reg(&act.wgrad_scratch, {out_dim, in_dim}, psz);
    }
    void register_inference(Allocator& alloc, EncoderActivations& act, int B_inf) {
        alloc.reg(&act.inf_out, {B_inf, out_dim}, PRECISION_SIZE);
    }
    void init_weights(uint64_t& seed, cudaStream_t stream) {
        PufTensor w2d = {
            .bytes = weight.bytes,
            .shape = {out_dim, in_dim},
            .dtype_size = weight.dtype_size
        };
        puf_orthogonal_init(w2d, std::sqrt(2.0f), seed++, stream);
    }
};

struct NativeDecoder {
    PufTensor weight, weight_grad;
    PufTensor logstd, logstd_grad;
    int hidden_dim, output_dim;
    bool continuous;

    NativeDecoder() : hidden_dim(0), output_dim(0), continuous(false) {}
    NativeDecoder(AllocSet& alloc, int hidden, int output, bool continuous)
        : hidden_dim(hidden), output_dim(output), continuous(continuous) {
        alloc.params.reg(&weight, {output + 1, hidden}, alloc.esz);
        alloc.grads.reg(&weight_grad, {output + 1, hidden}, alloc.esz);
        if (continuous) {
            alloc.params.reg(&logstd, {1, output}, alloc.esz);
            alloc.grads.reg(&logstd_grad, {1, output}, alloc.esz);
        }
    }
    void register_activations(Allocator& alloc, DecoderActivations& act, int B_TT) {
        int psz = PRECISION_SIZE;
        alloc.reg(&act.saved_input, {B_TT, hidden_dim}, psz);
        alloc.reg(&act.out, {B_TT, output_dim + 1}, psz);
        alloc.reg(&act.grad_out, {B_TT, output_dim + 1}, psz);
        alloc.reg(&act.wgrad_scratch, {output_dim + 1, hidden_dim}, psz);
    }
    void register_inference(Allocator& alloc, DecoderActivations& act, int B_inf) {
        alloc.reg(&act.inf_out, {B_inf, output_dim + 1}, PRECISION_SIZE);
    }
    void init_weights(uint64_t& seed, cudaStream_t stream) {
        PufTensor w2d = {
            .bytes = weight.bytes,
            .shape = {output_dim + 1, hidden_dim},
            .dtype_size = weight.dtype_size
        };
        puf_orthogonal_init(w2d, 0.01f, seed++, stream);
    }
};

struct MinGRUActivations {
    int num_layers;
    vector<PufTensor> saved_inputs;
    vector<PrefixScan> scan_bufs;
    vector<PufTensor> combined_bufs;
    PufTensor grad_input_buf;
    PufTensor grad_next_state;
    PufTensor wgrad_scratch;
    vector<PufTensor> inf_combined;
    PufTensor inf_out;
    PufTensor inf_next_state;

    MinGRUActivations() : num_layers(0) {}
    MinGRUActivations(int num_layers)
        : num_layers(num_layers),
          saved_inputs(num_layers), scan_bufs(num_layers),
          combined_bufs(num_layers), inf_combined(num_layers) {}
};

struct MinGRU {
    int hidden, num_layers;
    vector<PufTensor> weights;
    vector<PufTensor> weight_grads;

    MinGRU() : hidden(0), num_layers(0) {}
    MinGRU(AllocSet& alloc, int hidden, int num_layers)
        : hidden(hidden), num_layers(num_layers),
          weights(num_layers), weight_grads(num_layers) {
        for (int i = 0; i < num_layers; i++) {
            alloc.params.reg(&weights[i], {3 * hidden, hidden}, alloc.esz);
            alloc.grads.reg(&weight_grads[i], {3 * hidden, hidden}, alloc.esz);
        }
    }
    void register_activations(Allocator& alloc, MinGRUActivations& act, int B, int TT) {
        int H = hidden, B_TT = B * TT, psz = PRECISION_SIZE;
        alloc.reg(&act.grad_input_buf, {B_TT, H}, psz);
        alloc.reg(&act.grad_next_state, {B, 1, H}, psz);
        alloc.reg(&act.wgrad_scratch, {3 * H, H}, psz);
        for (int i = 0; i < num_layers; i++) {
            alloc.reg(&act.saved_inputs[i], {B, TT, H}, psz);
            alloc.reg(&act.combined_bufs[i], {B_TT, 3 * H}, psz);
            alloc.reg(&act.scan_bufs[i].out, {B, TT, H}, psz);
            alloc.reg(&act.scan_bufs[i].next_state, {B, 1, H}, psz);
            alloc.reg(&act.scan_bufs[i].a_star, {B, TT + 1, H}, sizeof(float));
            alloc.reg(&act.scan_bufs[i].s_vals, {B, TT + 1, H}, sizeof(float));
            alloc.reg(&act.scan_bufs[i].log_values_buf, {B, TT + 1, H}, sizeof(float));
            alloc.reg(&act.scan_bufs[i].grad_combined, {B, TT, 3 * H}, psz);
            alloc.reg(&act.scan_bufs[i].grad_state, {B, 1, H}, psz);
        }
    }
    void register_inference(Allocator& alloc, MinGRUActivations& act, int B_inf) {
        int H = hidden, dsz = PRECISION_SIZE;
        for (int i = 0; i < num_layers; i++)
            alloc.reg(&act.inf_combined[i], {B_inf, 3 * H}, dsz);
        alloc.reg(&act.inf_out, {B_inf, H}, dsz);
        alloc.reg(&act.inf_next_state, {B_inf, H}, dsz);
    }
    PufTensor state_layer(PufTensor& state, int i) {
        int64_t B = state.size(1), H = state.size(2);
        return {.bytes = state.bytes + i * B * H * state.dtype_size, .shape = {B, H}, .dtype_size = state.dtype_size};
    }
    void append_param_shapes(vector<ParamShape>& out) {
        for (int i = 0; i < num_layers; i++) {
            vector<int64_t> s(weights[i].shape, weights[i].shape + weights[i].ndim());
            out.push_back({weights[i].numel(), s});
        }
    }
    void init_weights(uint64_t& seed, cudaStream_t stream) {
        for (int i = 0; i < num_layers; i++) {
            PufTensor w2d = {.bytes = weights[i].bytes, .shape = {3 * hidden, hidden}, .dtype_size = weights[i].dtype_size};
            puf_orthogonal_init(w2d, 1.0f, seed++, stream);
        }
    }
    PufTensor forward(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
        for (int i = 0; i < num_layers; i++) {
            PufTensor state_i = state_layer(state, i);
            puf_mm(x, weights[i], act.inf_combined[i], stream);
            int Bi = static_cast<int>(state_i.size(0));
            int Hi = static_cast<int>(state_i.size(1));
            mingru_gate_inference_kernel<<<grid_size(Bi * Hi), BLOCK_SIZE, 0, stream>>>(
                (precision_t*)act.inf_out.bytes, (precision_t*)act.inf_next_state.bytes,
                (const precision_t*)act.inf_combined[i].bytes, (const precision_t*)state_i.bytes, Hi, Bi);
            puf_copy(state_i, act.inf_next_state, stream);
            x = act.inf_out;
        }
        return x;
    }
    PufTensor forward_train(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
        int B = x.size(0), TT = x.size(1);
        for (int i = 0; i < num_layers; i++) {
            puf_copy(act.saved_inputs[i], x, stream);
            PufTensor state_i = state_layer(state, i);
            PufTensor state_3d = {.bytes = state_i.bytes, .shape = {B, 1, hidden}, .dtype_size = state_i.dtype_size};
            PufTensor x_flat = {.bytes = x.bytes, .shape = {B * TT, hidden}, .dtype_size = x.dtype_size};
            puf_mm(x_flat, weights[i], act.combined_bufs[i], stream);
            PufTensor combined_3d = {.bytes = act.combined_bufs[i].bytes, .shape = {B, TT, 3 * hidden}, .dtype_size = act.combined_bufs[i].dtype_size};
            act.scan_bufs[i].combined_ptr = combined_3d.bytes;
            act.scan_bufs[i].state_ptr = state_3d.bytes;
            act.scan_bufs[i].B = B; act.scan_bufs[i].T = TT; act.scan_bufs[i].H = hidden;
            fused_scan_forward_kernel_checkpointed<<<grid_size(B * hidden), BLOCK_SIZE, 0, stream>>>(
                (precision_t*)act.scan_bufs[i].out.bytes, (precision_t*)act.scan_bufs[i].next_state.bytes,
                (float*)act.scan_bufs[i].a_star.bytes, (float*)act.scan_bufs[i].s_vals.bytes,
                (float*)act.scan_bufs[i].log_values_buf.bytes,
                (const precision_t*)combined_3d.bytes, (const precision_t*)state_3d.bytes, TT, hidden, B);
            x = act.scan_bufs[i].out;
        }
        return x;
    }
    PufTensor backward(PufTensor grad, MinGRUActivations& act, MinGRU* target, cudaStream_t stream) {
        int B = grad.size(0), TT = grad.size(1), H = grad.size(2);
        for (int i = num_layers - 1; i >= 0; i--) {
            fused_scan_backward_kernel_checkpointed<<<grid_size(act.scan_bufs[i].B * act.scan_bufs[i].H), BLOCK_SIZE, 0, stream>>>(
                (precision_t*)act.scan_bufs[i].grad_combined.bytes, (precision_t*)act.scan_bufs[i].grad_state.bytes,
                (const precision_t*)grad.bytes, (const precision_t*)act.grad_next_state.bytes,
                (const precision_t*)act.scan_bufs[i].combined_ptr, (const precision_t*)act.scan_bufs[i].state_ptr,
                (float*)act.scan_bufs[i].a_star.bytes, (float*)act.scan_bufs[i].s_vals.bytes,
                (float*)act.scan_bufs[i].log_values_buf.bytes,
                act.scan_bufs[i].T, act.scan_bufs[i].H, act.scan_bufs[i].B);
            PufTensor gc_flat = {.bytes = act.scan_bufs[i].grad_combined.bytes, .shape = {B * TT, 3 * H}, .dtype_size = act.scan_bufs[i].grad_combined.dtype_size};
            PufTensor inp_flat = {.bytes = act.saved_inputs[i].bytes, .shape = {B * TT, H}, .dtype_size = act.saved_inputs[i].dtype_size};
            puf_mm_tn(gc_flat, inp_flat, act.wgrad_scratch, stream);
            puf_add(target->weight_grads[i], act.wgrad_scratch, stream);
            puf_mm_nn(gc_flat, weights[i], act.grad_input_buf, stream);
            grad = {.bytes = act.grad_input_buf.bytes, .shape = {B, TT, H}, .dtype_size = act.grad_input_buf.dtype_size};
        }
        return grad;
    }
};

struct PolicyActivations {
    EncoderActivations enc;
    DecoderActivations dec;
    MinGRUActivations rnn;
    PolicyActivations() {}
    PolicyActivations(int num_layers) : rnn(num_layers) {}
};

struct Policy {
    NativeEncoder encoder;
    NativeDecoder decoder;
    MinGRU rnn;
    int num_atns;
    PolicyActivations act;

    Policy() : num_atns(0) {}
    Policy(AllocSet& alloc, int input, int hidden, int output, int num_layers, int num_atns, bool continuous)
        : encoder(alloc, input, hidden),
          decoder(alloc, hidden, output, continuous),
          rnn(alloc, hidden, num_layers),
          num_atns(num_atns),
          act(num_layers) {}
    void register_activations(Allocator& alloc, int B, int TT) {
        register_activations(alloc, act, B, TT);
    }
    void register_activations(Allocator& alloc, PolicyActivations& a, int B, int TT) {
        encoder.register_activations(alloc, a.enc, B * TT);
        decoder.register_activations(alloc, a.dec, B * TT);
        rnn.register_activations(alloc, a.rnn, B, TT);
    }
    void register_inference(Allocator& alloc, PolicyActivations& a, int B_inf) {
        encoder.register_inference(alloc, a.enc, B_inf);
        decoder.register_inference(alloc, a.dec, B_inf);
        rnn.register_inference(alloc, a.rnn, B_inf);
    }
    void init_weights(cudaStream_t stream, uint64_t seed = 42) {
        encoder.init_weights(seed, stream);
        decoder.init_weights(seed, stream);
        rnn.init_weights(seed, stream);
    }
    PufTensor forward(PufTensor obs, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
        puf_mm(obs, encoder.weight, a.enc.inf_out, stream);
        PufTensor h = rnn.forward(a.enc.inf_out, state, a.rnn, stream);
        puf_mm(h, decoder.weight, a.dec.inf_out, stream);
        return a.dec.inf_out;
    }
    PufTensor forward_train(PufTensor x, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
        int B = x.size(0), TT = x.size(1);
        PufTensor x_flat = {.bytes = x.bytes, .shape = {B * TT, encoder.in_dim}, .dtype_size = x.dtype_size};
        puf_copy(a.enc.saved_input, x_flat, stream);
        puf_mm(a.enc.saved_input, encoder.weight, a.enc.out, stream);
        PufTensor h = {.bytes = a.enc.out.bytes, .shape = {B, TT, encoder.out_dim}, .dtype_size = a.enc.out.dtype_size};
        h = rnn.forward_train(h, state, a.rnn, stream);
        PufTensor flat_h = {.bytes = h.bytes, .shape = {B * TT, encoder.out_dim}, .dtype_size = h.dtype_size};
        puf_copy(a.dec.saved_input, flat_h, stream);
        puf_mm(flat_h, decoder.weight, a.dec.out, stream);
        PufTensor result = {.bytes = a.dec.out.bytes, .shape = {B, TT, decoder.output_dim + 1}, .dtype_size = a.dec.out.dtype_size};
        return result;
    }
    void backward(PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
                  PolicyActivations& a, Policy* target, cudaStream_t stream) {
        int B_TT = a.dec.saved_input.size(0);
        int B = grad_logits.size(0), TT = grad_logits.size(1);
        PufTensor gl_flat = {.bytes = grad_logits.bytes, .shape = {B_TT, decoder.output_dim}, .dtype_size = grad_logits.dtype_size};
        PufTensor gv_flat = {.bytes = grad_value.bytes, .shape = {B_TT}, .dtype_size = grad_value.dtype_size};
        {
            int od = decoder.output_dim, od1 = od + 1;
            assemble_decoder_grad_kernel<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)a.dec.grad_out.bytes, (const float*)gl_flat.bytes,
                (const float*)gv_flat.bytes, B_TT, od, od1);
        }
        puf_mm_tn(a.dec.grad_out, a.dec.saved_input, a.dec.wgrad_scratch, stream);
        puf_add(target->decoder.weight_grad, a.dec.wgrad_scratch, stream);
        if (decoder.continuous && grad_logstd.bytes != nullptr) {
            PufTensor gls_flat = {.bytes = grad_logstd.bytes, .shape = {B_TT, decoder.output_dim}, .dtype_size = grad_logstd.dtype_size};
            sum_rows_add_kernel<<<grid_size(decoder.output_dim), BLOCK_SIZE, 0, stream>>>(
                (float*)target->decoder.logstd_grad.bytes, (const float*)gls_flat.bytes,
                B_TT, decoder.output_dim);
        }
        puf_mm_nn(a.dec.grad_out, decoder.weight, a.enc.out, stream);
        PufTensor grad_h = {.bytes = a.enc.out.bytes, .shape = {B, TT, encoder.out_dim}, .dtype_size = a.enc.out.dtype_size};
        grad_h = rnn.backward(grad_h, a.rnn, &target->rnn, stream);
        PufTensor grad_enc = {.bytes = grad_h.bytes, .shape = {B_TT, encoder.out_dim}, .dtype_size = grad_h.dtype_size};
        puf_mm_tn(grad_enc, a.enc.saved_input, a.enc.wgrad_scratch, stream);
        puf_add(target->encoder.weight_grad, a.enc.wgrad_scratch, stream);
    }
    vector<ParamShape> param_shapes() {
        vector<ParamShape> shapes;
        auto push = [&](PufTensor& p) {
            vector<int64_t> s(p.shape, p.shape + p.ndim());
            shapes.push_back({p.numel(), s});
        };
        push(encoder.weight);
        push(decoder.weight);
        if (decoder.continuous) push(decoder.logstd);
        rnn.append_param_shapes(shapes);
        return shapes;
    }
};

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
    PufTensor slice(PufTensor& buf, int64_t rows, int64_t cols) {
        return {.bytes = buf.bytes, .shape = {rows, cols}, .dtype_size = buf.dtype_size};
    }
};

struct Muon {
    double momentum, weight_decay, eps;
    float lr_val_init;
    float* lr_ptr = nullptr;
    float* lr_derived_ptr = nullptr;
    PufTensor lr_puf, lr_derived_puf, ns_norm_puf;
    PufTensor wb_puf, gb_puf, mb_puf, gc_puf, up_puf;
    bool bufs_initialized = false;
    NSScratch ns;
    std::vector<ParamShape> param_shapes;
    ncclComm_t nccl_comm = nullptr;
    int world_size = 1;

    Muon(std::vector<ParamShape> param_shapes, PufTensor weight_buffer,
             PufTensor grad_buffer, double lr_val, double momentum,
             double eps, double weight_decay)
            : momentum(momentum), weight_decay(weight_decay), eps(eps),
              lr_val_init((float)lr_val),
              wb_puf(weight_buffer), gb_puf(grad_buffer),
              param_shapes(std::move(param_shapes)) {
        assert(lr_val >= 0 && "Invalid learning rate");
        assert(eps >= 0 && "Invalid epsilon value");
        assert(weight_decay >= 0 && "Invalid weight_decay value");
    }
    ~Muon() {}
    Muon(const Muon&) = delete;
    Muon& operator=(const Muon&) = delete;

    void register_buffers(Allocator& alloc) {
        int64_t n = wb_puf.numel();
        alloc.reg(&lr_puf, {1}, sizeof(float));
        alloc.reg(&lr_derived_puf, {2}, sizeof(float));
        alloc.reg(&mb_puf, {n}, sizeof(float));
        alloc.reg(&gc_puf, {n}, sizeof(float));
        alloc.reg(&up_puf, {n}, sizeof(float));
        int64_t max_M = 0, max_N = 0;
        for (auto& ps : param_shapes) {
            if (ps.shape.size() >= 2) {
                int64_t R = ps.shape[0], C = ps.numel / R;
                max_M = std::max(max_M, std::min(R, C));
                max_N = std::max(max_N, std::max(R, C));
            }
        }
        if (max_M > 0) {
            ns.max_M = max_M; ns.max_N = max_N;
            alloc.reg(&ns.x, {max_M, max_N}, 2);
            alloc.reg(&ns.A, {max_M, max_M}, 2);
            alloc.reg(&ns.gram, {max_M, max_M}, 2);
            alloc.reg(&ns.tmp, {max_M, max_N}, 2);
            alloc.reg(&ns.result_f32, {max_M, max_N}, sizeof(float));
            alloc.reg(&ns_norm_puf, {1}, sizeof(float));
        }
        bufs_initialized = true;
    }
    void post_create() {
        lr_ptr = (float*)lr_puf.bytes;
        lr_derived_ptr = (float*)lr_derived_puf.bytes;
        if (ns_norm_puf.bytes) ns.norm_ptr = (float*)ns_norm_puf.bytes;
        cudaMemcpy(lr_ptr, &lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(lr_derived_ptr, 0, 2 * sizeof(float));
        cudaMemset(mb_puf.bytes, 0, mb_puf.numel() * sizeof(float));
    }
    void step(cudaStream_t stream = 0) {
        if (wb_puf.bytes == nullptr) return;
        puf_copy(gc_puf, gb_puf, stream);
        if (nccl_comm != nullptr && world_size > 1) {
            ncclAllReduce(gc_puf.bytes, gc_puf.bytes, gc_puf.numel(),
                          ncclFloat, ncclAvg, nccl_comm, stream);
        }
        nesterov_f32_kernel<<<grid_size(mb_puf.numel()), BLOCK_SIZE, 0, stream>>>(
            (float*)mb_puf.bytes, (float*)gc_puf.bytes, (float)momentum, mb_puf.numel());
        puf_zero(up_puf, stream);
        int64_t offset = 0;
        for (auto& ps : param_shapes) {
            float* gc_ptr = (float*)gc_puf.bytes + offset;
            float* up_ptr = (float*)up_puf.bytes + offset;
            if (ps.shape.size() >= 2) {
                int64_t R = ps.shape[0], C = ps.numel / R;
                bool transposed = R > C;
                int64_t M = transposed ? C : R, N = transposed ? R : C;
                PufTensor G_f32 = {.bytes = (char*)gc_ptr, .shape = {R, C}, .dtype_size = 4};
                PufTensor x = ns.slice(ns.x, M, N);
                PufTensor A = ns.slice(ns.A, M, M);
                PufTensor gram = ns.slice(ns.gram, M, M);
                PufTensor tmp = ns.slice(ns.tmp, M, N);
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
                    norm_reduce_kernel<<<1, 256, 0, stream>>>(ns.norm_ptr, norm_partials_buf, nblk);
                }
                normalize_bf16_kernel<<<grid_size(x.numel()), BLOCK_SIZE, 0, stream>>>(
                    (__nv_bfloat16*)x.bytes, ns.norm_ptr, 1e-7f, x.numel());
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
                PufTensor res_f32 = ns.slice(ns.result_f32, M, N);
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
        muon_weight_update_kernel<<<grid_size(wb_puf.numel()), BLOCK_SIZE, 0, stream>>>(
            (float*)wb_puf.bytes, (const float*)up_puf.bytes, lr_ptr, (float)weight_decay, wb_puf.numel());
    }
    void zero_grad(cudaStream_t stream) {
        if (gb_puf.bytes != nullptr) puf_zero(gb_puf, stream);
    }
};

} // namespace pufferlib

#endif // PUFFERLIB_MODELS_CU
