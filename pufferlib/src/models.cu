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

using std::tuple;
using std::vector;

// ============================================================================
// PufTensor — minimal tensor view (no torch dependency)
// ============================================================================

#define PUF_MAX_DIMS 8

// Minimal tensor: raw pointer + shape, no torch dependency in the struct itself.
// Memory is owned by an Allocator buffer — PufTensor is just a view.
struct PufTensor {
    char* bytes;
    int64_t shape[PUF_MAX_DIMS];
    int ndim;
    int dtype_size;      // bytes per element (2 for bf16/f16, 4 for f32, 8 for f64)

    PufTensor() : bytes(nullptr), ndim(0), dtype_size(0) {
        for (int i = 0; i < PUF_MAX_DIMS; i++) shape[i] = 0;
    }

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < ndim; i++) n *= shape[i];
        return n;
    }

    int64_t size(int dim) const { return shape[dim]; }
    int64_t nbytes() const { return numel() * dtype_size; }

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
        for (int i = 0; i < ndim; i++) {
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
// Allocator — contiguous memory for params/grads/activations
// ============================================================================

struct Allocator {
    struct PufRegistration {
        PufTensor* ptr;
        int64_t size;
        std::vector<int64_t> shape;
        int elem_size;
    };
    std::vector<PufRegistration> params, grads, puf_activations;
    void* param_mem = nullptr;
    void* grad_mem = nullptr;
    void* puf_mem = nullptr;

    void register_param(PufTensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        params.push_back({ptr, size, shape, 0});
    }

    void register_grad(PufTensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        grads.push_back({ptr, size, shape, 0});
    }

    void register_puf(PufTensor* ptr, std::vector<int64_t> shape, int elem_size) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        puf_activations.push_back({ptr, size, shape, elem_size});
    }

    void assign_puf_views(std::vector<PufRegistration>& regs, char* base, int esz);
    void create(int esz);
    void destroy();

    int elem_size = 0;
    int64_t total_param_elems = 0;
    int64_t total_grad_elems = 0;
};

// Pre-allocated buffers for prio_replay
struct PrioBuffers {
    PufTensor prio_probs, cdf, idx, mb_prio;
    void register_buffers(Allocator& alloc, int S, int minibatch_segments) {
        alloc.register_puf(&prio_probs, {S}, sizeof(float));
        alloc.register_puf(&cdf, {S}, sizeof(float));
        alloc.register_puf(&idx, {minibatch_segments}, sizeof(int64_t));
        alloc.register_puf(&mb_prio, {minibatch_segments, 1}, sizeof(float));
    }
};

// Pre-allocated buffers for PPO loss
struct PPOBuffersPuf {
    PufTensor loss_output, saved_for_bwd, grad_loss;
    PufTensor grad_logits, grad_values, grad_logstd, adv_scratch;
    void register_buffers(Allocator& alloc, int N, int T, int A_total, bool is_continuous) {
        int64_t total = (int64_t)N * T;
        alloc.register_puf(&loss_output, {1}, sizeof(float));
        alloc.register_puf(&saved_for_bwd, {total, 5}, sizeof(double));
        alloc.register_puf(&grad_loss, {1}, sizeof(float));
        alloc.register_puf(&grad_logits, {N, T, A_total}, sizeof(float));
        alloc.register_puf(&grad_values, {N, T, 1}, sizeof(float));
        if (is_continuous) alloc.register_puf(&grad_logstd, {N, T, A_total}, sizeof(float));
        alloc.register_puf(&adv_scratch, {2}, sizeof(float));
    }
    void post_create();
};

// ============================================================================
// Native Policy, Muon optimizer, and supporting structs
// ============================================================================

namespace pufferlib {

using std::tuple;
using std::vector;

// Param shape info (used by Muon and Policy)
struct ParamShape {
    int64_t numel;
    std::vector<int64_t> shape;
    int ndim;
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

// Native encoder — weights only, activations created separately
struct NativeEncoder {
    PufTensor weight, weight_grad;
    int in_dim, out_dim;

    NativeEncoder() : in_dim(0), out_dim(0) {}

    NativeEncoder(Allocator& alloc, int input, int hidden)
        : in_dim(input), out_dim(hidden) {
        alloc.register_param(&weight, {hidden, input});
        alloc.register_grad(&weight_grad, {hidden, input});
    }

    void register_activations(Allocator& alloc, EncoderActivations& act, int B_TT) {
        int psz = PRECISION_SIZE;
        alloc.register_puf(&act.saved_input, {B_TT, in_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, out_dim}, psz);
        alloc.register_puf(&act.wgrad_scratch, {out_dim, in_dim}, psz);
    }

    void register_inference(Allocator& alloc, EncoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, out_dim}, PRECISION_SIZE);
    }

    void init_weights(uint64_t& seed, cudaStream_t stream);
};

// Native decoder — weights only, activations created separately
struct NativeDecoder {
    PufTensor weight, weight_grad;
    PufTensor logstd, logstd_grad;
    int hidden_dim, output_dim;
    bool continuous;

    NativeDecoder() : hidden_dim(0), output_dim(0), continuous(false) {}

    NativeDecoder(Allocator& alloc, int hidden, int output, bool continuous)
        : hidden_dim(hidden), output_dim(output), continuous(continuous) {
        alloc.register_param(&weight, {output + 1, hidden});
        alloc.register_grad(&weight_grad, {output + 1, hidden});
        if (continuous) {
            alloc.register_param(&logstd, {1, output});
            alloc.register_grad(&logstd_grad, {1, output});
        }
    }

    void register_activations(Allocator& alloc, DecoderActivations& act, int B_TT) {
        int psz = PRECISION_SIZE;
        alloc.register_puf(&act.saved_input, {B_TT, hidden_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.grad_out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.wgrad_scratch, {output_dim + 1, hidden_dim}, psz);
    }

    void register_inference(Allocator& alloc, DecoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, output_dim + 1}, PRECISION_SIZE);
    }

    void init_weights(uint64_t& seed, cudaStream_t stream);
};

// Activation buffers for MinGRU — separate from weights so multiple copies can exist
struct MinGRUActivations {
    int num_layers;  // needed to size vectors

    // Training forward activations
    vector<PufTensor> saved_inputs;    // (B, TT, H) per layer
    vector<PrefixScan> scan_bufs;      // per layer
    vector<PufTensor> combined_bufs;   // (B*TT, 3*H) per layer — mm_out dest

    // Training backward buffers
    PufTensor grad_input_buf;          // (B*TT, H) — shared across layers
    PufTensor grad_next_state;         // (B, 1, H)
    PufTensor wgrad_scratch;           // (3*H, H) — scratch for weight grad mm_out

    // Inference buffers
    vector<PufTensor> inf_combined;    // (B_inf, 3*H) per layer
    PufTensor inf_out;                 // (B_inf, H) — shared across layers (written by mingru_gate)
    PufTensor inf_next_state;          // (B_inf, H) — shared across layers (written by mingru_gate)

    MinGRUActivations() : num_layers(0) {}
    MinGRUActivations(int num_layers)
        : num_layers(num_layers),
          saved_inputs(num_layers), scan_bufs(num_layers),
          combined_bufs(num_layers), inf_combined(num_layers) {}
};

// Native MinGRU — weights only, activations created separately
struct MinGRU {
    int hidden, num_layers;
    vector<PufTensor> weights;       // (3*H, H) per layer, views into allocator
    vector<PufTensor> weight_grads;  // (3*H, H) per layer, views into allocator

    MinGRU() : hidden(0), num_layers(0) {}

    MinGRU(Allocator& alloc, int hidden, int num_layers)
        : hidden(hidden), num_layers(num_layers),
          weights(num_layers), weight_grads(num_layers) {
        for (int i = 0; i < num_layers; i++) {
            alloc.register_param(&weights[i], {3 * hidden, hidden});
            alloc.register_grad(&weight_grads[i], {3 * hidden, hidden});
        }
    }

    void register_activations(Allocator& alloc, MinGRUActivations& act, int B, int TT) {
        int H = hidden;
        int B_TT = B * TT;
        int psz = PRECISION_SIZE;
        alloc.register_puf(&act.grad_input_buf, {B_TT, H}, psz);
        alloc.register_puf(&act.grad_next_state, {B, 1, H}, psz);
        alloc.register_puf(&act.wgrad_scratch, {3 * H, H}, psz);
        for (int i = 0; i < num_layers; i++) {
            alloc.register_puf(&act.saved_inputs[i], {B, TT, H}, psz);
            alloc.register_puf(&act.combined_bufs[i], {B_TT, 3 * H}, psz);
            alloc.register_puf(&act.scan_bufs[i].out, {B, TT, H}, psz);
            alloc.register_puf(&act.scan_bufs[i].next_state, {B, 1, H}, psz);
            alloc.register_puf(&act.scan_bufs[i].a_star, {B, TT + 1, H}, sizeof(float));
            alloc.register_puf(&act.scan_bufs[i].s_vals, {B, TT + 1, H}, sizeof(float));
            alloc.register_puf(&act.scan_bufs[i].log_values_buf, {B, TT + 1, H}, sizeof(float));
            alloc.register_puf(&act.scan_bufs[i].grad_combined, {B, TT, 3 * H}, psz);
            alloc.register_puf(&act.scan_bufs[i].grad_state, {B, 1, H}, psz);
        }
    }

    void register_inference(Allocator& alloc, MinGRUActivations& act, int B_inf) {
        int H = hidden;
        int dsz = PRECISION_SIZE;
        for (int i = 0; i < num_layers; i++) {
            alloc.register_puf(&act.inf_combined[i], {B_inf, 3 * H}, dsz);
        }
        alloc.register_puf(&act.inf_out, {B_inf, H}, dsz);
        alloc.register_puf(&act.inf_next_state, {B_inf, H}, dsz);
    }

    // Helper: get PufTensor view of layer i from (num_layers, B, H) state
    PufTensor state_layer(PufTensor& state, int i) {
        PufTensor s;
        int64_t B = state.size(1);
        int64_t H = state.size(2);
        s.bytes = state.bytes + i * B * H * state.dtype_size;
        s.shape[0] = B;
        s.shape[1] = H;
        s.ndim = 2;
        s.dtype_size = state.dtype_size;
        return s;
    }

    void append_param_shapes(vector<ParamShape>& out) {
        for (int i = 0; i < num_layers; i++) {
            vector<int64_t> s(weights[i].shape, weights[i].shape + weights[i].ndim);
            out.push_back({weights[i].numel(), s, weights[i].ndim});
        }
    }

    void init_weights(uint64_t& seed, cudaStream_t stream);
    PufTensor forward(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream);
    PufTensor forward_train(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream);
    PufTensor backward(PufTensor grad, MinGRUActivations& act, MinGRU* target, cudaStream_t stream);
};

struct PolicyActivations {
    EncoderActivations enc;
    DecoderActivations dec;
    MinGRUActivations rnn;

    PolicyActivations() {}
    PolicyActivations(int num_layers) : rnn(num_layers) {}
};

// Native Policy — no nn::Module, no autograd. Kernels only.
// Each sub-struct self-registers with the Allocator.
// Construction order (encoder → decoder → rnn) determines param_buffer layout,
// which must match parameters() order for Muon.
// Native Policy — weights only, activations created separately
struct Policy {
    NativeEncoder encoder;
    NativeDecoder decoder;
    MinGRU rnn;
    int num_atns;
    PolicyActivations act;  // owned activations (1 copy for now)

    Policy() : num_atns(0) {}

    Policy(Allocator& alloc, int input, int hidden, int output, int num_layers, int num_atns, bool continuous)
        : encoder(alloc, input, hidden),
          decoder(alloc, hidden, output, continuous),
          rnn(alloc, hidden, num_layers),
          num_atns(num_atns),
          act(num_layers) {}

    // Register training activation buffers — call after constructor, before alloc.create()
    void register_activations(Allocator& alloc, int B, int TT) {
        register_activations(alloc, act, B, TT);
    }

    void register_activations(Allocator& alloc, PolicyActivations& a, int B, int TT) {
        encoder.register_activations(alloc, a.enc, B * TT);
        decoder.register_activations(alloc, a.dec, B * TT);
        rnn.register_activations(alloc, a.rnn, B, TT);
    }

    // Register inference-only activations into a separate allocator (for per-buffer copies)
    void register_inference(Allocator& alloc, PolicyActivations& a, int B_inf) {
        encoder.register_inference(alloc, a.enc, B_inf);
        decoder.register_inference(alloc, a.dec, B_inf);
        rnn.register_inference(alloc, a.rnn, B_inf);
    }

    void init_weights(cudaStream_t stream, uint64_t seed = 42);
    PufTensor forward(PufTensor obs, PufTensor state, PolicyActivations& a, cudaStream_t stream);
    PufTensor forward_train(PufTensor x, PufTensor state, PolicyActivations& a, cudaStream_t stream);
    void backward(PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
                  PolicyActivations& a, Policy* target, cudaStream_t stream);
    vector<ParamShape> param_shapes();
};

// Fast clip_grad_norm_ for contiguous grad buffer using PufTensor kernel
void clip_grad_norm_(PufTensor& grad, float max_norm, float* scratch, cudaStream_t stream);

inline float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

// ============================================================================
// Muon optimizer
// ============================================================================

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
        PufTensor s = buf;
        s.shape[0] = rows;
        s.shape[1] = cols;
        s.ndim = 2;
        return s;
    }
};

struct Muon {
    double momentum;
    double weight_decay;
    double eps;
    float lr_val_init;

    float* lr_ptr = nullptr;
    float* lr_derived_ptr = nullptr;
    PufTensor lr_puf;
    PufTensor lr_derived_puf;
    PufTensor ns_norm_puf;
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
          param_shapes(std::move(param_shapes))
    {
        assert(lr_val >= 0 && "Invalid learning rate");
        assert(eps >= 0 && "Invalid epsilon value");
        assert(weight_decay >= 0 && "Invalid weight_decay value");
    }

    ~Muon() {}
    Muon(const Muon&) = delete;
    Muon& operator=(const Muon&) = delete;

    void register_buffers(Allocator& alloc) {
        int64_t n = wb_puf.numel();
        alloc.register_puf(&lr_puf, {1}, sizeof(float));
        alloc.register_puf(&lr_derived_puf, {2}, sizeof(float));
        alloc.register_puf(&mb_puf, {n}, sizeof(float));
        alloc.register_puf(&gc_puf, {n}, sizeof(float));
        alloc.register_puf(&up_puf, {n}, sizeof(float));

        int64_t max_M = 0, max_N = 0;
        for (auto& ps : param_shapes) {
            if (ps.ndim >= 2) {
                int64_t R = ps.shape[0], C = ps.numel / R;
                max_M = std::max(max_M, std::min(R, C));
                max_N = std::max(max_N, std::max(R, C));
            }
        }
        if (max_M > 0) {
            ns.max_M = max_M; ns.max_N = max_N;
            alloc.register_puf(&ns.x, {max_M, max_N}, 2);
            alloc.register_puf(&ns.A, {max_M, max_M}, 2);
            alloc.register_puf(&ns.gram, {max_M, max_M}, 2);
            alloc.register_puf(&ns.tmp, {max_M, max_N}, 2);
            alloc.register_puf(&ns.result_f32, {max_M, max_N}, sizeof(float));
            alloc.register_puf(&ns_norm_puf, {1}, sizeof(float));
        }
        bufs_initialized = true;
    }

    void post_create();
    void step(cudaStream_t stream = 0);
    void zero_grad(cudaStream_t stream);
};

} // namespace pufferlib

// ============================================================================
// Kernels
// ============================================================================

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

// Forward declare — defined after orthogonal init
void puf_scale(PufTensor& dst, float alpha, cudaStream_t stream);

// ============================================================================
// Orthogonal init (cuSOLVER + cuRAND)
// ============================================================================

void puf_orthogonal_init(PufTensor& dst, float gain, uint64_t seed, cudaStream_t stream) {
    assert(dst.ndim == 2);
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
            PufTensor tmp = dst;
            puf_scale(tmp, gain, stream);
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

static void puf_norm(const PufTensor& src, float* out_ptr, cudaStream_t stream) {
    assert(src.dtype_size == 2 && "puf_norm: expected bf16");
    ensure_norm_partials();
    int blocks = std::min((int)grid_size(src.numel()), 256);
    norm_bf16_kernel<<<blocks, 256, 0, stream>>>(
        norm_partials_buf, (const __nv_bfloat16*)src.bytes, src.numel());
    norm_reduce_kernel<<<1, 256, 0, stream>>>(out_ptr, norm_partials_buf, blocks);
}

void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_cast_f32_to_bf16: size mismatch");
    assert(dst.dtype_size == 2 && src.dtype_size == 4);
    cast_f32_to_bf16_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (__nv_bfloat16*)dst.bytes, (const float*)src.bytes, dst.numel());
}

void puf_transpose_01(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    int A = src.shape[0], B = src.shape[1];
    int C = (src.ndim >= 3) ? src.shape[2] : 1;
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

void puf_fill(PufTensor& dst, float val, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        fill_bf16_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)dst.bytes, __float2bfloat16(val), dst.numel());
    } else {
        assert(dst.dtype_size == 4 && "puf_fill: expected bf16 or f32");
        fill_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
            (float*)dst.bytes, val, dst.numel());
    }
}

void puf_clamp(PufTensor& dst, float lo, float hi, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        clamp_bf16_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)dst.bytes, lo, hi, dst.numel());
    } else {
        assert(false && "puf_clamp: only bf16 supported for now");
    }
}

void puf_scale(PufTensor& dst, float alpha, cudaStream_t stream) {
    assert(dst.dtype_size == 4 && "puf_scale: expected f32");
    scale_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, alpha, dst.numel());
}

void puf_axpy(PufTensor& dst, const PufTensor& src, float alpha, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_axpy: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy: expected f32");
    axpy_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, (const float*)src.bytes, alpha, dst.numel());
}

void puf_scale_dev(PufTensor& dst, const float* alpha_ptr, cudaStream_t stream) {
    assert(dst.dtype_size == 4 && "puf_scale_dev: expected f32");
    scale_f32_dev_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, alpha_ptr, dst.numel());
}

void puf_axpy_dev(PufTensor& dst, const PufTensor& src, const float* alpha_ptr, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_axpy_dev: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 4 && "puf_axpy_dev: expected f32");
    axpy_f32_dev_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, (const float*)src.bytes, alpha_ptr, dst.numel());
}

void puf_add(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    assert(dst.numel() == src.numel() && "puf_add: size mismatch");
    assert(dst.dtype_size == 4 && src.dtype_size == 2 && "puf_add: expected fp32 += bf16");
    add_bf16_to_f32_kernel<<<grid_size(dst.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)dst.bytes, (const __nv_bfloat16*)src.bytes, dst.numel());
}

void puf_var_mean(const PufTensor& src, float* var_out, float* mean_out, cudaStream_t stream) {
    assert(src.dtype_size == 4 && "puf_var_mean: expected f32");
    var_mean_kernel<<<1, 256, 0, stream>>>(
        (const float*)src.bytes, var_out, mean_out, src.numel());
}

void puf_add_scalar(float* ptr, float val, cudaStream_t stream) {
    add_scalar_kernel<<<1, 1, 0, stream>>>(ptr, val);
}

void puf_index_copy(PufTensor& dst, const PufTensor& idx, const PufTensor& src, cudaStream_t stream) {
    int num_idx = idx.numel();
    int row_bytes = src.numel() / src.shape[0] * src.dtype_size;
    index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, stream>>>(
        dst.bytes, (const int64_t*)idx.bytes, (const char*)src.bytes, num_idx, row_bytes);
}

void puf_cast_u8_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    cast_u8_to_precision_kernel<<<grid_size(src.numel()), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)dst.bytes, (const unsigned char*)src.bytes, src.numel());
}

void puf_cast_f32_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream) {
    if (dst.dtype_size == 2) {
        puf_cast_f32_to_bf16(dst, src, stream);
    } else {
        puf_copy(dst, src, stream);
    }
}

// ============================================================================
// High-level PufTensor orchestration (dispatch to kernels)
// ============================================================================

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
    int N = logits.shape[0], T = logits.shape[1], A_total = logits.shape[2];
    int num_atns = act_sizes.numel();
    int total = N * T;

    float* adv_var_ptr = (float*)bufs.adv_scratch.bytes;
    float* adv_mean_ptr = adv_var_ptr + 1;
    puf_var_mean(advantages, adv_var_ptr, adv_mean_ptr, stream);

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
    puf_zero(bufs.saved_for_bwd, stream);

    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        (float*)bufs.loss_output.bytes, (float*)losses_acc.bytes, ppo_partials_buf,
        (double*)bufs.saved_for_bwd.bytes, (precision_t*)ratio_out.bytes, (precision_t*)newvalue_out.bytes,
        (const precision_t*)logits.bytes, is_continuous ? (const precision_t*)logstd.bytes : nullptr,
        (const precision_t*)values_pred.bytes, (double*)actions.bytes,
        (const precision_t*)old_logprobs.bytes, (float*)advantages.bytes,
        (const precision_t*)prio.bytes, (const precision_t*)values.bytes, (const precision_t*)returns.bytes,
        adv_mean_ptr, adv_var_ptr, (int*)act_sizes.bytes, num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef, T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a, values_stride_n, values_stride_t, is_continuous);

    ppo_loss_reduce_kernel<<<1, LOSS_N + 1, 0, stream>>>(
        (float*)bufs.loss_output.bytes, (float*)losses_acc.bytes, ppo_partials_buf, ppo_grid);

    puf_add_scalar((float*)losses_acc.bytes + LOSS_N, 1.0f, stream);

    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS, 0, stream>>>(
        (float*)bufs.grad_logits.bytes, is_continuous ? (float*)bufs.grad_logstd.bytes : nullptr,
        (float*)bufs.grad_values.bytes, (float*)bufs.grad_loss.bytes,
        (const precision_t*)logits.bytes, is_continuous ? (const precision_t*)logstd.bytes : nullptr,
        (const precision_t*)values_pred.bytes, (double*)actions.bytes,
        (const precision_t*)old_logprobs.bytes, (float*)advantages.bytes,
        (const precision_t*)prio.bytes, (const precision_t*)values.bytes, (const precision_t*)returns.bytes,
        adv_mean_ptr, adv_var_ptr, (int*)act_sizes.bytes, num_atns,
        clip_coef, vf_clip_coef, vf_coef, ent_coef, T, A_total, N,
        logits_stride_n, logits_stride_t, logits_stride_a, values_stride_n, values_stride_t, is_continuous);
}

void sample_logits(
    PufTensor& logits, PufTensor& logstd, PufTensor& value,
    PufTensor& actions_out, PufTensor& logprobs_out, PufTensor& value_out,
    PufTensor& act_sizes, uint64_t seed, int64_t* offset_ptr,
    int logits_stride, int logstd_stride, int value_stride,
    cudaStream_t stream
) {
    bool is_continuous = logstd.bytes != nullptr && logstd.numel() > 0;
    int B = actions_out.shape[0];
    int num_atns = act_sizes.numel();
    sample_logits_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        (double*)actions_out.bytes, (precision_t*)logprobs_out.bytes, (precision_t*)value_out.bytes,
        (const precision_t*)logits.bytes, is_continuous ? (const precision_t*)logstd.bytes : nullptr,
        (const precision_t*)value.bytes, (int*)act_sizes.bytes,
        seed, offset_ptr, num_atns, B, logits_stride, logstd_stride, value_stride, is_continuous);
}

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
    int obs_rb = (observations.numel() / observations.shape[0]) * observations.dtype_size;
    int act_rb = (actions.numel() / actions.shape[0]) * actions.dtype_size;
    int lp_rb = (logprobs.numel() / logprobs.shape[0]) * logprobs.dtype_size;
    puf_zero(dst_state, stream);
    select_copy_kernel<<<dim3(mb_segs, 5), SELECT_COPY_THREADS, 0, stream>>>(
        (int64_t*)idx.bytes,
        (const char*)observations.bytes, dst_obs.bytes, obs_rb,
        (const char*)actions.bytes, dst_actions.bytes, act_rb,
        (const char*)logprobs.bytes, dst_logprobs.bytes, lp_rb,
        (const precision_t*)values.bytes, (precision_t*)dst_values.bytes,
        (float*)advantages.bytes, (float*)dst_advantages.bytes,
        (precision_t*)dst_returns.bytes, horizon,
        (float*)mb_prio.bytes, (precision_t*)dst_prio.bytes);
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
// Method implementations
// ============================================================================

// --- Allocator ---

void Allocator::assign_puf_views(std::vector<PufRegistration>& regs, char* base, int esz) {
    int64_t offset = 0;
    for (auto& r : regs) {
        r.ptr->bytes = base + offset * esz;
        r.ptr->ndim = r.shape.size();
        r.ptr->dtype_size = esz;
        for (int i = 0; i < PUF_MAX_DIMS; i++) {
            r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 0;
        }
        offset += r.size;
    }
}

void Allocator::destroy() {
    if (param_mem) { cudaFree(param_mem); param_mem = nullptr; }
    if (grad_mem) { cudaFree(grad_mem); grad_mem = nullptr; }
    if (puf_mem) { cudaFree(puf_mem); puf_mem = nullptr; }
}

void Allocator::create(int esz) {
    int64_t total_params = 0;
    for (auto& r : params) total_params += r.size;
    if (total_params > 0) {
        cudaMalloc(&param_mem, total_params * esz);
        cudaMemset(param_mem, 0, total_params * esz);
        assign_puf_views(params, (char*)param_mem, esz);
    }
    total_param_elems = total_params;

    int64_t total_grads = 0;
    for (auto& r : grads) total_grads += r.size;
    if (total_grads > 0) {
        cudaMalloc(&grad_mem, total_grads * esz);
        cudaMemset(grad_mem, 0, total_grads * esz);
        assign_puf_views(grads, (char*)grad_mem, esz);
    }
    total_grad_elems = total_grads;

    int64_t total_puf_bytes = 0;
    for (auto& r : puf_activations) total_puf_bytes += r.size * r.elem_size;
    if (total_puf_bytes > 0) {
        cudaMalloc(&puf_mem, total_puf_bytes);
        cudaMemset(puf_mem, 0, total_puf_bytes);
        char* base = (char*)puf_mem;
        int64_t offset = 0;
        for (auto& r : puf_activations) {
            r.ptr->bytes = base + offset;
            r.ptr->ndim = r.shape.size();
            r.ptr->dtype_size = r.elem_size;
            for (int i = 0; i < PUF_MAX_DIMS; i++) {
                r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 0;
            }
            offset += r.size * r.elem_size;
        }
    }
    elem_size = esz;
}

// --- PPOBuffersPuf ---

void PPOBuffersPuf::post_create() {
    float one = 1.0f;
    cudaMemcpy(grad_loss.bytes, &one, sizeof(float), cudaMemcpyHostToDevice);
}

// --- pufferlib namespace methods ---

namespace pufferlib {

using std::vector;

// --- NativeEncoder ---

void NativeEncoder::init_weights(uint64_t& seed, cudaStream_t stream) {
    PufTensor w2d;
    w2d.bytes = weight.bytes;
    w2d.shape[0] = out_dim;
    w2d.shape[1] = in_dim;
    w2d.ndim = 2;
    w2d.dtype_size = weight.dtype_size;
    puf_orthogonal_init(w2d, std::sqrt(2.0f), seed++, stream);
}

// --- NativeDecoder ---

void NativeDecoder::init_weights(uint64_t& seed, cudaStream_t stream) {
    PufTensor w2d;
    w2d.bytes = weight.bytes;
    w2d.shape[0] = output_dim + 1;
    w2d.shape[1] = hidden_dim;
    w2d.ndim = 2;
    w2d.dtype_size = weight.dtype_size;
    puf_orthogonal_init(w2d, 0.01f, seed++, stream);
    // logstd is already zero from allocator's cudaMemset
}

// --- MinGRU ---

void MinGRU::init_weights(uint64_t& seed, cudaStream_t stream) {
    for (int i = 0; i < num_layers; i++) {
        PufTensor w2d;
        w2d.bytes = weights[i].bytes;
        w2d.shape[0] = 3 * hidden;
        w2d.shape[1] = hidden;
        w2d.ndim = 2;
        w2d.dtype_size = weights[i].dtype_size;
        puf_orthogonal_init(w2d, 1.0f, seed++, stream);
    }
}

PufTensor MinGRU::forward(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
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

PufTensor MinGRU::forward_train(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
    int B = x.size(0);
    int TT = x.size(1);

    for (int i = 0; i < num_layers; i++) {
        puf_copy(act.saved_inputs[i], x, stream);
        PufTensor state_i = state_layer(state, i);

        // Reshape state_i (B, H) -> (B, 1, H) for prefix_scan
        PufTensor state_3d = state_i;
        state_3d.shape[0] = B;
        state_3d.shape[1] = 1;
        state_3d.shape[2] = hidden;
        state_3d.ndim = 3;

        // Flatten x from (B, TT, H) to (B*TT, H) for mm
        PufTensor x_flat = x;
        x_flat.shape[0] = B * TT;
        x_flat.shape[1] = hidden;
        x_flat.ndim = 2;
        puf_mm(x_flat, weights[i], act.combined_bufs[i], stream);

        // Reinterpret (B*TT, 3*H) as (B, TT, 3*H) for scan
        PufTensor combined_3d = act.combined_bufs[i];
        combined_3d.shape[0] = B;
        combined_3d.shape[1] = TT;
        combined_3d.shape[2] = 3 * hidden;
        combined_3d.ndim = 3;
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

PufTensor MinGRU::backward(PufTensor grad, MinGRUActivations& act, MinGRU* target, cudaStream_t stream) {
    int B = grad.size(0);
    int TT = grad.size(1);
    int H = grad.size(2);
    for (int i = num_layers - 1; i >= 0; i--) {
        fused_scan_backward_kernel_checkpointed<<<grid_size(act.scan_bufs[i].B * act.scan_bufs[i].H), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)act.scan_bufs[i].grad_combined.bytes, (precision_t*)act.scan_bufs[i].grad_state.bytes,
            (const precision_t*)grad.bytes, (const precision_t*)act.grad_next_state.bytes,
            (const precision_t*)act.scan_bufs[i].combined_ptr, (const precision_t*)act.scan_bufs[i].state_ptr,
            (float*)act.scan_bufs[i].a_star.bytes, (float*)act.scan_bufs[i].s_vals.bytes,
            (float*)act.scan_bufs[i].log_values_buf.bytes,
            act.scan_bufs[i].T, act.scan_bufs[i].H, act.scan_bufs[i].B);

        // Reinterpret grad_combined (B, TT, 3*H) as (B*TT, 3*H) for matmuls
        PufTensor gc_flat = act.scan_bufs[i].grad_combined;
        gc_flat.shape[0] = B * TT;
        gc_flat.shape[1] = 3 * H;
        gc_flat.ndim = 2;

        // Reinterpret saved_inputs (B, TT, H) as (B*TT, H)
        PufTensor inp_flat = act.saved_inputs[i];
        inp_flat.shape[0] = B * TT;
        inp_flat.shape[1] = H;
        inp_flat.ndim = 2;

        // Weight grad: gc_flat^T @ inp_flat → wgrad_scratch, then accumulate
        puf_mm_tn(gc_flat, inp_flat, act.wgrad_scratch, stream);
        puf_add(target->weight_grads[i], act.wgrad_scratch, stream);

        // Input grad: gc_flat @ weights[i] → grad_input_buf
        puf_mm_nn(gc_flat, weights[i], act.grad_input_buf, stream);

        // Reshape (B*TT, H) → (B, TT, H)
        grad = act.grad_input_buf;
        grad.shape[0] = B;
        grad.shape[1] = TT;
        grad.shape[2] = H;
        grad.ndim = 3;
    }
    return grad;
}

// --- Policy ---

void Policy::init_weights(cudaStream_t stream, uint64_t seed) {
    encoder.init_weights(seed, stream);
    decoder.init_weights(seed, stream);
    rnn.init_weights(seed, stream);
}

PufTensor Policy::forward(PufTensor obs, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
    puf_mm(obs, encoder.weight, a.enc.inf_out, stream);
    PufTensor h = rnn.forward(a.enc.inf_out, state, a.rnn, stream);
    puf_mm(h, decoder.weight, a.dec.inf_out, stream);
    return a.dec.inf_out;
}

PufTensor Policy::forward_train(PufTensor x, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
    int B = x.size(0);
    int TT = x.size(1);

    // Flatten to (B*TT, input)
    PufTensor x_flat = x;
    x_flat.shape[0] = B * TT;
    x_flat.shape[1] = encoder.in_dim;
    x_flat.ndim = 2;

    puf_copy(a.enc.saved_input, x_flat, stream);
    puf_mm(a.enc.saved_input, encoder.weight, a.enc.out, stream);

    // Reshape enc output to (B, TT, H)
    PufTensor h = a.enc.out;
    h.shape[0] = B;
    h.shape[1] = TT;
    h.shape[2] = encoder.out_dim;
    h.ndim = 3;

    h = rnn.forward_train(h, state, a.rnn, stream);

    // Flatten for decoder mm
    PufTensor flat_h = h;
    flat_h.shape[0] = B * TT;
    flat_h.shape[1] = encoder.out_dim;
    flat_h.ndim = 2;

    puf_copy(a.dec.saved_input, flat_h, stream);
    puf_mm(flat_h, decoder.weight, a.dec.out, stream);

    // Reshape to (B, TT, output+1)
    PufTensor result = a.dec.out;
    result.shape[0] = B;
    result.shape[1] = TT;
    result.shape[2] = decoder.output_dim + 1;
    result.ndim = 3;
    return result;
}

void Policy::backward(PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
              PolicyActivations& a, Policy* target, cudaStream_t stream) {
    int B_TT = a.dec.saved_input.size(0);
    int B = grad_logits.size(0);
    int TT = grad_logits.size(1);

    // Assemble fused grad_out (B_TT, output+1) from separate fp32 grads → bf16
    PufTensor gl_flat = grad_logits;
    gl_flat.shape[0] = B_TT;
    gl_flat.shape[1] = decoder.output_dim;
    gl_flat.ndim = 2;
    PufTensor gv_flat = grad_value;
    gv_flat.shape[0] = B_TT;
    gv_flat.ndim = 1;
    {
        int od = decoder.output_dim, od1 = od + 1;
        assemble_decoder_grad_kernel<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)a.dec.grad_out.bytes, (const float*)gl_flat.bytes,
            (const float*)gv_flat.bytes, B_TT, od, od1);
    }

    // Decoder weight grad: bf16 matmul into scratch, then accumulate into fp32 grad
    puf_mm_tn(a.dec.grad_out, a.dec.saved_input, a.dec.wgrad_scratch, stream);
    puf_add(target->decoder.weight_grad, a.dec.wgrad_scratch, stream);

    // logstd grad: column-wise sum reduction into fp32 master grad
    if (decoder.continuous && grad_logstd.bytes != nullptr) {
        PufTensor gls_flat = grad_logstd;
        gls_flat.shape[0] = B_TT;
        gls_flat.shape[1] = decoder.output_dim;
        gls_flat.ndim = 2;
        sum_rows_add_kernel<<<grid_size(decoder.output_dim), BLOCK_SIZE, 0, stream>>>(
            (float*)target->decoder.logstd_grad.bytes, (const float*)gls_flat.bytes,
            B_TT, decoder.output_dim);
    }

    // Decoder input grad: mm into enc.out (same shape: B_TT x hidden, reused buffer)
    puf_mm_nn(a.dec.grad_out, decoder.weight, a.enc.out, stream);

    // Reshape to (B, TT, H) for RNN backward
    PufTensor grad_h = a.enc.out;
    grad_h.shape[0] = B;
    grad_h.shape[1] = TT;
    grad_h.shape[2] = encoder.out_dim;
    grad_h.ndim = 3;

    // RNN backward
    grad_h = rnn.backward(grad_h, a.rnn, &target->rnn, stream);

    // Flatten for encoder weight grad
    PufTensor grad_enc = grad_h;
    grad_enc.shape[0] = B_TT;
    grad_enc.shape[1] = encoder.out_dim;
    grad_enc.ndim = 2;
    puf_mm_tn(grad_enc, a.enc.saved_input, a.enc.wgrad_scratch, stream);
    puf_add(target->encoder.weight_grad, a.enc.wgrad_scratch, stream);
}

vector<ParamShape> Policy::param_shapes() {
    vector<ParamShape> shapes;
    auto push = [&](PufTensor& p) {
        vector<int64_t> s(p.shape, p.shape + p.ndim);
        shapes.push_back({p.numel(), s, p.ndim});
    };
    push(encoder.weight);
    push(decoder.weight);
    if (decoder.continuous) push(decoder.logstd);
    rnn.append_param_shapes(shapes);
    return shapes;
}

// --- clip_grad_norm_ ---

void clip_grad_norm_(PufTensor& grad, float max_norm, float* scratch, cudaStream_t stream) {
    if (grad.bytes == nullptr || grad.numel() == 0) return;
    ensure_norm_partials();
    int blocks = std::min((int)grid_size(grad.numel()), 256);
    norm_f32_kernel<<<blocks, 256, 0, stream>>>(norm_partials_buf, (float*)grad.bytes, grad.numel());
    norm_reduce_kernel<<<1, 256, 0, stream>>>(scratch, norm_partials_buf, blocks);
    clip_by_norm_f32_kernel<<<grid_size(grad.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)grad.bytes, scratch, max_norm, 1e-6f, grad.numel());
}

// --- Muon ---

void Muon::post_create() {
    lr_ptr = (float*)lr_puf.bytes;
    lr_derived_ptr = (float*)lr_derived_puf.bytes;
    if (ns_norm_puf.bytes) ns.norm_ptr = (float*)ns_norm_puf.bytes;
    cudaMemcpy(lr_ptr, &lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(mb_puf.bytes, 0, mb_puf.numel() * sizeof(float));
}

void Muon::step(cudaStream_t stream) {
    if (wb_puf.bytes == nullptr) return;
    puf_copy(gc_puf, gb_puf, stream);
    if (nccl_comm != nullptr && world_size > 1) {
        ncclAllReduce(gc_puf.bytes, gc_puf.bytes, gc_puf.numel(),
                      ncclFloat, ncclAvg, nccl_comm, stream);
    }
    puf_scale(mb_puf, (float)momentum, stream);
    puf_axpy(mb_puf, gc_puf, 1.0f, stream);
    puf_axpy(gc_puf, mb_puf, (float)momentum, stream);

    puf_zero(up_puf, stream);
    int64_t offset = 0;
    for (auto& ps : param_shapes) {
        float* gc_ptr = (float*)gc_puf.bytes + offset;
        float* up_ptr = (float*)up_puf.bytes + offset;
        if (ps.ndim >= 2) {
            int64_t R = ps.shape[0], C = ps.numel / R;
            bool transposed = R > C;
            int64_t M = transposed ? C : R, N = transposed ? R : C;
            PufTensor G_f32;
            G_f32.bytes = (char*)gc_ptr;
            G_f32.shape[0] = R;
            G_f32.shape[1] = C;
            G_f32.ndim = 2;
            G_f32.dtype_size = 4;
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
            puf_norm(x, ns.norm_ptr, stream);
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
            PufTensor out_f32;
            out_f32.bytes = (char*)up_ptr;
            out_f32.shape[0] = R;
            out_f32.shape[1] = C;
            out_f32.ndim = 2;
            out_f32.dtype_size = 4;
            PufTensor res_f32 = ns.slice(ns.result_f32, M, N);
            res_f32.dtype_size = 4;
            cast_bf16_to_f32_kernel<<<grid_size(res_f32.numel()), BLOCK_SIZE, 0, stream>>>(
                (float*)res_f32.bytes, (const __nv_bfloat16*)result_bf16.bytes, res_f32.numel());
            if (scale != 1.0f) puf_scale(res_f32, scale, stream);
            if (transposed) {
                transpose_f32_kernel<<<grid_size(R * C), BLOCK_SIZE, 0, stream>>>(
                    (float*)out_f32.bytes, (const float*)res_f32.bytes, (int)M, (int)N);
            } else {
                puf_copy(out_f32, res_f32, stream);
            }
        } else {
            PufTensor src_puf, dst_puf;
            src_puf.bytes = (char*)gc_ptr;
            src_puf.shape[0] = ps.numel;
            src_puf.ndim = 1;
            src_puf.dtype_size = 4;
            dst_puf.bytes = (char*)up_ptr;
            dst_puf.shape[0] = ps.numel;
            dst_puf.ndim = 1;
            dst_puf.dtype_size = 4;
            puf_copy(dst_puf, src_puf, stream);
        }
        offset += ps.numel;
    }
    compute_lr_scalars_kernel<<<1, 1, 0, stream>>>(lr_ptr, (float)weight_decay, lr_derived_ptr, lr_derived_ptr + 1);
    if (weight_decay != 0) puf_scale_dev(wb_puf, lr_derived_ptr + 1, stream);
    puf_axpy_dev(wb_puf, up_puf, lr_derived_ptr, stream);
}

void Muon::zero_grad(cudaStream_t stream) {
    if (gb_puf.bytes != nullptr) {
        puf_zero(gb_puf, stream);
    }
}

} // namespace pufferlib

#endif // PUFFERLIB_MODELS_CU
