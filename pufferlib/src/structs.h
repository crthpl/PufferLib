#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>

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

    // Merge shape[dim] into shape[dim+1]: {B, TT, H} -> {B*TT, H}
    PufTensor squeeze(int dim) {
        int n = ndim();
        shape[dim + 1] *= shape[dim];
        for (int i = dim; i < n - 1; i++) shape[i] = shape[i + 1];
        shape[n - 1] = 0;
        return *this;
    }

    // Split shape[dim] into two: {B*TT, H} with unsqueeze(0, B, TT) -> {B, TT, H}
    PufTensor unsqueeze(int dim, int64_t d0, int64_t d1) {
        assert(d0 * d1 == shape[dim] && "unsqueeze: d0 * d1 must equal shape[dim]");
        int n = ndim();
        for (int i = n; i > dim; i--) {
            shape[i] = shape[i - 1];
        }
        shape[dim] = d0;
        shape[dim + 1] = d1;
        return *this;
    }

    // Product of all dims except the last two (1 if ndim <= 2)
    int64_t batch_size() const {
        int n = ndim();
        int64_t b = 1;
        for (int i = 0; i < n - 2; i++) {
            b *= shape[i];
        }
        return b;
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
        if (!bytes) {
            snprintf(buf, sizeof(buf), "PufTensor(empty)");
            return buf;
        }
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
    void* combined_ptr = nullptr;
    void* state_ptr = nullptr;
    void* input_ptr = nullptr;      // (B, T, H) original input before projection (for highway gate)
    int B = 0, T = 0, H = 0;
    PufTensor a_star, s_vals, log_values_buf;
    PufTensor out, next_state;
    PufTensor grad_combined, grad_state;
    PufTensor grad_input;           // (B, T, H) highway gate gradient w.r.t. input
};

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

// Fused PPO forward + backward kernel: computes loss partials AND gradients in one pass.
// Avoids redundant recomputation of logits, logsumexp, ratio, advantage normalization.
struct PPOGraphArgs {
    precision_t* out_ratio;
    precision_t* out_newvalue;
    const double* actions;
    const precision_t* old_logprobs;
    const float* advantages;
    const precision_t* prio;
    const precision_t* values;
    const precision_t* returns;
};

struct PPOKernelArgs {
    // Gradient outputs
    float* grad_logits;          // For continuous: grad_mean
    float* grad_logstd;          // For continuous: grad_logstd (nullptr for discrete)
    float* grad_values_pred;
    // Inputs (from dec_out)
    const precision_t* logits;
    const precision_t* logstd;   // nullptr for discrete
    const precision_t* values_pred;
    const float* adv_mean;
    const float* adv_var;
    const int* act_sizes;
    // Scalars
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef, ent_coef;
    int T_seq, A_total, N;
    int logits_stride_n, logits_stride_t, logits_stride_a;
    int values_stride_n, values_stride_t;
    bool is_continuous;
};

// Pre-allocated buffers for PPO loss
struct PPOBuffersPuf {
    PufTensor loss_output, saved_for_bwd, grad_loss;
    PufTensor grad_logits, grad_values, grad_logstd, adv_scratch;
};

// Pre-allocated buffers for prio_replay
struct PrioBuffers {
    PufTensor prio_probs, cdf, idx, mb_prio;
};

struct Allocator {
    PufTensor** regs = nullptr;
    int num_regs = 0;
    void* mem = nullptr;
    long total_elems = 0;
    long total_bytes = 0;
};
