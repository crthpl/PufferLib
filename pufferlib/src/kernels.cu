#ifndef PUFFERLIB_KERNELS_CU
#define PUFFERLIB_KERNELS_CU

/* Kernels must launch on the current torch stream to be traced by cudagraphs.
 * This file is included by models.cu which calls kernels directly with <<<>>>.
 * Callers should use at::cuda::getCurrentCUDAStream() when using with torch.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
// ============================================================================
// Device math ops (formerly ops.cuh)
// ============================================================================

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f

__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    return (x_scaled > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
}

__device__ __forceinline__ float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) return grad_output;
    float exp_beta_x = expf(beta_x);
    return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
}

__device__ __forceinline__ float relu(float x) { return fmaxf(0.0f, x); }
__device__ __forceinline__ float relu_backward(float x, float grad_output) { return (x > 0.0f) ? grad_output : 0.0f; }

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

__device__ __forceinline__ float tilde_relu_fwd(float x) {
    return (x >= 0.0f) ? x + 0.5f : fast_sigmoid(x);
}

__device__ __forceinline__ float tilde_relu_bwd(float x, float grad_output) {
    if (x >= 0.0f) return grad_output;
    float sig = fast_sigmoid(x);
    return grad_output * sig * (1.0f - sig);
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}
__device__ __forceinline__ float lerp_backward_a(float a, float b, float w, float grad_output) { return grad_output * (1.0f - w); }
__device__ __forceinline__ float lerp_backward_b(float a, float b, float w, float grad_output) { return grad_output * w; }
__device__ __forceinline__ float lerp_backward_w(float a, float b, float w, float grad_output) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? grad_output * diff : grad_output * diff;
}

__device__ __forceinline__ float mingru_gate(float h) { return h >= 0.0f ? h + 0.5f : sigmoid(h); }
__device__ __forceinline__ float mingru_gate_backward(float h, float grad_output) {
    if (h > 0.0f) return grad_output;
    float sig = sigmoid(h);
    return grad_output * sig * (1.0f - sig);
}

__device__ __forceinline__ float logaddexp(float a, float b) {
    float m = fmaxf(a, b), diff = fminf(a, b) - m;
    return (diff < -88.0f) ? m : m + log1pf(expf(diff));
}

__device__ __forceinline__ float exp_safe(float x) {
    return (x > 88.0f) ? 1.651e38f : ((x < -88.0f) ? 0.0f : expf(x));
}

__device__ __forceinline__ void cumsum_forward(const float* x, float* y, int T) {
    float sum = 0.0f;
    for (int t = 0; t < T; t++) { sum += x[t]; y[t] = sum; }
}

__device__ __forceinline__ void cumsum_backward(const float* grad_y, float* grad_x, int T) {
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) { running += grad_y[t]; grad_x[t] = running; }
}

__device__ __forceinline__ void logcumsumexp_forward(const float* x, float* y, int T) {
    float lse = -CUDART_INF_F;
    for (int t = 0; t < T; t++) { lse = (lse == -CUDART_INF_F) ? x[t] : logaddexp(lse, x[t]); y[t] = lse; }
}

__device__ __forceinline__ void logcumsumexp_backward(const float* x, const float* y, const float* grad_y, float* grad_x, int T) {
    float running = 0.0f;
    for (int t = T - 1; t >= 0; t--) { running += grad_y[t] * exp_safe(x[t] - y[t]); grad_x[t] = running; }
}

__device__ __forceinline__ void log_coeffs_and_values_fwd(float gate, float hidden, float* log_coeff_out, float* log_value_out) {
    float abs_gate = fabsf(gate);
    float sp_neg = log1pf(expf(-abs_gate));
    float softplus_gate = (gate >= 0.0f) ? gate + sp_neg : sp_neg;
    float softplus_neg_gate = (gate >= 0.0f) ? sp_neg : -gate + sp_neg;
    *log_coeff_out = -softplus_gate;
    float log_tilde_h = (hidden >= 0.0f) ? logf(hidden + 0.5f) : -softplus_fwd(-hidden);
    *log_value_out = -softplus_neg_gate + log_tilde_h;
}

__device__ __forceinline__ void log_coeffs_and_values_bwd(float grad_log_coeffs, float grad_log_values, float gate, float hidden, float* grad_gate_out, float* grad_hidden_out) {
    float sig_gate = sigmoid(gate);
    *grad_gate_out = -grad_log_coeffs * sig_gate + grad_log_values * (1.0f - sig_gate);
    *grad_hidden_out = (hidden >= 0.0f) ? grad_log_values / (hidden + 0.5f) : grad_log_values * sigmoid(-hidden);
}
#include <curand_kernel.h>

#include <cstdio>
#include <cstdint>

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

#define PPO_THREADS 256

#define SEQ_SIZE 256
#define BLOCK_SIZE 256
#define CHECKPOINT_INTERVAL 4  // Sparse checkpoint interval for optimized kernels

// Maximum number of action heads supported for MultiDiscrete
// Using register arrays to avoid dynamic allocation
#define MAX_ATN_HEADS 16


inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}
inline int seq_size(int N) {
    return (N + SEQ_SIZE - 1) / SEQ_SIZE;
}

// Compile-time precision conversion macros
#ifdef PRECISION_FLOAT
#define to_float(x) (x)
#define from_float(x) (x)
#else
#define to_float(x) __bfloat162float(x)
#define from_float(x) __float2bfloat16(x)
#endif


// Fused kernel: chunk + mingru_gate + sigmoid(proj) * out
// combined is (B, 1, 3*H) containing [hidden, gate, proj] concatenated on last dim
// state is (B, 1, H)
// out is (B, H) = sigmoid(proj) * mingru_out (final output)
// next_state is (B, H) = mingru_out (recurrent state, without proj)
__global__ void mingru_gate_inference_kernel(
    precision_t* out,
    precision_t* next_state,
    const precision_t* combined,    // (B, 3*H) = [hidden, gate, proj]
    const precision_t* state_in,    // (B, H)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) return;

    int b = idx / H;
    int h = idx % H;

    // Read from combined: layout is [hidden(H), gate(H), proj(H)] for each batch
    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2 * H + h]);
    float state = to_float(state_in[idx]);

    // mingru_gate computation
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = tilde_relu_fwd(hidden);
    float mingru_out = lerp(state, hidden_tilde, gate_sigmoid);

    // next_state is mingru_out (for recurrence)
    next_state[idx] = from_float(mingru_out);

    // out is sigmoid(proj) * mingru_out (final output)
    float proj_sigmoid = sigmoid(proj);
    out[idx] = from_float(proj_sigmoid * mingru_out);
}


__device__ __forceinline__ double logcumsumexp_forward(double x, double acc) {
    if (acc == -INFINITY) {
        return x;
    } else {
        double min_val = fmin(acc, x);
        double max_val = fmax(acc, x);
        return max_val + log1pf(expf(min_val - max_val));
    }
}

__device__ __forceinline__ double logcumsumexp_backward(double x, double* acc, double grad, double s, double* s_nxt) {
    *acc = grad + *acc * exp(s - *s_nxt);
    *s_nxt = s;
    return *acc * exp(x - s);
}

// Optimized forward kernel with checkpointing
// Writes checkpoints only every CHECKPOINT_INTERVAL timesteps (vs every time)
// Uses fast math intrinsics for better performance
__global__ void fused_scan_forward_kernel_checkpointed(
    precision_t* __restrict__ out,                 // (B, T, H)
    precision_t* __restrict__ next_state,          // (B, 1, H)
    float* __restrict__ a_star_buf,      // (B, T+1, H)
    float* __restrict__ s_buf,           // (B, T+1, H)
    float* __restrict__ log_values_buf,  // (B, T+1, H)
    const precision_t* __restrict__ combined,      // (B, T, 3*H)
    const precision_t* __restrict__ state,         // (B, 1, H)
    int T_seq,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;

    // Handle t=0 outside the loop: use log(state), coeff = 0
    float s = __logf(to_float(state[bH + h]));
    log_value = s;

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    // Loop t=1..T_seq with sparse checkpointing
    float scan_result = 0.0f;
    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < T_seq + 1; t++) {
        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value);

        // a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += log_coeff_val;

        float z = log_value - a_star;
        float max_val = fmaxf(s, z);
        s = max_val + log1pf(__expf(-fabsf(s - z)));

        scan_result = __expf(a_star + s);
        float proj_sigmoid = sigmoid(proj_val);

        out[out_curr] = from_float(proj_sigmoid * scan_result);

        buf_curr += H;
        out_curr += H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Write timestep T to next_state (raw scan_result, no proj, for recurrence)
    next_state[bH + h] = from_float(scan_result);
}

// Optimized backward kernel with sparse checkpoint loading
// Reads sparse checkpoints from forward pass, recomputes intermediate values in chunks
// Uses fast math intrinsics for better performance
__global__ void fused_scan_backward_kernel_checkpointed(
    precision_t* __restrict__ grad_combined,         // (B, T, 3*H)
    precision_t* __restrict__ grad_state,            // (B, 1, H)
    const precision_t* __restrict__ grad_out,        // (B, T, H)
    const precision_t* __restrict__ grad_next_state, // (B, 1, H)
    const precision_t* __restrict__ combined,        // (B, T, 3*H)
    const precision_t* __restrict__ state,           // (B, 1, H)
    const float* __restrict__ a_star_buf,  // (B, T+1, H)
    const float* __restrict__ s_buf,       // (B, T+1, H)
    const float* __restrict__ log_values_buf, // (B, T+1, H)
    int T_seq,                             // (T)
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    precision_t* grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* grad_combined_p_base = &grad_combined[cbase + H2 + h];

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;

    float acc = 0.0;
    float s_val_next = 0.0;
    float carry_grad_a = 0.0;

    for (int chunk_end = T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in registers
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        // Load checkpoint from global memory
        int ckpt_buf_idx = buf_base + chunk_start * H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Recompute and store from chunk_start to chunk_end
        for (int i = 0; i < chunk_len; ++i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = to_float(combined_h_base[t_offset]);
            float gv = to_float(combined_g_base[t_offset]);

            float lc;
            log_coeffs_and_values_fwd(gv, hv, &lc, &recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            float mv = fmaxf(recomp_s, z);
            recomp_s = mv + log1pf(__expf(-fabsf(recomp_s - z)));

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        for (int i = chunk_len - 1; i >= 0; --i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];

            float proj_val = to_float(combined_p_base[t_offset]);

            float scan_result = __expf(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = to_float(grad_out[out_base + (t - 1) * H]);

            float grad_scan_from_next = (t == T_seq) ? to_float(grad_next_state[state_idx]) : 0.0f;

            float proj_sigmoid = sigmoid(proj_val);
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * scan_result * proj_sigmoid * (1.0f - proj_sigmoid);

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * __expf(s_t - s_val_next);
            }
            float grad_z = acc * __expf(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, &grad_g, &grad_h);

            grad_combined_h_base[t_offset] = from_float(grad_h);
            grad_combined_g_base[t_offset] = from_float(grad_g);
            grad_combined_p_base[t_offset] = from_float(grad_proj);
        }
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    float scan_result_0 = __expf(a_star_0 + s_0);
    float z_0 = log_value_0 - a_star_0;

    float grad_scan_result_0 = 0.0f;
    float grad_log_h_0 = grad_scan_result_0 * scan_result_0;
    float grad_s_0 = grad_log_h_0;

    acc = grad_s_0 + acc * __expf(s_0 - s_val_next);
    float grad_z_0 = acc * __expf(z_0 - s_0);

    grad_state[state_idx] = from_float(grad_z_0 / to_float(state[state_idx]));
}



// This exactly matches pytorch in double, but not in float
__global__ void logcumsumexp_forward_kernel(
    precision_t* __restrict__ out,           // exp(s[t])
    double* __restrict__ s_buf,     // s[t] = logcumsumexp(x[0..t])
    const precision_t* __restrict__ x,       // input: log_values
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double s = -INFINITY;

    for (int t = 0; t < T_total; t++) {
        int curr = base + t * H;
        double x_val = (double)to_float(x[curr]);
        s = logcumsumexp_forward(x_val, s);
        out[curr] = from_float((float)s);
        s_buf[curr] = s;
    }
}
__global__ void logcumsumexp_backward_kernel(
    precision_t* __restrict__ grad_x,
    const precision_t* __restrict__ grad_out,
    const precision_t* __restrict__ x,
    const double* __restrict__ s_buf,
    int T_total,
    int H,
    int B
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) return;

    int b = idx / H;
    int h = idx % H;

    int base = b * T_total * H + h;

    double acc = 0.0;
    double s_val_next = 0.0;

    for (int t = T_total - 1; t >= 0; --t) {
        int curr = base + t * H;

        double x_val = (double)to_float(x[curr]);
        double s_val = double(s_buf[curr]);
        double g_val = (double)to_float(grad_out[curr]);
        grad_x[curr] = from_float((float)logcumsumexp_backward(x_val, &acc, g_val, s_val, &s_val_next));
    }
}


// ============================================================================
// PPO shared device helpers (used by both forward and backward kernels)
// ============================================================================

// Compute log-probability, entropy, and logsumexp for a single discrete action head.
__device__ __forceinline__ void ppo_discrete_head(
    const precision_t* __restrict__ logits,
    int logits_base, int logits_stride_a, int logits_offset,
    int A, int act,
    float* out_logsumexp, float* out_entropy, float* out_logp
) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    float act_logit = 0.0f;

    for (int a = 0; a < A; ++a) {
        float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
        if (a == act) act_logit = l;
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    float logsumexp = max_logit + __logf(sum);

    float ent = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
        float logp = l - logsumexp;
        float p = __expf(logp);
        ent -= p * logp;
    }

    *out_logsumexp = logsumexp;
    *out_entropy = ent;
    *out_logp = act_logit - logsumexp;
}

// Compute log-probability and entropy for a single continuous action dimension.
__device__ __forceinline__ void ppo_continuous_head(
    float mean, float log_std, float action,
    float* out_logp, float* out_entropy
) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}


// Fused PPO forward + backward kernel: computes loss partials AND gradients in one pass.
// Avoids redundant recomputation of logits, logsumexp, ratio, advantage normalization.
__global__ void ppo_loss_fwd_bwd_kernel(
    float* __restrict__ ppo_partials,    // per-block partial sums for deterministic reduction
    // Gradient outputs
    float* __restrict__ grad_logits,     // For continuous: grad_mean
    float* __restrict__ grad_logstd,     // For continuous: grad_logstd (nullptr for discrete)
    float* __restrict__ grad_values_pred,
    // Inputs
    const precision_t* __restrict__ logits,
    const precision_t* __restrict__ logstd,       // nullptr for discrete
    const precision_t* __restrict__ values_pred,
    const double* __restrict__ actions,
    const precision_t* __restrict__ old_logprobs,
    const float* __restrict__ advantages,
    const precision_t* __restrict__ prio,
    const precision_t* __restrict__ values,
    const precision_t* __restrict__ returns,
    const float* __restrict__ adv_mean,
    const float* __restrict__ adv_var,
    const int* __restrict__ act_sizes,
    int num_atns,
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef,
    int T_seq,
    int A_total,
    int N,
    int logits_stride_n,
    int logits_stride_t,
    int logits_stride_a,
    int values_stride_n,
    int values_stride_t,
    bool is_continuous
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = N * T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx >= total_elements) goto reduce;

    {
    int n = idx / T_seq;
    int t = idx % T_seq;
    int nt = n * T_seq + t;

    int logits_base = n * logits_stride_n + t * logits_stride_t;
    int values_idx = n * values_stride_n + t * values_stride_t;
    int grad_logits_base = nt * A_total;

    // --- Shared computation (used by both forward and backward) ---

    float old_logp = to_float(old_logprobs[nt]);
    float adv = float(advantages[nt]);
    float w = to_float(prio[n]);
    float val = to_float(values[nt]);
    float ret = to_float(returns[nt]);
    float val_pred = to_float(values_pred[values_idx]);

    float adv_std = sqrtf(float(adv_var[0]));
    float adv_normalized = (adv - float(adv_mean[0])) / (adv_std + 1e-8f);

    // grad_loss is always 1.0 (set in post_create, never changes)
    float dL = inv_NT;
    float d_pg_loss = dL;
    float d_entropy_term = dL * (-ent_coef);

    // --- Value loss (forward) + value gradient (backward) ---

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-vf_clip_coef, fminf(vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    // Value gradient
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -vf_clip_coef && v_error <= vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    grad_values_pred[nt] = dL * vf_coef * d_val_pred;

    // --- Policy loss + gradients (branch on continuous/discrete) ---

    if (is_continuous) {
        float total_log_prob = 0.0f;
        float total_entropy = 0.0f;
        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h * logits_stride_a]);
            float log_std = to_float(logstd[logits_base + h * logits_stride_a]);
            float action = float(actions[nt * num_atns + h]);
            float lp, ent;
            ppo_continuous_head(mean, log_std, action, &lp, &ent);
            total_log_prob += lp;
            total_entropy += ent;
        }

        float logratio = total_log_prob - old_logp;
        float ratio = __expf(logratio);
        float ratio_clipped = fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
        float wa = -w * adv_normalized;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        float pg_loss = fmaxf(pg_loss1, pg_loss2);

        // Backward: policy gradient
        float d_ratio = wa * d_pg_loss;
        if (pg_loss2 > pg_loss1) {
            if (ratio <= (1.0f - clip_coef) || ratio >= (1.0f + clip_coef)) {
                d_ratio = 0.0f;
            }
        }
        float d_new_logp = d_ratio * ratio;

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h * logits_stride_a]);
            float log_std = to_float(logstd[logits_base + h * logits_stride_a]);
            float std = __expf(log_std);
            float var = std * std;
            float action = float(actions[nt * num_atns + h]);
            float diff = action - mean;

            grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
            grad_logstd[grad_logits_base + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
        }

        // Forward: loss partials
        float thread_loss = (pg_loss + vf_coef * v_loss - ent_coef * total_entropy) * inv_NT;
        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] = (fabsf(ratio - 1.0f) > clip_coef ? 1.0f : 0.0f) * inv_NT;
    } else {
        // Discrete: compute per-head logsumexp, entropy, log_prob in one pass
        float head_logsumexp[MAX_ATN_HEADS];
        float head_entropy[MAX_ATN_HEADS];
        int head_act[MAX_ATN_HEADS];

        int logits_offset = 0;
        float total_log_prob = 0.0f;
        float total_entropy = 0.0f;

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];
            int act = static_cast<int>(actions[nt * num_atns + h]);
            head_act[h] = act;
            float lse, ent, lp;
            ppo_discrete_head(logits, logits_base, logits_stride_a, logits_offset, A, act, &lse, &ent, &lp);
            head_logsumexp[h] = lse;
            head_entropy[h] = ent;
            total_log_prob += lp;
            total_entropy += ent;
            logits_offset += A;
        }

        float logratio = total_log_prob - old_logp;
        float ratio = __expf(logratio);
        float ratio_clipped = fmaxf(1.0f - clip_coef, fminf(1.0f + clip_coef, ratio));
        float wa = -w * adv_normalized;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        float pg_loss = fmaxf(pg_loss1, pg_loss2);

        // Backward: policy gradient
        float d_ratio = wa * d_pg_loss;
        if (pg_loss2 > pg_loss1) {
            if (ratio <= (1.0f - clip_coef) || ratio >= (1.0f + clip_coef)) {
                d_ratio = 0.0f;
            }
        }
        float d_new_logp = d_ratio * ratio;

        // Gradient pass over logits (reuses head_logsumexp, head_entropy)
        logits_offset = 0;
        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];
            int act = head_act[h];
            float logsumexp = head_logsumexp[h];
            float ent = head_entropy[h];

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + (logits_offset + a) * logits_stride_a]);
                float logp = l - logsumexp;
                float p = __expf(logp);
                float d_logit = (a == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;
                d_logit += d_entropy_term * p * (-ent - logp);
                grad_logits[grad_logits_base + logits_offset + a] = d_logit;
            }
            logits_offset += A;
        }

        // Forward: loss partials
        float thread_loss = (pg_loss + vf_coef * v_loss - ent_coef * total_entropy) * inv_NT;
        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] = (fabsf(ratio - 1.0f) > clip_coef ? 1.0f : 0.0f) * inv_NT;
    }
    } // end if (idx < total_elements)

reduce:
    __syncthreads();

    for (int stride = PPO_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int c = 0; c < LOSS_N; c++) {
                block_losses[c][tid] += block_losses[c][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int base = blockIdx.x * (LOSS_N + 1);
        ppo_partials[base] = block_losses[LOSS_TOTAL][0];
        for (int c = 0; c < LOSS_N; c++) {
            ppo_partials[base + 1 + c] = block_losses[c][0];
        }
    }
}

// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce_kernel(
    float* __restrict__ loss,
    float* __restrict__ losses_acc,
    const float* __restrict__ partials,
    int num_blocks
) {
    int tid = threadIdx.x;
    if (tid > LOSS_N) return;

    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * (LOSS_N + 1) + tid];
    }

    if (tid == 0) {
        *loss += sum;
    } else {
        losses_acc[tid - 1] += sum;
    }

    // Fold add_scalar: increment epoch count
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}



// ============================================================================
// Fused sample_logits kernel: nan_to_num + log_softmax + multinomial + gather + value copy
// Inference-only (no gradients needed)
// Uses inline cuRAND to avoid separate torch::rand() kernel launch
//
// NOTE: This kernel supports strided (non-contiguous) logits/value input.
// This is needed for fused logit+value decoder output where logits is a view
// of a larger (B, V+A) tensor. The stride parameters handle this case
// to avoid .contiguous() kernel launches.
// ============================================================================

// Single kernel that handles both discrete and continuous action sampling
// Discrete: nan_to_num, log_softmax, multinomial sampling, logprob gather, value copy
// Continuous: sample from Normal(mean, exp(logstd)), compute log_prob, value copy
// Input: logits (B, num_atns) for continuous or (B, sum(act_sizes)) for discrete
//        logstd (B, num_atns) for continuous, nullptr for discrete
// Output: actions (B, num_atns) as float64, logprobs (B,), value_out (B,)
// NOTE: offset is read from a pointer (not passed by value) so it works correctly with CUDA graphs.
__global__ void sample_logits_kernel(
    PufTensor dec_out,                    // (B, fused_cols) fused logits+value from decoder
    PufTensor logstd_puf,                 // (1, od) log std for continuous, or empty
    PufTensor act_sizes_puf,              // (num_atns,) action head sizes
    double* __restrict__ actions,         // (B, num_atns) output
    precision_t* __restrict__ logprobs,   // (B,) output
    precision_t* __restrict__ value_out,  // (B,) output
    uint64_t seed,
    const int64_t* __restrict__ offset_ptr
) {
    int B = dec_out.shape[0];
    int fused_cols = dec_out.shape[1];
    int num_atns = act_sizes_puf.numel();
    const int* act_sizes = (const int*)act_sizes_puf.bytes;
    const precision_t* logits = (const precision_t*)dec_out.bytes;
    int logits_stride = fused_cols;
    int value_stride = fused_cols;
    bool is_continuous = logstd_puf.bytes != nullptr && logstd_puf.numel() > 0;
    const precision_t* logstd = (const precision_t*)logstd_puf.bytes;
    int logstd_stride = is_continuous ? 0 : 0;  // 1D broadcast: stride 0
    const precision_t* value = logits + (fused_cols - 1);  // last column

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    // Read offset at execution time (important for CUDA graph replay)
    uint64_t offset = static_cast<uint64_t>(*offset_ptr);

    // Initialize RNG state once per thread
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);

    int logits_base = idx * logits_stride;
    float total_log_prob = 0.0f;

    if (is_continuous) {
        // Continuous action sampling from Normal(mean, exp(logstd))
        constexpr float LOG_2PI = 1.8378770664093453f;  // log(2*pi)
        int logstd_base = idx * logstd_stride;  // separate stride for logstd (may be 0 for broadcast)

        for (int h = 0; h < num_atns; ++h) {
            float mean = to_float(logits[logits_base + h]);
            float log_std = to_float(logstd[logstd_base + h]);
            float std = expf(log_std);

            // Sample from N(0,1) and transform: action = mean + std * noise
            float noise = curand_normal(&state);
            float action = mean + std * noise;

            // Log probability: -0.5 * ((action - mean) / std)^2 - 0.5 * log(2*pi) - log(std)
            float normalized = (action - mean) / std;
            float log_prob = -0.5f * normalized * normalized - 0.5f * LOG_2PI - log_std;

            actions[idx * num_atns + h] = double(action);
            total_log_prob += log_prob;
        }
    } else {
        // Discrete action sampling (original multinomial logic)
        int logits_offset = 0;  // offset within row for current action head

        for (int h = 0; h < num_atns; ++h) {
            int A = act_sizes[h];  // size of this action head

            // Step 1: Find max for numerical stability (with nan_to_num)
            float max_val = -INFINITY;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                max_val = fmaxf(max_val, l);
            }

            // Step 2: Compute logsumexp for log_softmax denominator
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                sum_exp += expf(l - max_val);
            }
            float logsumexp = max_val + logf(sum_exp);

            // Step 3: Generate random value for this action head
            float rand_val = curand_uniform(&state);

            // Step 4: Multinomial sampling using inverse CDF
            float cumsum = 0.0f;
            int sampled_action = A - 1;  // default to last action

            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) l = 0.0f;
                if (isinf(l)) l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = to_float(logits[logits_base + logits_offset + sampled_action]);
            if (isnan(sampled_logit)) sampled_logit = 0.0f;
            if (isinf(sampled_logit)) sampled_logit = (sampled_logit > 0) ? 3.4028e+38f : -3.4028e+38f;
            float log_prob = sampled_logit - logsumexp;

            // Write action for this head
            actions[idx * num_atns + h] = double(sampled_action);
            total_log_prob += log_prob;

            // Advance to next action head
            logits_offset += A;
        }
    }

    // Write summed log probability (log of joint probability)
    logprobs[idx] = from_float(total_log_prob);

    // Copy value (fused to avoid separate elementwise kernel for strided->contiguous copy)
    value_out[idx] = value[idx * value_stride];

    // Increment RNG offset for next call (thread 0 only, fused to avoid separate kernel)
    if (idx == 0) {
        atomicAdd((unsigned long long*)offset_ptr, 1ULL);
    }
}



// =============================================================================
// FCMax: Fused FC -> Max kernel
// Input: x (B, N, D_in), W (D_out, D_in), b (D_out)
// Output: out (B, D_out) = max_over_N(x @ W.T + b)
// Each thread computes one (b, d_out) output element
// N-fold memory bandwidth reduction vs separate FC + Max kernels
// W and b are always float32 (mixed precision for bf16 activations)
// =============================================================================

__global__ void fc_max_forward_kernel(
    precision_t* __restrict__ out,                // (B, D_out)
    int* __restrict__ argmax_indices,   // (B, D_out) - which N produced the max
    const precision_t* __restrict__ x,            // (B, N, D_in)
    const float* __restrict__ W,        // (D_out, D_in) - always float32
    const float* __restrict__ b,        // (D_out) - always float32
    int B, int N, int D_in, int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_out) return;

    int batch = idx / D_out;
    int d_out = idx % D_out;

    float bias = b[d_out];
    float max_val = -INFINITY;
    int argmax_n = 0;

    // Iterate over all N points, compute FC output, track max
    for (int n = 0; n < N; n++) {
        float val = bias;
        for (int di = 0; di < D_in; di++) {
            val += to_float(x[batch * N * D_in + n * D_in + di]) * W[d_out * D_in + di];
        }
        if (val > max_val) {
            max_val = val;
            argmax_n = n;
        }
    }

    out[idx] = from_float(max_val);
    argmax_indices[idx] = argmax_n;
}

// Deterministic backward: three separate kernels with no atomicAdd
// Each kernel assigns threads so that each output element is written by exactly one thread.

// grad_b[d_out] = sum over batch,t of grad_out[batch*D_out + d_out]
// One thread per d_out, serial sum over batch dimension
__global__ void fc_max_backward_grad_b_kernel(
    float* __restrict__ grad_b,             // (D_out)
    const precision_t* __restrict__ grad_out,         // (B, D_out)
    int B, int D_out
) {
    int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_out >= D_out) return;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += to_float(grad_out[b * D_out + d_out]);
    }
    grad_b[d_out] = sum;
}

// grad_W[d_out, d_in] = sum over batch of grad_out[batch, d_out] * x[batch, argmax[batch, d_out], d_in]
// One thread per (d_out, d_in) pair, serial sum over batch dimension
__global__ void fc_max_backward_grad_W_kernel(
    float* __restrict__ grad_W,             // (D_out, D_in)
    const precision_t* __restrict__ grad_out,         // (B, D_out)
    const precision_t* __restrict__ x,                // (B, N, D_in)
    const int* __restrict__ argmax_indices, // (B, D_out)
    int B, int N, int D_in, int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D_out * D_in) return;

    int d_out = idx / D_in;
    int d_in = idx % D_in;

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        float g_out = to_float(grad_out[b * D_out + d_out]);
        int argmax_n = argmax_indices[b * D_out + d_out];
        float x_val = to_float(x[b * N * D_in + argmax_n * D_in + d_in]);
        sum += g_out * x_val;
    }
    grad_W[idx] = sum;
}

// grad_x[batch, n, d_in] = sum over d_out where argmax[batch, d_out]==n of grad_out[batch, d_out] * W[d_out, d_in]
// One thread per (batch, d_in) pair, serial loop over d_out
// Each thread writes to its own batch's grad_x — no cross-thread contention
__global__ void fc_max_backward_grad_x_kernel(
    float* __restrict__ grad_x,             // (B, N, D_in)
    const precision_t* __restrict__ grad_out,         // (B, D_out)
    const float* __restrict__ W,            // (D_out, D_in)
    const int* __restrict__ argmax_indices, // (B, D_out)
    int B, int N, int D_in, int D_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D_in) return;

    int batch = idx / D_in;
    int d_in = idx % D_in;

    // Zero this batch's grad_x column first
    for (int n = 0; n < N; n++) {
        grad_x[batch * N * D_in + n * D_in + d_in] = 0.0f;
    }

    // Accumulate: for each d_out, add grad_out * W to the argmax position
    for (int d_out = 0; d_out < D_out; d_out++) {
        float g_out = to_float(grad_out[batch * D_out + d_out]);
        int argmax_n = argmax_indices[batch * D_out + d_out];
        grad_x[batch * N * D_in + argmax_n * D_in + d_in] += g_out * W[d_out * D_in + d_in];
    }
}


#define SELECT_COPY_THREADS 256

__device__ __forceinline__ void copy_bytes(
    const char* __restrict__ src, char* __restrict__ dst,
    int src_row, int dst_row, int row_bytes
) {
    const int* soffset = (const int*)(src + (int64_t)src_row * row_bytes);
    int* doffset = (int*)(dst + (int64_t)dst_row * row_bytes);
    for (int i = threadIdx.x; i < row_bytes / 4; i += blockDim.x)
        doffset[i] = soffset[i];
}

__device__ __forceinline__ void copy_values_adv_returns(
    const precision_t* __restrict__ src_values, precision_t* __restrict__ dst_values,
    const float* __restrict__ src_advantages, float* __restrict__ dst_advantages,
    precision_t* __restrict__ dst_returns,
    int src_row, int dst_row, int horizon
) {
    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)dst_row * horizon;
    const precision_t* s_values = src_values + srh;
    const float* s_adv = src_advantages + srh;
    precision_t* d_values = dst_values + drh;
    float* d_adv = dst_advantages + drh;
    precision_t* d_returns = dst_returns + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        precision_t val = s_values[i];
        float adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
        d_returns[i] = from_float(to_float(val) + adv);
    }
}

__global__ void select_copy_kernel(
    RolloutBuf rollouts, TrainGraph graph,
    const int64_t* __restrict__ idx,
    const float* __restrict__ advantages, const float* __restrict__ mb_prio
) {
    int mb = blockIdx.x;
    int ch = blockIdx.y;
    int src_row = (int)idx[mb];

    // Compute row byte counts from tensor shapes
    int obs_row_bytes = (rollouts.observations.numel() / rollouts.observations.shape[0]) * rollouts.observations.dtype_size;
    int act_row_bytes = (rollouts.actions.numel() / rollouts.actions.shape[0]) * rollouts.actions.dtype_size;
    int lp_row_bytes = (rollouts.logprobs.numel() / rollouts.logprobs.shape[0]) * rollouts.logprobs.dtype_size;
    int horizon = rollouts.values.shape[1];

    switch (ch) {
    case 0:
        copy_bytes((const char*)rollouts.observations.bytes, graph.mb_obs.bytes, src_row, mb, obs_row_bytes);
        break;
    case 1:
        copy_bytes((const char*)rollouts.actions.bytes, graph.mb_actions.bytes, src_row, mb, act_row_bytes);
        break;
    case 2:
        copy_bytes((const char*)rollouts.logprobs.bytes, graph.mb_logprobs.bytes, src_row, mb, lp_row_bytes);
        break;
    case 3:
        copy_values_adv_returns((const precision_t*)rollouts.values.bytes, (precision_t*)graph.mb_values.bytes,
                advantages, (float*)graph.mb_advantages.bytes,
                (precision_t*)graph.mb_returns.bytes, src_row, mb, horizon);
        break;
    case 4:
        if (threadIdx.x == 0) {
            ((precision_t*)graph.mb_prio.bytes)[mb] = from_float(mb_prio[mb]);
            break;
        }
    }
}

#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)

__global__ void compute_prio_adv_reduction(
    const float* __restrict__ advantages,
    float* prio_weights,
    float prio_alpha,
    int stride
) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(advantages[offset + t]);
    }

    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (tx == 0) {
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) pw = 0.0f;
        prio_weights[row] = pw;
    }
}

__global__ void compute_prio_normalize(
    float* prio_weights,
    int length
) {
    __shared__ float shmem[PRIO_NUM_WARPS];
    __shared__ float block_sum;

    int tx = threadIdx.x;
    int lane = tx % PRIO_WARP_SIZE;
    int warp_id = tx / PRIO_WARP_SIZE;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (lane == 0) shmem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) block_sum = val + eps;
    }
    __syncthreads();

    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

// Part 3: compute importance weights for sampled indices
// mb_prio[i] = pow(total_agents * prio_probs[idx[i]], -anneal_beta)
__global__ void compute_prio_imp_weights(
    const int64_t* __restrict__ indices,
    const float* __restrict__ prio_probs,
    float* mb_prio,
    int total_agents,
    float anneal_beta,
    int minibatch_segments
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = __powf(value, -anneal_beta);
    }
}


// =============================================================================
// Puff Advantage kernels (moved from cuda/advantage.cu)
// =============================================================================

// Advantage kernels: precision_t input, float output (advantages always f32)

__device__ __forceinline__ float p2f(float x) { return x; }
__device__ __forceinline__ float p2f(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ void puff_advantage_row_scalar(
    const precision_t* values, const precision_t* rewards, const precision_t* dones,
    const precision_t* importance, float* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - p2f(dones[t_next]);
        float imp = p2f(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float delta = rho_t*(p2f(rewards[t_next]) + gamma*p2f(values[t_next])*nextnonterminal - p2f(values[t]));
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

__device__ __forceinline__ void adv_vec_load(const float* ptr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

__device__ __forceinline__ void adv_vec_load(const __nv_bfloat16* ptr, float* out) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = __bfloat162float(bf[i]);
}

__device__ __forceinline__ void puff_advantage_row_vec(
    const precision_t* values, const precision_t* rewards, const precision_t* dones,
    const precision_t* importance, float* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    constexpr int N = 16 / sizeof(precision_t);

    float lastpufferlam = 0.0f;
    int num_chunks = horizon / N;

    float next_value = p2f(values[horizon - 1]);
    float next_done = p2f(dones[horizon - 1]);
    float next_reward = p2f(rewards[horizon - 1]);

    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * N;

        float v[N], r[N], d[N], imp[N];
        adv_vec_load(values + base, v);
        adv_vec_load(rewards + base, r);
        adv_vec_load(dones + base, d);
        adv_vec_load(importance + base, imp);

        float adv[N] = {0};
        int start_idx = (chunk == num_chunks - 1) ? (N - 2) : (N - 1);

        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }

        *reinterpret_cast<float4*>(advantages + base) =
            make_float4(adv[0], adv[1], adv[2], adv[3]);
        if (N > 4) {
            *reinterpret_cast<float4*>(advantages + base + 4) =
                make_float4(adv[4], adv[5], adv[6], adv[7]);
        }
    }
}

__global__ void puff_advantage_kernel(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row*horizon;
    puff_advantage_row_vec(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

__global__ void puff_advantage_kernel_scalar(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row*horizon;
    puff_advantage_row_scalar(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}


// ============================================================================
// Element-wise kernels (cast, fill, scale, norm, etc.)
// Host wrappers live in models.cu
// ============================================================================

__global__ void cast_bf16_to_f32_kernel(float* __restrict__ dst, const __nv_bfloat16* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __bfloat162float(src[idx]);
}

__global__ void cast_f32_to_bf16_kernel(__nv_bfloat16* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ void cast_f32_to_bf16_transpose_kernel(__nv_bfloat16* __restrict__ dst, const float* __restrict__ src, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    dst[(idx % C) * R + idx / C] = __float2bfloat16(src[idx]);
}

__global__ void transpose_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, int R, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R * C) return;
    dst[(idx % C) * R + idx / C] = src[idx];
}

template <typename T>
__global__ void transpose_01_kernel(T* __restrict__ dst, const T* __restrict__ src, int A, int B, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A * B * C;
    if (idx >= total) return;
    int a = idx / (B * C), rem = idx % (B * C), b = rem / C, c = rem % C;
    dst[b * A * C + a * C + c] = src[idx];
}

__global__ void norm_bf16_kernel(float* __restrict__ partials, const __nv_bfloat16* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = __bfloat162float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void norm_f32_kernel(float* __restrict__ partials, const float* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) sum += src[i] * src[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void norm_reduce_kernel(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) *out = sdata[0];
}

__global__ void clip_by_norm_f32_kernel(float* __restrict__ dst, const float* __restrict__ sum_sq_ptr,
                                         float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= clip_coef;
}

__global__ void normalize_bf16_kernel(__nv_bfloat16* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(__bfloat162float(dst[idx]) * inv_norm);
}

__global__ void fill_bf16_kernel(__nv_bfloat16* __restrict__ dst, __nv_bfloat16 val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = val;
}

__global__ void fill_f32_kernel(float* __restrict__ dst, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = val;
}

__global__ void clamp_bf16_kernel(__nv_bfloat16* __restrict__ dst, float lo, float hi, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float v = __bfloat162float(dst[idx]); dst[idx] = __float2bfloat16(fminf(fmaxf(v, lo), hi)); }
}

__global__ void scale_f32_kernel(float* __restrict__ dst, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

__global__ void axpy_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

// Fused Nesterov momentum: mb = mu*mb + gc; gc = gc + mu*mb
__global__ void nesterov_f32_kernel(float* __restrict__ mb, float* __restrict__ gc, float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + gc[idx];
        mb[idx] = m;
        gc[idx] += mu * m;
    }
}

__global__ void scale_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= alpha;
}

__global__ void axpy_f32_dev_kernel(float* __restrict__ dst, const float* __restrict__ src, const float* __restrict__ alpha_ptr, int n) {
    float alpha = *alpha_ptr;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += alpha * src[idx];
}

__global__ void compute_lr_scalars_kernel(const float* __restrict__ lr, float wd,
                                           float* __restrict__ neg_lr, float* __restrict__ wd_scale) {
    *neg_lr = -(*lr);
    *wd_scale = 1.0f - (*lr) * wd;
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * up
__global__ void muon_weight_update_kernel(float* __restrict__ wb, const float* __restrict__ up,
                                           const float* __restrict__ lr_ptr, float wd, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * up[idx];
    }
}

__global__ void add_bf16_to_f32_kernel(float* __restrict__ dst, const __nv_bfloat16* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += __bfloat162float(src[idx]);
}

__global__ void sum_rows_add_kernel(float* __restrict__ dst, const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) return;
    float sum = 0.0f;
    for (int r = 0; r < R; r++) sum += src[r * C + col];
    dst[col] += sum;
}

// Sum f32 rows → bf16 output (set, not accumulate)
__global__ void sum_rows_to_bf16_kernel(__nv_bfloat16* __restrict__ dst, const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) return;
    float sum = 0.0f;
    for (int r = 0; r < R; r++) sum += src[r * C + col];
    dst[col] = __float2bfloat16(sum);
}

__global__ void assemble_decoder_grad_kernel(
    __nv_bfloat16* __restrict__ dst, const float* __restrict__ grad_logits,
    const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) return;
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = __float2bfloat16((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

__global__ void var_mean_kernel(const float* __restrict__ src, float* __restrict__ var_out,
        float* __restrict__ mean_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += src[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float mean = sdata[0] / (float)n;
    if (tid == 0) *mean_out = mean;
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) { float d = src[i] - mean; ss += d * d; }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) *var_out = sdata[0] / (float)(n - 1);
}

__global__ void add_scalar_kernel(float* __restrict__ ptr, float val) {
    *ptr += val;
}

__global__ void index_copy_kernel(char* __restrict__ dst, const int64_t* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int64_t dst_row = idx[i];
        memcpy(dst + dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
}

__global__ void cast_u8_to_precision_kernel(precision_t* __restrict__ dst,
        const unsigned char* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float((float)src[idx]);
}

// Orthogonal init helpers (used by puf_orthogonal_init in models.cu)
__global__ void sign_correct_columns_kernel(float* Q, const float* diag_signs, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    Q[idx] *= diag_signs[idx / m];
}

__global__ void extract_diag_sign_kernel(float* signs, const float* A, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    signs[i] = (A[i + (int64_t)i * m] >= 0.0f) ? 1.0f : -1.0f;
}

__global__ void colmaj_to_rowmaj_scale_f32_kernel(float* dst, const float* Q, float gain, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    dst[idx] = Q[(idx / n) + (int64_t)(idx % n) * m] * gain;
}

__global__ void colmaj_to_rowmaj_scale_bf16_kernel(__nv_bfloat16* dst, const float* Q, float gain, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    dst[idx] = __float2bfloat16(Q[(idx / n) + (int64_t)(idx % n) * m] * gain);
}

#endif // PUFFERLIB_KERNELS_CU
