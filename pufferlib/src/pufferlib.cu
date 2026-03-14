#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>

#include "models.cu"
#include "muon.cu"
#include "vecenv.h"

// Loss component indices
enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_N = 7, NUM_LOSSES = 8,
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


void register_prio_buffers(PrioBuffers& bufs, Allocator* alloc, int S, int minibatch_segments) {
    bufs = (PrioBuffers){
        .prio_probs = {.shape = {S}, .dtype_size = sizeof(float)},
        .cdf = {.shape = {S}, .dtype_size = sizeof(float)},
        .idx = {.shape = {minibatch_segments}, .dtype_size = sizeof(long)},
        .mb_prio = {.shape = {minibatch_segments, 1}, .dtype_size = sizeof(float)},
    };
    alloc_register(alloc, &bufs.prio_probs);
    alloc_register(alloc, &bufs.cdf);
    alloc_register(alloc, &bufs.idx);
    alloc_register(alloc, &bufs.mb_prio);
}

void register_ppo_buffers(PPOBuffersPuf& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    int f = sizeof(float);
    bufs = (PPOBuffersPuf){
        .loss_output = {.shape = {1}, .dtype_size = f},
        .saved_for_bwd = {.shape = {total, 5}, .dtype_size = (int)sizeof(double)},
        .grad_loss = {.shape = {1}, .dtype_size = f},
        .grad_logits = {.shape = {N, T, A_total}, .dtype_size = f},
        .grad_values = {.shape = {N, T, 1}, .dtype_size = f},
        .grad_logstd = {.shape = {N, T, A_total}, .dtype_size = f},
        .adv_scratch = {.shape = {2}, .dtype_size = f},
    };
    alloc_register(alloc, &bufs.loss_output);
    alloc_register(alloc, &bufs.saved_for_bwd);
    alloc_register(alloc, &bufs.grad_loss);
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) alloc_register(alloc, &bufs.grad_logstd);
    alloc_register(alloc, &bufs.adv_scratch);
}

void register_rollout_buffers(RolloutBuf& bufs, Allocator* alloc, int H, int S, int input_size, int num_atns) {
    int p = PRECISION_SIZE;
    bufs = (RolloutBuf){
        .observations = {.shape = {H, S, input_size}, .dtype_size = p},
        .actions = {.shape = {H, S, num_atns}, .dtype_size = (int)sizeof(double)},
        .values = {.shape = {H, S}, .dtype_size = p},
        .logprobs = {.shape = {H, S}, .dtype_size = p},
        .rewards = {.shape = {H, S}, .dtype_size = p},
        .terminals = {.shape = {H, S}, .dtype_size = p},
        .ratio = {.shape = {H, S}, .dtype_size = p},
        .importance = {.shape = {H, S}, .dtype_size = p},
    };
    alloc_register(alloc, &bufs.observations);
    alloc_register(alloc, &bufs.actions);
    alloc_register(alloc, &bufs.values);
    alloc_register(alloc, &bufs.logprobs);
    alloc_register(alloc, &bufs.rewards);
    alloc_register(alloc, &bufs.terminals);
    alloc_register(alloc, &bufs.ratio);
    alloc_register(alloc, &bufs.importance);
}

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int S, int H, int input_size,
        int hidden_size, int num_atns, int num_layers) {
    int p = PRECISION_SIZE;
    bufs = (TrainGraph){
        .mb_obs = {.shape = {S, H, input_size}, .dtype_size = p},
        .mb_state = {.shape = {num_layers, S, 1, hidden_size}, .dtype_size = p},
        .mb_actions = {.shape = {S, H, num_atns}, .dtype_size = (int)sizeof(double)},
        .mb_logprobs = {.shape = {S, H}, .dtype_size = p},
        .mb_advantages = {.shape = {S, H}, .dtype_size = (int)sizeof(float)},
        .mb_prio = {.shape = {S, 1}, .dtype_size = p},
        .mb_values = {.shape = {S, H}, .dtype_size = p},
        .mb_returns = {.shape = {S, H}, .dtype_size = p},
        .mb_ratio = {.shape = {S, H}, .dtype_size = p},
        .mb_newvalue = {.shape = {S, H, 1}, .dtype_size = p},
    };
    alloc_register(alloc, &bufs.mb_obs);
    alloc_register(alloc, &bufs.mb_state);
    alloc_register(alloc, &bufs.mb_actions);
    alloc_register(alloc, &bufs.mb_logprobs);
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_prio);
    alloc_register(alloc, &bufs.mb_values);
    alloc_register(alloc, &bufs.mb_returns);
    alloc_register(alloc, &bufs.mb_ratio);
    alloc_register(alloc, &bufs.mb_newvalue);
}

// Minimal CUDA graph wrapper using raw APIs (no torch dependency)
struct RawCudaGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    void capture_begin(cudaStream_t stream) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }
    void capture_end(cudaStream_t stream) {
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, 0);
    }
    void replay(cudaStream_t stream) {
        cudaGraphLaunch(exec, stream);
    }
    void reset() {
        if (exec) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
};

// Slice a PufTensor: select dim0 index t, then narrow dim0 from start for count.
// For a 3D (H, S, F) tensor: returns a view of shape (count, F) at row t*S + start.
// For a 2D (H, S) tensor: returns a view of shape (count,) at offset t*S + start.
inline PufTensor puf_slice(PufTensor& p, int t, int start, int count) {
    if (p.ndim() == 3) {
        long S = p.shape[1], F = p.shape[2];
        return {.bytes = p.bytes + (t*S + start)*F*p.dtype_size,
            .shape = {count, F}, .dtype_size = p.dtype_size};
    } else {
        long S = p.shape[1];
        return {.bytes = p.bytes + (t*S + start)*p.dtype_size,
            .shape = {count}, .dtype_size = p.dtype_size};
    }
}

int obs_dtype_size(int dtype) {
    if (dtype == FLOAT || dtype == INT) {
        return sizeof(float);
    }
    if (dtype == DOUBLE) {
        return sizeof(double);
    }
    return sizeof(char);
}

struct EnvBuf {
    PufTensor obs;        // (total_agents, obs_size) — dtype depends on env (uint8/f32/etc)
    int obs_raw_dtype;    // raw env dtype (FLOAT, INT, UNSIGNED_CHAR, etc.) for bindings to convert
    PufTensor actions;    // (total_agents, num_atns) f64
    PufTensor rewards;    // (total_agents,) f32
    PufTensor terminals;  // (total_agents,) f32
};

StaticVec* create_environments(int num_buffers, int total_agents,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    int obs_type = get_obs_type();
    env.obs = {
        .bytes = (char*)vec->gpu_observations,
        .shape = {total_agents, get_obs_size()},
        .dtype_size = obs_dtype_size(obs_type)
    };
    env.obs_raw_dtype = obs_type;
    env.actions = {
        .bytes = (char*)vec->gpu_actions,
        .shape = {total_agents, get_num_atns()},
        .dtype_size = (int)sizeof(double)
    };
    env.rewards = {
        .bytes = (char*)vec->gpu_rewards,
        .shape = {total_agents},
        .dtype_size = (int)sizeof(float)
    };
    env.terminals = {
        .bytes = (char*)vec->gpu_terminals,
        .shape = {total_agents},
        .dtype_size = (int)sizeof(float)
    };
    return vec;
}

typedef struct {
    // Layout
    int horizon;
    int total_agents;
    int num_buffers;
    // Model architecture
    int num_atns;
    int hidden_size;
    int num_layers;
    // Learning rate
    float lr;
    float min_lr_ratio;
    bool anneal_lr;
    // Optimizer
    float beta1;
    float beta2;
    float eps;
    // Training
    int minibatch_size;
    float replay_ratio;
    long total_timesteps;
    float max_grad_norm;
    // PPO
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    // GAE
    float gamma;
    float gae_lambda;
    // VTrace
    float vtrace_rho_clip;
    float vtrace_c_clip;
    // Priority
    float prio_alpha;
    float prio_beta0;
    // Flags
    bool use_rnn;
    int cudagraphs;
    bool profile;
    // Multi-GPU
    int rank;
    int world_size;
    int gpu_id;
    std::string nccl_id;  // raw bytes of ncclUniqueId (empty for single-GPU)
    // Threading
    int num_threads;
    int seed;
} HypersT;

enum ProfileIdx {
    PROF_ROLLOUT = 0,
    PROF_EVAL_GPU,
    PROF_EVAL_ENV,
    PROF_TRAIN_MISC,
    PROF_TRAIN_FORWARD,
    NUM_PROF,
};

static const char* PROF_NAMES[NUM_PROF] = {
    "rollout",
    "eval_gpu",
    "eval_env",
    "train_misc",
    "train_forward",
};

#define NUM_TRAIN_EVENTS 5  // preloop start/end, loop misc start, forward start/end
typedef struct {
    cudaEvent_t events[NUM_TRAIN_EVENTS];
    float accum[NUM_PROF];
} ProfileT;

typedef struct {
    Policy policy;
    PolicyWeights weights;       // current precision_t weights (structured)
    PolicyActivations train_activations;
    Allocator params_alloc;
    Allocator grads_alloc;
    Allocator activations_alloc;
    StaticVec* vec;
    Muon muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    PufTensor* buffer_states;  // Per-buffer states for contiguous access
    PolicyActivations* buffer_activations;  // Per-buffer inference activations
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    PufTensor advantages_puf;   // Pre-allocated for train_impl (S, H) f32
    RawCudaGraph* fused_rollout_cudagraphs;  // [horizon][num_buffers]
    RawCudaGraph train_cudagraph;
    cudaStream_t* streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    PufTensor act_sizes_puf;   // CUDA int32 PufTensor of action head sizes
    PufTensor losses_puf;      // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for PufTensor ppo_loss_fwd_bwd (kernels path)
    PrioBuffers prio_bufs;      // Pre-allocated buffers for PufTensor prio_replay (kernels path)
    PufTensor master_weights;    // fp32 master weights (flat); same buffer as param_puf in fp32 mode
    PufTensor param_puf;
    PufTensor grad_puf;
    PufTensor rng_offset_puf;    // (num_buffers+1,) int64 CUDA device counters, one per buffer + one for training
    ProfileT profile;
    nvmlDevice_t nvml_device;
    long epoch;
    long global_step;
    double start_time;
    double last_log_time;
    long last_log_step;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    ulong seed;
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    static_vec_log(pufferl.vec, out);
    return out;
}

inline void profile_begin(const char* tag, bool enable) {
    if (enable) {
        cudaDeviceSynchronize();
        nvtxRangePushA(tag);
    }
}

inline void profile_end(bool enable) {
    if (enable) {
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
}

// Thread-local stream for per-buffer threads (set once by thread_init_wrapper)
static thread_local cudaStream_t tl_stream = 0;

// Thread initialization callback - sets thread-local stream once per thread
extern "C" void thread_init_wrapper(void* ctx, int buf) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    tl_stream = pufferl->streams[buf];
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
    if (idx >= B) {
        return;
    }

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
                if (isnan(l)) {
                    l = 0.0f;
                }
                if (isinf(l)) {
                    l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                }
                max_val = fmaxf(max_val, l);
            }

            // Step 2: Compute logsumexp for log_softmax denominator
            float sum_exp = 0.0f;
            for (int a = 0; a < A; ++a) {
                float l = to_float(logits[logits_base + logits_offset + a]);
                if (isnan(l)) {
                    l = 0.0f;
                }
                if (isinf(l)) {
                    l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                }
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
                if (isnan(l)) {
                    l = 0.0f;
                }
                if (isinf(l)) {
                    l = (l > 0) ? 3.4028e+38f : -3.4028e+38f;
                }
                float prob = expf(l - logsumexp);
                cumsum += prob;
                if (rand_val < cumsum) {
                    sampled_action = a;
                    break;
                }
            }

            // Step 5: Gather log probability of sampled action
            float sampled_logit = to_float(logits[logits_base + logits_offset + sampled_action]);
            if (isnan(sampled_logit)) {
                sampled_logit = 0.0f;
            }
            if (isinf(sampled_logit)) {
                sampled_logit = (sampled_logit > 0) ? 3.4028e+38f : -3.4028e+38f;
            }
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

// Called by vecenv per buffer thread
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;
    int graph = t * hypers.num_buffers + buf;
    profile_begin("fused_rollout", hypers.profile);

    cudaStream_t current_stream = tl_stream;
    if (pufferl->rollout_captured) {
        pufferl->fused_rollout_cudagraphs[graph].replay(current_stream);
        profile_end(hypers.profile);
        return;
    }

    bool capturing = pufferl->epoch == hypers.cudagraphs;
    cudaStream_t cap_stream_raw = 0;
    if (capturing) {
        cudaStreamCreate(&cap_stream_raw);
        current_stream = cap_stream_raw;
        pufferl->fused_rollout_cudagraphs[graph].capture_begin(cap_stream_raw);
    }

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    cudaStream_t stream = current_stream;

    // Copy env data to rollout buffer (contiguous slices -> cudaMemcpyAsync)
    PufTensor& obs_env = env.obs;
    PufTensor obs_src = {
        .bytes = obs_env.bytes + (long)start*obs_env.shape[1]*obs_env.dtype_size,
        .shape = {block_size, obs_env.shape[1]},
        .dtype_size = obs_env.dtype_size
    };

    PufTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Cast env obs (uint8/f32/etc) directly into rollout buffer (precision_t)
    int n = obs_src.numel();
    if (obs_env.dtype_size == sizeof(char)) {
        cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)obs_dst.bytes, (const unsigned char*)obs_src.bytes, n);
    } else if (obs_env.dtype_size == sizeof(float)) {
        cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)obs_dst.bytes, (const float*)obs_src.bytes, n);
    } else {
        assert(false && "Unsupported obs dtype: only uint8 and float32 are supported");
    }

    // Rewards/terminals: env is f32, rollout is precision_t - cast via PufTensor
    PufTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    PufTensor rew_src = {
        .bytes = env.rewards.bytes + start * (int)sizeof(float),
        .shape = {block_size},
        .dtype_size = (int)sizeof(float)
    };

    n = rew_src.numel();
    cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)rew_dst.bytes, (const float*)rew_src.bytes, n);

    PufTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    PufTensor term_src = {
        .bytes = env.terminals.bytes + start * (int)sizeof(float),
        .shape = {block_size},
        .dtype_size = (int)sizeof(float)
    };
    n = term_src.numel();
    cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)term_dst.bytes, (const float*)term_src.bytes, n);

    // Forward pass — obs_dst already contains the cast obs in precision_t
    PufTensor state_puf = pufferl->buffer_states[buf];
    PufTensor dec_puf = policy_forward(&pufferl->policy, pufferl->weights, pufferl->buffer_activations[buf], obs_dst, state_puf, stream);

    // Sample actions, logprobs, values into rollout buffer
    PufTensor act_slice = puf_slice(rollouts.actions, t, start, block_size);
    PufTensor lp_slice = puf_slice(rollouts.logprobs, t, start, block_size);
    PufTensor val_slice = puf_slice(rollouts.values, t, start, block_size);

    PufTensor p_logstd;
    DecoderWeights* dw = (DecoderWeights*)pufferl->weights.decoder;
    if (dw->continuous) {
        p_logstd = dw->logstd;
    }

    // Each buffer uses its own RNG seed and offset slot for deterministic parallel rollouts
    long* buf_rng_offset = (long*)pufferl->rng_offset_puf.bytes + buf;
    ulong buf_rng_seed = pufferl->seed + buf;
    sample_logits_kernel<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        dec_puf, p_logstd, pufferl->act_sizes_puf,
        (double*)act_slice.bytes, (precision_t*)lp_slice.bytes, (precision_t*)val_slice.bytes,
        buf_rng_seed, buf_rng_offset);

    // Copy actions to env
    long act_cols = env.actions.shape[1];
    cudaMemcpyAsync(
        env.actions.bytes + start * act_cols * env.actions.dtype_size,
        act_slice.bytes, act_slice.numel() * act_slice.dtype_size, cudaMemcpyDeviceToDevice, stream);

    if (capturing) {
        pufferl->fused_rollout_cudagraphs[graph].capture_end(cap_stream_raw);
        cudaStreamSynchronize(cap_stream_raw);
        cudaDeviceSynchronize();
        cudaStreamDestroy(cap_stream_raw);
    }
    profile_end(hypers.profile);
}

void rollouts_impl(PuffeRL& pufferl) {
    HypersT& hypers = pufferl.hypers;

    int horizon = hypers.horizon;
    int num_buffers = hypers.num_buffers;
    // TODO: You removed state zeros and reward clamping

    for (int i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
        int h = i / num_buffers;

        net_callback_wrapper(&pufferl, buf, h);
        cudaDeviceSynchronize();
    }
}

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
        if (a == act) {
            act_logit = l;
        }
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

__global__ void ppo_loss_fwd_bwd_kernel(
    float* __restrict__ ppo_partials,
    PPOKernelArgs a, PPOGraphArgs g
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx >= total_elements) {
        goto reduce;
    }

    {
    int n = idx / a.T_seq;
    int t = idx % a.T_seq;
    int nt = n * a.T_seq + t;

    int logits_base = n * a.logits_stride_n + t * a.logits_stride_t;
    int values_idx = n * a.values_stride_n + t * a.values_stride_t;
    int grad_logits_base = nt * a.A_total;

    // --- Shared computation (used by both forward and backward) ---

    float old_logp = to_float(g.old_logprobs[nt]);
    float adv = float(g.advantages[nt]);
    float w = to_float(g.prio[n]);
    float val = to_float(g.values[nt]);
    float ret = to_float(g.returns[nt]);
    float val_pred = to_float(a.values_pred[values_idx]);
    g.out_newvalue[nt] = from_float(val_pred);

    float adv_std = sqrtf(float(a.adv_var[0]));
    float adv_normalized = (adv - float(a.adv_mean[0])) / (adv_std + 1e-8f);

    // grad_loss is always 1.0 (set in post_create, never changes)
    float dL = inv_NT;
    float d_pg_loss = dL;
    float d_entropy_term = dL * (-a.ent_coef);

    // --- Value loss (forward) + value gradient (backward) ---

    float v_error = val_pred - val;
    float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
    float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
    float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
    float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);

    // Value gradient
    bool use_clipped_vf = (v_loss_clipped > v_loss_unclipped);
    float d_val_pred = 0.0f;
    if (use_clipped_vf) {
        if (v_error >= -a.vf_clip_coef && v_error <= a.vf_clip_coef) {
            d_val_pred = v_clipped - ret;
        }
    } else {
        d_val_pred = val_pred - ret;
    }
    a.grad_values_pred[nt] = dL * a.vf_coef * d_val_pred;

    // --- Policy loss + gradients ---

    float pg_loss, total_entropy, logratio, ratio;
    float total_log_prob = 0.0f;
    total_entropy = 0.0f;

    // Discrete-only: per-head arrays needed across forward + backward
    float head_logsumexp[MAX_ATN_HEADS];
    float head_entropy[MAX_ATN_HEADS];
    int head_act[MAX_ATN_HEADS];

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = static_cast<int>(g.actions[nt * a.num_atns + h]);
            head_act[h] = act;
            float lse, ent, lp;
            ppo_discrete_head(a.logits, logits_base, a.logits_stride_a, logits_offset, A, act, &lse, &ent, &lp);
            head_logsumexp[h] = lse;
            head_entropy[h] = ent;
            total_log_prob += lp;
            total_entropy += ent;
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = to_float(a.logits[logits_base + h * a.logits_stride_a]);
            float log_std = to_float(a.logstd[logits_base + h * a.logits_stride_a]);
            float action = float(g.actions[nt * a.num_atns + h]);
            float lp, ent;
            ppo_continuous_head(mean, log_std, action, &lp, &ent);
            total_log_prob += lp;
            total_entropy += ent;
        }
    }

    // Shared pg loss computation
    logratio = total_log_prob - old_logp;
    ratio = __expf(logratio);
    g.out_ratio[nt] = from_float(ratio);
    float ratio_clipped = fmaxf(1.0f - a.clip_coef, fminf(1.0f + a.clip_coef, ratio));
    float wa = -w * adv_normalized;
    float pg_loss1 = wa * ratio;
    float pg_loss2 = wa * ratio_clipped;
    pg_loss = fmaxf(pg_loss1, pg_loss2);

    float d_ratio = wa * d_pg_loss;
    if (pg_loss2 > pg_loss1) {
        if (ratio <= (1.0f - a.clip_coef) || ratio >= (1.0f + a.clip_coef)) {
            d_ratio = 0.0f;
        }
    }
    float d_new_logp = d_ratio * ratio;

    if (!a.is_continuous) {
        int logits_offset = 0;
        for (int h = 0; h < a.num_atns; ++h) {
            int A = a.act_sizes[h];
            int act = head_act[h];
            float logsumexp = head_logsumexp[h];
            float ent = head_entropy[h];

            for (int j = 0; j < A; ++j) {
                float l = to_float(a.logits[logits_base + (logits_offset + j) * a.logits_stride_a]);
                float logp = l - logsumexp;
                float p = __expf(logp);
                float d_logit = (j == act) ? d_new_logp : 0.0f;
                d_logit -= p * d_new_logp;
                d_logit += d_entropy_term * p * (-ent - logp);
                a.grad_logits[grad_logits_base + logits_offset + j] = d_logit;
            }
            logits_offset += A;
        }
    } else {
        for (int h = 0; h < a.num_atns; ++h) {
            float mean = to_float(a.logits[logits_base + h * a.logits_stride_a]);
            float log_std = to_float(a.logstd[logits_base + h * a.logits_stride_a]);
            float std = __expf(log_std);
            float var = std * std;
            float action = float(g.actions[nt * a.num_atns + h]);
            float diff = action - mean;

            a.grad_logits[grad_logits_base + h] = d_new_logp * diff / var;
            a.grad_logstd[grad_logits_base + h] = d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
        }
    }

    // Forward: loss partials
    float thread_loss = (pg_loss + a.vf_coef * v_loss - a.ent_coef * total_entropy) * inv_NT;
    block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
    block_losses[LOSS_VF][tid] = v_loss * inv_NT;
    block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
    block_losses[LOSS_TOTAL][tid] = thread_loss;
    block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
    block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
    block_losses[LOSS_CLIPFRAC][tid] = (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
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
    if (tid > LOSS_N) {
        return;
    }

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

__global__ void var_mean_kernel(const float* __restrict__ src, float* __restrict__ var_out,
        float* __restrict__ mean_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += src[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)n;
    if (tid == 0) {
        *mean_out = mean;
    }
    __syncthreads();
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = src[i] - mean;
        ss += d * d;
    }
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *var_out = sdata[0] / (float)(n - 1);
    }
}

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
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;

    // Pointers into fused decoder output
    const precision_t* logits_ptr = (const precision_t*)dec_out.bytes;

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

    PPOGraphArgs graph_args = {
        .out_ratio = (precision_t*)graph.mb_ratio.bytes,
        .out_newvalue = (precision_t*)graph.mb_newvalue.bytes,
        .actions = (const double*)graph.mb_actions.bytes,
        .old_logprobs = (const precision_t*)graph.mb_logprobs.bytes,
        .advantages = (const float*)graph.mb_advantages.bytes,
        .prio = (const precision_t*)graph.mb_prio.bytes,
        .values = (const precision_t*)graph.mb_values.bytes,
        .returns = (const precision_t*)graph.mb_returns.bytes,
    };

    PPOKernelArgs args = {
        .grad_logits = (float*)bufs.grad_logits.bytes,
        .grad_logstd = is_continuous ? (float*)bufs.grad_logstd.bytes : nullptr,
        .grad_values_pred = (float*)bufs.grad_values.bytes,
        .logits = logits_ptr,
        .logstd = is_continuous ? (const precision_t*)logstd.bytes : nullptr,
        .values_pred = logits_ptr + A_total,
        .adv_mean = adv_mean_ptr,
        .adv_var = adv_var_ptr,
        .act_sizes = (const int*)act_sizes.bytes,
        .num_atns = act_sizes.numel(),
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef, .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .logits_stride_n = T * fused_cols, .logits_stride_t = fused_cols, .logits_stride_a = 1,
        .values_stride_n = T * fused_cols, .values_stride_t = fused_cols,
        .is_continuous = is_continuous,
    };

    ppo_loss_fwd_bwd_kernel<<<ppo_grid, PPO_THREADS, 0, stream>>>(ppo_partials_buf, args, graph_args);

    ppo_loss_reduce_kernel<<<1, LOSS_N + 1, 0, stream>>>(
        (float*)bufs.loss_output.bytes, (float*)losses_acc.bytes, ppo_partials_buf, ppo_grid);
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
        if (isnan(pw) || isinf(pw)) {
            pw = 0.0f;
        }
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
    if (lane == 0) {
        shmem[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) {
            block_sum = val + eps;
        }
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

// Multinomial with replacement (uses cuRAND)
__global__ void multinomial_with_replacement_kernel(
        int64_t* __restrict__ out_idx, const float* __restrict__ probs,
        float* __restrict__ cdf, int S, int num_samples,
        uint64_t seed, int64_t* __restrict__ offset_ptr) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float cum = 0.0f;
        for (int i = 0; i < S; i++) {
            cum += probs[i];
            cdf[i] = cum;
        }
    }
    __syncthreads();
    if (tid < num_samples) {
        uint64_t base_off = *offset_ptr;
        curandStatePhilox4_32_10_t rng_state;
        curand_init(seed, base_off + tid, 0, &rng_state);
        float u = curand_uniform(&rng_state);
        int lo = 0, hi = S - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cdf[mid] < u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out_idx[tid] = lo;
    }
    if (tid == 0) {
        atomicAdd((unsigned long long*)offset_ptr, (unsigned long long)num_samples);
    }
}

void prio_replay_cuda(PufTensor& advantages, float prio_alpha,
        int minibatch_segments, int total_agents, float anneal_beta,
        PrioBuffers& bufs, ulong seed, long* offset_ptr, cudaStream_t stream) {
    int S = advantages.shape[0], T = advantages.shape[1];
    compute_prio_adv_reduction<<<S, PRIO_WARP_SIZE, 0, stream>>>(
        (float*)advantages.bytes, (float*)bufs.prio_probs.bytes, prio_alpha, T);
    compute_prio_normalize<<<1, PRIO_BLOCK_SIZE, 0, stream>>>(
        (float*)bufs.prio_probs.bytes, S);
    int block = fmaxf(((minibatch_segments + 31) / 32) * 32, 32);
    multinomial_with_replacement_kernel<<<1, block, 0, stream>>>(
        (long*)bufs.idx.bytes, (float*)bufs.prio_probs.bytes,
        (float*)bufs.cdf.bytes, S, minibatch_segments, seed, offset_ptr);
    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_imp_weights<<<p3_blocks, PRIO_BLOCK_SIZE, 0, stream>>>(
        (long*)bufs.idx.bytes, (float*)bufs.prio_probs.bytes,
        (float*)bufs.mb_prio.bytes, total_agents, anneal_beta, minibatch_segments);
}

__device__ void puff_advantage_row_scalar(
    const precision_t* values, const precision_t* rewards, const precision_t* dones,
    const precision_t* importance, float* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - to_float(dones[t_next]);
        float imp = to_float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float r_nxt = to_float(rewards[t_next]);
        float v = to_float(values[t]);
        float v_nxt = to_float(values[t_next]);
        float delta = rho_t*r_nxt + gamma*v_nxt*nextnonterminal - v;
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
    for (int i = 0; i < 8; i++) {
        out[i] = __bfloat162float(bf[i]);
    }
}

__device__ __forceinline__ void puff_advantage_row_vec(
    const precision_t* values, const precision_t* rewards, const precision_t* dones,
    const precision_t* importance, float* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    constexpr int N = 16 / sizeof(precision_t);

    float lastpufferlam = 0.0f;
    int num_chunks = horizon / N;

    float next_value = to_float(values[horizon - 1]);
    float next_done = to_float(dones[horizon - 1]);
    float next_reward = to_float(rewards[horizon - 1]);

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
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_vec(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

__global__ void puff_advantage_kernel_scalar(const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_scalar(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void puff_advantage_cuda(PufTensor& values, PufTensor& rewards,
        PufTensor& dones, PufTensor& importance, PufTensor& advantages,
        float gamma, float lambda, float rho_clip, float c_clip, cudaStream_t stream) {
    int num_steps = values.shape[0], horizon = values.shape[1];
    assert(advantages.dtype_size == 4 && "advantages must be f32");
    int blocks = grid_size(num_steps);
    constexpr int N = 16 / sizeof(precision_t);
    auto kernel = (horizon % N == 0) ? puff_advantage_kernel : puff_advantage_kernel_scalar;
    kernel<<<blocks, 256, 0, stream>>>(
        (precision_t*)values.bytes, (precision_t*)rewards.bytes,
        (precision_t*)dones.bytes, (precision_t*)importance.bytes,
        (float*)advantages.bytes, gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

__global__ void index_copy_kernel(char* __restrict__ dst, const int64_t* __restrict__ idx,
        const char* __restrict__ src, int num_idx, int row_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_idx) {
        int64_t dst_row = idx[i];
        memcpy(dst + dst_row * row_bytes, src + (int64_t)i * row_bytes, row_bytes);
    }
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

inline float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

void train_impl(PuffeRL& pufferl) {
    // Update to HypersT& p
    HypersT& hypers = pufferl.hypers;

    // Buffers are stored as {horizon, segments, ...} for contiguous rollout writes
    // Transpose to {segments, horizon, ...} for train logic
    cudaEventRecord(pufferl.profile.events[0]);  // pre-loop start
    cudaStream_t train_stream = pufferl.default_stream;

    // Use pre-allocated transposed buffers (segments, horizon, ...)
    RolloutBuf& src = pufferl.rollouts;
    RolloutBuf& rollouts = pufferl.train_rollouts;

    int H = src.observations.shape[0], S = src.observations.shape[1];
    int obs_size = (src.observations.ndim() >= 3) ? src.observations.shape[2] : 1;
    int num_atns = (src.actions.ndim() >= 3) ? src.actions.shape[2] : 1;

    transpose_102<<<grid_size(H*S*obs_size), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.observations.bytes, (precision_t*)src.observations.bytes, H, S, obs_size);
    transpose_102<<<grid_size(H*S*num_atns), BLOCK_SIZE, 0, train_stream>>>(
        (double*)rollouts.actions.bytes, (double*)src.actions.bytes, H, S, num_atns);
    transpose_102<<<grid_size(H*S), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.logprobs.bytes, (precision_t*)src.logprobs.bytes, H, S, 1);
    transpose_102<<<grid_size(H*S), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.rewards.bytes, (precision_t*)src.rewards.bytes, H, S, 1);
    transpose_102<<<grid_size(H*S), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.terminals.bytes, (precision_t*)src.terminals.bytes, H, S, 1);
    transpose_102<<<grid_size(H*S), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.ratio.bytes, (precision_t*)src.ratio.bytes, H, S, 1);
    transpose_102<<<grid_size(H*S), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.values.bytes, (precision_t*)src.values.bytes, H, S, 1);

    // Clamp rewards and fill ratio
    clamp_precision_kernel<<<grid_size(rollouts.rewards.numel()), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.rewards.bytes, -1.0f, 1.0f, rollouts.rewards.numel());
    fill_precision_kernel<<<grid_size(rollouts.ratio.numel()), BLOCK_SIZE, 0, train_stream>>>(
        (precision_t*)rollouts.ratio.bytes, from_float(1.0f), rollouts.ratio.numel());

    // Zero pre-allocated advantages buffer
    PufTensor& advantages_puf = pufferl.advantages_puf;

    // Inline any of these only used once
    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int current_epoch = pufferl.epoch;

    Muon* muon = &pufferl.muon;

    int total_epochs = hypers.total_timesteps / batch_size;

    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        cudaMemcpy(muon->lr_ptr, &lr, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Annealed priority exponent
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;

    TrainGraph& graph = pufferl.train_buf;
    cudaEventRecord(pufferl.profile.events[1]);  // pre-loop end

    for (int mb = 0; mb < total_minibatches; ++mb) {
        cudaEventRecord(pufferl.profile.events[2]);  // start of misc (overwritten each iter)
        puf_zero(&advantages_puf, train_stream);

        profile_begin("compute_advantage", hypers.profile);
        puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        long* train_rng_offset = (long*)pufferl.rng_offset_puf.bytes + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        puf_zero(&graph.mb_state, train_stream);
        {
            RolloutBuf sel_src = rollouts;
            sel_src.values = rollouts.values;
            int mb_segs = pufferl.prio_bufs.idx.shape[0];
            select_copy_kernel<<<dim3(mb_segs, 5), SELECT_COPY_THREADS, 0, train_stream>>>(
                sel_src, graph, (const long*)pufferl.prio_bufs.idx.bytes,
                (const float*)advantages_puf.bytes, (const float*)pufferl.prio_bufs.mb_prio.bytes);
        }
        profile_end(hypers.profile);

        cudaEventRecord(pufferl.profile.events[3]);  // end misc / start forward
        if (pufferl.train_captured) {
            pufferl.train_cudagraph.replay(train_stream);
        } else {
            bool capturing = pufferl.train_warmup == hypers.cudagraphs;
            cudaStream_t cap_stream_raw = train_stream;
            if (capturing) {
                cudaStreamCreate(&cap_stream_raw);
                pufferl.train_cudagraph.capture_begin(cap_stream_raw);
            }

            cudaStream_t stream = cap_stream_raw;
            PufTensor obs_puf = graph.mb_obs;
            PufTensor state_puf = graph.mb_state;
            PufTensor dec_puf = policy_forward_train(&pufferl.policy, pufferl.weights, pufferl.train_activations, obs_puf, state_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights.decoder;
            int od = dw_train->output_dim;
            int fused_cols = od + 1;

            PufTensor p_logstd;
            if (dw_train->continuous) {
                p_logstd = dw_train->logstd;
            }

            ppo_loss_fwd_bwd(dec_puf, p_logstd, graph,
                pufferl.act_sizes_puf, pufferl.losses_puf,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, hypers.ent_coef,
                pufferl.ppo_bufs_puf, pufferl.is_continuous, stream);

            PufTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            PufTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : PufTensor();
            PufTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);

            muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf, hypers.max_grad_norm, stream);
            if (USE_BF16) {
                int n = pufferl.param_puf.numel();
                cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    (precision_t*)pufferl.param_puf.bytes, (const float*)pufferl.master_weights.bytes, n);
            }

            if (capturing) {
                pufferl.train_cudagraph.capture_end(cap_stream_raw);
                cudaStreamSynchronize(cap_stream_raw);
                cudaDeviceSynchronize();
                cudaStreamDestroy(cap_stream_raw);
                pufferl.train_captured = true;
            }
            pufferl.train_warmup++;
        }

        // Bugged version did not have the below updates correct but worked better.
        // Keeping this version until we can resweep hypers etc
        // mb_ratio is (S, H) precision — scatter into rollouts.ratio (S_total, H)
        {
            int num_idx = pufferl.prio_bufs.idx.numel();
            int row_bytes = graph.mb_ratio.numel() / graph.mb_ratio.shape[0] * graph.mb_ratio.dtype_size;
            index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                rollouts.ratio.bytes, (const long*)pufferl.prio_bufs.idx.bytes,
                (const char*)graph.mb_ratio.bytes, num_idx, row_bytes);
        }
        // mb_newvalue is (S, H, 1) — treat as (S, H) for scatter into rollouts.values
        {
            int num_idx = pufferl.prio_bufs.idx.numel();
            int S = graph.mb_newvalue.shape[0], H = graph.mb_newvalue.shape[1];
            int row_bytes = H * graph.mb_newvalue.dtype_size;
            index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                rollouts.values.bytes, (const long*)pufferl.prio_bufs.idx.bytes,
                (const char*)graph.mb_newvalue.bytes, num_idx, row_bytes);
        }
        cudaEventRecord(pufferl.profile.events[4]);  // end forward
    }
    pufferl.epoch += 1;

    cudaStreamSynchronize(pufferl.default_stream);

    if (total_minibatches > 0) {
        float ms;
        // Pre-loop setup (transpose, advantage, allocs)
        cudaEventElapsedTime(&ms, pufferl.profile.events[0], pufferl.profile.events[1]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms;
        // In-loop misc (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[2], pufferl.profile.events[3]);
        pufferl.profile.accum[PROF_TRAIN_MISC] += ms * total_minibatches;
        // In-loop forward (last iteration, representative) scaled by count
        cudaEventElapsedTime(&ms, pufferl.profile.events[3], pufferl.profile.events[4]);
        pufferl.profile.accum[PROF_TRAIN_FORWARD] += ms * total_minibatches;
    }

}

std::unique_ptr<PuffeRL> create_pufferl_impl(HypersT& hypers,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
    auto pufferl = std::make_unique<PuffeRL>();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;

    cudaSetDevice(hypers.gpu_id);

    // Multi-GPU: initialize NCCL
    if (hypers.world_size > 1) {
        if (hypers.nccl_id.size() != sizeof(ncclUniqueId))
            throw std::runtime_error("nccl_id must be " + std::to_string(sizeof(ncclUniqueId)) + " bytes");
        ncclUniqueId nccl_id;
        memcpy(&nccl_id, hypers.nccl_id.data(), sizeof(nccl_id));
        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    // Use CUDA default stream (stream 0) for main-thread work
    pufferl->default_stream = 0;

    ulong seed = hypers.seed + hypers.rank;
    pufferl->seed = seed;

    // Load environment first to get input_size and action info from env
    // Create environments and set up action sizes
    StaticVec* vec = create_environments(hypers.num_buffers, hypers.total_agents,
        env_name, vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;

    int num_action_heads = pufferl->env.actions.shape[1];
    int* raw_act_sizes = get_act_sizes();  // CPU int32 pointer from env
    int act_n = 0;
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        int val = raw_act_sizes[i];
        if (val == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
        act_n += val;
    }
    assert((num_continuous == 0 || num_discrete == 0) &&
        "Mixed continuous/discrete action spaces not supported");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
    }
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));

    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.gpu_id, &pufferl->nvml_device);

    int input_size = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;

    // Decoder output size: discrete = act_n (sum of action sizes), continuous = num_action_heads
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;

    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;
    int B_TT = minibatch_segments * hypers.horizon;
    int psz = PRECISION_SIZE;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .free_weights = encoder_free_weights,
        .in_dim = input_size, .out_dim = hidden_size,
    };
    create_custom_encoder(env_name, &encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .free_weights = decoder_free_weights,
        .hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous,
    };
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .free_weights = mingru_free_weights,
        .hidden = hidden_size, .num_layers = num_layers, .horizon = hypers.horizon,
    };
    pufferl->policy = Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
        .input_dim = input_size, .hidden_dim = hidden_size, .output_dim = decoder_output_size,
        .num_atns = act_n,
    };

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, psz, params);
    pufferl->train_activations = policy_reg_train(&pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    pufferl->buffer_states = (PufTensor*)calloc(num_buffers, sizeof(PufTensor));
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_activations[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {
            .shape = {num_layers, batch, hidden_size},
            .dtype_size = PRECISION_SIZE
        };
        alloc_register(acts, &pufferl->buffer_states[i]);
    }
    register_rollout_buffers(pufferl->rollouts,
        acts, horizon, total_agents, input_size, num_action_heads);
    register_train_buffers(pufferl->train_buf,
        acts, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers);
    register_rollout_buffers(pufferl->train_rollouts,
        acts, total_agents, horizon, input_size, num_action_heads);
    register_ppo_buffers(pufferl->ppo_bufs_puf,
        acts, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);
    register_prio_buffers(pufferl->prio_bufs,
        acts, hypers.total_agents, minibatch_segments);

    // Extra cuda buffers just reuse activ allocator
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}, .dtype_size = (int)sizeof(long)};
    alloc_register(acts, &pufferl->rng_offset_puf);

    pufferl->act_sizes_puf  = {.shape = {num_action_heads}, .dtype_size = (int)sizeof(int)};
    alloc_register(acts, &pufferl->act_sizes_puf);

    pufferl->losses_puf = {.shape = {NUM_LOSSES}, .dtype_size = (int)sizeof(float)};
    alloc_register(acts, &pufferl->losses_puf);

    pufferl->advantages_puf = {.shape = {total_agents, horizon}, .dtype_size = (int)sizeof(float)};
    alloc_register(acts, &pufferl->advantages_puf);

    muon_init(&pufferl->muon, params, hypers.lr, hypers.beta1, hypers.eps, 0.0, acts);
    pufferl->muon.nccl_comm = pufferl->nccl_comm;
    pufferl->muon.world_size = hypers.world_size;

    alloc_create(params);
    alloc_create(grads);
    alloc_create(acts);

    pufferl->grad_puf = {.bytes = (char*)grads->mem,
        .shape = {grads->total_elems}, .dtype_size = psz};
    pufferl->param_puf = {.bytes = (char*)params->mem,
        .shape = {params->total_elems}, .dtype_size = psz};

    ulong init_seed = hypers.seed;
    policy_init_weights(&pufferl->policy, pufferl->weights, &init_seed, pufferl->default_stream);
    pufferl->master_weights = pufferl->param_puf;
    if (USE_BF16) {
        pufferl->master_weights = {.shape = {params->total_elems}, .dtype_size = 4};
        cudaMalloc(&pufferl->master_weights.bytes, params->total_elems * sizeof(float));
        int n = pufferl->param_puf.numel();
        cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
            (float*)pufferl->master_weights.bytes, (const precision_t*)pufferl->param_puf.bytes, n);
    }

    // Post-create initialization
    cudaMemset(pufferl->rng_offset_puf.bytes, 0, (num_buffers + 1) * sizeof(long));
    cudaMemcpy(pufferl->act_sizes_puf.bytes, raw_act_sizes, num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.bytes, 0, NUM_LOSSES * sizeof(float));
    float one = 1.0f;
    cudaMemcpy(&pufferl->ppo_bufs_puf.grad_loss.bytes, &one, sizeof(float), cudaMemcpyHostToDevice);
    muon_post_create(&pufferl->muon);

    if (hypers.cudagraphs >= 0) {
        pufferl->fused_rollout_cudagraphs = (RawCudaGraph*)calloc(horizon*num_buffers, sizeof(RawCudaGraph));
        pufferl->train_warmup = 0;

        // Snapshot weights + optimizer state before init-time capture
        long wb_bytes = pufferl->master_weights.numel() * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->master_weights.bytes, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon.mb_puf.bytes, wb_bytes, cudaMemcpyDeviceToDevice);

        // Checklist for avoiding diabolical capture bugs:
        // 1. Don't start separate streams before tracing (i.e. env gpu buffers)
        // 2. Make sure input/output buffer pointers don't change
        // 3. Make sure to restore the original stream after tracing
        // 4. Scalars get captured by value. They cannot change between calls.
        cudaStream_t saved_default = pufferl->default_stream;
        cudaStream_t saved_tl = tl_stream;
        cudaStream_t warmup_stream;
        cudaStreamCreate(&warmup_stream);
        pufferl->default_stream = warmup_stream;
        tl_stream = warmup_stream;

        for (pufferl->epoch = 0; pufferl->epoch <= hypers.cudagraphs; pufferl->epoch++) {
            rollouts_impl(*pufferl);
        }
        pufferl->rollout_captured = true;

        for (int i = 0; i <= hypers.cudagraphs; i++) {
            train_impl(*pufferl);
        }

        cudaStreamSynchronize(warmup_stream);
        cudaDeviceSynchronize();
        pufferl->default_stream = saved_default;
        tl_stream = saved_tl;
        cudaStreamDestroy(warmup_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        cudaMemcpy(pufferl->master_weights.bytes, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon.mb_puf.bytes, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            int n = pufferl->param_puf.numel();
            cast_kernel<<<grid_size(n), BLOCK_SIZE, 0, pufferl->default_stream>>>(
                (precision_t*)pufferl->param_puf.bytes, (const float*)pufferl->master_weights.bytes, n);
        }

        pufferl->epoch = 0;
        pufferl->global_step = 0;
    }

    // Create per-buffer streams
    pufferl->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
    for (int i = 0; i < num_buffers; i++) {
        cudaStream_t s;
        cudaStreamCreate(&s);
        pufferl->streams[i] = s;
        vec->streams[i] = s;
    }

    create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(),
        net_callback_wrapper, thread_init_wrapper);
    static_vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    pufferl->start_time = now;
    pufferl->last_log_time = now;
    pufferl->last_log_step = 0;

    return pufferl;
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    pufferl.train_cudagraph.reset();
    for (int i = 0; i < pufferl.hypers.horizon * pufferl.hypers.num_buffers; i++) {
        pufferl.fused_rollout_cudagraphs[i].reset();
    }

    policy_weights_free(&pufferl.policy, &pufferl.weights);
    policy_activations_free(pufferl.train_activations);
    for (int buf = 0; buf < pufferl.hypers.num_buffers; buf++) {
        policy_activations_free(pufferl.buffer_activations[buf]);
    }

    if (USE_BF16) {
        cudaFree(pufferl.master_weights.bytes);
    }

    alloc_free(&pufferl.params_alloc);
    alloc_free(&pufferl.grads_alloc);
    alloc_free(&pufferl.activations_alloc);

    for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
        cudaStreamDestroy(pufferl.streams[i]);
    }
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    nvmlShutdown();

    static_vec_close(pufferl.vec);

    free(pufferl.buffer_states);
    free(pufferl.buffer_activations);
    free(pufferl.fused_rollout_cudagraphs);
    free(pufferl.streams);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
}
