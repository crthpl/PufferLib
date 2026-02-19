/* Checklist for avoiding diabolical capture bugs:
 * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
 * 2. Make sure input/output buffer pointers don't change
 * 3. Make sure to restore the original stream after tracing
 * 4. All custom kernels need to use the default torch stream
 * 5. Make sure you are using the torch stream fns, not the c10 ones.
 * 6. Scalars get captured by value. They cannot change between calls.
 */

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nccl.h>
#include <unistd.h>
#include <vector>
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>

#include "models.cu"
#include "vecenv.h"

namespace pufferlib {
Policy* create_policy(const std::string& env_name, Allocator& alloc,
        int input_size, int hidden_size,
        int decoder_output_size, int num_layers, int act_n, bool is_continuous, bool kernels) {
    return new Policy(alloc, input_size, hidden_size, decoder_output_size, num_layers, act_n, is_continuous);
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
    PufTensor s;
    s.dtype_size = p.dtype_size;
    if (p.ndim == 3) {
        int64_t S = p.shape[1], F = p.shape[2];
        s.bytes = p.bytes + (t * S + start) * F * p.dtype_size;
        s.shape[0] = count;
        s.shape[1] = F;
        for (int i = 2; i < PUF_MAX_DIMS; i++) s.shape[i] = 0;
        s.ndim = 2;
    } else {
        int64_t S = p.shape[1];
        s.bytes = p.bytes + (t * S + start) * p.dtype_size;
        s.shape[0] = count;
        for (int i = 1; i < PUF_MAX_DIMS; i++) s.shape[i] = 0;
        s.ndim = 1;
    }
    return s;
}

int obs_dtype_size(int dtype) {
    if (dtype == FLOAT || dtype == INT) {
        return sizeof(float);
    }
    if (dtype == DOUBLE) {
        return sizeof(double);
    }
    return sizeof(char);  // UNSIGNED_CHAR, CHAR
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
    printf("DEBUG create_environments: vec->size=%d, vec->total_agents=%d\n",
        vec->size, vec->total_agents);

    int obs_size = get_obs_size();
    int num_atns = get_num_atns();
    int obs_type = get_obs_type();

    auto mk = [](void* ptr, std::initializer_list<int64_t> dims, int dsz) {
        PufTensor p;
        p.bytes = (char*)ptr;
        p.dtype_size = dsz;
        p.ndim = dims.size();
        int i = 0;
        for (auto d : dims) {
            p.shape[i++] = d;
        }
        for (; i < PUF_MAX_DIMS; i++) {
            p.shape[i] = 0;
        }
        return p;
    };
    env.obs = mk(vec->gpu_observations, {total_agents, obs_size}, obs_dtype_size(obs_type));
    env.obs_raw_dtype = obs_type;
    env.actions = mk(vec->gpu_actions, {total_agents, num_atns}, sizeof(double));
    env.rewards = mk(vec->gpu_rewards, {total_agents}, sizeof(float));
    env.terminals = mk(vec->gpu_terminals, {total_agents}, sizeof(float));

    return vec;
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

    void register_buffers(Allocator& alloc, int S, int H, int input_size,
            int hidden_size, int num_atns, int num_layers) {
        int psz = PRECISION_SIZE;
        alloc.register_puf(&mb_obs, {S, H, input_size}, psz);
        alloc.register_puf(&mb_state, {num_layers, S, 1, hidden_size}, psz);
        alloc.register_puf(&mb_actions, {S, H, num_atns}, sizeof(double));
        alloc.register_puf(&mb_logprobs, {S, H}, psz);
        alloc.register_puf(&mb_advantages, {S, H}, sizeof(float));
        alloc.register_puf(&mb_prio, {S, 1}, psz);
        alloc.register_puf(&mb_values, {S, H}, psz);
        alloc.register_puf(&mb_returns, {S, H}, psz);
        alloc.register_puf(&mb_ratio, {S, H}, psz);
        alloc.register_puf(&mb_newvalue, {S, H, 1}, psz);
    }
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

    // Rename these. H, S
    void register_buffers(Allocator& alloc, int dim0, int dim1, int input_size, int num_atns) {
        int psz = PRECISION_SIZE;
        alloc.register_puf(&observations, {dim0, dim1, input_size}, psz);
        alloc.register_puf(&actions, {dim0, dim1, num_atns}, sizeof(double));
        alloc.register_puf(&values, {dim0, dim1}, psz);
        alloc.register_puf(&logprobs, {dim0, dim1}, psz);
        alloc.register_puf(&rewards, {dim0, dim1}, psz);
        alloc.register_puf(&terminals, {dim0, dim1}, psz);
        alloc.register_puf(&ratio, {dim0, dim1}, psz);
        alloc.register_puf(&importance, {dim0, dim1}, psz);
    }
};

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
    int cudagraphs;  // epoch at which to capture graph, -1 to disable
    bool kernels;
    bool profile;
    // Multi-GPU
    int rank;
    int world_size;
    std::string nccl_id_path;
    // Threading
    int num_threads;
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
    Policy* policy_bf16;  // Working weights (bf16) - used for forward/backward
    Policy* policy_fp32;  // Master weights (fp32) - used for optimizer
    Allocator alloc_fp32; // Contiguous param+grad buffers for fp32 policy
    Allocator alloc_bf16; // Contiguous param buffer for bf16 policy
    Allocator pufferl_alloc; // Consolidated allocator for all init-to-close buffers
    StaticVec* vec;
    Muon* muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    vector<PufTensor> buffer_states;  // Per-buffer states for contiguous access
    vector<PolicyActivations> buffer_acts;  // Per-buffer inference activations
    vector<Allocator> buffer_allocs;        // Per-buffer allocators for inference buffers
    RolloutBuf rollouts;
    RolloutBuf train_rollouts;  // Pre-allocated transposed copy for train_impl
    EnvBuf env;
    TrainGraph train_buf;
    PufTensor old_values_puf;   // Pre-allocated for train_impl (S, H) PRECISION
    PufTensor advantages_puf;   // Pre-allocated for train_impl (S, H) f32
    vector<vector<RawCudaGraph>> fused_rollout_cudagraphs;  // [horizon][num_buffers]
    RawCudaGraph train_cudagraph;
    vector<cudaStream_t> streams;  // per-buffer raw CUDA streams
    cudaStream_t default_stream;  // main-thread stream (captured once at init)
    PufTensor act_sizes_puf;   // CUDA int32 PufTensor of action head sizes
    PufTensor losses_puf;      // (NUM_LOSSES,) f32 accumulator
    PPOBuffersPuf ppo_bufs_puf; // Pre-allocated buffers for PufTensor ppo_loss_fwd_bwd (kernels path)
    PrioBuffers prio_bufs;      // Pre-allocated buffers for PufTensor prio_replay (kernels path)
    PufTensor grad_buffer_puf;   // cached PufTensor view of alloc_fp32.grad_buffer
    PufTensor param_fp32_puf;    // cached PufTensor view of alloc_fp32.param_buffer
    PufTensor param_bf16_puf;    // cached PufTensor view of alloc_bf16.param_buffer
    PufTensor grad_norm_puf;     // (1,) f32 scratch for clip_grad_norm
    PufTensor rng_offset_puf;    // (num_buffers+1,) int64 CUDA device counters, one per buffer + one for training
    ProfileT profile;
    nvmlDevice_t nvml_device;
    int epoch;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    uint64_t rng_seed;
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    static_vec_log(pufferl.vec, out);
    return out;
}

// ============================================================================
// Rollout and train section functions
// ============================================================================

//TODO: Profile without sync
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

// Called by vecenv per buffer thread
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;
    profile_begin("fused_rollout", hypers.profile);

    cudaStream_t current_stream = tl_stream;

    if (pufferl->rollout_captured) {
        pufferl->fused_rollout_cudagraphs[t][buf].replay(current_stream);
        profile_end(hypers.profile);
        return;
    }

    bool capturing = pufferl->epoch == hypers.cudagraphs;
    cudaStream_t cap_stream_raw = 0;
    if (capturing) {
        cudaStreamCreate(&cap_stream_raw);
        current_stream = cap_stream_raw;
        pufferl->fused_rollout_cudagraphs[t][buf].capture_begin(cap_stream_raw);
    }

    RolloutBuf& rollouts = pufferl->rollouts;
    EnvBuf& env = pufferl->env;
    int block_size = pufferl->vec->total_agents / hypers.num_buffers;
    int start = buf * block_size;
    cudaStream_t stream = current_stream;

    // Copy env data to rollout buffer (contiguous slices → cudaMemcpyAsync)
    PufTensor& obs_env = env.obs;
    PufTensor obs_src;
    obs_src.bytes = obs_env.bytes + (int64_t)start * obs_env.shape[1] * obs_env.dtype_size;
    obs_src.shape[0] = block_size;
    obs_src.shape[1] = obs_env.shape[1];
    obs_src.ndim = 2;
    obs_src.dtype_size = obs_env.dtype_size;

    PufTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Cast env obs (uint8/f32/etc) directly into rollout buffer (precision_t)
    if (obs_env.dtype_size == sizeof(char)) {
        puf_cast_u8_to_precision(obs_dst, obs_src, stream);
    } else if (obs_env.dtype_size == sizeof(float)) {
        puf_cast_f32_to_precision(obs_dst, obs_src, stream);
    } else {
        assert(false && "Unsupported obs dtype: only uint8 and float32 are supported");
    }

    // Rewards/terminals: env is f32, rollout is precision_t — cast via PufTensor
    PufTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    PufTensor rew_src;
    rew_src.bytes = env.rewards.bytes + start * sizeof(float);
    rew_src.shape[0] = block_size;
    rew_src.ndim = 1;
    rew_src.dtype_size = sizeof(float);

    if (USE_BF16) {
        puf_cast_f32_to_bf16(rew_dst, rew_src, stream);
    } else {
        puf_copy(rew_dst, rew_src, stream);
    }

    PufTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    PufTensor term_src;
    term_src.bytes = env.terminals.bytes + start * sizeof(float);
    term_src.shape[0] = block_size;
    term_src.ndim = 1;
    term_src.dtype_size = sizeof(float);
    if (USE_BF16) {
        puf_cast_f32_to_bf16(term_dst, term_src, stream);
    } else {
        puf_copy(term_dst, term_src, stream);
    }

    // Forward pass — obs_dst already contains the cast obs in precision_t
    PufTensor state_puf = pufferl->buffer_states[buf];
    PufTensor dec_puf = pufferl->policy_bf16->forward(obs_dst, state_puf, pufferl->buffer_acts[buf], stream);

    // Sample actions, logprobs, values into rollout buffer
    PufTensor act_slice = puf_slice(rollouts.actions, t, start, block_size);
    PufTensor lp_slice = puf_slice(rollouts.logprobs, t, start, block_size);
    PufTensor val_slice = puf_slice(rollouts.values, t, start, block_size);

    int od = pufferl->policy_bf16->decoder.output_dim;
    // dec_puf is (B, od+1): first od cols = logits, last col = value
    int fused_cols = od + 1;
    PufTensor p_logits;
    p_logits.bytes = dec_puf.bytes;
    p_logits.shape[0] = block_size;
    p_logits.shape[1] = od;
    p_logits.ndim = 2;
    p_logits.dtype_size = PRECISION_SIZE;

    PufTensor p_value;
    p_value.bytes = dec_puf.bytes + od * PRECISION_SIZE;
    p_value.shape[0] = block_size;
    p_value.shape[1] = 1;
    p_value.ndim = 2;
    p_value.dtype_size = PRECISION_SIZE;

    PufTensor p_logstd;
    if (pufferl->policy_bf16->decoder.continuous) {
        p_logstd = pufferl->policy_bf16->decoder.logstd;
    }

    PufTensor p_act_sizes = pufferl->act_sizes_puf;
    // logstd is 1D (od,) broadcast across batch → stride 0
    // Each buffer uses its own RNG seed and offset slot for deterministic parallel rollouts
    int64_t* buf_rng_offset = (int64_t*)pufferl->rng_offset_puf.bytes + buf;
    uint64_t buf_rng_seed = pufferl->rng_seed + buf;
    sample_logits(p_logits, p_logstd, p_value, act_slice, lp_slice, val_slice,
        p_act_sizes, buf_rng_seed, buf_rng_offset,
        fused_cols, 0, fused_cols, stream);

    // Copy actions to env
    int64_t act_cols = env.actions.shape[1];
    cudaMemcpyAsync(
        env.actions.bytes + start * act_cols * env.actions.dtype_size,
        act_slice.bytes, act_slice.nbytes(), cudaMemcpyDeviceToDevice, stream);

    if (capturing) {
        pufferl->fused_rollout_cudagraphs[t][buf].capture_end(cap_stream_raw);
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

    puf_transpose_01(rollouts.observations, src.observations, train_stream);
    puf_transpose_01(rollouts.actions, src.actions, train_stream);
    puf_transpose_01(rollouts.logprobs, src.logprobs, train_stream);
    puf_transpose_01(rollouts.rewards, src.rewards, train_stream);
    puf_transpose_01(rollouts.terminals, src.terminals, train_stream);
    puf_transpose_01(rollouts.ratio, src.ratio, train_stream);
    puf_transpose_01(rollouts.values, src.values, train_stream);

    // Clamp rewards and fill ratio with PufTensor ops
    puf_clamp(rollouts.rewards, -1.0f, 1.0f, train_stream);
    puf_fill(rollouts.ratio, 1.0f, train_stream);

    // old_values = values.clone() via pre-allocated PufTensor
    PufTensor& old_values_puf = pufferl.old_values_puf;
    puf_copy(old_values_puf, rollouts.values, train_stream);

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

    Muon* muon = pufferl.muon;

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
        puf_zero(advantages_puf, train_stream);

        profile_begin("compute_advantage", hypers.profile);
        puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
            rollouts.ratio, advantages_puf, hypers.gamma, hypers.gae_lambda,
            hypers.vtrace_rho_clip, hypers.vtrace_c_clip, train_stream);
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        int64_t* train_rng_offset = (int64_t*)pufferl.rng_offset_puf.bytes + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.rng_seed, train_rng_offset, train_stream);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        train_select_and_copy_cuda(rollouts.observations, rollouts.actions,
            rollouts.logprobs, old_values_puf, advantages_puf,
            pufferl.prio_bufs.idx, pufferl.prio_bufs.mb_prio,
            graph.mb_obs, graph.mb_state, graph.mb_actions,
            graph.mb_logprobs, graph.mb_advantages, graph.mb_prio,
            graph.mb_values, graph.mb_returns, train_stream);
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
            PufTensor dec_puf = pufferl.policy_bf16->forward_train(obs_puf, state_puf, pufferl.policy_bf16->act, stream);
            int od = pufferl.policy_bf16->decoder.output_dim;
            int fused_cols = od + 1;

            PufTensor grad_logits_puf, grad_logstd_puf, grad_values_puf;

            // Split fused decoder output into logits (B, od) and value (B, 1) PufTensors
            {
                PufTensor p_logits;
                p_logits.bytes = dec_puf.bytes;
                p_logits.shape[0] = graph.mb_obs.shape[0];
                p_logits.shape[1] = graph.mb_obs.shape[1];
                p_logits.shape[2] = od;
                p_logits.ndim = 3;
                p_logits.dtype_size = PRECISION_SIZE;

                PufTensor p_value;
                p_value.bytes = dec_puf.bytes + od * PRECISION_SIZE;
                p_value.shape[0] = graph.mb_obs.shape[0];
                p_value.shape[1] = graph.mb_obs.shape[1];
                p_value.shape[2] = 1;
                p_value.ndim = 3;
                p_value.dtype_size = PRECISION_SIZE;

                PufTensor p_logstd;
                if (pufferl.policy_bf16->decoder.continuous) {
                    p_logstd = pufferl.policy_bf16->decoder.logstd;
                }

                PufTensor p_act_sizes = pufferl.act_sizes_puf;

                // Strides for the fused (N, T, od+1) layout
                int N = graph.mb_obs.shape[0], T = graph.mb_obs.shape[1];
                int logits_stride_n = T * fused_cols;
                int logits_stride_t = fused_cols;
                int logits_stride_a = 1;
                int values_stride_n = T * fused_cols;
                int values_stride_t = fused_cols;

                // PufTensor views for newvalue_out shaped as (N, T) for ratio scatter
                PufTensor p_newvalue_out;
                p_newvalue_out.bytes = graph.mb_newvalue.bytes;
                p_newvalue_out.shape[0] = N;
                p_newvalue_out.shape[1] = T;
                p_newvalue_out.ndim = 2;
                p_newvalue_out.dtype_size = PRECISION_SIZE;

                ppo_loss_fwd_bwd(
                    p_logits, p_logstd, p_value,
                    graph.mb_actions, graph.mb_logprobs, graph.mb_advantages,
                    graph.mb_prio, graph.mb_values, graph.mb_returns,
                    graph.mb_ratio, p_newvalue_out,
                    p_act_sizes, pufferl.losses_puf,
                    hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, hypers.ent_coef,
                    pufferl.ppo_bufs_puf, pufferl.is_continuous,
                    logits_stride_n, logits_stride_t, logits_stride_a,
                    values_stride_n, values_stride_t, stream);

                grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
                grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : PufTensor();
                grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            }
            pufferl.policy_bf16->backward(
                grad_logits_puf, grad_logstd_puf, grad_values_puf,
                pufferl.policy_bf16->act, pufferl.policy_fp32, stream);

            clip_grad_norm_(pufferl.grad_buffer_puf, hypers.max_grad_norm, (float*)pufferl.grad_norm_puf.bytes, stream);
            pufferl.muon->step(stream);
            pufferl.muon->zero_grad(stream);
            if (USE_BF16) {
                puf_cast_f32_to_bf16(pufferl.param_bf16_puf, pufferl.param_fp32_puf, stream);
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

        // mb_ratio is (S, H) precision — scatter into rollouts.ratio (S_total, H)
        puf_index_copy(rollouts.ratio, pufferl.prio_bufs.idx, graph.mb_ratio, train_stream);
        // mb_newvalue is (S, H, 1) — treat as (S, H) for scatter into rollouts.values
        PufTensor nv_2d;
        nv_2d.bytes = graph.mb_newvalue.bytes;
        nv_2d.shape[0] = graph.mb_newvalue.shape[0];
        nv_2d.shape[1] = graph.mb_newvalue.shape[1];
        nv_2d.ndim = 2;
        nv_2d.dtype_size = graph.mb_newvalue.dtype_size;
        puf_index_copy(rollouts.values, pufferl.prio_bufs.idx, nv_2d, train_stream);
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

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers,
        const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->hypers = hypers;
    pufferl->nccl_comm = nullptr;

    // Multi-GPU: initialize NCCL (device already set by Python)
    if (hypers.world_size > 1) {
        ncclUniqueId nccl_id;
        if (hypers.rank == 0) {
            ncclGetUniqueId(&nccl_id);
            FILE* f = fopen(hypers.nccl_id_path.c_str(), "wb");
            fwrite(&nccl_id, sizeof(nccl_id), 1, f);
            fclose(f);
        }
        // Wait for rank 0 to write the ID file
        while (access(hypers.nccl_id_path.c_str(), F_OK) != 0) {
            usleep(10000);  // 10ms
        }
        if (hypers.rank != 0) {
            // Small delay to ensure file is fully written
            usleep(50000);
            FILE* f = fopen(hypers.nccl_id_path.c_str(), "rb");
            if (fread(&nccl_id, sizeof(nccl_id), 1, f) != 1) {
                fclose(f);
                throw std::runtime_error("Failed to read NCCL ID file");
            }
            fclose(f);
        }

        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    // Use CUDA default stream (stream 0) for main-thread work
    pufferl->default_stream = 0;

    // TODO: Base seed should come from train config
    int seed = 42 + hypers.rank;
    pufferl->rng_seed = seed;

    // Load environment first to get input_size and action info from env
    // Create environments and set up action sizes
    StaticVec* vec = create_environments(hypers.num_buffers, hypers.total_agents,
        env_name, vec_kwargs, env_kwargs, pufferl->env);
    pufferl->vec = vec;

    int num_action_heads = pufferl->env.actions.size(1);
    int* raw_act_sizes = get_act_sizes();  // CPU int32 pointer from env
    int act_n = 0;
    for (int i = 0; i < num_action_heads; i++) {
        act_n += raw_act_sizes[i];
    }

    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventCreate(&pufferl->profile.events[i]);
    }
    memset(pufferl->profile.accum, 0, sizeof(pufferl->profile.accum));

    nvmlInit();
    nvmlDeviceGetHandleByIndex(hypers.rank, &pufferl->nvml_device);

    // Determine action space type
    int* act_sizes_ptr = get_act_sizes();
    int num_continuous = 0;
    int num_discrete = 0;
    for (int i = 0; i < num_action_heads; i++) {
        if (act_sizes_ptr[i] == 1) {
            num_continuous++;
        } else {
            num_discrete++;
        }
    }
    assert((num_continuous == 0 || num_discrete == 0) &&
        "Mixed continuous/discrete action spaces not supported");
    pufferl->is_continuous = (num_continuous > 0);
    if (pufferl->is_continuous) {
        printf("Detected continuous action space with %d dimensions\n", num_action_heads);
    } else {
        printf("Detected discrete action space with %d heads\n", num_action_heads);
    }

    int input_size = pufferl->env.obs.shape[1];
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;

    // Decoder output size: discrete = act_n (sum of action sizes), continuous = num_action_heads
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;

    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int inf_batch = vec->total_agents / hypers.num_buffers;

    // Create fp32 master policy (for optimizer - precise gradient accumulation)
    Policy* policy_fp32 = create_policy(env_name, pufferl->alloc_fp32,
        input_size, hidden_size, decoder_output_size, num_layers, act_n, is_continuous, kernels);
    if (!USE_BF16) {
        // fp32-only mode: fp32 policy also runs forward/backward, needs activations
        policy_fp32->register_activations(pufferl->alloc_fp32, minibatch_segments, hypers.horizon);
    }
    pufferl->alloc_fp32.create(sizeof(float));
    auto mk_1d = [](void* data, int64_t n, int dsz) {
        PufTensor p;
        p.bytes = (char*)data;
        p.shape[0] = n;
        p.ndim = 1;
        p.dtype_size = dsz;
        for (int i = 1; i < PUF_MAX_DIMS; i++) {
            p.shape[i] = 0;
        }
        return p;
    };
    pufferl->grad_buffer_puf = mk_1d(pufferl->alloc_fp32.grad_mem, pufferl->alloc_fp32.total_grad_elems, sizeof(float));
    pufferl->param_fp32_puf = mk_1d(pufferl->alloc_fp32.param_mem, pufferl->alloc_fp32.total_param_elems, sizeof(float));
    policy_fp32->init_weights(pufferl->default_stream);
    pufferl->policy_fp32 = policy_fp32;

    if (USE_BF16) {
        // create bf16 working policy (for fwd/bwd — no grads needed)
        Policy* policy_bf16 = create_policy(env_name, pufferl->alloc_bf16,
            input_size, hidden_size, decoder_output_size, num_layers, act_n, is_continuous, kernels);
        policy_bf16->register_activations(pufferl->alloc_bf16, minibatch_segments, hypers.horizon);
        pufferl->alloc_bf16.create(2);
        pufferl->param_bf16_puf = mk_1d(pufferl->alloc_bf16.param_mem, pufferl->alloc_bf16.total_param_elems, 2);
        pufferl->policy_bf16 = policy_bf16;
        // Initial sync: fp32 → bf16
        puf_cast_f32_to_bf16(pufferl->param_bf16_puf, pufferl->param_fp32_puf,
            pufferl->default_stream);
    } else {
        pufferl->policy_bf16 = policy_fp32;
    }

    // Optimizer uses fp32 master weights with contiguous buffers from allocator
    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new Muon(policy_fp32->param_shapes(),
        pufferl->param_fp32_puf, pufferl->grad_buffer_puf,
        lr, beta1, eps, 0.0);
    pufferl->muon->nccl_comm = pufferl->nccl_comm;
    pufferl->muon->world_size = hypers.world_size;
    printf("DEBUG: Contiguous weight buffer: %ld elements\n", pufferl->muon->wb_puf.numel());

    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    printf("DEBUG: num_envs=%d, total_agents=%d, batch=%d, num_buffers=%d\n",
        vec->size, total_agents, batch, num_buffers);

    // ========================================================================
    // Register all init-to-close buffers into pufferl_alloc, then create once
    // ========================================================================
    Allocator& alloc = pufferl->pufferl_alloc;

    // Misc scalars
    alloc.register_puf(&pufferl->rng_offset_puf, {num_buffers + 1}, sizeof(int64_t));
    alloc.register_puf(&pufferl->act_sizes_puf, {num_action_heads}, sizeof(int32_t));
    alloc.register_puf(&pufferl->losses_puf, {NUM_LOSSES}, sizeof(float));
    alloc.register_puf(&pufferl->grad_norm_puf, {1}, sizeof(float));

    // Per-buffer RNN states
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        alloc.register_puf(&pufferl->buffer_states[i],
            {num_layers, batch, hidden_size}, PRECISION_SIZE);
    }

    // Rollout buffers (horizon, total_agents, ...)
    pufferl->rollouts.register_buffers(alloc, horizon, total_agents, input_size, num_action_heads);

    // Train graph buffers
    pufferl->train_buf.register_buffers(alloc, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers);

    // Pre-allocated transposed rollouts for train_impl (total_agents, horizon, ...)
    pufferl->train_rollouts.register_buffers(alloc, total_agents, horizon, input_size, num_action_heads);

    // Pre-allocated train temporaries
    alloc.register_puf(&pufferl->old_values_puf, {total_agents, horizon}, PRECISION_SIZE);
    alloc.register_puf(&pufferl->advantages_puf, {total_agents, horizon}, sizeof(float));

    // PPO loss buffers
    pufferl->ppo_bufs_puf.register_buffers(alloc, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);

    // Priority replay buffers
    pufferl->prio_bufs.register_buffers(alloc, hypers.total_agents, minibatch_segments);

    // Muon optimizer buffers
    pufferl->muon->register_buffers(alloc);

    // Single allocation for all registered buffers
    alloc.create(1);

    // Post-create initialization: copy data, set constants
    cudaMemset(pufferl->rng_offset_puf.bytes, 0, (num_buffers + 1) * sizeof(int64_t));
    cudaMemcpy(pufferl->act_sizes_puf.bytes, raw_act_sizes, num_action_heads * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.bytes, 0, NUM_LOSSES * sizeof(float));
    pufferl->ppo_bufs_puf.post_create();
    pufferl->muon->post_create();

    // Per-buffer inference activations (separate allocators — different lifetime)
    pufferl->buffer_acts.reserve(num_buffers);
    pufferl->buffer_allocs.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_acts.emplace_back(num_layers);
    }
    // Register and allocate per-buffer inference activations
    // (must be done after emplace_back so buffer_acts won't move)
    for (int i = 0; i < num_buffers; i++) {
        pufferl->policy_bf16->register_inference(
            pufferl->buffer_allocs[i], pufferl->buffer_acts[i], inf_batch);
        pufferl->buffer_allocs[i].create(PRECISION_SIZE);
    }

    if (hypers.cudagraphs >= 0) {
        pufferl->train_warmup = 0;

        // Fused rollout cudagraphs: [horizon][num_buffers]
        pufferl->fused_rollout_cudagraphs.resize(horizon);
        for (int h = 0; h < horizon; ++h) {
            pufferl->fused_rollout_cudagraphs[h].resize(num_buffers);
        }

        // Snapshot weights + optimizer state before init-time capture
        int64_t wb_bytes = pufferl->muon->wb_puf.numel() * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->muon->wb_puf.bytes, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon->mb_puf.bytes, wb_bytes, cudaMemcpyDeviceToDevice);

        // Run warmup + capture on a fresh stream.
        // Swap default_stream (used by train_impl) and tl_stream (used by callback on main thread).
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
        cudaMemcpy(pufferl->muon->wb_puf.bytes, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon->mb_puf.bytes, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            puf_cast_f32_to_bf16(pufferl->param_bf16_puf, pufferl->param_fp32_puf,
                pufferl->default_stream);
        }
        pufferl->muon->zero_grad(pufferl->default_stream);

        pufferl->epoch = 0;
    }

    // Create per-buffer streams
    for (int i = 0; i < num_buffers; i++) {
        cudaStream_t s;
        cudaStreamCreate(&s);
        pufferl->streams.push_back(s);
        vec->streams[i] = s;
    }

    create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(),
        net_callback_wrapper, thread_init_wrapper);
    static_vec_reset(vec);

    if (hypers.profile) {
        cudaDeviceSynchronize();
        cudaProfilerStart();
    }

    return pufferl;
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    if (pufferl.hypers.profile) {
        cudaProfilerStop();
    }

    pufferl.train_cudagraph.reset();
    for (auto& row : pufferl.fused_rollout_cudagraphs) {
        for (auto& g : row) {
            g.reset();
        }
    }

    delete pufferl.muon;
    if (USE_BF16) {
        delete pufferl.policy_bf16;
    }
    delete pufferl.policy_fp32;

    pufferl.alloc_fp32.destroy();
    pufferl.alloc_bf16.destroy();
    pufferl.pufferl_alloc.destroy();
    for (auto& a : pufferl.buffer_allocs) {
        a.destroy();
    }

    for (auto s : pufferl.streams) {
        cudaStreamDestroy(s);
    }
    for (int i = 0; i < NUM_TRAIN_EVENTS; i++) {
        cudaEventDestroy(pufferl.profile.events[i]);
    }
    nvmlShutdown();

    static_vec_close(pufferl.vec);

    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
    }
}

} // namespace pufferlib
