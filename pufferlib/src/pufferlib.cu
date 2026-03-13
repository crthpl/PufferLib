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
#include <nvtx3/nvToolsExt.h>
#include <nvml.h>

#include "models.cu"
#include "vecenv.h"


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
    if (obs_env.dtype_size == sizeof(char)) {
        cast_u8_to_precision_kernel<<<grid_size(obs_src.numel()), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)obs_dst.bytes, (const unsigned char*)obs_src.bytes, obs_src.numel());
    } else if (obs_env.dtype_size == sizeof(float)) {
        puf_f32_to_precision(obs_dst, obs_src, stream);
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

    puf_f32_to_precision(rew_dst, rew_src, stream);

    PufTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    PufTensor term_src = {
        .bytes = env.terminals.bytes + start * (int)sizeof(float),
        .shape = {block_size},
        .dtype_size = (int)sizeof(float)
    };
    puf_f32_to_precision(term_dst, term_src, stream);

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
        puf_zero(advantages_puf, train_stream);

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
        puf_zero(graph.mb_state, train_stream);
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
                puf_cast_f32_to_bf16(pufferl.param_puf, pufferl.master_weights, stream);
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

    int B_TT = minibatch_segments * hypers.horizon;
    int psz = PRECISION_SIZE;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    // Create and allocate params
    Allocator* params = &pufferl->params_alloc;
    Allocator* acts = &pufferl->activations_alloc;
    Allocator* grads = &pufferl->grads_alloc;

    // Buffers for weights, grads, and activations
    pufferl->weights = policy_weights_create(&pufferl->policy, psz, params);
    pufferl->train_activations = policy_reg_train(&pufferl->policy, pufferl->weights, acts, grads, B_TT);
    pufferl->buffer_activations = (PolicyActivations*)calloc(num_buffers, sizeof(PolicyActivations));
    pufferl->buffer_states = (PufTensor*)calloc(num_buffers, sizeof(PufTensor));
    int p = PRECISION_SIZE;
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_activations[i] = policy_reg_rollout(
            &pufferl->policy, pufferl->weights, acts, inf_batch);
        pufferl->buffer_states[i] = {.shape = {num_layers, batch, hidden_size}, .dtype_size = p};
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

    policy_init_weights(&pufferl->policy, pufferl->weights, &seed, pufferl->default_stream);
    pufferl->master_weights = pufferl->param_puf;
    if (USE_BF16) {
        pufferl->master_weights = {.shape = {params->total_elems}, .dtype_size = 4};
        cudaMalloc(&pufferl->master_weights.bytes, params->total_elems * sizeof(float));
        puf_cast_precision_to_f32(pufferl->master_weights, pufferl->param_puf, pufferl->default_stream);
    }

    // Post-create initialization
    cudaMemset(pufferl->rng_offset_puf.bytes, 0, (num_buffers + 1) * sizeof(long));
    cudaMemcpy(pufferl->act_sizes_puf.bytes, raw_act_sizes, num_action_heads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.bytes, 0, NUM_LOSSES * sizeof(float));

    float one = 1.0f;
    cudaMemcpy(&pufferl->ppo_bufs_puf.grad_loss.bytes, &one, sizeof(float), cudaMemcpyHostToDevice);

    muon_post_create(&pufferl->muon);

    if (hypers.cudagraphs >= 0) {
        pufferl->train_warmup = 0;

        // Fused rollout cudagraphs: [horizon][num_buffers]
        pufferl->fused_rollout_cudagraphs = (RawCudaGraph*)calloc(horizon*num_buffers, sizeof(RawCudaGraph));

        // Snapshot weights + optimizer state before init-time capture
        long wb_bytes = pufferl->master_weights.numel() * sizeof(float);
        void* saved_weights;
        cudaMalloc(&saved_weights, wb_bytes);
        cudaMemcpy(saved_weights, pufferl->master_weights.bytes, wb_bytes, cudaMemcpyDeviceToDevice);
        void* saved_momentum;
        cudaMalloc(&saved_momentum, wb_bytes);
        cudaMemcpy(saved_momentum, pufferl->muon.mb_puf.bytes, wb_bytes, cudaMemcpyDeviceToDevice);

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
        cudaMemcpy(pufferl->master_weights.bytes, saved_weights, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_weights);
        cudaMemcpy(pufferl->muon.mb_puf.bytes, saved_momentum, wb_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(saved_momentum);
        if (USE_BF16) {
            puf_cast_f32_to_bf16(pufferl->param_puf, pufferl->master_weights,
                pufferl->default_stream);
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
