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
        int64_t S = p.shape[1], F = p.shape[2];
        return {.bytes = p.bytes + (t * S + start) * F * p.dtype_size, .shape = {count, F}, .dtype_size = p.dtype_size};
    } else {
        int64_t S = p.shape[1];
        return {.bytes = p.bytes + (t * S + start) * p.dtype_size, .shape = {count}, .dtype_size = p.dtype_size};
    }
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

    env.obs = {.bytes = (char*)vec->gpu_observations, .shape = {total_agents, obs_size}, .dtype_size = obs_dtype_size(obs_type)};
    env.obs_raw_dtype = obs_type;
    env.actions = {.bytes = (char*)vec->gpu_actions, .shape = {total_agents, num_atns}, .dtype_size = (int)sizeof(double)};
    env.rewards = {.bytes = (char*)vec->gpu_rewards, .shape = {total_agents}, .dtype_size = (int)sizeof(float)};
    env.terminals = {.bytes = (char*)vec->gpu_terminals, .shape = {total_agents}, .dtype_size = (int)sizeof(float)};

    return vec;
}

// RolloutBuf and TrainGraph defined in models.cu

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
    Policy* policy;       // Vtables (encoder, decoder, network)
    PolicyWeights weights_fp32;  // Master weights (fp32) - for optimizer
    PolicyWeights weights_bf16;  // Working weights (bf16) - for forward/backward
    PolicyActivations train_activations; // Training activation/grad buffers
    AllocSet alloc_fp32; // Contiguous param+grad+activation buffers for fp32 policy
    AllocSet alloc_bf16; // Contiguous param+activation buffers for bf16 policy
    Allocator pufferl_alloc; // Consolidated allocator for all init-to-close buffers
    StaticVec* vec;
    Muon* muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    vector<PufTensor> buffer_states;  // Per-buffer states for contiguous access
    vector<PolicyActivations> buffer_activations;  // Per-buffer inference activations
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
    PufTensor param_fp32_puf;    // cached PufTensor view of alloc_fp32.param_buffer
    PufTensor param_bf16_puf;    // cached PufTensor view of alloc_bf16.param_buffer
    PufTensor grad_bf16_puf;     // cached PufTensor view of alloc_bf16.grads (contiguous bf16 weight grads)
    PufTensor grad_norm_puf;     // (1,) f32 scratch for clip_grad_norm
    PufTensor rng_offset_puf;    // (num_buffers+1,) int64 CUDA device counters, one per buffer + one for training
    ProfileT profile;
    nvmlDevice_t nvml_device;
    int epoch;
    int train_warmup;
    bool rollout_captured;
    bool train_captured;
    uint64_t rng_seed;
    bool is_nmmo3;
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
    PufTensor obs_src = {
        .bytes = obs_env.bytes + (int64_t)start * obs_env.shape[1] * obs_env.dtype_size,
        .shape = {block_size, obs_env.shape[1]},
        .dtype_size = obs_env.dtype_size
    };

    PufTensor obs_dst = puf_slice(rollouts.observations, t, start, block_size);
    // Cast env obs (uint8/f32/etc) directly into rollout buffer (precision_t)
    if (obs_env.dtype_size == sizeof(char)) {
        cast_u8_to_precision_kernel<<<grid_size(obs_src.numel()), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)obs_dst.bytes, (const unsigned char*)obs_src.bytes, obs_src.numel());
        CHECK_LAST_KERNEL();
    } else if (obs_env.dtype_size == sizeof(float)) {
        if (USE_BF16) {
            puf_cast_f32_to_bf16(obs_dst, obs_src, stream);
            CHECK_LAST_KERNEL();
        } else {
            puf_copy(obs_dst, obs_src, stream);
        }
    } else {
        assert(false && "Unsupported obs dtype: only uint8 and float32 are supported");
    }

    // Rewards/terminals: env is f32, rollout is precision_t — cast via PufTensor
    PufTensor rew_dst = puf_slice(rollouts.rewards, t, start, block_size);
    PufTensor rew_src = {
        .bytes = env.rewards.bytes + start * (int)sizeof(float),
        .shape = {block_size},
        .dtype_size = (int)sizeof(float)
    };

    if (USE_BF16) {
        puf_cast_f32_to_bf16(rew_dst, rew_src, stream);
        CHECK_LAST_KERNEL();
    } else {
        puf_copy(rew_dst, rew_src, stream);
    }

    PufTensor term_dst = puf_slice(rollouts.terminals, t, start, block_size);
    PufTensor term_src = {
        .bytes = env.terminals.bytes + start * (int)sizeof(float),
        .shape = {block_size},
        .dtype_size = (int)sizeof(float)
    };
    if (USE_BF16) {
        puf_cast_f32_to_bf16(term_dst, term_src, stream);
        CHECK_LAST_KERNEL();
    } else {
        puf_copy(term_dst, term_src, stream);
    }

    // Forward pass — obs_dst already contains the cast obs in precision_t
    PufTensor state_puf = pufferl->buffer_states[buf];
    PufTensor dec_puf = policy_forward(pufferl->policy, pufferl->weights_bf16, pufferl->buffer_activations[buf], obs_dst, state_puf, stream);

    // Sample actions, logprobs, values into rollout buffer
    PufTensor act_slice = puf_slice(rollouts.actions, t, start, block_size);
    PufTensor lp_slice = puf_slice(rollouts.logprobs, t, start, block_size);
    PufTensor val_slice = puf_slice(rollouts.values, t, start, block_size);

    PufTensor p_logstd;
    {
        DecoderWeights* dw = (DecoderWeights*)pufferl->weights_bf16.decoder;
        if (dw->continuous) { p_logstd = dw->logstd; }
    }

    // Each buffer uses its own RNG seed and offset slot for deterministic parallel rollouts
    int64_t* buf_rng_offset = (int64_t*)pufferl->rng_offset_puf.bytes + buf;
    uint64_t buf_rng_seed = pufferl->rng_seed + buf;
    sample_logits_kernel<<<grid_size(block_size), BLOCK_SIZE, 0, stream>>>(
        dec_puf, p_logstd, pufferl->act_sizes_puf,
        (double*)act_slice.bytes, (precision_t*)lp_slice.bytes, (precision_t*)val_slice.bytes,
        buf_rng_seed, buf_rng_offset);
    CHECK_LAST_KERNEL();

    // Copy actions to env
    int64_t act_cols = env.actions.shape[1];
    cudaMemcpyAsync(
        env.actions.bytes + start * act_cols * env.actions.dtype_size,
        act_slice.bytes, act_slice.numel() * act_slice.dtype_size, cudaMemcpyDeviceToDevice, stream);

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
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_LAST_KERNEL();
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

    // Clamp rewards and fill ratio
    if (USE_BF16) {
        clamp_bf16_kernel<<<grid_size(rollouts.rewards.numel()), BLOCK_SIZE, 0, train_stream>>>(
            (__nv_bfloat16*)rollouts.rewards.bytes, -1.0f, 1.0f, rollouts.rewards.numel());
        CHECK_LAST_KERNEL();
        fill_bf16_kernel<<<grid_size(rollouts.ratio.numel()), BLOCK_SIZE, 0, train_stream>>>(
            (__nv_bfloat16*)rollouts.ratio.bytes, __float2bfloat16(1.0f), rollouts.ratio.numel());
        CHECK_LAST_KERNEL();
    } else {
        clamp_f32_kernel<<<grid_size(rollouts.rewards.numel()), BLOCK_SIZE, 0, train_stream>>>(
            (float*)rollouts.rewards.bytes, -1.0f, 1.0f, rollouts.rewards.numel());
        CHECK_LAST_KERNEL();
        fill_f32_kernel<<<grid_size(rollouts.ratio.numel()), BLOCK_SIZE, 0, train_stream>>>(
            (float*)rollouts.ratio.bytes, 1.0f, rollouts.ratio.numel());
        CHECK_LAST_KERNEL();
    }

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
        CHECK_LAST_KERNEL();
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        // Use the training RNG offset slot (last slot, index num_buffers)
        int64_t* train_rng_offset = (int64_t*)pufferl.rng_offset_puf.bytes + hypers.num_buffers;
        prio_replay_cuda(advantages_puf, prio_alpha, minibatch_segments,
            hypers.total_agents, anneal_beta,
            pufferl.prio_bufs, pufferl.rng_seed, train_rng_offset, train_stream);
        CHECK_LAST_KERNEL();
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        puf_zero(graph.mb_state, train_stream);
        {
            // Build a RolloutBuf view with old_values and advantages swapped in
            RolloutBuf sel_src = rollouts;
            sel_src.values = old_values_puf;
            int mb_segs = pufferl.prio_bufs.idx.shape[0];
            select_copy_kernel<<<dim3(mb_segs, 5), SELECT_COPY_THREADS, 0, train_stream>>>(
                sel_src, graph, (const int64_t*)pufferl.prio_bufs.idx.bytes,
                (const float*)advantages_puf.bytes, (const float*)pufferl.prio_bufs.mb_prio.bytes);
            CHECK_LAST_KERNEL();
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
            PufTensor dec_puf = policy_forward_train(pufferl.policy, pufferl.weights_bf16, pufferl.train_activations, obs_puf, state_puf, stream);
            DecoderWeights* dw_train = (DecoderWeights*)pufferl.weights_bf16.decoder;
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
            CHECK_LAST_KERNEL();

            PufTensor grad_logits_puf = pufferl.ppo_bufs_puf.grad_logits;
            PufTensor grad_logstd_puf = pufferl.is_continuous ? pufferl.ppo_bufs_puf.grad_logstd : PufTensor();
            PufTensor grad_values_puf = pufferl.ppo_bufs_puf.grad_values;
            policy_backward(pufferl.policy, pufferl.weights_bf16, pufferl.train_activations,
                grad_logits_puf, grad_logstd_puf, grad_values_puf, stream);
            CHECK_LAST_KERNEL();

            // Cast contiguous grads → f32 into muon gc_puf, then clip grad norm
            if (USE_BF16) {
                cast_bf16_to_f32_kernel<<<grid_size(pufferl.grad_bf16_puf.numel()), BLOCK_SIZE, 0, stream>>>(
                    (float*)pufferl.muon->gc_puf.bytes, (const __nv_bfloat16*)pufferl.grad_bf16_puf.bytes,
                    pufferl.grad_bf16_puf.numel());
                CHECK_LAST_KERNEL();
            } else {
                cudaMemcpyAsync(pufferl.muon->gc_puf.bytes, pufferl.grad_bf16_puf.bytes,
                    pufferl.grad_bf16_puf.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            {
                PufTensor& grad = pufferl.muon->gc_puf;
                float* scratch = (float*)pufferl.grad_norm_puf.bytes;
                ensure_norm_partials();
                int blocks = std::min((int)grid_size(grad.numel()), 256);
                norm_f32_kernel<<<blocks, 256, 0, stream>>>(norm_partials_buf, (float*)grad.bytes, grad.numel());
                CHECK_LAST_KERNEL();
                norm_reduce_kernel<<<1, 256, 0, stream>>>(scratch, norm_partials_buf, blocks);
                CHECK_LAST_KERNEL();
                clip_by_norm_f32_kernel<<<grid_size(grad.numel()), BLOCK_SIZE, 0, stream>>>(
                    (float*)grad.bytes, scratch, hypers.max_grad_norm, 1e-6f, grad.numel());
                CHECK_LAST_KERNEL();
            }
            muon_step(pufferl.muon, stream);
            CHECK_LAST_KERNEL();
            if (USE_BF16) {
                puf_cast_f32_to_bf16(pufferl.param_bf16_puf, pufferl.param_fp32_puf, stream);
                CHECK_LAST_KERNEL();
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
        /*
        // mb_ratio is (S, H) precision — scatter into rollouts.ratio (S_total, H)
        {
            int num_idx = pufferl.prio_bufs.idx.numel();
            int row_bytes = graph.mb_ratio.numel() / graph.mb_ratio.shape[0] * graph.mb_ratio.dtype_size;
            index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                rollouts.ratio.bytes, (const int64_t*)pufferl.prio_bufs.idx.bytes,
                (const char*)graph.mb_ratio.bytes, num_idx, row_bytes);
        }
        // mb_newvalue is (S, H, 1) — treat as (S, H) for scatter into rollouts.values
        {
            int num_idx = pufferl.prio_bufs.idx.numel();
            int S = graph.mb_newvalue.shape[0], H = graph.mb_newvalue.shape[1];
            int row_bytes = H * graph.mb_newvalue.dtype_size;
            index_copy_kernel<<<grid_size(num_idx), BLOCK_SIZE, 0, train_stream>>>(
                rollouts.values.bytes, (const int64_t*)pufferl.prio_bufs.idx.bytes,
                (const char*)graph.mb_newvalue.bytes, num_idx, row_bytes);
        }
        */
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

    int num_action_heads = pufferl->env.actions.shape[1];
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

    // ========================================================================
    // fp32 master weights (for optimizer)
    // ========================================================================

    int esz_fp32 = sizeof(float);
    pufferl->alloc_fp32.esz = esz_fp32;
    Allocator& fp32_params = pufferl->alloc_fp32.params;

    bool is_nmmo3 = (env_name == "puffer_nmmo3");
    pufferl->is_nmmo3 = is_nmmo3;

    Encoder encoder = is_nmmo3 ? Encoder{
        .forward = nmmo3_encoder_forward,
        .backward = nmmo3_encoder_backward,
        .init_weights = nmmo3_encoder_init_weights,
        .reg_params = nmmo3_encoder_reg_params,
        .reg_train = nmmo3_encoder_reg_train,
        .reg_rollout = nmmo3_encoder_reg_rollout,
    } : Encoder{
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
    };
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
    };
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
    };

    // Policy vtables (shared across fp32/bf16)
    Policy* policy = new Policy{
        .encoder = encoder, .decoder = decoder, .network = network,
        .input_dim = input_size, .hidden_dim = hidden_size, .output_dim = decoder_output_size,
        .num_atns = act_n,
    };
    pufferl->policy = policy;

    // fp32 master weights
    auto new_weights = [&](int esz) -> PolicyWeights {
        PolicyWeights w;
        if (is_nmmo3) {
            w.encoder = new NMMO3EncoderWeights{.obs_size = input_size, .hidden = hidden_size};
        } else {
            w.encoder = new EncoderWeights{.in_dim = input_size, .out_dim = hidden_size};
        }
        w.decoder = new DecoderWeights{.hidden_dim = hidden_size, .output_dim = decoder_output_size, .continuous = is_continuous};
        w.network = new MinGRUWeights{.hidden = hidden_size, .num_layers = num_layers, .horizon = hypers.horizon};
        ((MinGRUWeights*)w.network)->weights.resize(num_layers);
        return w;
    };

    pufferl->weights_fp32 = new_weights(esz_fp32);
    PolicyWeights& wfp32 = pufferl->weights_fp32;
    encoder.reg_params(wfp32.encoder, &fp32_params, esz_fp32);
    decoder.reg_params(wfp32.decoder, &fp32_params, esz_fp32);
    network.reg_params(wfp32.network, &fp32_params, esz_fp32);

    pufferl->alloc_fp32.create();
    pufferl->param_fp32_puf = {.bytes = (char*)fp32_params.mem, .shape = {fp32_params.total_elems}, .dtype_size = esz_fp32};

    // Init weights on fp32 master
    {
        uint64_t seed = 42;
        encoder.init_weights(wfp32.encoder, &seed, pufferl->default_stream);
        decoder.init_weights(wfp32.decoder, &seed, pufferl->default_stream);
        network.init_weights(wfp32.network, &seed, pufferl->default_stream);
    }

    // ========================================================================
    // bf16 compute policy: working copy for fwd/bwd + activations + grads
    // ========================================================================

    int B_TT = minibatch_segments * hypers.horizon;
    int psz = PRECISION_SIZE;

    if (USE_BF16) {
        pufferl->alloc_bf16.esz = 2;
        Allocator& bf16_params = pufferl->alloc_bf16.params;
        Allocator& acts = pufferl->alloc_bf16.acts;
        Allocator& grads = pufferl->alloc_bf16.grads;

        pufferl->weights_bf16 = new_weights(psz);
        PolicyWeights& wbf16 = pufferl->weights_bf16;

        encoder.reg_params(wbf16.encoder, &bf16_params, psz);
        decoder.reg_params(wbf16.decoder, &bf16_params, psz);
        network.reg_params(wbf16.network, &bf16_params, psz);

        PolicyActivations& tb = pufferl->train_activations;
        tb.encoder = is_nmmo3 ? (void*)new NMMO3EncoderActivations{} : (void*)new EncoderActivations{};
        tb.decoder = new DecoderActivations{};
        tb.network = new MinGRUActivations{};
        encoder.reg_train(wbf16.encoder, tb.encoder, &acts, &grads, B_TT);
        decoder.reg_train(wbf16.decoder, tb.decoder, &acts, &grads, B_TT);
        network.reg_train(wbf16.network, tb.network, &acts, &grads, B_TT);

        pufferl->alloc_bf16.create();
        pufferl->param_bf16_puf = {.bytes = (char*)bf16_params.mem, .shape = {bf16_params.total_elems}, .dtype_size = 2};
        pufferl->grad_bf16_puf = {.bytes = (char*)pufferl->alloc_bf16.grads.mem, .shape = {pufferl->alloc_bf16.grads.total_elems}, .dtype_size = 2};

        puf_cast_f32_to_bf16(pufferl->param_bf16_puf, pufferl->param_fp32_puf,
            pufferl->default_stream);
    } else {
        pufferl->weights_bf16 = pufferl->weights_fp32;

        // In fp32 mode, train activations and grads use the fp32 alloc
        pufferl->alloc_fp32.esz = esz_fp32;
        Allocator& acts = pufferl->alloc_fp32.acts;
        Allocator& grads = pufferl->alloc_fp32.grads;

        PolicyActivations& tb = pufferl->train_activations;
        tb.encoder = is_nmmo3 ? (void*)new NMMO3EncoderActivations{} : (void*)new EncoderActivations{};
        tb.decoder = new DecoderActivations{};
        tb.network = new MinGRUActivations{};
        encoder.reg_train(wfp32.encoder, tb.encoder, &acts, &grads, B_TT);
        decoder.reg_train(wfp32.decoder, tb.decoder, &acts, &grads, B_TT);
        network.reg_train(wfp32.network, tb.network, &acts, &grads, B_TT);

        pufferl->alloc_fp32.acts.create();
        pufferl->alloc_fp32.grads.create();
        pufferl->grad_bf16_puf = {.bytes = (char*)grads.mem, .shape = {grads.total_elems}, .dtype_size = esz_fp32};
    }

    // ========================================================================
    // Optimizer (Muon) — operates on fp32 master weights
    // ========================================================================

    // Muon reads param shapes directly from fp32_params allocator

    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new Muon{};
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
    int p = PRECISION_SIZE;
    pufferl->rng_offset_puf = {.shape = {num_buffers + 1}, .dtype_size = (int)sizeof(int64_t)};
    pufferl->act_sizes_puf = {.shape = {num_action_heads}, .dtype_size = (int)sizeof(int32_t)};
    pufferl->losses_puf = {.shape = {NUM_LOSSES}, .dtype_size = (int)sizeof(float)};
    pufferl->grad_norm_puf = {.shape = {1}, .dtype_size = (int)sizeof(float)};
    alloc.reg(&pufferl->rng_offset_puf);
    alloc.reg(&pufferl->act_sizes_puf);
    alloc.reg(&pufferl->losses_puf);
    alloc.reg(&pufferl->grad_norm_puf);

    // Per-buffer RNN states
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = {.shape = {num_layers, batch, hidden_size}, .dtype_size = p};
        alloc.reg(&pufferl->buffer_states[i]);
    }

    // Rollout buffers (horizon, total_agents, ...)
    register_rollout_buffers(pufferl->rollouts, alloc, horizon, total_agents, input_size, num_action_heads);

    // Train graph buffers
    register_train_buffers(pufferl->train_buf, alloc, minibatch_segments, horizon, input_size,
        hidden_size, num_action_heads, num_layers);

    // Pre-allocated transposed rollouts for train_impl (total_agents, horizon, ...)
    register_rollout_buffers(pufferl->train_rollouts, alloc, total_agents, horizon, input_size, num_action_heads);

    // Pre-allocated train temporaries
    pufferl->old_values_puf = {.shape = {total_agents, horizon}, .dtype_size = p};
    pufferl->advantages_puf = {.shape = {total_agents, horizon}, .dtype_size = (int)sizeof(float)};
    alloc.reg(&pufferl->old_values_puf);
    alloc.reg(&pufferl->advantages_puf);

    // PPO loss buffers
    register_ppo_buffers(pufferl->ppo_bufs_puf, alloc, minibatch_segments, hypers.horizon, decoder_output_size, is_continuous);

    // Priority replay buffers
    register_prio_buffers(pufferl->prio_bufs, alloc, hypers.total_agents, minibatch_segments);

    // Muon optimizer (init + register buffers)
    muon_init(pufferl->muon, &fp32_params,
        pufferl->param_fp32_puf, lr, beta1, eps, 0.0, alloc);
    pufferl->muon->nccl_comm = pufferl->nccl_comm;
    pufferl->muon->world_size = hypers.world_size;
    printf("DEBUG: Contiguous weight buffer: %ld elements\n", pufferl->muon->wb_puf.numel());

    // Single allocation for all registered buffers
    alloc.create();

    // Post-create initialization: copy data, set constants
    cudaMemset(pufferl->rng_offset_puf.bytes, 0, (num_buffers + 1) * sizeof(int64_t));
    cudaMemcpy(pufferl->act_sizes_puf.bytes, raw_act_sizes, num_action_heads * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(pufferl->losses_puf.bytes, 0, NUM_LOSSES * sizeof(float));
    post_create_ppo_buffers(pufferl->ppo_bufs_puf);
    muon_post_create(pufferl->muon);

    // Per-buffer inference activations (separate allocators — different lifetime)
    pufferl->buffer_activations.resize(num_buffers);
    pufferl->buffer_allocs.resize(num_buffers);
    // Register and allocate per-buffer inference activations
    for (int i = 0; i < num_buffers; i++) {
        PolicyActivations& rbuf = pufferl->buffer_activations[i];
        Allocator& ralloc = pufferl->buffer_allocs[i];
        rbuf.encoder = is_nmmo3 ? (void*)new NMMO3EncoderActivations{} : (void*)new EncoderActivations{};
        rbuf.decoder = new DecoderActivations{};
        rbuf.network = new MinGRUActivations{};
        encoder.reg_rollout(pufferl->weights_bf16.encoder, rbuf.encoder, &ralloc, inf_batch);
        decoder.reg_rollout(pufferl->weights_bf16.decoder, rbuf.decoder, &ralloc, inf_batch);
        network.reg_rollout(pufferl->weights_bf16.network, rbuf.network, &ralloc, inf_batch);
        ralloc.create();
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

    auto delete_weights = [&](PolicyWeights& w) {
        if (pufferl.is_nmmo3) delete (NMMO3EncoderWeights*)w.encoder;
        else delete (EncoderWeights*)w.encoder;
        delete (DecoderWeights*)w.decoder;
        delete (MinGRUWeights*)w.network;
    };
    if (USE_BF16) {
        delete_weights(pufferl.weights_bf16);
    }
    if (pufferl.is_nmmo3) delete (NMMO3EncoderActivations*)pufferl.train_activations.encoder;
    else delete (EncoderActivations*)pufferl.train_activations.encoder;
    delete (DecoderActivations*)pufferl.train_activations.decoder;
    delete (MinGRUActivations*)pufferl.train_activations.network;
    delete_weights(pufferl.weights_fp32);
    for (auto& rbuf : pufferl.buffer_activations) {
        if (pufferl.is_nmmo3) delete (NMMO3EncoderActivations*)rbuf.encoder;
        else delete (EncoderActivations*)rbuf.encoder;
        delete (DecoderActivations*)rbuf.decoder;
        delete (MinGRUActivations*)rbuf.network;
    }
    delete pufferl.policy;

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
