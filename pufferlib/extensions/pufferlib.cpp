//TODO:clamped
//5.6% cat overhead from grad clip. Preallocate?
//11% seqwise overhead from fused scan
//30% elemwise form random ops
//5% on log_coeffs_and_values

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nccl.h>

#include <atomic>
#include <dlfcn.h>
#include <unistd.h>
#include "muon.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>

#include <nvtx3/nvToolsExt.h>

#include <functional>
#include <iostream>
#include <vector>

#include "env_binding.h"

using std::tuple;
using std::vector;
using std::shared_ptr;
namespace nn = torch::nn;

typedef torch::Tensor Tensor;

// CUDA kernel wrappers (implemented in modules.cu, compiled separately by nvcc)
#include "modules.h"

// get dtype based on bf16 flag
inline torch::ScalarType get_dtype(bool bf16) {
    return bf16 ? torch::kBFloat16 : torch::kFloat32;
}

namespace pufferlib {

// Common tensor options
auto cuda_f32 = torch::dtype(torch::kFloat32).device(torch::kCUDA);
auto cuda_f64 = torch::dtype(torch::kFloat64).device(torch::kCUDA);
auto cuda_i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
auto cuda_i64 = torch::dtype(torch::kInt64).device(torch::kCUDA);

void print_cuda_mem(const char* label) {
    cudaDeviceSynchronize();
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[CUDA MEM %s] used=%.2f MB\n", label, (total_mem - free_mem) / 1e6);
}

// Advantage computation is in advantage.cpp
#include "advantage.cpp"

// Model classes are in models.cpp
#include "models.cpp"

// Environment-specific encoder/decoder models are in ocean.cpp
#include "ocean.cpp"

torch::Dtype to_torch_dtype(int dtype) {
    if (dtype == FLOAT) {
        return torch::kFloat32;
    } else if (dtype == INT) {
        return torch::kInt32;
    } else if (dtype == UNSIGNED_CHAR) {
        return torch::kUInt8;
    } else if (dtype == DOUBLE) {
        return torch::kFloat64;
    } else if (dtype == CHAR) {
        return torch::kInt8;
    } else {
        assert(false && "to_torch_dtype failed to convert dtype");
    }
    return torch::kFloat32;
}

// Fast clip_grad_norm_ for contiguous weights
// Cats all grads for one-shot norm computation, then scales each grad
void clip_grad_norm_(
    const vector<Tensor>& parameters,
    double max_norm
    ) {
  // Collect flattened grads
  vector<Tensor> flat_grads;
  flat_grads.reserve(parameters.size());

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      flat_grads.push_back(grad.flatten());
    }
  }

  if (flat_grads.empty()) {
    return;
  }

  // Single cat + norm (avoids per-param norm calls)
  Tensor all_grads = torch::cat(flat_grads);
  // Getting errors here? See if your net is definint a layeyr and not using it.
  // TODO: That shouldn't error
  Tensor total_norm = all_grads.to(torch::kFloat32).norm(2);

  // Compute clip coefficient
  Tensor clip_coef = torch::clamp_max(max_norm / (total_norm + 1e-6), 1.0);

  // Scale each grad in-place
  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      grad.mul_(clip_coef);
    }
  }
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

typedef struct {
    Tensor obs;
    Tensor actions;
    Tensor rewards;
    Tensor terminals;
} EnvBuf;

tuple<StaticVec*, Tensor>
create_environments(int num_buffers, int total_agents, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    printf("DEBUG create_environments: vec->size=%d, vec->total_agents=%d\n",
        vec->size, vec->total_agents);

    int obs_size = get_obs_size();
    int num_atns = get_num_atns();

    env.obs = torch::from_blob(vec->gpu_observations, {total_agents, obs_size}, torch::dtype(to_torch_dtype(get_obs_type())).device(torch::kCUDA));
    env.actions = torch::from_blob(vec->gpu_actions, {total_agents, num_atns}, cuda_f64);
    env.rewards = torch::from_blob(vec->gpu_rewards, {total_agents}, cuda_f32);
    env.terminals = torch::from_blob(vec->gpu_terminals, {total_agents}, cuda_f32);

    // Create act_sizes tensor on CUDA (needed for sample_logits kernel)
    Tensor act_sizes = torch::from_blob(get_act_sizes(), {num_atns}, torch::dtype(torch::kInt32)).to(torch::kCUDA);

    return std::make_tuple(vec, act_sizes);
}

typedef struct {
    Tensor mb_obs;
    Tensor mb_state;
    Tensor mb_actions;
    Tensor mb_logprobs;
    Tensor mb_advantages;
    Tensor mb_prio;
    Tensor mb_values;
    Tensor mb_returns;
    Tensor mb_ratio;
    Tensor mb_newvalue;
} TrainGraph;

TrainGraph create_train_graph(int mb_segments, int horizon, int input_size,
        int num_layers, int hidden_size, int num_atns, bool bf16) {
    TrainGraph g;
    auto opts = torch::dtype(get_dtype(bf16)).device(torch::kCUDA);
    g.mb_obs = torch::zeros({mb_segments, horizon, input_size}, opts);
    g.mb_state = torch::zeros({num_layers, mb_segments, 1, hidden_size}, opts);
    g.mb_newvalue = torch::zeros({mb_segments, horizon, 1}, opts);
    g.mb_ratio = torch::zeros({mb_segments, horizon}, opts);
    g.mb_actions = torch::zeros({mb_segments, horizon, num_atns}, cuda_f64);
    g.mb_logprobs = torch::zeros({mb_segments, horizon}, opts);
    g.mb_advantages = torch::zeros({mb_segments, horizon}, cuda_f32);  // always fp32 for precision
    g.mb_prio = torch::zeros({mb_segments, 1}, opts);
    g.mb_values = torch::zeros({mb_segments, horizon}, opts);
    g.mb_returns = torch::zeros({mb_segments, horizon}, opts);
    return g;
}

typedef struct {
    Tensor observations;
    Tensor actions;
    Tensor values;
    Tensor logprobs;
    Tensor rewards;
    Tensor terminals;
    Tensor ratio;
    Tensor importance;
} RolloutBuf;

RolloutBuf create_rollouts(int horizon, int segments, int input_size, int num_atns, bool bf16) {
    RolloutBuf r;
    auto opts = torch::dtype(get_dtype(bf16)).device(torch::kCUDA);
    r.observations = torch::zeros({horizon, segments, input_size}, opts);
    r.actions = torch::zeros({horizon, segments, num_atns}, cuda_f64);
    r.values = torch::zeros({horizon, segments}, opts);
    r.logprobs = torch::zeros({horizon, segments}, opts);
    r.rewards = torch::zeros({horizon, segments}, opts);
    r.terminals = torch::zeros({horizon, segments}, opts);
    r.ratio = torch::zeros({horizon, segments}, opts);
    r.importance = torch::zeros({horizon, segments}, opts);
    return r;
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
    bool cudagraphs;
    bool kernels;
    bool profile;
    bool use_omp;
    bool bf16;  // bfloat16 mixed precision training
    // Multi-GPU
    int rank;
    int world_size;
    std::string nccl_id_path;
    // Threading
    int num_threads;
} HypersT;

typedef struct {
    Policy* policy_bf16;  // Working weights (bf16) - used for forward/backward
    Policy* policy_fp32;  // Master weights (fp32) - used for optimizer
    StaticVec* vec;
    torch::optim::Muon* muon;
    ncclComm_t nccl_comm;  // NCCL communicator for multi-GPU
    HypersT hypers;
    bool is_continuous;  // True if all action dimensions are continuous (size==1)
    vector<Tensor> buffer_states;  // Per-buffer states for contiguous access
    RolloutBuf rollouts;
    EnvBuf env;
    TrainGraph train_buf;
    vector<vector<at::cuda::CUDAGraph>> fused_rollout_cudagraphs;  // [horizon][num_buffers]
    at::cuda::CUDAGraph train_cudagraph;
    at::cuda::MempoolId_t train_pool_id;     // Pool ID for releasing graph memory
    at::cuda::MempoolId_t rollout_pool_id;   // Pool ID for releasing graph memory
    vector<at::cuda::CUDAStream> torch_streams;  // PyTorch-managed streams for OMP
    Tensor adv_mean;
    Tensor adv_std;
    Tensor act_sizes;      // CUDA int32 tensor of action head sizes for MultiDiscrete
    Tensor act_sizes_cpu;  // CPU int64 tensor (pre-computed to avoid alloc during graph replay)
    int epoch;
    int train_warmup;
    bool rollout_captured;
    uint64_t rng_seed;
    Tensor rng_offset;  // CUDA tensor so increment is graphable
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    static_vec_log(pufferl.vec, out);
    return out;
}

/* Checklist for avoiding diabolical capture bugs:
 * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
 * 2. Make sure input/output buffer pointers don't change
 * 3. Make sure to restore the original stream after tracing
 * 4. All custom kernels need to use the default torch stream
 * 5. Make sure you are using the torch stream fns, not the c10 ones.
 * 6. Scalars get captured by value. They cannot change between calls.
 */

constexpr int CAPTURE_WARMUP = 10;

// ============================================================================
// Rollout and train section functions
// ============================================================================

inline void profile_begin(const char* tag, bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePushA(tag); }
}

inline void profile_end(bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePop(); }
}

void compute_advantage(RolloutBuf& rollouts, Tensor& advantages, HypersT& hypers) {
    compute_puff_advantage_cuda(rollouts.values, rollouts.rewards, rollouts.terminals,
        rollouts.ratio, advantages, hypers.gamma, hypers.gae_lambda,
        hypers.vtrace_rho_clip, hypers.vtrace_c_clip);
}

// Thread initialization callback - sets CUDA stream once per thread
extern "C" void thread_init_wrapper(void* ctx, int buf) {
    PuffeRL* pufferl = (PuffeRL*)ctx;
    at::cuda::setCurrentCUDAStream(pufferl->torch_streams[buf]);
}

// Callback for OMP threadmanager - also called at init for warmup + capture
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    torch::NoGradGuard no_grad;
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;

    profile_begin("fused_rollout", hypers.profile);
    if (hypers.cudagraphs && pufferl->rollout_captured) {
        pufferl->fused_rollout_cudagraphs[t][buf].replay();
    } else {
        bool capturing = hypers.cudagraphs && pufferl->epoch == CAPTURE_WARMUP;
        auto saved_stream = at::cuda::getCurrentCUDAStream();
        auto cap_stream = capturing ? at::cuda::getStreamFromPool() : saved_stream;
        if (capturing) {
            at::cuda::setCurrentCUDAStream(cap_stream);
            pufferl->fused_rollout_cudagraphs[t][buf].capture_begin(
                pufferl->rollout_pool_id);
        }

        int total_agents = pufferl->vec->total_agents;
        int num_buffers = hypers.num_buffers;
        int block_size = total_agents / num_buffers;

        Tensor obs_slice = pufferl->env.obs.narrow(0, buf*block_size, block_size);
        Tensor& state = pufferl->buffer_states[buf];

        auto [logits, value, state_out] = pufferl->policy_bf16->forward(obs_slice, state);

        RolloutBuf& rollouts = pufferl->rollouts;
        Tensor actions_out = rollouts.actions.select(0, t).narrow(0, buf*block_size, block_size);
        Tensor logprobs_out = rollouts.logprobs.select(0, t).narrow(0, buf*block_size, block_size);
        Tensor values_out = rollouts.values.select(0, t).narrow(0, buf*block_size, block_size);

        sample_actions(logits, value, actions_out, logprobs_out, values_out,
            pufferl->act_sizes, pufferl->act_sizes_cpu,
            pufferl->is_continuous, hypers.kernels, pufferl->rng_seed, pufferl->rng_offset);

        state.copy_(state_out, false);

        rollouts.observations.select(0, t).narrow(0, buf*block_size, block_size).copy_(obs_slice, true);
        rollouts.rewards.select(0, t).narrow(0, buf*block_size, block_size).copy_(
            pufferl->env.rewards.narrow(0, buf*block_size, block_size), true);
        rollouts.terminals.select(0, t).narrow(0, buf*block_size, block_size).copy_(
            pufferl->env.terminals.narrow(0, buf*block_size, block_size), true);

        pufferl->env.actions.narrow(0, buf*block_size, block_size).copy_(actions_out, true);

        if (capturing) {
            pufferl->fused_rollout_cudagraphs[t][buf].capture_end();
            cap_stream.synchronize();
            cudaDeviceSynchronize();
            at::cuda::setCurrentCUDAStream(saved_stream);
        }
    }
    profile_end(hypers.profile);
}

void rollouts_impl(PuffeRL& pufferl) {
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;

    int horizon = hypers.horizon;
    int num_buffers = hypers.num_buffers;
    // TODO: You removed state zeros and reward clamping

    for (int i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
        int h = i / num_buffers;

        net_callback_wrapper(&pufferl, buf, h);

        // TODO: There should be a lighter way to sync. You need to make sure the torch data streams
        // are ready because puffer vec uses different streams. Setting to non-blocking is not enough.
        cudaDeviceSynchronize();
    }
}

void train_impl(PuffeRL& pufferl) {
    // Update to HypersT& p
    HypersT& hypers = pufferl.hypers;

    // Buffers are stored as {horizon, segments, ...} for contiguous rollout writes
    // Transpose to {segments, horizon, ...} for train logic
    // Need .contiguous() because compute_puff_advantage_cuda uses raw data pointers
    RolloutBuf rollouts;
    rollouts.observations = pufferl.rollouts.observations.permute({1, 0, 2}).contiguous();
    rollouts.actions = pufferl.rollouts.actions.transpose(0, 1).contiguous();
    rollouts.logprobs = pufferl.rollouts.logprobs.transpose(0, 1).contiguous();
    rollouts.rewards = pufferl.rollouts.rewards.transpose(0, 1).contiguous();
    rollouts.rewards.clamp_(-1.0, 1.0);  // Clamp rewards here instead of in eval to save a kernel call per step
    rollouts.terminals = pufferl.rollouts.terminals.transpose(0, 1).contiguous();
    rollouts.ratio = pufferl.rollouts.ratio.transpose(0, 1).contiguous();
    rollouts.values = pufferl.rollouts.values.transpose(0, 1).contiguous();

    // Inline any of these only used once
    int minibatch_size = hypers.minibatch_size;
    int batch_size = hypers.total_agents * hypers.horizon;
    int minibatch_segments = minibatch_size / hypers.horizon;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int current_epoch = pufferl.epoch;

    // Accumulators
    torch::Device device = rollouts.values.device();
    torch::TensorOptions scalar_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    Tensor pg_sum = torch::zeros({}, scalar_opts);
    Tensor v_sum = torch::zeros({}, scalar_opts);
    Tensor ent_sum = torch::zeros({}, scalar_opts);
    Tensor total_sum = torch::zeros({}, scalar_opts);
    Tensor old_approx_kl_sum = torch::zeros({}, scalar_opts);
    Tensor approx_kl_sum = torch::zeros({}, scalar_opts);
    Tensor clipfrac_sum = torch::zeros({}, scalar_opts);
    Tensor importance_sum = torch::zeros({}, scalar_opts);

    Policy* policy_bf16 = pufferl.policy_bf16;
    // Policy* policy_fp32 = pufferl.policy_fp32;
    torch::optim::Muon* muon = pufferl.muon;

    int total_epochs = hypers.total_timesteps / batch_size;

    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, total_epochs);
        muon->lr.fill_(lr);
    }

    // Annealed priority exponent - TODO: graphed?
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    rollouts.ratio.fill_(1.0);

    Tensor advantages = torch::zeros_like(rollouts.values, torch::kFloat32);  // fp32 precision

    compute_advantage(rollouts, advantages, hypers);
    pufferl.adv_mean.copy_(advantages.mean().detach());
    pufferl.adv_std.copy_(advantages.std().detach());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto dtype = get_dtype(hypers.bf16);
    Tensor mb_state = torch::zeros(
        {hypers.num_layers, minibatch_segments, 1, (int64_t)hypers.hidden_size},
        torch::dtype(dtype).device(rollouts.values.device())
    );

    int total_minibatches = hypers.replay_ratio * batch_size / hypers.minibatch_size;

    TrainGraph& graph = pufferl.train_buf;

    for (int mb = 0; mb < total_minibatches; ++mb) {
        advantages.fill_(0.0);

        profile_begin("compute_advantage", hypers.profile);
        compute_advantage(rollouts, advantages, hypers);
        profile_end(hypers.profile);

        // Inlined compute_prio
        profile_begin("compute_prio", hypers.profile);
        Tensor adv = advantages.abs().sum(1);
        Tensor prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
        Tensor prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
        Tensor idx = at::multinomial(prio_probs, minibatch_segments, true);
        Tensor mb_prio = torch::pow(hypers.total_agents*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);
        profile_end(hypers.profile);

        // Inlined train_select_and_copy
        profile_begin("train_select_and_copy", hypers.profile);
        {
            Tensor mb_obs = rollouts.observations.index_select(0, idx);
            Tensor mb_actions = rollouts.actions.index_select(0, idx);
            Tensor mb_logprobs = rollouts.logprobs.index_select(0, idx);
            Tensor mb_values = rollouts.values.index_select(0, idx);
            Tensor mb_advantages = advantages.index_select(0, idx);
            Tensor mb_returns = mb_advantages + mb_values;

            mb_state.zero_();
            graph.mb_obs.copy_(mb_obs, false);
            graph.mb_state.copy_(mb_state, false);
            graph.mb_actions.copy_(mb_actions, false);
            graph.mb_logprobs.copy_(mb_logprobs, false);
            graph.mb_advantages.copy_(mb_advantages, false);
            graph.mb_prio.copy_(mb_prio, false);
            graph.mb_values.copy_(mb_values, false);
            graph.mb_returns.copy_(mb_returns, false);
        }
        profile_end(hypers.profile);

        profile_begin("train_forward_graph", hypers.profile);
        if (hypers.cudagraphs && pufferl.train_warmup > CAPTURE_WARMUP) {
            pufferl.train_cudagraph.replay();
        } else {
            bool capturing = hypers.cudagraphs && pufferl.train_warmup == CAPTURE_WARMUP;
            auto saved_stream = at::cuda::getCurrentCUDAStream();
            auto cap_stream = capturing ? at::cuda::getStreamFromPool() : saved_stream;
            if (capturing) {
                at::cuda::setCurrentCUDAStream(cap_stream);
                pufferl.train_cudagraph.capture_begin(pufferl.train_pool_id);
            }

            auto [logits, newvalue] = pufferl.policy_bf16->forward_train(graph.mb_obs, graph.mb_state);

            Tensor loss = compute_train_loss(logits, newvalue,
                graph.mb_actions, graph.mb_logprobs, graph.mb_advantages, graph.mb_prio,
                graph.mb_values, graph.mb_returns,
                graph.mb_ratio, graph.mb_newvalue.view({graph.mb_ratio.size(0), graph.mb_ratio.size(1)}),
                pufferl.act_sizes, pufferl.act_sizes_cpu,
                hypers.minibatch_size, hypers.horizon,
                hypers.clip_coef, hypers.vf_clip_coef, hypers.vf_coef, hypers.ent_coef,
                pufferl.is_continuous, hypers.kernels);

            loss.backward();

            if (hypers.bf16) {
                copy_gradients_to_fp32(pufferl.policy_bf16, pufferl.policy_fp32);
            }
            clip_grad_norm_(pufferl.policy_fp32->parameters(), hypers.max_grad_norm);
            pufferl.muon->step();
            pufferl.muon->zero_grad();
            if (hypers.bf16) {
                pufferl.policy_bf16->zero_grad();
                sync_policy_weights(pufferl.policy_bf16, pufferl.policy_fp32);
            }

            if (capturing) {
                pufferl.train_cudagraph.capture_end();
                cap_stream.synchronize();
                cudaDeviceSynchronize();
                at::cuda::setCurrentCUDAStream(saved_stream);
            }
            if (hypers.cudagraphs) pufferl.train_warmup++;
        }
        profile_end(hypers.profile);

        // Update global ratio and values in-place (matches Python)
        pufferl.rollouts.ratio.index_copy_(1, idx, graph.mb_ratio.detach().squeeze(-1).to(dtype).transpose(0, 1));
        pufferl.rollouts.values.index_copy_(1, idx, graph.mb_newvalue.detach().squeeze(-1).to(dtype).transpose(0, 1));

    }
    pufferl.epoch += 1;

    // Compute explained variance at end of epoch
    /*
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    */
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
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
            fread(&nccl_id, sizeof(nccl_id), 1, f);
            fclose(f);
        }

        ncclCommInitRank(&pufferl->nccl_comm, hypers.world_size, nccl_id, hypers.rank);
        printf("Rank %d/%d: NCCL initialized\n", hypers.rank, hypers.world_size);
    }

    // Seeding (vary by rank for different random exploration)
    int seed = 42 + hypers.rank;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    pufferl->rng_seed = seed;
    pufferl->rng_offset = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    // Enable cuDNN benchmarking
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    torch::globalContext().setBenchmarkLimitCuDNN(32);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);

    // Enable faster FP16 reductions
    torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    // BF16 reduction (if using bfloat16)
    torch::globalContext().setAllowBF16ReductionCuBLAS(true);

    // Load environment first to get input_size and action info from env
    // act_sizes: 1D tensor of action space sizes per head
    // num_action_heads: number of action heads (for MultiDiscrete)
    // act_n: sum of action space sizes (decoder output dim)
    auto [vec, act_sizes] = create_environments(hypers.num_buffers, hypers.total_agents, env_name, vec_kwargs, env_kwargs, pufferl->env);
    int num_action_heads = pufferl->env.actions.size(1);
    int act_n = act_sizes.sum().item<int>();

    pufferl->vec = vec;
    pufferl->act_sizes = act_sizes;
    pufferl->act_sizes_cpu = act_sizes.cpu().to(torch::kInt64).contiguous();

    // Determine if action space is continuous or discrete
    // Continuous: all action dimensions have size 1
    // Discrete: all action dimensions have size > 1
    // Mixed: not supported (assert)
    {
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
        TORCH_CHECK(num_continuous == 0 || num_discrete == 0,
            "Mixed continuous/discrete action spaces not supported. "
            "All action dimensions must be either continuous (size==1) or discrete (size>1). "
            "Got ", num_continuous, " continuous and ", num_discrete, " discrete.");
        pufferl->is_continuous = (num_continuous > 0);
        if (pufferl->is_continuous) {
            printf("Detected continuous action space with %d dimensions\n", num_action_heads);
        } else {
            printf("Detected discrete action space with %d heads\n", num_action_heads);
        }
    }

    int input_size = pufferl->env.obs.size(1);
    int hidden_size = hypers.hidden_size;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;

    // Create encoder/decoder based on env_name
    // Decoder output size: discrete = act_n (sum of action sizes), continuous = num_action_heads
    // We need two sets for mixed-precision: fp32 (master) and bf16 (working)
    bool is_continuous = pufferl->is_continuous;
    int decoder_output_size = is_continuous ? num_action_heads : act_n;
    // TODO: Move conditional policy selection into ocean.cpp
    auto create_policy = [&]() -> Policy* {
        shared_ptr<Encoder> enc;
        shared_ptr<Decoder> dec;
        if (env_name == "puffer_snake") {
            enc = std::make_shared<SnakeEncoder>(input_size, hidden_size, 8);
            dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
        } else if (env_name == "falsepuffer_g2048") {
            enc = std::make_shared<SimpleG2048Encoder>(input_size, hidden_size);
            dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
        } else if (env_name == "puffer_nmmo3") {
            enc = std::make_shared<NMMO3Encoder>(input_size, hidden_size);
            dec = std::make_shared<NMMO3Decoder>(hidden_size, decoder_output_size);
        } else if (env_name == "puffer_drive") {
            enc = std::make_shared<DriveEncoder>(input_size, hidden_size);
            dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
        } else {
            enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
            dec = std::make_shared<DefaultDecoder>(hidden_size, decoder_output_size, is_continuous);
        }
        auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, kernels);
        return new Policy(enc, dec, rnn, input_size, act_n, hidden_size);
    };

    // Create fp32 master policy (for optimizer - precise gradient accumulation)
    Policy* policy_fp32 = create_policy();
    policy_fp32->to(torch::kCUDA);
    policy_fp32->to(torch::kFloat32);
    pufferl->policy_fp32 = policy_fp32;

    if (hypers.bf16) {
        // create bf16 working policy (for fwd/bwd)
        Policy* policy_bf16 = create_policy();
        policy_bf16->to(torch::kCUDA);
        policy_bf16->to(torch::kBFloat16);
        pufferl->policy_bf16 = policy_bf16;
        sync_policy_weights(policy_bf16, policy_fp32); // initial sync
    } else {
        // just use same policy for both
        pufferl->policy_bf16 = policy_fp32;
    }

    // Optimizer uses fp32 master weights for precise gradient accumulation
    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new torch::optim::Muon(policy_fp32->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));
    pufferl->muon->init_contiguous_weights();
    pufferl->muon->nccl_comm = pufferl->nccl_comm;
    pufferl->muon->world_size = hypers.world_size;
    printf("DEBUG: Contiguous weight buffer: %ld elements\n", pufferl->muon->weight_buffer.numel());


    // Allocate buffers
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;

    printf("DEBUG: num_envs=%d, total_agents=%d, batch=%d, num_buffers=%d\n",
        vec->size, total_agents, batch, num_buffers);

    int minibatch_segments = hypers.minibatch_size / horizon;

    pufferl->rollouts = create_rollouts(horizon, total_agents, input_size, num_action_heads, hypers.bf16);
    pufferl->train_buf = create_train_graph(minibatch_segments, horizon, input_size,
        num_layers, hidden_size, num_action_heads, hypers.bf16);

    // always fp32 since advantages are computed in fp32
    pufferl->adv_mean = torch::zeros({1}, cuda_f32);
    pufferl->adv_std = torch::ones({1}, cuda_f32);

    // Per-buffer states: each is {num_layers, block_size, hidden} for contiguous access
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = pufferl->policy_bf16->initial_state(batch, torch::kCUDA);
    }

    if (hypers.cudagraphs) {
        pufferl->train_cudagraph = at::cuda::CUDAGraph();
        pufferl->train_pool_id = at::cuda::graph_pool_handle();
        pufferl->train_warmup = 0;

        // Fused rollout cudagraphs: [horizon][num_buffers]
        pufferl->rollout_pool_id = at::cuda::graph_pool_handle();
        pufferl->fused_rollout_cudagraphs.resize(horizon);
        for (int h = 0; h < horizon; ++h) {
            pufferl->fused_rollout_cudagraphs[h].resize(num_buffers);
            for (int b = 0; b < num_buffers; ++b) {
                pufferl->fused_rollout_cudagraphs[h][b] = at::cuda::CUDAGraph();
            }
        }

        // Snapshot weights + optimizer state before init-time capture
        Tensor saved_weights = pufferl->muon->weight_buffer.clone();
        Tensor saved_momentum;
        if (pufferl->muon->momentum_buffer.defined()) {
            saved_momentum = pufferl->muon->momentum_buffer.clone();
        }

        // Run warmup + capture on a fresh stream (matching original capture_graph).
        // Tensors get associated with warmup_stream, not the default stream.
        // Captured graphs' event-waits reference warmup_stream which is dead at runtime.
        auto saved_stream = at::cuda::getCurrentCUDAStream();
        auto warmup_stream = at::cuda::getStreamFromPool();
        at::cuda::setCurrentCUDAStream(warmup_stream);

        // Init-time warmup + capture BEFORE creating streams/threads.
        // No per-buffer streams exist yet = no cross-stream deps baked into graphs.
        for (pufferl->epoch = 0; pufferl->epoch <= CAPTURE_WARMUP; pufferl->epoch++) {
            rollouts_impl(*pufferl);
        }
        pufferl->rollout_captured = true;

        for (int i = 0; i <= CAPTURE_WARMUP; i++) {
            train_impl(*pufferl);
        }

        warmup_stream.synchronize();
        cudaDeviceSynchronize();
        at::cuda::setCurrentCUDAStream(saved_stream);

        // Restore weights + optimizer state corrupted by warmup/capture
        {
        torch::NoGradGuard no_grad;
        pufferl->muon->weight_buffer.copy_(saved_weights);
        if (saved_momentum.defined()) {
            pufferl->muon->momentum_buffer.copy_(saved_momentum);
        } else {
            pufferl->muon->momentum_buffer = Tensor();
        }
        if (hypers.bf16) {
            sync_policy_weights(pufferl->policy_bf16, pufferl->policy_fp32);
        }
        pufferl->muon->zero_grad();
        if (hypers.bf16) {
            pufferl->policy_bf16->zero_grad();
        }
        } // end NoGradGuard

        pufferl->epoch = 0;
    }

    // Create PyTorch-managed streams and assign to vec
    for (int i = 0; i < num_buffers; i++) {
        pufferl->torch_streams.push_back(at::cuda::getStreamFromPool(false));
        vec->streams[i] = pufferl->torch_streams[i].stream();
    }

    // Static breakout - OMP only
    if (hypers.use_omp) {
        create_static_threads(vec, hypers.num_threads, horizon, pufferl.get(), net_callback_wrapper, thread_init_wrapper);
    }
    static_vec_reset(vec);

    return pufferl;
}

void print_tensor_info(const char* name, const Tensor& t) {
    if (t.defined() && t.numel() > 0) {
        size_t bytes = t.numel() * t.element_size();
        int64_t refcount = t.use_count();
        printf("  %s: %.2f MB, refcount=%ld, device=%s\n",
               name, bytes / 1e6, refcount,
               t.device().str().c_str());
    }
}

void close_impl(PuffeRL& pufferl) {
    cudaDeviceSynchronize();
    for (size_t i = 0; i < pufferl.buffer_states.size(); i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "buffer_states[%zu]", i);
    }
    // Policy params total
    size_t policy_bytes = 0;
    for (const auto& p : pufferl.policy_fp32->parameters()) {
        policy_bytes += p.numel() * p.element_size();
    }
    if (pufferl.hypers.bf16) {
        size_t bf16_bytes = 0;
        for (const auto& p : pufferl.policy_bf16->parameters()) {
            bf16_bytes += p.numel() * p.element_size();
        }
    }

    // Reset CUDA graphs first (they hold references to tensor memory)
    pufferl.train_cudagraph.reset();
    pufferl.fused_rollout_cudagraphs.clear();

    // Clear optimizer buffers explicitly (policy params are views into weight_buffer)
    pufferl.muon->weight_buffer = Tensor();
    pufferl.muon->momentum_buffer = Tensor();
    pufferl.muon->lr = Tensor();
    // Clear the param_groups to release parameter references
    pufferl.muon->param_groups().clear();
    delete pufferl.muon;
    pufferl.muon = nullptr;

    // Delete policies - check if bf16 and fp32 are the same pointer
    if (pufferl.hypers.bf16 && pufferl.policy_bf16 != pufferl.policy_fp32) {
        delete pufferl.policy_bf16;
    }
    delete pufferl.policy_fp32;
    pufferl.policy_bf16 = nullptr;
    pufferl.policy_fp32 = nullptr;

    // Clear buffer states (releases CUDA tensors)
    pufferl.buffer_states.clear();

    // Clear rollout buffers (releases CUDA tensors)
    pufferl.rollouts.observations = Tensor();
    pufferl.rollouts.actions = Tensor();
    pufferl.rollouts.values = Tensor();
    pufferl.rollouts.logprobs = Tensor();
    pufferl.rollouts.rewards = Tensor();
    pufferl.rollouts.terminals = Tensor();
    pufferl.rollouts.ratio = Tensor();
    pufferl.rollouts.importance = Tensor();

    // Clear train buffers (releases CUDA tensors)
    pufferl.train_buf.mb_obs = Tensor();
    pufferl.train_buf.mb_state = Tensor();
    pufferl.train_buf.mb_actions = Tensor();
    pufferl.train_buf.mb_logprobs = Tensor();
    pufferl.train_buf.mb_advantages = Tensor();
    pufferl.train_buf.mb_prio = Tensor();
    pufferl.train_buf.mb_values = Tensor();
    pufferl.train_buf.mb_returns = Tensor();
    pufferl.train_buf.mb_ratio = Tensor();
    pufferl.train_buf.mb_newvalue = Tensor();

    // Clear misc tensors
    pufferl.adv_mean = Tensor();
    pufferl.adv_std = Tensor();
    pufferl.act_sizes = Tensor();
    pufferl.act_sizes_cpu = Tensor();
    pufferl.rng_offset = Tensor();

    // Clear env tensors (from_blob wrappers - don't own memory but hold refs)
    pufferl.env.obs = Tensor();
    pufferl.env.actions = Tensor();
    pufferl.env.rewards = Tensor();
    pufferl.env.terminals = Tensor();

    // Clear torch streams
    pufferl.torch_streams.clear();

    // Close environment vectorization (frees env GPU buffers)
    static_vec_close(pufferl.vec);
    pufferl.vec = nullptr;

    // Cleanup NCCL
    if (pufferl.nccl_comm != nullptr) {
        ncclCommDestroy(pufferl.nccl_comm);
        pufferl.nccl_comm = nullptr;
    }

    // Force CUDA to release cached memory first
    c10::cuda::CUDACachingAllocator::emptyCache();
    cudaDeviceSynchronize();

}

// nsys capture control (--capture-range=cudaProfilerApi). Different from profile_begin/end which are nvtx ranges.
void profiler_start() {
    cudaDeviceSynchronize();
    printf("cudaProfilerStart()\n");
    cudaProfilerStart();
}

void profiler_stop() {
    cudaDeviceSynchronize();
    cudaProfilerStop();
    printf("cudaProfilerStop()\n");
}

} // namespace pufferlib
