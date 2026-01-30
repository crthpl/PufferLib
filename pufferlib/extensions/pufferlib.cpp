//TODO:clamped
//5.6% cat overhead from grad clip. Preallocate?
//11% seqwise overhead from fused scan
//30% elemwise form random ops
//5% on log_coeffs_and_values

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <atomic>
#include "vecenv.h"
#include <dlfcn.h>
#include "muon.h"

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>

#include <nvtx3/nvToolsExt.h>

#include <functional>
#include <iostream>
#include <vector>


typedef torch::Tensor Tensor;

// CUDA kernel wrappers
#include "modules.cpp"

auto DTYPE = torch::kFloat32;

namespace pufferlib {

// Advantage computation is in advantage.cpp
#include "advantage.cpp"

// Model classes are in models.cpp
#include "models.cpp"

// Function pointer type for loading exports
typedef EnvExports* (*get_env_exports_fn)(void);


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

// Torch is stupid. Had to clip out a redundant cuda sync.
void clip_grad_norm_(
    const std::vector<Tensor>& parameters,
    double max_norm,
    double norm_type = 2.0
    ) {
  std::vector<Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }

  if (params_with_grad.empty()) {
    return;
  }

  Tensor total_norm_tensor;
  if (norm_type == std::numeric_limits<double>::infinity()) {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().abs().max());
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
  } else if (norm_type == 0) {
    total_norm_tensor =
        torch::full({}, static_cast<double>(params_with_grad.size()));
  } else {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().norm(norm_type));
    }
    total_norm_tensor =
        (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
  }

  Tensor clip_coef = max_norm / (total_norm_tensor + 1e-6);
  Tensor clip_coef_clamped =
      torch::clamp(clip_coef, std::nullopt /* min */, 1.0 /* max */);
  for (auto& param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
  }
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = static_cast<float>(t) / static_cast<float>(T);
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}


typedef struct {
    Tensor obs;
    Tensor actions;
    Tensor rewards;
    Tensor terminals;
} EnvBuf;

std::tuple<EnvExports*, VecEnv*, Tensor>
create_environments(int num_buffers, int total_agents, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs, EnvBuf& env) {
    std::string name = env_name;
    if (name.rfind("puffer_", 0) == 0) {
        name = name.substr(7);
    }
    std::string so_path = "./" + name + ".so";
    void* handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
        exit(1);
    }
    dlerror();

    // Single dlsym call to get all exports
    get_env_exports_fn get_exports = (get_env_exports_fn)dlsym(handle, "get_env_exports");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        exit(1);
    }
    EnvExports* env_exports = get_exports();

    VecEnv* vec = env_exports->create_environments(num_buffers, true, 0, vec_kwargs, env_kwargs);
    printf("DEBUG create_environments: vec->size=%d, vec->total_agents=%d\n",
        vec->size, vec->total_agents);

    auto obs_dtype = to_torch_dtype(env_exports->obs_type);

    env.obs = torch::from_blob(vec->gpu_observations, {total_agents, env_exports->obs_n}, torch::dtype(obs_dtype).device(torch::kCUDA));
    env.actions = torch::from_blob(vec->gpu_actions, {total_agents, env_exports->num_atns}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    env.rewards = torch::from_blob(vec->gpu_rewards, {total_agents}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    env.terminals = torch::from_blob(vec->gpu_terminals, {total_agents}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Create act_sizes tensor on CUDA (needed for sample_logits kernel)
    Tensor act_sizes = torch::from_blob(env_exports->act_sizes, {env_exports->num_atns}, torch::dtype(torch::kInt32)).to(torch::kCUDA);

    return std::make_tuple(env_exports, vec, act_sizes);
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

TrainGraph create_train_graph(int minibatch_segments, int horizon, int input_size,
        int num_layers, int hidden_size, int expansion_factor, int num_atns) {
    TrainGraph g;
    auto options = torch::TensorOptions().dtype(DTYPE).device(torch::kCUDA);
    g.mb_obs = torch::zeros({minibatch_segments, horizon, input_size}, options);
    g.mb_state = torch::zeros({num_layers, minibatch_segments, 1, hidden_size * expansion_factor}, options);
    g.mb_newvalue = torch::zeros({minibatch_segments, horizon, 1}, options);
    g.mb_ratio = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_actions = torch::zeros({minibatch_segments, horizon, num_atns}, options).to(torch::kInt64);
    g.mb_logprobs = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_advantages = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_prio = torch::zeros({minibatch_segments, 1}, options);
    g.mb_values = torch::zeros({minibatch_segments, horizon}, options);
    g.mb_returns = torch::zeros({minibatch_segments, horizon}, options);
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

RolloutBuf create_rollouts(int horizon, int segments, int input_size, int num_atns) {
    RolloutBuf r;
    r.observations = torch::zeros({horizon, segments, input_size}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.actions = torch::zeros({horizon, segments, num_atns}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    r.values = torch::zeros({horizon, segments}, torch::dtype(DTYPE).device(torch::kCUDA));
    r.logprobs = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.rewards = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.terminals = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.ratio = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    r.importance = torch::zeros({horizon, segments}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return r;
}

typedef struct {
    // Layout
    int segments;
    int horizon;
    int total_agents;
    int num_buffers;
    int minibatch_segments;
    int total_minibatches;
    int accumulate_minibatches;
    // Model architecture
    int num_atns;
    int hidden_size;
    int expansion_factor;
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
    int max_epochs;
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
} HypersT;

typedef struct {
    PolicyMinGRU* policy;
    VecEnv* vec;
    torch::optim::Muon* muon;
    EnvExports* env_exports;
    HypersT hypers;
    std::vector<Tensor> buffer_states;  // Per-buffer states for contiguous access
    RolloutBuf rollouts;
    EnvBuf env;
    TrainGraph train_buf;
    std::vector<std::vector<at::cuda::CUDAGraph>> fused_rollout_cudagraphs;  // [horizon][num_buffers]
    at::cuda::CUDAGraph train_cudagraph;
    std::vector<at::cuda::CUDAStream> torch_streams;  // PyTorch-managed streams for OMP
    Tensor adv_mean;
    Tensor adv_std;
    Tensor act_sizes;      // CUDA int32 tensor of action head sizes for MultiDiscrete
    Tensor act_sizes_cpu;  // CPU int64 tensor (pre-computed to avoid alloc during graph replay)
    int epoch;
    uint64_t rng_seed;
    Tensor rng_offset;  // CUDA tensor so increment is graphable
} PuffeRL;

Dict* log_environments_impl(PuffeRL& pufferl) {
    Dict* out = create_dict(32);
    pufferl.env_exports->vec_log(pufferl.vec, out);
    return out;
}

// Fused rollout step: reads from env, runs forward, writes directly to rollouts storage
// Eliminates intermediate RolloutGraph buffers when cudagraphed
void fused_rollout_step(PuffeRL& pufferl, int h, int buf) {
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;
    int total_agents = pufferl.vec->total_agents;
    int num_buffers = hypers.num_buffers;
    int block_size = total_agents / num_buffers;

    // Get slices for this buffer
    Tensor obs_slice = pufferl.env.obs.narrow(0, buf*block_size, block_size);
    Tensor& state = pufferl.buffer_states[buf];

    // Run policy forward
    auto [logits, value, state_out] = pufferl.policy->forward(obs_slice, state);

    // Get output slices in rollouts storage
    Tensor actions_out = pufferl.rollouts.actions.select(0, h).narrow(0, buf*block_size, block_size);
    Tensor logprobs_out = pufferl.rollouts.logprobs.select(0, h).narrow(0, buf*block_size, block_size);
    Tensor values_out = pufferl.rollouts.values.select(0, h).narrow(0, buf*block_size, block_size);

    // Sample actions and write directly to rollouts
    if (hypers.kernels) {
        sample_logits(logits, value, actions_out, logprobs_out,
            values_out, pufferl.act_sizes, pufferl.rng_seed, pufferl.rng_offset);
    } else {
        int num_action_heads = actions_out.size(1);
        logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);

        auto split_logits = torch::split(logits, c10::IntArrayRef(pufferl.act_sizes_cpu.data_ptr<int64_t>(), num_action_heads), 1);
        std::vector<Tensor> actions_vec;
        std::vector<Tensor> logprobs_vec;

        for (int i = 0; i < num_action_heads; i++) {
            Tensor head_logits = split_logits[i];
            Tensor log_probs = torch::log_softmax(head_logits, 1);
            Tensor action = at::multinomial(log_probs.exp(), 1, true);
            Tensor logprob = log_probs.gather(1, action);
            actions_vec.push_back(action);
            logprobs_vec.push_back(logprob);
        }
        actions_out.copy_(torch::cat(actions_vec, 1).to(torch::kFloat64), false);
        logprobs_out.copy_(torch::cat(logprobs_vec, 1).sum(1), false);
        values_out.copy_(value.flatten(), false);
    }

    // Update state
    state.copy_(state_out, false);

    // Copy obs to rollouts
    pufferl.rollouts.observations.select(0, h).narrow(0, buf*block_size, block_size).copy_(obs_slice, true);

    // Copy rewards and terminals from env to rollouts
    pufferl.rollouts.rewards.select(0, h).narrow(0, buf*block_size, block_size).copy_(
        pufferl.env.rewards.narrow(0, buf*block_size, block_size), true);
    pufferl.rollouts.terminals.select(0, h).narrow(0, buf*block_size, block_size).copy_(
        pufferl.env.terminals.narrow(0, buf*block_size, block_size), true);

    // Copy actions to env for next step
    pufferl.env.actions.narrow(0, buf*block_size, block_size).copy_(actions_out, true);
}

void train_forward_call(TrainGraph& graph, PolicyMinGRU* policy,
        torch::optim::Muon* muon, HypersT& hypers, Tensor& adv_mean, Tensor& adv_std, Tensor& act_sizes_cpu, bool kernels) {
    auto [logits, newvalue] = policy->forward_train(graph.mb_obs.to(DTYPE), graph.mb_state);

    Tensor loss;
    if (kernels) {
        auto [mb_adv_var, mb_adv_mean] = torch::var_mean(graph.mb_advantages);  // single kernel launch
        loss = fused_ppo_loss_optimized(
            logits,
            newvalue,
            graph.mb_actions,
            graph.mb_logprobs.to(logits.dtype()),
            graph.mb_advantages.to(logits.dtype()),
            graph.mb_prio.to(logits.dtype()),
            graph.mb_values.to(logits.dtype()),
            graph.mb_returns.to(logits.dtype()),
            mb_adv_mean,
            mb_adv_var,  // variance, not std - kernel does sqrtf to avoid second kernel launch here
            graph.mb_ratio,
            graph.mb_newvalue.view({graph.mb_ratio.size(0), graph.mb_ratio.size(1)}),
            hypers.clip_coef,
            hypers.vf_clip_coef,
            hypers.vf_coef,
            hypers.ent_coef
        )[0];
    } else {
        int num_action_heads = graph.mb_actions.size(-1);
        int batch = hypers.minibatch_segments * hypers.horizon;

        // Split logits by action head sizes and compute log probs for each head
        Tensor flat_logits = logits.reshape({batch, -1});
        flat_logits = torch::nan_to_num(flat_logits, 1e-8, 1e-8, 1e-8);
        auto split_logits = torch::split(flat_logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_action_heads), 1);

        std::vector<Tensor> logprobs_vec;
        std::vector<Tensor> entropies_vec;

        for (int h = 0; h < num_action_heads; h++) {
            Tensor head_logits = split_logits[h];
            Tensor log_probs = torch::log_softmax(head_logits, 1);
            Tensor probs = log_probs.exp();
            Tensor head_actions = graph.mb_actions.select(-1, h).reshape({batch}).to(torch::kInt64);
            Tensor logprob = log_probs.gather(1, head_actions.unsqueeze(1));
            logprobs_vec.push_back(logprob);
            entropies_vec.push_back(-(probs * log_probs).sum(1, true));
        }

        // Stack and reduce - no per-iteration allocations
        Tensor newlogprob = torch::cat(logprobs_vec, 1).sum(1).reshape({hypers.minibatch_segments, hypers.horizon});
        Tensor entropy = torch::cat(entropies_vec, 1).sum(1).mean();

        // Compute ratio
        Tensor logratio = newlogprob - graph.mb_logprobs;
        Tensor ratio_new = logratio.exp();
        graph.mb_ratio.copy_(ratio_new, false);
        graph.mb_newvalue.copy_(newvalue, false);

        // Normalize advantages: (adv - mean) / std, then weight
        Tensor adv_normalized = graph.mb_advantages;
        adv_normalized = graph.mb_prio * (adv_normalized - adv_normalized.mean()) / (adv_normalized.std() + 1e-8);

        // Policy loss
        Tensor pg_loss1 = -adv_normalized * ratio_new;
        Tensor pg_loss2 = -adv_normalized * torch::clamp(ratio_new, 1.0 - hypers.clip_coef, 1.0 + hypers.clip_coef);
        Tensor pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(graph.mb_returns.sizes());
        Tensor v_clipped = graph.mb_values + torch::clamp(newvalue - graph.mb_values,
            -hypers.vf_clip_coef, hypers.vf_clip_coef);
        Tensor v_loss_unclipped = (newvalue - graph.mb_returns).pow(2);
        Tensor v_loss_clipped = (v_clipped - graph.mb_returns).pow(2);
        Tensor v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

        // Total loss
        loss = pg_loss + hypers.vf_coef*v_loss - hypers.ent_coef*entropy;
        /*
        {
            torch::NoGradGuard no_grad;

            // Accumulate stats
            pg_sum += pg_loss.detach();
            v_sum += v_loss.detach();
            ent_sum += entropy.detach();
            total_sum += loss.detach();

            // KL and clipping diagnostics (matches Python)
            auto old_kl = (-logratio).mean();
            auto kl = ((ratio_new - 1) - logratio).mean();
            auto cf = (ratio_new - 1.0).abs().gt(hypers.clip_coef).to(torch::kFloat32).mean();
            auto imp = ratio_new.mean();

            old_approx_kl_sum += old_kl.detach();
            approx_kl_sum += kl.detach();
            clipfrac_sum += cf.detach();
            importance_sum += imp.detach();
        }
        */
    }

    loss.backward();
    clip_grad_norm_(policy->parameters(), hypers.max_grad_norm);
    muon->step();
    muon->zero_grad();
}

// Capture
void capture_graph(at::cuda::CUDAGraph* graph, std::function<void()> func) {
    /* Checklist for avoiding diabolical capture bugs:
     * 1. Don't start separate streams before tracing (i.e. env gpu buffers)
     * 2. Make sure input/output buffer pointers don't change
     * 3. Make sure to restore the original stream after tracing
     * 4. All custom kernels need to use the default torch stream
     * 5. Make sure you are using the torch stream fns, not the c10 ones.
     * 6. Scalars get captured by value. They cannot change between calls.
     */
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < 10; ++i) {
        func();
    }
    warmup_stream.synchronize();

    auto cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    graph->capture_begin();
    func();
    graph->capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();

    at::cuda::setCurrentCUDAStream(current_stream);
}


// ============================================================================
// Rollout and train section functions
// ============================================================================

inline void profile_begin(const char* tag, bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePushA(tag); }
}

inline void profile_end(bool enable) {
    if (enable) { cudaDeviceSynchronize(); nvtxRangePop(); }
}

void env_recv(PuffeRL& pufferl, int buf) {
    pufferl.env_exports->vec_recv(pufferl.vec, buf, pufferl.vec->streams[buf]);
}

void env_send(PuffeRL& pufferl, int buf) {
    pufferl.env_exports->vec_send(pufferl.vec, buf, pufferl.vec->streams[buf]);
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

// Callback for OMP threadmanager - runs policy forward for one (buf, t) step
extern "C" void net_callback_wrapper(void* ctx, int buf, int t) {
    torch::NoGradGuard no_grad;
    PuffeRL* pufferl = (PuffeRL*)ctx;
    HypersT& hypers = pufferl->hypers;

    profile_begin("fused_rollout", hypers.profile);
    if (hypers.cudagraphs) {
        // Fused cudagraph: input copy + forward + output copy in one shot
        pufferl->fused_rollout_cudagraphs[t][buf].replay();
    } else {
        fused_rollout_step(*pufferl, t, buf);
    }
    profile_end(hypers.profile);
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl_impl(HypersT& hypers, const std::string& env_name, Dict* vec_kwargs, Dict* env_kwargs) {
    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->hypers = hypers;

    // Seeding
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);
    pufferl->rng_seed = 42;
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
    auto [env_exports, vec, act_sizes] = create_environments(hypers.num_buffers, hypers.total_agents, env_name, vec_kwargs, env_kwargs, pufferl->env);
    int num_action_heads = pufferl->env.actions.size(1);
    int act_n = act_sizes.sum().item<int>();

    pufferl->env_exports = env_exports;
    pufferl->vec = vec;
    pufferl->act_sizes = act_sizes;
    pufferl->act_sizes_cpu = act_sizes.cpu().to(torch::kInt64).contiguous();

    int input_size = pufferl->env.obs.size(1);
    int hidden_size = hypers.hidden_size;
    int expansion_factor = hypers.expansion_factor;
    int num_layers = hypers.num_layers;
    bool kernels = hypers.kernels;

    // Create encoder/decoder based on env_name
    // Decoder output size is act_n (sum of all action space sizes)
    std::shared_ptr<Encoder> enc;
    std::shared_ptr<Decoder> dec;
    if (env_name == "puffer_snake") {
        enc = std::make_shared<SnakeEncoder>(input_size, hidden_size, 8);
        dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    } else if (env_name == "puffer_g2048") {
        enc = std::make_shared<G2048Encoder>(input_size, hidden_size);
        dec = std::make_shared<G2048Decoder>(hidden_size, act_n);
    } else if (env_name == "puffer_nmmo3") {
        enc = std::make_shared<NMMO3Encoder>(input_size, hidden_size);
        dec = std::make_shared<NMMO3Decoder>(hidden_size, act_n);
    } else if (env_name == "puffer_drive") {
        enc = std::make_shared<DriveEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    } else {
        enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
        dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    }

    PolicyMinGRU* policy = new PolicyMinGRU(enc, dec, input_size, act_n, hidden_size, expansion_factor, num_layers, kernels);
    policy->to(torch::kCUDA);
    policy->to(DTYPE);
    pufferl->policy = policy;

    float lr = hypers.lr;
    float beta1 = hypers.beta1;
    float eps = hypers.eps;
    pufferl->muon = new torch::optim::Muon(policy->parameters(),
        torch::optim::MuonOptions(lr).momentum(beta1).eps(eps));

    // Allocate buffers
    int segments = hypers.segments;
    int horizon = hypers.horizon;
    int total_agents = vec->total_agents;
    int batch = total_agents / hypers.num_buffers;
    int num_buffers = hypers.num_buffers;
    int minibatch_segments = hypers.minibatch_segments;

    printf("DEBUG: num_envs=%d, total_agents=%d, segments=%d, batch=%d, num_buffers=%d\n",
        vec->size, total_agents, segments, batch, num_buffers);

    pufferl->rollouts = create_rollouts(horizon, total_agents, input_size, num_action_heads);
    pufferl->train_buf = create_train_graph(minibatch_segments, horizon, input_size,
        policy->num_layers, policy->hidden_size, policy->expansion_factor, num_action_heads);

    pufferl->adv_mean = torch::zeros({1}, torch::dtype(DTYPE).device(torch::kCUDA));
    pufferl->adv_std = torch::ones({1}, torch::dtype(DTYPE).device(torch::kCUDA));

    // Per-buffer states: each is {num_layers, block_size, hidden} for contiguous access
    pufferl->buffer_states.resize(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        pufferl->buffer_states[i] = policy->initial_state(batch, torch::kCUDA);
    }

    if (hypers.cudagraphs) {
        pufferl->train_cudagraph = at::cuda::CUDAGraph();

        auto* p = pufferl.get();
        capture_graph(&pufferl->train_cudagraph, [p]() {
            train_forward_call(p->train_buf, p->policy, p->muon,
                p->hypers, p->adv_mean, p->adv_std, p->act_sizes_cpu, p->hypers.kernels);
        });

        // Fused rollout cudagraphs: [horizon][num_buffers]
        // Each graph does input copy + forward + output copy in one shot
        pufferl->fused_rollout_cudagraphs.resize(horizon);
        for (int h = 0; h < horizon; ++h) {
            pufferl->fused_rollout_cudagraphs[h].resize(num_buffers);
            for (int b = 0; b < num_buffers; ++b) {
                pufferl->fused_rollout_cudagraphs[h][b] = at::cuda::CUDAGraph();
                capture_graph(&pufferl->fused_rollout_cudagraphs[h][b], [p, h, b]() {
                    fused_rollout_step(*p, h, b);
                });
            }
        }

    }

    // FAILS IF DONE AFTER CREATE_ENVIRONMENTS
    // Try num_threads=0 to disable threading for debugging
    int num_threads = 16;
    int block_size = vec->size / 16;
    if (vec->size < num_threads) {
        num_threads = vec->size;
    }
    if (block_size < 1) {
        block_size = 1;
    }
    if (hypers.use_omp) {
        pufferl->env_exports->create_threads(vec, num_threads, block_size, true, pufferl.get(), net_callback_wrapper, thread_init_wrapper, horizon);

        // Create PyTorch-managed streams and replace vec->streams with their raw cudaStream_t
        // This ensures PyTorch properly recognizes the streams for all operations
        for (int i = 0; i < num_buffers; i++) {
            pufferl->torch_streams.push_back(at::cuda::getStreamFromPool(false));
            vec->streams[i] = pufferl->torch_streams[i].stream();
        }
    } else {
        pufferl->env_exports->create_threads(vec, num_threads, block_size, false, nullptr, nullptr, nullptr, 0);
    }
    pufferl->env_exports->vec_reset(vec);

    return pufferl;
}

std::tuple<Tensor, Tensor> compute_prio(Tensor& advantages,
        int minibatch_segments, int segments,
        float prio_alpha, float anneal_beta) {
    Tensor adv = advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, minibatch_segments, true);
    Tensor mb_prio = torch::pow(segments*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);
    return {idx, mb_prio};
}

void train_select_and_copy(TrainGraph& graph, RolloutBuf& rollouts,
        Tensor& advantages, Tensor& idx, Tensor& mb_state, Tensor& mb_prio) {
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

void rollouts_impl(PuffeRL& pufferl) {
    torch::NoGradGuard no_grad;
    HypersT& hypers = pufferl.hypers;

    int horizon = hypers.horizon;
    int num_buffers = hypers.num_buffers;
    // TODO: You removed state zeros and reward clamping

    for (int i = 0; i < num_buffers*horizon; ++i) {
        int buf = i % num_buffers;
        int h = i / num_buffers;

        profile_begin("env_recv", hypers.profile);
        env_recv(pufferl, buf);
        profile_end(hypers.profile);

        net_callback_wrapper(&pufferl, buf, h);

        // TODO: There should be a lighter way to sync. You need to make sure the torch data streams
        // are ready because puffer vec uses different streams. Setting to non-blocking is not enough.
        cudaDeviceSynchronize();

        profile_begin("env_send", hypers.profile);
        env_send(pufferl, buf);
        profile_end(hypers.profile);
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
    int total_minibatches = hypers.total_minibatches;
    int minibatch_segments = hypers.minibatch_segments;
    int segments = hypers.segments;
    float prio_beta0 = hypers.prio_beta0;
    float prio_alpha = hypers.prio_alpha;
    bool anneal_lr = hypers.anneal_lr;
    int total_epochs = hypers.max_epochs;
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

    PolicyMinGRU* policy = pufferl.policy;
    torch::optim::Muon* muon = pufferl.muon;

    if (anneal_lr) {
        float lr_min = hypers.min_lr_ratio * hypers.lr;
        float lr = cosine_annealing(hypers.lr, lr_min, current_epoch, hypers.max_epochs);
        muon->lr.fill_(lr);
    }

    // Annealed priority exponent - TODO: graphed?
    float anneal_beta = prio_beta0 + (1.0f - prio_beta0) * prio_alpha * (float)current_epoch/(float)total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    rollouts.ratio.fill_(1.0);

    Tensor advantages = torch::zeros_like(rollouts.values);

    compute_advantage(rollouts, advantages, hypers);

    pufferl.adv_mean.copy_(advantages.mean().detach());
    pufferl.adv_std.copy_(advantages.std().detach());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Tensor mb_state = torch::zeros(
        {policy->num_layers, minibatch_segments, 1, (int64_t)(policy->hidden_size*policy->expansion_factor)},
        torch::dtype(DTYPE).device(rollouts.values.device())
    );

    // Temporary: random indices and uniform weights
    /*
    auto idx = torch::randint(0, segments, {minibatch_segments}, torch::dtype(torch::kInt64).device(device));
    auto mb_prio = torch::ones({minibatch_segments, 1}, torch::dtype(torch::kFloat32).device(device));
    */

    for (int mb = 0; mb < total_minibatches; ++mb) {
        advantages.fill_(0.0);

        profile_begin("compute_advantage", hypers.profile);
        compute_advantage(rollouts, advantages, hypers);
        profile_end(hypers.profile);

        profile_begin("compute_prio", hypers.profile);
        auto [idx, mb_prio] = compute_prio(advantages, minibatch_segments, segments,
            prio_alpha, anneal_beta);
        profile_end(hypers.profile);

        profile_begin("train_select_and_copy", hypers.profile);
        TrainGraph& graph = pufferl.train_buf;
        train_select_and_copy(graph, rollouts, advantages, idx, mb_state, mb_prio);
        profile_end(hypers.profile);

        profile_begin("train_forward_graph", hypers.profile);
        if (hypers.cudagraphs) {
            pufferl.train_cudagraph.replay();
        } else {
            train_forward_call(graph, pufferl.policy, pufferl.muon,
                hypers, pufferl.adv_mean, pufferl.adv_std, pufferl.act_sizes_cpu, hypers.kernels);
        }
        profile_end(hypers.profile);

        // Update global ratio and values in-place (matches Python)
        // Buffers are {horizon, segments}, so index_copy_ along dim 1 (segments)
        // Source is {minibatch_segments, horizon}, need to transpose to {horizon, minibatch_segments}
        // Temporary: use slice instead of index_copy_ for contiguous test
        /*
        pufferl.rollouts.ratio.slice(1, 0, minibatch_segments).copy_(graph.ratio.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        pufferl.rollouts.values.slice(1, 0, minibatch_segments).copy_(graph.newvalue.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        */
        // Original index_copy_ version:
        pufferl.rollouts.ratio.index_copy_(1, idx, graph.mb_ratio.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));
        pufferl.rollouts.values.index_copy_(1, idx, graph.mb_newvalue.detach().squeeze(-1).to(torch::kFloat32).transpose(0, 1));

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

// Profiler control for nsys --capture-range=cudaProfilerApi
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
