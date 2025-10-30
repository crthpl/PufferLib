#include <torch/extension.h>
#include <torch/torch.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

namespace py = pybind11;

namespace pufferlib {

void puff_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}


// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, importance + offset, advantages + offset,
            gamma, lambda, rho_clip, c_clip, horizon
        );
    }
}


void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon
    );
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
 }

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_squared_environments(int64_t num_envs, int64_t grid_size, torch::Tensor dummy);

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

void step_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

void reset_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

Log log_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

void compute_puff_advantage_cuda(
    torch::Tensor values,
    torch::Tensor rewards,
    torch::Tensor dones,
    torch::Tensor importance,
    torch::Tensor advantages,
    double gamma,
    double lambda,  // Note: 'lambda' is fine as a param name in C++
    double rho_clip,
    double c_clip
);

/*
static torch::jit::Module g_policy;
void set_policy(torch::Tensor serialized_policy) {
    std::string model_str(reinterpret_cast<const char*>(serialized_policy.data_ptr<uint8_t>()), serialized_policy.numel());
    std::istringstream model_stream(model_str);
    g_policy = torch::jit::load(model_stream);
    g_policy.eval();
}
*/

class PolicyLSTM : public torch::nn::Module {
private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_atns_;
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Linear decoder{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::LSTMCell cell{nullptr};

public:
    // Constructor: input_size instead of grid_size
    PolicyLSTM(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128)
        : input_size_(input_size), hidden_size_(hidden_size), num_atns_(num_atns) {

        torch::globalContext().setDeterministicCuDNN(true);
        torch::globalContext().setBenchmarkCuDNN(false);
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size_, hidden_size_),
            torch::nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<torch::nn::LinearImpl>();
        torch::nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", torch::nn::Linear(hidden_size_, num_atns_));
        torch::nn::init::orthogonal_(decoder->weight, 0.01);
        torch::nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", torch::nn::Linear(hidden_size_, 1));
        torch::nn::init::orthogonal_(value->weight, 1.0);
        torch::nn::init::constant_(value->bias, 0.0);

        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size_, hidden_size_).num_layers(1)));
        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_ih_l0"], 1.0);
        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_hh_l0"], 1.0);
        lstm->named_parameters()["bias_ih_l0"].data().zero_();
        lstm->named_parameters()["bias_hh_l0"].data().zero_();

        // Share weights between LSTM and LSTMCell. Do not register or you'll double-update during optim.
        cell = torch::nn::LSTMCell(hidden_size_, hidden_size_);
        cell->named_parameters()["weight_ih"].data() = lstm->named_parameters()["weight_ih_l0"].data();
        cell->named_parameters()["weight_hh"].data() = lstm->named_parameters()["weight_hh_l0"].data();
        cell->named_parameters()["bias_ih"].data() = lstm->named_parameters()["bias_ih_l0"].data();
        cell->named_parameters()["bias_hh"].data() = lstm->named_parameters()["bias_hh_l0"].data();
        //cell->to(torch::kCUDA);
    }

    // Forward for evaluation/inference (uses LSTMCell)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor observations, torch::Tensor h, torch::Tensor c) {
        int64_t B = observations.size(0);

        // Ensure flat input: [B, input_size]
        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_size_,
                    "Observations must be [B, input_size]");

        if (h.defined() && h.numel() > 0) {
            TORCH_CHECK(h.dim() == 2 && h.size(0) == B && h.size(1) == hidden_size_,
                        "h must be [B, hidden_size]");
            TORCH_CHECK(c.dim() == 2 && c.size(0) == B && c.size(1) == hidden_size_,
                        "c must be [B, hidden_size]");
        }

        auto hidden = encoder->forward(observations);

        std::tuple<torch::Tensor, torch::Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        auto hidden_out = std::get<0>(cell_out);
        auto c_out = std::get<1>(cell_out);

        //std::std::cout << std::fixed << std::setprecision(10);
        //std::std::cout << "Hidden 0 cpp: " << hidden_out[0][0].item<float>() << std::std::endl;


        auto logits = decoder->forward(hidden_out);
        auto values = value->forward(hidden_out);

        return {logits, values, hidden_out, c_out};
    }

    // Forward for training (uses LSTM)
    std::tuple<torch::Tensor, torch::Tensor> forward_train(
        torch::Tensor observations, torch::Tensor lstm_h, torch::Tensor lstm_c) {
        auto x = observations;
        auto x_shape = x.sizes();

        // Expecting [B, TT, input_size] or [B, input_size]
        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input_size] or [B, TT, input_size]");
        TORCH_CHECK(x.size(-1) == input_size_,
                    "Last dimension of observations must match input_size");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        if (lstm_h.defined() && lstm_h.numel() > 0) {
            TORCH_CHECK(lstm_h.dim() == 3 && lstm_h.size(0) == 1 && lstm_h.size(1) == B,
                        "lstm_h must be [1, B, hidden_size]");
            TORCH_CHECK(lstm_c.dim() == 3 && lstm_c.size(0) == 1 && lstm_c.size(1) == B,
                        "lstm_c must be [1, B, hidden_size]");
        }

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_size_});
        } else {
            TT = 1;
        }

        auto hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_size_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden_size]

        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden_size]

        auto flat_hidden = hidden.reshape({-1, hidden_size_});
        auto logits = decoder->forward(flat_hidden);
        auto values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns_});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};

double cosine_annealing(double lr_base, int64_t t, int64_t T) {
    if (T == 0) return lr_base;  // avoid division by zero
    double ratio = static_cast<double>(t) / static_cast<double>(T);
    ratio = std::max(0.0, std::min(1.0, ratio));  // clamp to [0, 1]
    return lr_base * 0.5 * (1 + std::cos(M_PI * ratio));
}

void sync_fp16_fp32(pufferlib::PolicyLSTM* policy_16, pufferlib::PolicyLSTM* policy_32) {
    auto params_32 = policy_32->parameters();
    auto params_16 = policy_16->parameters();
    for (size_t i = 0; i < params_32.size(); ++i) {
        params_16[i].copy_(params_32[i].to(torch::kFloat32));
    }
}

typedef struct {
    PolicyLSTM* policy_16;
    PolicyLSTM* policy_32;
    //torch::optim::Adam* optimizer;
    torch::optim::SGD* optimizer;
    double lr;
    int64_t max_epochs;
} PuffeRL;

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(int64_t input_size,
        int64_t num_atns, int64_t hidden_size, double lr, double beta1, double beta2, double eps, int64_t max_epochs) {

    // Enable cuDNN benchmarking
    //torch::globalContext().setBenchmarkCuDNN(true);
    //torch::globalContext().setDeterministicCuDNN(false);
    //torch::globalContext().setBenchmarkLimitCuDNN(32);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    //torch::globalContext().setAllowTF32CuBLAS(true);
    //torch::globalContext().setAllowTF32CuDNN(true);

    // Enable faster FP16 reductions
    //torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    // BF16 reduction (if using bfloat16)
    //torch::globalContext().setAllowBF16ReductionCuBLAS(true);

    // Random seed
    torch::manual_seed(42);

    auto policy_16 = new PolicyLSTM(input_size, num_atns, hidden_size);
    //policy_16->to(torch::kCUDA);
    policy_16->to(torch::kFloat32);

    auto policy_32 = new PolicyLSTM(input_size, num_atns, hidden_size);
    //policy_32->to(torch::kCUDA);

    //auto optimizer = new torch::optim::Adam(policy_32->parameters(), torch::optim::AdamOptions(lr).betas({beta1, beta2}).eps(eps));
    auto optimizer = new torch::optim::SGD(policy_32->parameters(), torch::optim::SGDOptions(lr));

    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->policy_16 = policy_16;
    pufferl->policy_32 = policy_32;
    pufferl->optimizer = optimizer;
    pufferl->lr = lr;
    pufferl->max_epochs = max_epochs;
    return pufferl;
}

// Updated compiled_evaluate
std::tuple<torch::Tensor, torch::Tensor> compiled_evaluate(
    pybind11::object pufferl_obj,
    torch::Tensor envs_tensor,
    torch::Tensor indices_tensor,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor terminals,
    torch::Tensor lstm_h,
    torch::Tensor lstm_c,
    torch::Tensor obs_buffer,
    torch::Tensor act_buffer,
    torch::Tensor logprob_buffer,
    torch::Tensor rew_buffer,
    torch::Tensor term_buffer,
    torch::Tensor val_buffer,
    int64_t horizon,
    int64_t num_envs
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy_32;

    torch::NoGradGuard no_grad;

    for (int64_t i = 0; i < horizon; ++i) {
        auto [logits, value, lstm_h_out, lstm_c_out] = policy->forward(obs.to(torch::kFloat32).to(torch::kCPU), lstm_h, lstm_c);
        lstm_h = lstm_h_out;
        lstm_c = lstm_c_out;

        auto logprobs = torch::log_softmax(logits, 1);
        auto action = at::multinomial(logprobs.exp(), 1, true).squeeze(1);
        auto logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);

        // Store
        obs_buffer.select(1, i).copy_(obs.to(torch::kFloat32).to(torch::kCPU));
        act_buffer.select(1, i).copy_(action.to(torch::kInt32).to(torch::kCPU));
        logprob_buffer.select(1, i).copy_(logprob.to(torch::kFloat32).to(torch::kCPU));
        rew_buffer.select(1, i).copy_(rewards.to(torch::kFloat32).to(torch::kCPU));
        term_buffer.select(1, i).copy_(terminals.to(torch::kFloat32).to(torch::kCPU));
        val_buffer.select(1, i).copy_(value.flatten().to(torch::kFloat32).to(torch::kCPU));

        actions.copy_(action.to(torch::kCUDA));
        {
            pybind11::gil_scoped_release no_gil;
            step_environments_cuda(envs_tensor, indices_tensor);
        }
        rewards.clamp_(-1.0f, 1.0f);
    }

    return std::make_tuple(lstm_h, lstm_c);
}

std::tuple<torch::Tensor, torch::Tensor> evaluate_step(
    pybind11::object pufferl_obj,
    torch::Tensor envs_tensor,
    torch::Tensor indices_tensor,
    torch::Tensor obs,
    torch::Tensor actions,
    torch::Tensor rewards,
    torch::Tensor terminals,
    torch::Tensor lstm_h,
    torch::Tensor lstm_c,
    torch::Tensor obs_buffer,
    torch::Tensor act_buffer,
    torch::Tensor logprob_buffer,
    torch::Tensor rew_buffer,
    torch::Tensor term_buffer,
    torch::Tensor val_buffer,
    int64_t i
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy = pufferl.policy_32;

    torch::NoGradGuard no_grad;

    auto [logits, value, lstm_h_out, lstm_c_out] = policy->forward(obs.to(torch::kFloat32), lstm_h, lstm_c);
    lstm_h = lstm_h_out;
    lstm_c = lstm_c_out;

    auto logprobs = torch::log_softmax(logits, 1);
    auto action = at::multinomial(logprobs.exp(), 1, true).squeeze(1);
    auto logprob = logprobs.gather(1, action.unsqueeze(1)).squeeze(1);

    // Store
    obs_buffer.select(1, i).copy_(obs.to(torch::kFloat32));
    act_buffer.select(1, i).copy_(action.to(torch::kInt32));
    logprob_buffer.select(1, i).copy_(logprob.to(torch::kFloat32));
    rew_buffer.select(1, i).copy_(rewards.to(torch::kFloat32));
    term_buffer.select(1, i).copy_(terminals.to(torch::kFloat32));
    val_buffer.select(1, i).copy_(value.flatten().to(torch::kFloat32));

    actions.copy_(action);
    return std::make_tuple(lstm_h, lstm_c);
}

pybind11::dict compiled_train(
    pybind11::object pufferl_obj,
    torch::Tensor observations,  // [num_envs, horizon, ...]
    torch::Tensor actions,       // [num_envs, horizon]
    torch::Tensor logprobs,      // [num_envs, horizon]
    torch::Tensor rewards,       // [num_envs, horizon]
    torch::Tensor terminals_input, // [num_envs, horizon]
    torch::Tensor truncations,   // [num_envs, horizon] (not used in puff advantage?)
    torch::Tensor ratio,         // [num_envs, horizon]
    torch::Tensor values,        // [num_envs, horizon]
    int64_t total_minibatches,
    int64_t minibatch_segments,
    int64_t segments,
    int64_t accumulate_minibatches,
    int64_t horizon,
    double prio_beta0,
    double prio_alpha,
    double clip_coef,
    double vf_clip_coef,
    double gamma,
    double gae_lambda,
    double vtrace_rho_clip,
    double vtrace_c_clip,
    double vf_coef,
    double ent_coef,
    double max_grad_norm,
    bool use_rnn,
    bool anneal_lr,
    int64_t total_epochs,
    int64_t current_epoch
) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& policy_32 = pufferl.policy_32;
    auto& optimizer = pufferl.optimizer;

    auto device = values.device();
    auto terminals = terminals_input.to(torch::kFloat32);

    if (anneal_lr) {
        double lr = cosine_annealing(pufferl.lr, current_epoch, pufferl.max_epochs);
        optimizer->param_groups().at(0).options().set_lr(lr);
    }

    // Annealed priority exponent
    double anneal_beta = prio_beta0 + (1.0 - prio_beta0) * prio_alpha * static_cast<double>(current_epoch) / total_epochs;

    // Zero out ratio at start of epoch (matches Python: self.ratio[:] = 1)
    ratio.fill_(1.0);

    // Accumulators
    torch::Tensor pg_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor v_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor ent_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor total_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor old_approx_kl_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor approx_kl_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor clipfrac_sum = torch::zeros({}, torch::kFloat32).to(device);
    torch::Tensor importance_sum = torch::zeros({}, torch::kFloat32).to(device);

    auto advantages = torch::zeros_like(values);

    for (int64_t mb = 0; mb < total_minibatches; ++mb) {
    //for (int64_t mb = 0; mb < 1; ++mb) {
        advantages = torch::zeros_like(values);
        compute_puff_advantage_cpu(
            values, rewards, terminals, ratio,
            advantages, gamma, gae_lambda,
            vtrace_rho_clip, vtrace_c_clip
        );
        //std::cout << "Adv: " << advantages.mean().item<float>() << std::endl;

        // Prioritization
        auto adv = advantages.abs().sum(1);  // [num_envs]
        auto prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
        auto prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
        auto idx = at::multinomial(prio_probs, minibatch_segments, true);
        auto mb_prio = torch::pow(segments*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);

        //std::cout << "Prio: " << mb_prio.mean().item<float>() << std::endl;

        // Index into data
        torch::Tensor mb_obs = observations.index_select(0, idx);
        torch::Tensor mb_actions = actions.index_select(0, idx);
        torch::Tensor mb_logprobs = logprobs.index_select(0, idx);
        torch::Tensor mb_values = values.index_select(0, idx);
        torch::Tensor mb_advantages = advantages.index_select(0, idx);
        torch::Tensor mb_returns = mb_advantages + mb_values;

        /*
        std::cout << "mb_obs: " << mb_obs.sum() << std::endl;
        std::cout << "mb_actions: " << mb_actions.sum() << std::endl;
        std::cout << "mb_logprobs: " << mb_logprobs.min() << std::endl;
        std::cout << "mb_values: " << mb_values.min() << std::endl;
        std::cout << "mb_advantages: " << mb_advantages.min() << std::endl;
        std::cout << "mb_returns: " << mb_returns.min() << std::endl;
        */

        // Reshape obs if not using RNN
        if (!use_rnn) {
            auto flat_shape = std::vector<int64_t>{-1, mb_obs.size(2), mb_obs.size(3)};
            mb_obs = mb_obs.reshape(flat_shape);
        }

        // HARDCODED LSTM SIZE 128
        // Initial LSTM states (zero or none)
        torch::Tensor mb_lstm_h = torch::zeros(
            {1, minibatch_segments, 128},
            torch::kFloat32
        ).to(device);
        torch::Tensor mb_lstm_c = torch::zeros_like(mb_lstm_h);

        // Forward pass
        auto [logits, newvalue] = policy_32->forward_train(mb_obs.to(torch::kFloat32).to(torch::kCPU), mb_lstm_h, mb_lstm_c);

        //std::cout << "logits: " << logits.mean() << std::endl;

        // Flatten for action lookup
        auto flat_logits = logits.reshape({-1, logits.size(-1)});
        auto flat_actions = mb_actions.reshape({-1});
        auto logprobs_new = torch::log_softmax(flat_logits, /*dim=*/1);
        auto probs_new = logprobs_new.exp();

        // Gather logprobs for taken actions
        auto newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
        auto newlogprob = newlogprob_flat.reshape({minibatch_segments, horizon});
        auto entropy = - (probs_new * logprobs_new).sum(1);

        //std::cout << "newlogprob: " << newlogprob.min() << std::endl;
        //std::cout << "entropy: " << entropy.min() << std::endl;

        entropy = entropy.mean();

        // Compute ratio
        auto logratio = newlogprob - mb_logprobs;
        auto ratio_new = logratio.exp();

        //std::cout << "logratio_new: " << std::fixed << std::setprecision(20) << logratio.min().item<float>() << std::endl;
        //std::cout << "ratio_new: " << std::fixed << std::setprecision(20) << ratio_new.min().item<float>() << std::endl;

        // Update global ratio and values in-place (matches Python)
        ratio.index_copy_(0, idx, ratio_new.detach().squeeze(-1).to(torch::kFloat32));
        values.index_copy_(0, idx, newvalue.detach().squeeze(-1).to(torch::kFloat32));

        // Normalize advantages: (adv - mean) / std, then weight
        auto adv_normalized = mb_advantages;
        adv_normalized = mb_prio * (adv_normalized - adv_normalized.mean()) / (adv_normalized.std() + 1e-8);

        //std::cout << "adv_normalized: " << std::fixed << std::setprecision(20) << adv_normalized.min().item<float>() << std::endl;

        // Policy loss
        auto pg_loss1 = -adv_normalized * ratio_new;
        auto pg_loss2 = -adv_normalized * torch::clamp(ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
        auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        //std::cout << "pg_loss: " << pg_loss << std::endl;

        // Value loss
        newvalue = newvalue.view(mb_returns.sizes());
        auto v_clipped = mb_values + torch::clamp(newvalue - mb_values, -vf_clip_coef, vf_clip_coef);
        auto v_loss_unclipped = (newvalue - mb_returns).pow(2);
        auto v_loss_clipped = (v_clipped - mb_returns).pow(2);
        auto v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

        //std::cout << "v_loss: " << v_loss << std::endl;

        // Entropy
        auto entropy_loss = entropy;  // Already mean

        // Total loss
        auto loss = pg_loss + vf_coef*v_loss - ent_coef*entropy_loss;

        //std::cout << "loss: " << loss << std::endl;

        // Accumulate stats
        pg_sum += pg_loss.detach();
        v_sum += v_loss.detach();
        ent_sum += entropy_loss.detach();
        total_sum += loss.detach();

        // KL and clipping diagnostics (matches Python)
        {
            torch::NoGradGuard no_grad;
            auto old_kl = (-logratio).mean();
            auto kl = ((ratio_new - 1) - logratio).mean();
            auto cf = (ratio_new - 1.0).abs().gt(clip_coef).to(torch::kFloat32).mean();
            auto imp = ratio_new.mean();

            old_approx_kl_sum += old_kl.detach();
            approx_kl_sum += kl.detach();
            clipfrac_sum += cf.detach();
            importance_sum += imp.detach();
        }

        // Backward pass
        {
            pybind11::gil_scoped_release no_gil;
            loss.backward();
        }

        // Gradient accumulation and step
        if ((mb + 1) % accumulate_minibatches == 0) {
            torch::nn::utils::clip_grad_norm_(policy_32->parameters(), max_grad_norm);

            /*
            // Print grads
            for (auto& param : policy_32->parameters()) {
                std::cout << param.grad().abs().sum() << std::endl;
            }
            */

            // Print current lr
            //std::cout << "Current lr: " 
            //          << optimizer->param_groups()[0].options().get_lr() 
            //          << std::endl;
            optimizer->step();
            optimizer->zero_grad();
        }
    }

    // Compute explained variance at end of epoch
    auto y_true = advantages.flatten() + values.flatten();
    auto y_pred = values.flatten();
    auto var_y = y_true.var();
    //double explained_var = (var_y.abs() < 1e-8) ? NAN : (1 - (y_true - y_pred).var() / var_y).item<double>();

    // Return losses (averaged)
    pybind11::dict losses;
    auto num_mb = static_cast<double>(total_minibatches);
    losses["pg_loss"] = (pg_sum / num_mb).item<double>();
    losses["v_loss"] = (v_sum / num_mb).item<double>();
    losses["entropy"] = (ent_sum / num_mb).item<double>();
    losses["total_loss"] = (total_sum / num_mb).item<double>();
    losses["old_approx_kl"] = (old_approx_kl_sum / num_mb).item<double>();
    losses["approx_kl"] = (approx_kl_sum / num_mb).item<double>();
    losses["clipfrac"] = (clipfrac_sum / num_mb).item<double>();
    losses["importance"] = (importance_sum / num_mb).item<double>();
    //losses["explained_variance"] = explained_var;

    return losses;
}

// PYBIND11_MODULE with the extension name (pufferlib._C)
PYBIND11_MODULE(_C, m) {
    m.def("create_squared_environments", &create_squared_environments);
    m.def("step_environments", &step_environments_cuda);
    m.def("reset_environments", &reset_environments_cuda);
    m.def("log_environments", &log_environments_cuda);
    m.def("compiled_evaluate", &compiled_evaluate);
    m.def("evaluate_step", &evaluate_step);
    m.def("compiled_train", &compiled_train);

    py::class_<Log>(m, "Log")
    .def_readwrite("perf", &Log::perf)
    .def_readwrite("score", &Log::score)
    .def_readwrite("episode_return", &Log::episode_return)
    .def_readwrite("episode_length", &Log::episode_length)
    .def_readwrite("n", &Log::n);

    m.def("create_pufferl", &create_pufferl);
    py::class_<pufferlib::PuffeRL, std::unique_ptr<pufferlib::PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy_16", &pufferlib::PuffeRL::policy_16)
        .def_readwrite("policy_32", &pufferlib::PuffeRL::policy_32)
        .def_readwrite("optimizer", &pufferlib::PuffeRL::optimizer);

    py::class_<pufferlib::PolicyLSTM, std::shared_ptr<pufferlib::PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int64_t, int64_t, int64_t>());
    cls.def("forward", &pufferlib::PolicyLSTM::forward);
    cls.def("forward_train", &pufferlib::PolicyLSTM::forward_train);
}
}
