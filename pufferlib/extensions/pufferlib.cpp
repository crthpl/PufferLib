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

void step_environments_cuda(torch::Tensor envs_tensor, int64_t num_envs);

void reset_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor);

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

        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size_, hidden_size_),
            torch::nn::GELU()
        ));

        decoder = register_module("decoder", torch::nn::Linear(hidden_size_, num_atns_));

        value = register_module("value", torch::nn::Linear(hidden_size_, 1));

        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(hidden_size_, hidden_size_).num_layers(1)));

        cell = register_module("cell", torch::nn::LSTMCell(hidden_size_, hidden_size_));

        // Share weights between LSTM and LSTMCell
        cell->named_parameters()["weight_ih"] = lstm->named_parameters()["weight_ih_l0"];
        cell->named_parameters()["weight_hh"] = lstm->named_parameters()["weight_hh_l0"];
        cell->named_parameters()["bias_ih"] = lstm->named_parameters()["bias_ih_l0"];
        cell->named_parameters()["bias_hh"] = lstm->named_parameters()["bias_hh_l0"];

        // Initialization
        auto encoder_linear = (*encoder)[0]->as<torch::nn::LinearImpl>();
        torch::nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        encoder_linear->bias.data().zero_();

        torch::nn::init::orthogonal_(decoder->weight, 0.01);
        decoder->bias.data().zero_();

        torch::nn::init::orthogonal_(value->weight, 1.0);
        value->bias.data().zero_();

        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_ih_l0"], 1.0);
        torch::nn::init::orthogonal_(lstm->named_parameters()["weight_hh_l0"], 1.0);
        lstm->named_parameters()["bias_ih_l0"].data().zero_();
        lstm->named_parameters()["bias_hh_l0"].data().zero_();
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

        auto hidden = encoder->forward(observations.to(torch::kFloat32));

        std::tuple<torch::Tensor, torch::Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        auto hidden_out = std::get<0>(cell_out);
        auto c_out = std::get<1>(cell_out);

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

        auto hidden = encoder->forward(x.to(torch::kFloat32));

        hidden = hidden.reshape({B, TT, hidden_size_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden_size]

        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.to(torch::kFloat32);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden_size]

        auto flat_hidden = hidden.reshape({-1, hidden_size_});
        auto logits = decoder->forward(flat_hidden);
        auto values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns_});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};

typedef struct {
    PolicyLSTM* policy;
    torch::optim::Adam* optimizer;
} PuffeRL;

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(int64_t input_size,
        int64_t num_atns, int64_t hidden_size, double lr, double beta1, double beta2, double eps) {
    auto policy = new PolicyLSTM(input_size, num_atns, hidden_size);
    auto optimizer = new torch::optim::Adam(policy->parameters(), torch::optim::AdamOptions(lr).betas({beta1, beta2}).eps(eps));

    auto pufferl = std::make_unique<pufferlib::PuffeRL>();
    pufferl->policy = policy;
    pufferl->optimizer = optimizer;
    return pufferl;
}

// Updated compiled_evaluate
std::tuple<torch::Tensor, torch::Tensor> compiled_evaluate(
    pybind11::object pufferl_obj,
    torch::Tensor envs_tensor,
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
    auto& policy = pufferl.policy;
    auto& optimizer = pufferl.optimizer;

    // No-grad guard
    torch::NoGradGuard no_grad;

    for (int64_t i = 0; i < horizon; ++i) {
        // Clamp rewards
        auto r = rewards.clamp(-1.0f, 1.0f);

        // Policy forward: Native C++ call
        auto [logits, value, lstm_h_out, lstm_c_out] = policy->forward(obs, lstm_h, lstm_c);
        lstm_h = lstm_h_out;
        lstm_c = lstm_c_out;

        // Sample action and logprob (assuming discrete categorical from logits)
        auto max_logits = logits.amax(1, true);
        auto logits_shifted = logits - max_logits;
        auto logsumexp = logits_shifted.exp().sum(1, true).log() + max_logits;
        auto logprobs = logits - logsumexp;
        auto probs = logprobs.exp();
        auto action = at::multinomial(probs, 1, /*replacement=*/true);
        auto logprob = logprobs.gather(1, action).squeeze(1);
        action = action.squeeze(1);

        // Store to buffers
        obs_buffer.select(1, i).copy_(obs);
        act_buffer.select(1, i).copy_(action);
        logprob_buffer.select(1, i).copy_(logprob);
        rew_buffer.select(1, i).copy_(r);
        term_buffer.select(1, i).copy_(terminals.to(torch::kFloat32));
        val_buffer.select(1, i).copy_(value.flatten());

        // Step the environments
        actions.copy_(action);
        step_environments_cuda(envs_tensor, num_envs);
    }

    return std::make_tuple(lstm_h, lstm_c);
}

// Updated compiled_train
pybind11::dict compiled_train(
    pybind11::object pufferl_obj,
    torch::Tensor observations,  // [num_envs, horizon, grid_size, grid_size] uint8
    torch::Tensor actions,       // [num_envs, horizon] int32
    torch::Tensor logprobs,      // [num_envs, horizon] float
    torch::Tensor rewards,       // [num_envs, horizon] float
    torch::Tensor terminals,     // [num_envs, horizon] float
    torch::Tensor truncations,   // [num_envs, horizon] float (included but not used in loop)
    torch::Tensor ratio,         // [num_envs, horizon] float
    torch::Tensor values,        // [num_envs, horizon] float
    //pybind11::object scheduler,
    int64_t total_minibatches,
    int64_t minibatch_segments,
    int64_t segments,  // num_envs
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
    auto& policy = pufferl.policy;
    auto& optimizer = pufferl.optimizer;

    // Compute anneal_beta
    double anneal_beta = prio_beta0 + (1.0 - prio_beta0) * prio_alpha * static_cast<double>(current_epoch) / total_epochs;

    // Compute advantages
    auto advantages = torch::zeros_like(values);
    compute_puff_advantage_cuda(values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip);

    // Prioritize
    auto adv = advantages.abs().sum(1);
    auto prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    auto sum_weights = prio_weights.sum() + static_cast<double>(adv.size(0)) * 1e-6;
    auto prio_probs = (prio_weights + 1e-6) / sum_weights;

    auto device = values.device();
    auto pg_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto v_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto ent_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto total_sum = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (int64_t mb = 0; mb < total_minibatches; ++mb) {
        auto idx = at::multinomial(prio_probs, minibatch_segments, /*replacement=*/true);
        auto prio_probs_mb = prio_probs.index_select(0, idx).unsqueeze(1);
        auto mb_prio = torch::pow(static_cast<double>(segments) * prio_probs_mb, -anneal_beta);

        // Select minibatch tensors
        auto mb_obs = observations.index_select(0, idx);
        auto mb_actions = actions.index_select(0, idx);
        auto mb_logprobs = logprobs.index_select(0, idx);
        auto mb_rewards = rewards.index_select(0, idx);
        auto mb_terminals = terminals.index_select(0, idx);
        auto mb_ratio = ratio.index_select(0, idx);  // Not used directly
        auto mb_values = values.index_select(0, idx);
        auto mb_advantages = advantages.index_select(0, idx);
        auto mb_returns = mb_advantages + mb_values;

        auto original_obs_shape = mb_obs.sizes();  // [minibatch_segments, horizon, grid_size, grid_size]
        if (!use_rnn) {
            mb_obs = mb_obs.reshape({-1, original_obs_shape[2], original_obs_shape[3]});
        }

        // Initial LSTM states (undefined for zero init)
        torch::Tensor mb_lstm_h;
        torch::Tensor mb_lstm_c;

        // Policy forward: Native C++ call
        auto [logits, newvalue] = policy->forward_train(mb_obs, mb_lstm_h, mb_lstm_c);

        // Compute newlogprob and entropy (discrete assumption)
        auto flat_batch = minibatch_segments * horizon;
        auto flat_logits = logits.reshape({flat_batch, -1});
        auto flat_actions = mb_actions.reshape({flat_batch});
        auto max_logits = flat_logits.amax(1, true);
        auto logits_shifted = flat_logits - max_logits;
        auto logsumexp = (logits_shifted.exp().sum(1, true)).log() + max_logits;
        auto logprobs = flat_logits - logsumexp;  // Correct
        auto probs = logprobs.exp();
        auto newlogprob_flat = logprobs.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
        auto newlogprob = newlogprob_flat.reshape({minibatch_segments, horizon});
        auto entropy = - (probs * logprobs).sum(1).mean();

        auto logratio = newlogprob - mb_logprobs;
        auto mb_ratio_new = logratio.exp();

        // Update full ratio (detach)
        ratio.index_copy_(0, idx, mb_ratio_new.detach());

        // Advantages normalization
        auto adv = mb_advantages;
        adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8);

        // Policy loss
        auto pg_loss1 = -adv * mb_ratio_new;
        auto pg_loss2 = -adv * torch::clamp(mb_ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
        auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

        // Value loss
        newvalue = newvalue.view(mb_returns.sizes());
        auto v_clipped = mb_values + torch::clamp(newvalue - mb_values, -vf_clip_coef, vf_clip_coef);
        auto v_loss_unclipped = (newvalue - mb_returns).pow(2);
        auto v_loss_clipped = (v_clipped - mb_returns).pow(2);
        auto v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

        // Entropy loss
        auto entropy_loss = entropy;  // Already mean

        // Total loss
        auto loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss;
        pg_sum += pg_loss.detach();
        v_sum += v_loss.detach();
        ent_sum += entropy.detach();
        total_sum += loss.detach();

        {
            pybind11::gil_scoped_release no_gil;
            loss.backward();
        }

        // Update values
        values.index_copy_(0, idx, newvalue.detach().to(torch::kFloat32));

        // Accumulate and step
        if ((mb + 1) % accumulate_minibatches == 0) {
            torch::nn::utils::clip_grad_norm_(policy->parameters(), max_grad_norm);
            optimizer->step();
            optimizer->zero_grad();
        }
    }

    // Scheduler step if anneal_lr
    //if (anneal_lr) {
    //    scheduler.attr("step")();
    //}

    pybind11::dict losses;
    auto num_mb = static_cast<double>(total_minibatches);
    losses["pg_loss"] = (pg_sum / num_mb).item<double>();
    losses["v_loss"] = (v_sum / num_mb).item<double>();
    losses["entropy"] = (ent_sum / num_mb).item<double>();
    losses["total_loss"] = (total_sum / num_mb).item<double>();
    return losses;
}


// PYBIND11_MODULE with the extension name (pufferlib._C)
PYBIND11_MODULE(_C, m) {
    m.def("create_squared_environments", &create_squared_environments);
    m.def("step_environments", &step_environments_cuda);
    m.def("reset_environments", &reset_environments_cuda);
    m.def("compiled_evaluate", &compiled_evaluate);
    m.def("compiled_train", &compiled_train);

    m.def("create_pufferl", &create_pufferl);
    py::class_<pufferlib::PuffeRL, std::unique_ptr<pufferlib::PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &pufferlib::PuffeRL::policy)
        .def_readwrite("optimizer", &pufferlib::PuffeRL::optimizer);

    py::class_<pufferlib::PolicyLSTM, std::shared_ptr<pufferlib::PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int64_t, int64_t, int64_t>());
    cls.def("forward", &pufferlib::PolicyLSTM::forward);
    cls.def("forward_train", &pufferlib::PolicyLSTM::forward_train);
}
}
