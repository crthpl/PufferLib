// bindings.cpp - Python/Torch bindings for pufferlib
// Separated from pufferlib.cpp to allow clean inclusion for profiling

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pufferlib.cpp"

using namespace pufferlib;
namespace py = pybind11;

// Wrapper functions for Python bindings
pybind11::dict log_environments(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    Dict* out = log_environments_impl(pufferl);
    pybind11::dict py_out;
    for (int i = 0; i < out->size; i++) {
        py_out[out->items[i].key] = out->items[i].float_value;
    }
    return py_out;
}

Tensor initial_state(pybind11::object pufferl_obj, int64_t batch_size, torch::Device device) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    return initial_state_impl(pufferl, batch_size, device);
}

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    python_vec_recv_impl(pufferl, buf);
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    python_vec_send_impl(pufferl, buf);
}

torch::autograd::tensor_list env_buffers(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    return env_buffers_impl(pufferl);
}

void rollouts(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::gil_scoped_release no_gil;
    rollouts_impl(pufferl);
}

pybind11::dict train(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    {
        pybind11::gil_scoped_release no_gil;
        train_impl(pufferl);
    }
    pybind11::dict losses;
    return losses;
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(pybind11::dict kwargs) {
    HypersT hypers;
    // Layout
    hypers.segments = kwargs["segments"].cast<int>();
    hypers.horizon = kwargs["horizon"].cast<int>();
    hypers.num_envs = kwargs["num_envs"].cast<int>();
    hypers.num_buffers = kwargs["num_buffers"].cast<int>();
    hypers.minibatch_segments = kwargs["minibatch_segments"].cast<int>();
    hypers.total_minibatches = kwargs["total_minibatches"].cast<int>();
    hypers.accumulate_minibatches = kwargs["accumulate_minibatches"].cast<int>();
    // Model architecture
    hypers.input_size = kwargs["input_size"].cast<int>();
    hypers.num_atns = kwargs["num_atns"].cast<int>();
    hypers.hidden_size = kwargs["hidden_size"].cast<int>();
    hypers.expansion_factor = kwargs["expansion_factor"].cast<int>();
    hypers.num_layers = kwargs["num_layers"].cast<int>();
    // Learning rate
    hypers.lr = kwargs["lr"].cast<float>();
    hypers.min_lr_ratio = kwargs["min_lr_ratio"].cast<float>();
    hypers.anneal_lr = kwargs["anneal_lr"].cast<bool>();
    // Optimizer
    hypers.beta1 = kwargs["beta1"].cast<float>();
    hypers.beta2 = kwargs["beta2"].cast<float>();
    hypers.eps = kwargs["eps"].cast<float>();
    // Training
    hypers.max_epochs = kwargs["max_epochs"].cast<int>();
    hypers.max_grad_norm = kwargs["max_grad_norm"].cast<float>();
    // PPO
    hypers.clip_coef = kwargs["clip_coef"].cast<float>();
    hypers.vf_clip_coef = kwargs["vf_clip_coef"].cast<float>();
    hypers.vf_coef = kwargs["vf_coef"].cast<float>();
    hypers.ent_coef = kwargs["ent_coef"].cast<float>();
    // GAE
    hypers.gamma = kwargs["gamma"].cast<float>();
    hypers.gae_lambda = kwargs["gae_lambda"].cast<float>();
    // VTrace
    hypers.vtrace_rho_clip = kwargs["vtrace_rho_clip"].cast<float>();
    hypers.vtrace_c_clip = kwargs["vtrace_c_clip"].cast<float>();
    // Priority
    hypers.prio_alpha = kwargs["prio_alpha"].cast<float>();
    hypers.prio_beta0 = kwargs["prio_beta0"].cast<float>();
    // Flags
    hypers.use_rnn = kwargs["use_rnn"].cast<bool>();
    hypers.cudagraphs = kwargs["cudagraphs"].cast<bool>();
    hypers.kernels = kwargs["kernels"].cast<bool>();
    hypers.profile = kwargs["profile"].cast<bool>();

    pybind11::gil_scoped_release no_gil;
    return create_pufferl_impl(hypers);
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
}

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
}

TORCH_LIBRARY(_C, m) {
    m.def("mingru_gate(Tensor state, Tensor combined) -> (Tensor, Tensor)");
    m.def("log_coeffs_and_values(Tensor gate, Tensor hidden) -> (Tensor, Tensor)");
    m.def("fused_scan(Tensor combined, Tensor state) -> (Tensor, Tensor)");
    m.def("fused_ppo_loss(Tensor logits, Tensor values, Tensor actions, Tensor old_logprobs, Tensor advantages, Tensor prio, Tensor values, Tensor returns, Tensor adv_mean, Tensor adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) -> Tensor");
    m.def("policy_forward(Tensor obs, Tensor state) -> (Tensor, Tensor, Tensor)");
}

PYBIND11_MODULE(_C, m) {
    m.def("log_environments", &log_environments);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("logcumsumexp_cuda", &logcumsumexp_cuda);
    m.def("policy_forward", &PolicyMinGRU::forward);
    m.def("initial_state", &initial_state);
    m.def("mingru_gate", &mingru_gate);
    m.def("log_coeffs_and_values", &log_coeffs_and_values);
    m.def("fused_scan", &fused_scan);
    m.def("fused_ppo_loss", &fused_ppo_loss);
    m.def("sample_logits", &sample_logits);
    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    m.def("env_buffers", &env_buffers);
    m.def("profiler_start", &profiler_start);
    m.def("profiler_stop", &profiler_stop);

    py::class_<torch::optim::MuonOptions>(m, "MuonOptions")
        .def(py::init<double>());

    py::class_<torch::optim::MuonParamState>(m, "MuonParamState")
        .def(py::init<>());

    py::class_<torch::optim::Muon>(m, "Muon")
        .def(py::init<std::vector<torch::optim::OptimizerParamGroup>, torch::optim::MuonOptions>());

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("segments", &HypersT::segments)
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("num_envs", &HypersT::num_envs)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("minibatch_segments", &HypersT::minibatch_segments)
        .def_readwrite("total_minibatches", &HypersT::total_minibatches)
        .def_readwrite("accumulate_minibatches", &HypersT::accumulate_minibatches)
        .def_readwrite("input_size", &HypersT::input_size)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)
        .def_readwrite("expansion_factor", &HypersT::expansion_factor)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("beta1", &HypersT::beta1)
        .def_readwrite("beta2", &HypersT::beta2)
        .def_readwrite("eps", &HypersT::eps)
        .def_readwrite("max_epochs", &HypersT::max_epochs)
        .def_readwrite("max_grad_norm", &HypersT::max_grad_norm)
        .def_readwrite("clip_coef", &HypersT::clip_coef)
        .def_readwrite("vf_clip_coef", &HypersT::vf_clip_coef)
        .def_readwrite("vf_coef", &HypersT::vf_coef)
        .def_readwrite("ent_coef", &HypersT::ent_coef)
        .def_readwrite("gamma", &HypersT::gamma)
        .def_readwrite("gae_lambda", &HypersT::gae_lambda)
        .def_readwrite("vtrace_rho_clip", &HypersT::vtrace_rho_clip)
        .def_readwrite("vtrace_c_clip", &HypersT::vtrace_c_clip)
        .def_readwrite("prio_alpha", &HypersT::prio_alpha)
        .def_readwrite("prio_beta0", &HypersT::prio_beta0)
        .def_readwrite("use_rnn", &HypersT::use_rnn)
        .def_readwrite("cudagraphs", &HypersT::cudagraphs)
        .def_readwrite("kernels", &HypersT::kernels)
        .def_readwrite("profile", &HypersT::profile);

    py::class_<RolloutBuf>(m, "RolloutBuf")
        .def_readwrite("observations", &RolloutBuf::observations)
        .def_readwrite("actions", &RolloutBuf::actions)
        .def_readwrite("values", &RolloutBuf::values)
        .def_readwrite("logprobs", &RolloutBuf::logprobs)
        .def_readwrite("rewards", &RolloutBuf::rewards)
        .def_readwrite("terminals", &RolloutBuf::terminals)
        .def_readwrite("ratio", &RolloutBuf::ratio)
        .def_readwrite("importance", &RolloutBuf::importance);

    m.def("create_pufferl", &create_pufferl);
    py::class_<PuffeRL, std::unique_ptr<PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &PuffeRL::policy)
        .def_readwrite("muon", &PuffeRL::muon)
        .def_readwrite("hypers", &PuffeRL::hypers)
        .def_readwrite("rollouts", &PuffeRL::rollouts);

    py::class_<PolicyLSTM, std::shared_ptr<PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int64_t, int64_t, int64_t>());
    cls.def("forward", &PolicyLSTM::forward);
    cls.def("forward_train", &PolicyLSTM::forward_train);

    py::class_<PolicyMinGRU, std::shared_ptr<PolicyMinGRU>, torch::nn::Module> cls2(m, "PolicyMinGRU");
    cls2.def(py::init<int64_t, int64_t, int64_t>());
    cls2.def("forward", &PolicyMinGRU::forward);
    cls2.def("forward_train", &PolicyMinGRU::forward_train);
}
