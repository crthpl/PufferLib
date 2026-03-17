// legacy_bindings.cpp - Torch reference implementations for numerical testing
// Builds the pufferlib._torch_ref Python module

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>

// advantage.cpp is normally #included inside namespace pufferlib in pufferlib.cpp.
// We do the same here so puff_advantage_cpu lives in the right namespace.
namespace pufferlib {
typedef torch::Tensor Tensor;
#include "advantage.cpp"
}

// legacy_modules.h has all *_cpp fns inside namespace pufferlib
// (guarded by PUFFERLIB_TORCH and !__CUDACC__)
#include "legacy_modules.h"


namespace py = pybind11;

PYBIND11_MODULE(_torch_ref, m) {
    m.doc() = "Torch reference implementations for numerical testing";

    // MinGRU
    m.def("mingru_gate", &mingru_gate_cpp);
    m.def("fused_scan", &fused_scan_cpp);
    m.def("logcumsumexp", &logcumsumexp_cpp);
    m.def("log_coeffs_and_values", &log_coeffs_and_values_cpp);

    // Sampling
    m.def("sample_discrete", &sample_discrete_cpp);
    m.def("sample_continuous", &sample_continuous_cpp);

    // Logprob + entropy
    m.def("discrete_logprob_entropy", &discrete_logprob_entropy_cpp);
    m.def("continuous_logprob_entropy", &continuous_logprob_entropy_cpp);

    // PPO loss
    m.def("ppo_loss", &ppo_loss_cpp);
    m.def("fused_ppo_loss", &fused_ppo_loss_cpp);

    // Training ops
    m.def("train_select_and_copy", &train_select_and_copy_cpp);
    m.def("prio_replay", &prio_replay_cpp);

    // FC max (for DriveEncoder testing)
    m.def("fc_max", &fc_max_cpp);
    m.def("fc_relu_fc_max", &fc_relu_fc_max_cpp);

    // Advantage (CPU reference)
    m.def("puff_advantage_cpu", &puff_advantage_cpu);

    // Encoder/Decoder classes (for policy testing)
    py::class_<DefaultEncoder, std::shared_ptr<DefaultEncoder>, torch::nn::Module>(m, "DefaultEncoder")
        .def(py::init<int, int>());
    py::class_<DefaultDecoder, std::shared_ptr<DefaultDecoder>, torch::nn::Module>(m, "DefaultDecoder")
        .def(py::init<int, int, bool>(), py::arg("hidden"), py::arg("output"), py::arg("is_continuous") = false);
    py::class_<PolicyLSTM, std::shared_ptr<PolicyLSTM>, torch::nn::Module>(m, "PolicyLSTM")
        .def(py::init<int, int, int>());
}
