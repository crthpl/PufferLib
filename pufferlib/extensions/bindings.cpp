// bindings.cpp - Python/Torch bindings for pufferlib
// Separated from pufferlib.cpp to allow clean inclusion for profiling

// PUFFERLIB_TORCH is defined via -DPUFFERLIB_TORCH compile flag from setup.py

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
#ifdef PUFFERLIB_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#endif
#include "pufferlib.cpp"

using namespace pufferlib;
namespace py = pybind11;

#ifdef PUFFERLIB_TORCH
// Convert raw env dtype to torch ScalarType (moved from pufferlib.cpp)
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
#endif // PUFFERLIB_TORCH

// Wrapper functions for Python bindings
pybind11::dict log_environments(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    Dict* out = log_environments_impl(pufferl);
    pybind11::dict py_out;
    for (int i = 0; i < out->size; i++) {
        py_out[out->items[i].key] = out->items[i].value;
    }
    return py_out;
}

#ifdef PUFFERLIB_TORCH
Tensor initial_state(pybind11::object pufferl_obj, int64_t batch_size, torch::Device device) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto& rnn = pufferl.policy_bf16->rnn;
    return torch::zeros({rnn.num_layers, batch_size, rnn.hidden},
        torch::dtype(pufferlib::PRECISION_DTYPE).device(device));
}
#endif // PUFFERLIB_TORCH

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

#ifdef PUFFERLIB_TORCH
torch::autograd::tensor_list env_buffers(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    torch::ScalarType obs_dtype = (torch::ScalarType)to_torch_dtype(pufferl.env.obs_raw_dtype);
    return {
        pufferl.env.obs.to_torch(obs_dtype),
        pufferl.env.actions.to_torch(torch::kFloat64),
        pufferl.env.rewards.to_torch(torch::kFloat32),
        pufferl.env.terminals.to_torch(torch::kFloat32),
    };
}
#endif // PUFFERLIB_TORCH

void rollouts(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::gil_scoped_release no_gil;
    auto t0 = std::chrono::high_resolution_clock::now();
    static_vec_omp_step(pufferl.vec);
    float sec = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t0).count();
    pufferl.profile.accum[PROF_ROLLOUT] += sec * 1000.0f;  // store as ms

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(pufferl.vec, eval_prof);
    pufferl.profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    pufferl.profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];
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

pybind11::dict log_losses(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    // Copy losses from device to host via cudaMemcpy (no torch dependency)
    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, pufferl.losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    float n = losses_host[LOSS_N];
    pybind11::dict result;
    if (n > 0) {
        float inv_n = 1.0f / n;
        result["pg_loss"] = losses_host[LOSS_PG] * inv_n;
        result["vf_loss"] = losses_host[LOSS_VF] * inv_n;
        result["entropy"] = losses_host[LOSS_ENT] * inv_n;
        result["total_loss"] = losses_host[LOSS_TOTAL] * inv_n;
        result["old_approx_kl"] = losses_host[LOSS_OLD_APPROX_KL] * inv_n;
        result["approx_kl"] = losses_host[LOSS_APPROX_KL] * inv_n;
        result["clipfrac"] = losses_host[LOSS_CLIPFRAC] * inv_n;
    }
    // Zero the accumulator
    cudaMemset(pufferl.losses_puf.data, 0, pufferl.losses_puf.nbytes());
    return result;
}

pybind11::dict log_profile(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;
    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = pufferl.profile.accum[i] / 1000.0f;  // ms -> seconds
        result[PROF_NAMES[i]] = sec;
        if (i >= PROF_TRAIN_MISC) train_total += sec;
    }
    result["train"] = train_total;
    memset(pufferl.profile.accum, 0, sizeof(pufferl.profile.accum));
    return result;
}

pybind11::dict log_utilization(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(pufferl.nvml_device, &util);
    result["gpu_util"] = (float)util.gpu;

    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(pufferl.nvml_device, &mem);
    result["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    result["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
    result["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

    // CPU memory from /proc/self/status
    long rss_kb = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
        }
        fclose(f);
    }
    result["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);

    return result;
}

void puf_close(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    close_impl(pufferl);
#ifdef PUFFERLIB_TORCH
    // Force CUDA to release cached memory
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}

double get_config(py::dict& kwargs, const char* key) {
    if (!kwargs.contains(key)) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    try {
        return kwargs[key].cast<double>();
    } catch (const py::cast_error& e) {
        throw std::runtime_error(std::string("Failed to cast config key '") + key + "': " + e.what());
    }
}

Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(c_dict, key, item.second.cast<double>());
        } catch (const py::cast_error&) {
            // Skip non-numeric values
        }
    }
    return c_dict;
}

std::unique_ptr<pufferlib::PuffeRL> create_pufferl(pybind11::dict kwargs, pybind11::dict vec_kwargs, pybind11::dict env_kwargs, pybind11::dict policy_kwargs) {
    HypersT hypers;
    // Layout (total_agents and num_buffers come from vec config)
    hypers.total_agents = get_config(vec_kwargs, "total_agents");
    hypers.num_buffers = get_config(vec_kwargs, "num_buffers");
    hypers.num_threads = get_config(vec_kwargs, "num_threads");
    hypers.horizon = get_config(kwargs, "horizon");
    // Model architecture (num_atns computed from env in C++)
    hypers.hidden_size = get_config(policy_kwargs, "hidden_size");
    hypers.num_layers = get_config(policy_kwargs, "num_layers");
    // Learning rate
    hypers.lr = get_config(kwargs, "learning_rate");
    hypers.min_lr_ratio = get_config(kwargs, "min_lr_ratio");
    hypers.anneal_lr = get_config(kwargs, "anneal_lr");
    // Optimizer
    hypers.beta1 = get_config(kwargs, "beta1");
    hypers.beta2 = get_config(kwargs, "beta2");
    hypers.eps = get_config(kwargs, "eps");
    // Training
    hypers.minibatch_size = get_config(kwargs, "minibatch_size");
    hypers.replay_ratio = get_config(kwargs, "replay_ratio");
    hypers.total_timesteps = get_config(kwargs, "total_timesteps");
    hypers.max_grad_norm = get_config(kwargs, "max_grad_norm");
    // PPO
    hypers.clip_coef = get_config(kwargs, "clip_coef");
    hypers.vf_clip_coef = get_config(kwargs, "vf_clip_coef");
    hypers.vf_coef = get_config(kwargs, "vf_coef");
    hypers.ent_coef = get_config(kwargs, "ent_coef");
    // GAE
    hypers.gamma = get_config(kwargs, "gamma");
    hypers.gae_lambda = get_config(kwargs, "gae_lambda");
    // VTrace
    hypers.vtrace_rho_clip = get_config(kwargs, "vtrace_rho_clip");
    hypers.vtrace_c_clip = get_config(kwargs, "vtrace_c_clip");
    // Priority
    hypers.prio_alpha = get_config(kwargs, "prio_alpha");
    hypers.prio_beta0 = get_config(kwargs, "prio_beta0");
    // Flags
    hypers.use_rnn = get_config(kwargs, "use_rnn");
    hypers.cudagraphs = get_config(kwargs, "cudagraphs");
    hypers.kernels = get_config(kwargs, "kernels");
    hypers.profile = get_config(kwargs, "profile");
    // Multi-GPU
    hypers.rank = get_config(kwargs, "rank");
    hypers.world_size = get_config(kwargs, "world_size");
    hypers.nccl_id_path = kwargs["nccl_id_path"].cast<std::string>();

    std::string env_name = kwargs["env_name"].cast<std::string>();
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs.cast<py::dict>());
    Dict* env_dict = py_dict_to_c_dict(env_kwargs.cast<py::dict>());

    std::unique_ptr<pufferlib::PuffeRL> pufferl;
    {
        pybind11::gil_scoped_release no_gil;
        pufferl = create_pufferl_impl(hypers, env_name, vec_dict, env_dict);
    }

#ifdef PUFFERLIB_TORCH
    // Create Tensor views over allocator buffers for Python/torch interop
    auto f32_opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    auto bf16_opts = torch::dtype(torch::kBFloat16).device(torch::kCUDA);
    auto& a32 = pufferl->alloc_fp32;
    if (a32.param_mem) {
        a32.param_buffer = torch::from_blob(a32.param_mem, {a32.total_param_elems}, f32_opts);
    }
    if (a32.grad_mem) {
        a32.grad_buffer = torch::from_blob(a32.grad_mem, {a32.total_grad_elems}, f32_opts);
    }
    auto& abf = pufferl->alloc_bf16;
    if (abf.param_mem) {
        abf.param_buffer = torch::from_blob(abf.param_mem, {abf.total_param_elems}, bf16_opts);
    }
    if (abf.grad_mem) {
        abf.grad_buffer = torch::from_blob(abf.grad_mem, {abf.total_grad_elems}, bf16_opts);
    }
#endif // PUFFERLIB_TORCH

    return pufferl;
}

PYBIND11_MODULE(_C, m) {
    // Core functions (torch-free)
    m.def("log_environments", &log_environments);
    m.def("log_losses", &log_losses);
    m.def("log_profile", &log_profile);
    m.def("log_utilization", &log_utilization);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("close", &puf_close);
    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    m.def("profiler_start", &profiler_start);
    m.def("profiler_stop", &profiler_stop);

#ifdef PUFFERLIB_TORCH
    // Torch-dependent bindings
    m.def("initial_state", &initial_state);
    m.def("fc_max_cpp", &fc_max_cpp);
    m.def("env_buffers", &env_buffers);
    m.def("test_orthogonal_init", [](torch::Tensor t, float gain, int64_t seed) {
        PufTensor p = PufTensor::from_torch(t);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        puf_orthogonal_init(p, gain, (uint64_t)seed, stream);
    });

    py::class_<Muon>(m, "Muon")
        .def_property("lr",
            [](Muon& self) {
                return torch::from_blob(self.lr_ptr, {1},
                    torch::dtype(torch::kFloat32).device(torch::kCUDA));
            },
            [](Muon& self, torch::Tensor val) {
                float v = val.item<float>();
                cudaMemcpy(self.lr_ptr, &v, sizeof(float), cudaMemcpyHostToDevice);
            })
        .def_property_readonly("weight_buffer", [](Muon& self) {
            return self.wb_puf.to_torch(torch::kFloat32);
        })
        .def_property_readonly("momentum_buffer", [](Muon& self) -> py::object {
            if (!self.bufs_initialized) return py::none();
            return py::cast(self.mb_puf.to_torch(torch::kFloat32));
        })
        .def("state_dict", [](Muon& self) {
            std::unordered_map<std::string, torch::Tensor> state;
            state["lr"] = torch::from_blob(self.lr_ptr, {1},
                torch::dtype(torch::kFloat32).device(torch::kCUDA)).clone();
            if (self.wb_puf.data) state["weight_buffer"] = self.wb_puf.to_torch(torch::kFloat32);
            if (self.bufs_initialized) state["momentum_buffer"] = self.mb_puf.to_torch(torch::kFloat32);
            return state;
        })
        .def("load_state_dict", [](Muon& self, const std::unordered_map<std::string, torch::Tensor>& state) {
            auto it = state.find("lr");
            if (it != state.end()) {
                float v = it->second.item<float>();
                cudaMemcpy(self.lr_ptr, &v, sizeof(float), cudaMemcpyHostToDevice);
            }
            it = state.find("weight_buffer");
            if (it != state.end() && self.wb_puf.data) {
                self.wb_puf.to_torch(torch::kFloat32).copy_(it->second);
            }
            it = state.find("momentum_buffer");
            if (it != state.end() && self.bufs_initialized) {
                self.mb_puf.to_torch(torch::kFloat32).copy_(it->second);
            }
        });

    py::class_<PolicyLSTM, std::shared_ptr<PolicyLSTM>, torch::nn::Module> cls(m, "PolicyLSTM");
    cls.def(py::init<int, int, int>());
    cls.def("forward", &PolicyLSTM::forward);
    cls.def("forward_train", &PolicyLSTM::forward_train);

    py::class_<Logits>(m, "Logits")
        .def_readwrite("mean", &Logits::mean)
        .def_readwrite("logstd", &Logits::logstd);

    py::class_<Encoder, std::shared_ptr<Encoder>, torch::nn::Module>(m, "Encoder");
    py::class_<Decoder, std::shared_ptr<Decoder>, torch::nn::Module>(m, "Decoder");
    py::class_<DefaultEncoder, std::shared_ptr<DefaultEncoder>, Encoder>(m, "DefaultEncoder")
        .def(py::init<int, int>());
    py::class_<SnakeEncoder, std::shared_ptr<SnakeEncoder>, Encoder>(m, "SnakeEncoder")
        .def(py::init<int, int, int>());
    py::class_<G2048Encoder, std::shared_ptr<G2048Encoder>, Encoder>(m, "G2048Encoder")
        .def(py::init<int, int>());
    py::class_<DefaultDecoder, std::shared_ptr<DefaultDecoder>, Decoder>(m, "DefaultDecoder")
        .def(py::init<int, int, bool>(), py::arg("hidden"), py::arg("output"), py::arg("is_continuous") = false);
    py::class_<G2048Decoder, std::shared_ptr<G2048Decoder>, Decoder>(m, "G2048Decoder")
        .def(py::init<int, int>());
    py::class_<NMMO3Encoder, std::shared_ptr<NMMO3Encoder>, Encoder>(m, "NMMO3Encoder")
        .def(py::init<int, int>());
    py::class_<NMMO3Decoder, std::shared_ptr<NMMO3Decoder>, Decoder>(m, "NMMO3Decoder")
        .def(py::init<int, int>());
    py::class_<DriveEncoder, std::shared_ptr<DriveEncoder>, Encoder>(m, "DriveEncoder")
        .def(py::init<int, int>());

    py::class_<Allocator>(m, "Allocator")
        .def(py::init<>())
        .def_readwrite("param_buffer", &Allocator::param_buffer)
        .def_readwrite("grad_buffer", &Allocator::grad_buffer);

    py::class_<MinGRU>(m, "MinGRU")
        .def(py::init<Allocator&, int, int>(), py::arg("alloc"), py::arg("hidden"), py::arg("num_layers") = 1);

    py::class_<Policy>(m, "Policy")
        .def(py::init([](Allocator& alloc, int input, int hidden, int output,
                         int num_layers, int num_atns, bool continuous) {
            return new Policy(alloc, input, hidden, output, num_layers, num_atns, continuous);
        }))
        .def("forward", [](Policy& self, Tensor observations, Tensor state) {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            Tensor obs_cast = observations.to(pufferlib::PRECISION_DTYPE);
            PufTensor obs_puf = PufTensor::from_torch(obs_cast);
            PufTensor state_puf = PufTensor::from_torch(state);
            PufTensor dec_out = self.forward(obs_puf, state_puf, self.act, stream);
            return std::make_tuple(dec_out.to_torch(pufferlib::PRECISION_DTYPE), state);
        })
        .def("forward_train", [](Policy& self, Tensor x, Tensor state) {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            PufTensor x_puf = PufTensor::from_torch(x);
            PufTensor state_puf = PufTensor::from_torch(state);
            PufTensor dec_out = self.forward_train(x_puf, state_puf, self.act, stream);
            return dec_out.to_torch(pufferlib::PRECISION_DTYPE);
        })
        .def("init_weights", [](Policy& self, uint64_t seed) {
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            self.init_weights(stream, seed);
        }, py::arg("seed") = 42)
        .def("parameters", [](Policy& self) {
            // Build torch views from PufTensor weights for Python interop
            std::vector<torch::Tensor> params;
            params.push_back(self.encoder.weight.to_torch(PRECISION_DTYPE));
            params.push_back(self.decoder.weight.to_torch(PRECISION_DTYPE));
            if (self.decoder.continuous) params.push_back(self.decoder.logstd.to_torch(PRECISION_DTYPE));
            for (int i = 0; i < self.rnn.num_layers; i++)
                params.push_back(self.rnn.weights[i].to_torch(PRECISION_DTYPE));
            return params;
        })
        .def("named_parameters", [](Policy& self) {
            std::vector<torch::Tensor> params;
            params.push_back(self.encoder.weight.to_torch(PRECISION_DTYPE));
            params.push_back(self.decoder.weight.to_torch(PRECISION_DTYPE));
            if (self.decoder.continuous) params.push_back(self.decoder.logstd.to_torch(PRECISION_DTYPE));
            for (int i = 0; i < self.rnn.num_layers; i++)
                params.push_back(self.rnn.weights[i].to_torch(PRECISION_DTYPE));
            std::vector<std::string> names = {"encoder.linear.weight", "decoder.linear.weight"};
            if (self.decoder.continuous) names.push_back("decoder.logstd");
            for (int i = 0; i < self.rnn.num_layers; i++)
                names.push_back("rnn.layer_" + std::to_string(i) + ".weight");
            std::vector<std::pair<std::string, torch::Tensor>> result;
            for (size_t i = 0; i < params.size(); i++)
                result.push_back({names[i], params[i]});
            return result;
        });
#else
    // Torch-free minimal Policy binding (no forward/parameters returning Tensors)
    py::class_<Policy>(m, "Policy")
        .def("num_params", [](Policy& self) -> int64_t {
            int64_t total = self.encoder.weight.numel + self.decoder.weight.numel;
            if (self.decoder.continuous) total += self.decoder.logstd.numel;
            for (int i = 0; i < self.rnn.num_layers; i++)
                total += self.rnn.weights[i].numel;
            return total;
        });
    py::class_<Muon>(m, "Muon");
    py::class_<Allocator>(m, "Allocator")
        .def(py::init<>());
#endif // PUFFERLIB_TORCH

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("total_agents", &HypersT::total_agents)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)

        .def_readwrite("replay_ratio", &HypersT::replay_ratio)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("beta1", &HypersT::beta1)
        .def_readwrite("beta2", &HypersT::beta2)
        .def_readwrite("eps", &HypersT::eps)
        .def_readwrite("total_timesteps", &HypersT::total_timesteps)
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
        .def_readwrite("profile", &HypersT::profile)
        .def_readwrite("rank", &HypersT::rank)
        .def_readwrite("world_size", &HypersT::world_size)
        .def_readwrite("nccl_id_path", &HypersT::nccl_id_path);

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
        .def_readwrite("policy_bf16", &PuffeRL::policy_bf16)
        .def_readwrite("policy_fp32", &PuffeRL::policy_fp32)
        .def_readwrite("muon", &PuffeRL::muon)
        .def_readwrite("hypers", &PuffeRL::hypers)
        .def_readwrite("rollouts", &PuffeRL::rollouts);
}
