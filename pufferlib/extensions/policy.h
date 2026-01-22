// policy.h - Neural network component classes for PufferLib
// Shared between pufferlib.cpp and profile_kernels.cu
#pragma once

#include <torch/torch.h>

// Forward declarations for kernel functions (defined in modules.cu)
std::vector<torch::Tensor> mingru_gate(
    torch::Tensor state,
    torch::Tensor combined
);

torch::autograd::tensor_list fused_scan(
    torch::Tensor combined,
    torch::Tensor state
);

// ============================================================================
// RMSNorm
// ============================================================================
class RMSNorm : public torch::nn::Module {
private:
    int64_t dim;
    torch::Tensor weight{nullptr};

public:
    RMSNorm(int64_t dim)
        : dim(dim) {
        weight = register_parameter("weight", torch::ones(dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        int ndim = x.dim();
        TORCH_CHECK(x.size(ndim - 1) == dim, "Last dimension must match expected size");
        double eps = 1.19e-07;
        auto rms = (x.pow(2).mean(-1, true) + eps).rsqrt();
        return x * rms * weight;
    }
};

// ============================================================================
// MinGRULayer
// ============================================================================
class MinGRULayer : public torch::nn::Module {
private:
    int64_t dim;
    torch::nn::Linear to_hidden_and_gate{nullptr};
    torch::nn::Linear to_out{nullptr};
    std::shared_ptr<RMSNorm> norm{nullptr};
    bool kernels;

public:
    int64_t expansion_factor;

    MinGRULayer(int64_t dim, int64_t expansion_factor = 1., bool kernels = true)
        : dim(dim), expansion_factor(expansion_factor), kernels(kernels) {

        int dim_inner = int(dim * expansion_factor);
        to_hidden_and_gate = register_module("to_hidden_and_gate",
                torch::nn::Linear(torch::nn::LinearOptions(dim, 3*dim_inner).bias(false)));
        torch::nn::init::orthogonal_(to_hidden_and_gate->weight);

        norm = register_module("norm", std::make_shared<RMSNorm>(dim));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor state = torch::Tensor()) {
        TORCH_CHECK(x.dim() == 3, "x must be [B, seq, input_size]");
        TORCH_CHECK(state.dim() == 3, "state must be [B, seq, hidden_size]");
        TORCH_CHECK(x.size(0) == state.size(0), "x and state must have the same batch size");

        auto seq_len = x.size(1);
        auto output = to_hidden_and_gate->forward(x);

        torch::Tensor out;
        torch::Tensor next_prev_hidden;

        if (seq_len == 1) {
            // Inference path: fused chunk + mingru + sigmoid(proj) * out
            if (kernels) {
                auto result = mingru_gate(state, output.contiguous());
                out = result[0];
                next_prev_hidden = result[1];
            } else {
                auto chunks = output.chunk(3, 2);
                auto hidden = chunks[0];
                auto gate = chunks[1];
                auto proj = chunks[2];
                hidden = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
                gate = gate.sigmoid();
                out = torch::lerp(state, hidden, gate);
                next_prev_hidden = out;
                proj = torch::sigmoid(proj);
                out = proj * out;
            }
        } else {
            // Training path: fully fused kernel
            if (kernels) {
                TORCH_CHECK(output.is_contiguous(), "output not contiguous before fused_scan");
                TORCH_CHECK(state.is_contiguous(), "state not contiguous before fused_scan");
                auto scan_out = fused_scan(output, state);
                out = scan_out[0];
                next_prev_hidden = scan_out[1];
            } else {
                // Non-kernel path: chunk for gate/hidden/proj
                auto chunks = output.chunk(3, 2);
                auto hidden = chunks[0];
                auto gate = chunks[1];
                auto proj = chunks[2];

                auto log_coeffs = -torch::nn::functional::softplus(gate);
                auto log_z = -torch::nn::functional::softplus(-gate);
                auto log_tilde_h = torch::where(hidden >= 0,
                    (torch::nn::functional::relu(hidden) + 0.5).log(),
                    -torch::nn::functional::softplus(-hidden));
                auto log_values = log_z + log_tilde_h;

                log_values = torch::cat({state.log(), log_values}, 1);
                log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});
                auto a_star = log_coeffs.cumsum(1);
                auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
                auto log_h = a_star + log_h0_plus_b_star;
                out = log_h.exp();
                out = out.narrow(1, out.size(1) - seq_len, seq_len);
                next_prev_hidden = out.narrow(1, out.size(1) - 1, 1);

                proj = torch::sigmoid(proj);
                out = proj * out;
            }
        }

        return std::make_tuple(out, next_prev_hidden);
    }
};

// ============================================================================
// DefaultEncoder
// ============================================================================
class DefaultEncoder : public torch::nn::Module {
public:
    torch::nn::Linear encoder{nullptr};
    int input_size;
    int hidden_size;

    DefaultEncoder(int64_t input_size, int64_t hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        encoder = register_module("encoder", torch::nn::Linear(
            torch::nn::LinearOptions(input_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(encoder->weight, std::sqrt(2.0));
    }

    torch::Tensor forward(torch::Tensor x) {
        return encoder->forward(x);
    }
};

// ============================================================================
// DefaultDecoder
// ============================================================================
class DefaultDecoder : public torch::nn::Module {
public:
    torch::nn::Linear decoder{nullptr};
    torch::nn::Linear value_function{nullptr};
    int hidden_size;
    int output_size;

    DefaultDecoder(int64_t hidden_size, int64_t output_size)
        : hidden_size(hidden_size), output_size(output_size) {

        decoder = register_module("decoder", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, output_size+1).bias(false)));
        torch::nn::init::orthogonal_(decoder->weight, 0.01);

        value_function = register_module("value_function", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, 1).bias(false)));
        torch::nn::init::orthogonal_(value_function->weight, 1.0);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor hidden) {
        torch::Tensor output = decoder->forward(hidden);
        torch::Tensor logits = output.narrow(1, 0, output_size);
        torch::Tensor value = output.narrow(1, output_size, 1);
        return {logits, value.squeeze(1)};
    }
};

// ============================================================================
// PolicyMinGRU
// ============================================================================
class PolicyMinGRU : public torch::nn::Module {
private:
    std::shared_ptr<DefaultEncoder> encoder{nullptr};
    std::shared_ptr<DefaultDecoder> decoder{nullptr};
    torch::nn::ModuleList mingru{nullptr};
    bool kernels;

public:
    torch::nn::Linear value{nullptr};
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_atns;
    int64_t num_layers;
    float expansion_factor;

    PolicyMinGRU(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128,
                 int64_t expansion_factor = 1, int64_t num_layers = 1, bool kernels = true)
        : input_size(input_size), hidden_size(hidden_size), expansion_factor(expansion_factor),
          num_atns(num_atns), num_layers(num_layers), kernels(kernels) {
        encoder = register_module("encoder", std::make_shared<DefaultEncoder>(input_size, hidden_size));
        decoder = register_module("decoder", std::make_shared<DefaultDecoder>(hidden_size, num_atns));

        mingru = torch::nn::ModuleList();
        for (int64_t i = 0; i < num_layers; ++i) {
            mingru->push_back(MinGRULayer(hidden_size, expansion_factor, kernels));
        }
        register_module("mingru", mingru);
    }

    torch::Tensor initial_state(int64_t batch_size, torch::Device device) {
        return torch::zeros(
            {num_layers, batch_size, (int)(hidden_size*expansion_factor)},
            torch::dtype(torch::kFloat32).device(device)
        );
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor observations, torch::Tensor state) {
        int64_t B = observations.size(0);

        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_size,
            "Observations must be [B, input_size]");
        TORCH_CHECK(state.dim() == 3 && state.size(0) == num_layers && state.size(1) == B &&
                    state.size(2) == hidden_size*expansion_factor,
            "state must be [num_layers, B, hidden_size]");

        auto hidden = encoder->forward(observations);

        hidden = hidden.unsqueeze(1);
        state = state.unsqueeze(2);

        std::tuple<torch::Tensor, torch::Tensor> mingru_out;

        for (int64_t i = 0; i < num_layers; ++i) {
            auto state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
            auto state_out = std::get<1>(mingru_out);
            state.select(0, i).copy_(state_out);
        }

        hidden = hidden.squeeze(1);
        state = state.squeeze(2);

        std::tuple<torch::Tensor, torch::Tensor> out = decoder->forward(hidden);
        auto logits = std::get<0>(out);
        auto values = std::get<1>(out);

        return {logits, values, state};
    }

    std::tuple<torch::Tensor, torch::Tensor> forward_train(
        torch::Tensor observations, torch::Tensor state) {

        auto x = observations;
        auto x_shape = x.sizes();

        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input_size] or [B, TT, input_size]");
        TORCH_CHECK(x.size(-1) == input_size,
                    "Last dimension of observations must match input_size");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        TORCH_CHECK(state.dim() == 4 && state.size(0) == num_layers && state.size(1) == B &&
                    state.size(2) == 1 && state.size(3) == hidden_size*expansion_factor,
            "state must be [num_layers, B, 1, hidden_size*expansion_factor]");

        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_size});
        } else {
            TT = 1;
        }

        auto hidden = encoder->forward(x);
        hidden = hidden.reshape({B, TT, hidden_size});

        std::tuple<torch::Tensor, torch::Tensor> mingru_out;
        for (int64_t i = 0; i < num_layers; ++i) {
            auto state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
        }

        auto flat_hidden = hidden.reshape({-1, hidden_size});

        std::tuple<torch::Tensor, torch::Tensor> out = decoder->forward(flat_hidden);
        auto logits = std::get<0>(out);
        auto values = std::get<1>(out);

        logits = logits.reshape({B, TT, num_atns});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};
