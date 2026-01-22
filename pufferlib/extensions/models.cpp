// models.cpp - MinGRU, LSTM, and related model classes for pufferlib
// Separated from pufferlib.cpp for cleaner organization
// NOTE: This file is included directly into pufferlib.cpp inside namespace pufferlib

struct ShareableLSTMCell : public torch::nn::LSTMCellImpl {
    ShareableLSTMCell(const torch::nn::LSTMCellOptions& options) : torch::nn::LSTMCellImpl(options) {}

    void set_shared_weights(Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) {
        weight_ih = w_ih;
        weight_hh = w_hh;
        bias_ih = b_ih;
        bias_hh = b_hh;

        // Remove the original (unused) tensors from the parameter dict to avoid waste
        parameters_.erase("weight_ih");
        parameters_.erase("weight_hh");
        parameters_.erase("bias_ih");
        parameters_.erase("bias_hh");
    }
};

class RMSNorm : public torch::nn::Module {
private:
    int64_t dim;
    Tensor weight{nullptr};

public:
    RMSNorm(int64_t dim)
        : dim(dim) {

        weight = register_parameter("weight", torch::ones(dim));
    }

    Tensor forward(Tensor x) {
        int ndim = x.dim();
        TORCH_CHECK(x.size(ndim - 1) == dim, "Last dimension must match expected size");
        double eps = 1.19e-07;
        //return torch::nn::functional::normalize(
        //    x, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(-1)) * weight;
        Tensor rms = (x.pow(2).mean(-1, true) + eps).rsqrt();
        return x * rms * weight;
        //auto mean_sq = (x*x).mean(ndim - 1, true);
        //return weight * x/mean_sq.sqrt();
    }
};

class DyT : public torch::nn::Module {
    private:
        int64_t dim;
        Tensor alpha{nullptr};
        Tensor weight{nullptr};
        Tensor bias{nullptr};

    public:
        DyT(int64_t dim)
            : dim(dim) {

            alpha = register_parameter("alpha", 0.5*torch::ones({dim}));
            weight = register_parameter("weight", torch::ones({dim}));
            bias = register_parameter("bias", torch::zeros({dim}));
        }

        Tensor forward(Tensor x) {
            x = torch::tanh(alpha*x);
            x = x*weight + bias;
            return x;
        }
};


class MinGRULayer : public torch::nn::Module {
private:
    int64_t dim;
    torch::nn::Linear to_hidden_and_gate{nullptr};
    //Tensor to_hidden_and_gate_bf16{nullptr};
    torch::nn::Linear to_out{nullptr};
    //Tensor rmsnorm_weight{nullptr};
    //RMSNorm rmsnorm{nullptr};
    std::shared_ptr<RMSNorm> norm{nullptr};
    //std::shared_ptr<DyT> dyt{nullptr};
    bool kernels;

public:
    int64_t expansion_factor;
    MinGRULayer(int64_t dim, int64_t expansion_factor = 1., bool kernels = true)
        : dim(dim), expansion_factor(expansion_factor), kernels(kernels) {

        int dim_inner = int(dim * expansion_factor);
        to_hidden_and_gate = register_module("to_hidden_and_gate",
                torch::nn::Linear(torch::nn::LinearOptions(dim, 3*dim_inner).bias(false)));
        torch::nn::init::orthogonal_(to_hidden_and_gate->weight);

        //to_hidden_and_gate_bf16 = register_parameter("to_hidden_and_gate_bf16", torch::zeros({dim, 2*dim_inner}, torch::dtype(torch::kBFloat16).device(torch::kCUDA)));
        //torch::nn::init::orthogonal_(to_hidden_and_gate_bf16);

        // TODO: Is there a way to have this be identity to keep param count correct?
        //if (expansion_factor != 1.)
        //to_out = register_module("to_out",
        //        torch::nn::Linear(torch::nn::LinearOptions(dim*expansion_factor, dim).bias(false)));
        //torch::nn::init::orthogonal_(to_out->weight);

        norm = register_module("norm", std::make_shared<RMSNorm>(dim));
        //dyt = register_module("dyt", std::make_shared<DyT>(dim));

        //rmsnorm_weight = register_parameter("rmsnorm_weight", torch::ones({dim}));
    }

    std::tuple<Tensor, Tensor> forward(Tensor x, Tensor state = Tensor()) {
        TORCH_CHECK(x.dim() == 3, "x must be [B, seq, input_size]");
        TORCH_CHECK(state.dim() == 3, "state must be [B, seq, hidden_size]");
        TORCH_CHECK(x.size(0) == state.size(0), "x and state must have the same batch size");

        int seq_len = x.size(1);
        Tensor output = to_hidden_and_gate->forward(x);

        Tensor out;
        Tensor next_prev_hidden;

        if (seq_len == 1) {
            // Inference path: fused chunk + mingru + sigmoid(proj) * out
            if (kernels) {
                auto result = mingru_gate(state, output.contiguous());
                out = result[0];              // sigmoid(proj) * mingru_out
                next_prev_hidden = result[1]; // mingru_out (for recurrence)
            } else {
                std::vector<Tensor> chunks = output.chunk(3, 2);
                Tensor hidden = chunks[0];
                Tensor gate = chunks[1];
                Tensor proj = chunks[2];
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
                // fused_scan takes combined (B, T, 3*H) directly
                // output already has layout [hidden, gate, proj] from to_hidden_and_gate
                TORCH_CHECK(output.is_contiguous(), "output not contiguous before fused_scan");
                TORCH_CHECK(state.is_contiguous(), "state not contiguous before fused_scan");
                auto scan_out = fused_scan(output, state);
                out = scan_out[0];                // (B, T, H) = sigmoid(proj) * scan_result
                next_prev_hidden = scan_out[1];   // (B, 1, H) = raw scan_result at T
            } else {
                // Non-kernel path: chunk for gate/hidden/proj
                std::vector<Tensor> chunks = output.chunk(3, 2);
                Tensor hidden = chunks[0];
                Tensor gate = chunks[1];
                Tensor proj = chunks[2];

                // Compute log_coeffs/values manually
                Tensor log_coeffs = -torch::nn::functional::softplus(gate);
                Tensor log_z = -torch::nn::functional::softplus(-gate);
                Tensor log_tilde_h = torch::where(hidden >= 0,
                    (torch::nn::functional::relu(hidden) + 0.5).log(),
                    -torch::nn::functional::softplus(-hidden));
                Tensor log_values = log_z + log_tilde_h;

                // Non-kernel path still needs cat+pad+narrow
                log_values = torch::cat({state.log(), log_values}, 1);
                log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});
                Tensor a_star = log_coeffs.cumsum(1);
                Tensor log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
                Tensor log_h = a_star + log_h0_plus_b_star;
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
        //torch::nn::init::constant_(encoder->bias, 0.0);
    }

    Tensor forward(Tensor x) {
        return encoder->forward(x);
        //Tensor hidden = encoder->forward(x);
        //return torch::nn::functional::gelu(hidden);
    }
};

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
        //torch::nn::init::constant_(decoder->bias, 0.0);

        value_function = register_module("value_function", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, 1).bias(false)));
        torch::nn::init::orthogonal_(value_function->weight, 1.0);
        //torch::nn::init::constant_(value_function->bias, 0.0);
    }

    std::tuple<Tensor, Tensor> forward(Tensor hidden) {
        Tensor output = decoder->forward(hidden);
        Tensor logits = output.narrow(1, 0, output_size);
        Tensor value = output.narrow(1, output_size, 1);
        return {logits, value.squeeze(1)};
        //Tensor logits = decoder->forward(hidden);
        //Tensor value = value_function->forward(hidden);
        //return {logits, value};
    }
};


class PolicyMinGRU : public torch::nn::Module {
private:
    //torch::nn::Sequential encoder{nullptr};
    //torch::nn::Linear decoder{nullptr};
    std::shared_ptr<DefaultEncoder> encoder{nullptr};
    std::shared_ptr<DefaultDecoder> decoder{nullptr};
    //std::shared_ptr<MinGRULayer> mingru{nullptr};
    torch::nn::ModuleList mingru{nullptr};
    bool kernels;

public:
    torch::nn::Linear value{nullptr};
    int64_t input_size;
    int64_t hidden_size;
    int64_t num_atns;
    int64_t num_layers;
    float expansion_factor;

    PolicyMinGRU(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128, int64_t expansion_factor = 1, int64_t num_layers = 1, bool kernels = true)
        : input_size(input_size), hidden_size(hidden_size), expansion_factor(expansion_factor),
          num_atns(num_atns), num_layers(num_layers), kernels(kernels) {
        encoder = register_module("encoder", std::make_shared<DefaultEncoder>(input_size, hidden_size));
        decoder = register_module("decoder", std::make_shared<DefaultDecoder>(hidden_size, num_atns));
        /*
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Linear(input_size, hidden_size),
            torch::nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<torch::nn::LinearImpl>();
        torch::nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", torch::nn::Linear(hidden_size, num_atns));
        torch::nn::init::orthogonal_(decoder->weight, 0.01);
        torch::nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", torch::nn::Linear(hidden_size, 1));
        torch::nn::init::orthogonal_(value->weight, 1.0);
        torch::nn::init::constant_(value->bias, 0.0);
        */

        //mingru = register_module("mingru", std::make_shared<MinGRULayer>(hidden_size, 1));
        mingru = torch::nn::ModuleList();
        for (int64_t i = 0; i < num_layers; ++i) {
            mingru->push_back(MinGRULayer(hidden_size, expansion_factor, kernels));
        }
        register_module("mingru", mingru);
    }

    Tensor initial_state(int64_t batch_size, torch::Device device) {
        // Layout: {num_layers, batch_size, hidden} - select(0, i) gives contiguous slice
        return torch::zeros(
            {num_layers, batch_size, (int64_t)(hidden_size*expansion_factor)},
            torch::dtype(torch::kFloat32).device(device)
        );
    }

    std::tuple<Tensor, Tensor, Tensor> forward(
        Tensor observations, Tensor state) {
        int64_t B = observations.size(0);

        // Ensure flat input: [B, input_size]
        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_size,
            "Observations must be [B, input_size]");

        TORCH_CHECK(state.dim() == 3 && state.size(0) == num_layers && state.size(1) == B && state.size(2) == (int64_t)(hidden_size*expansion_factor),
            "state must be [num_layers, B, hidden_size]");

        Tensor hidden = encoder->forward(observations);

        hidden = hidden.unsqueeze(1);
        state = state.unsqueeze(2);

        std::tuple<Tensor, Tensor> mingru_out;

        for (int64_t i = 0; i < num_layers; ++i) {
            Tensor state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
            Tensor state_out = std::get<1>(mingru_out);
            state.select(0, i).copy_(state_out);
        }

        hidden = hidden.squeeze(1);
        state = state.squeeze(2);

        std::tuple<Tensor, Tensor> out = decoder->forward(hidden);
        Tensor logits = std::get<0>(out);
        Tensor values = std::get<1>(out);
        //auto logits = decoder->forward(hidden);
        //auto values = value->forward(hidden);

        return {logits, values, state};
    }

    std::tuple<Tensor, Tensor> forward_train(
        Tensor observations, Tensor state) {

        Tensor x = observations;
        auto x_shape = x.sizes();

        // Expecting [B, TT, input_size] or [B, input_size]
        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input_size] or [B, TT, input_size]");
        TORCH_CHECK(x.size(-1) == input_size,
                    "Last dimension of observations must match input_size");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        TORCH_CHECK(state.dim() == 4 && state.size(0) == num_layers && state.size(1) == B && state.size(2) == 1 && state.size(3) == (int64_t)(hidden_size*expansion_factor),
            "state must be [num_layers, B, 1, hidden_size*expansion_factor]");

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_size});
        } else {
            TT = 1;
        }

        Tensor hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_size});

        std::tuple<Tensor, Tensor> mingru_out;
        for (int64_t i = 0; i < num_layers; ++i) {
            Tensor state_in = state.select(0, i);
            auto layer = (*mingru)[i]->as<MinGRULayer>();
            mingru_out = layer->forward(hidden, state_in);
            hidden = std::get<0>(mingru_out);
        }

        Tensor flat_hidden = hidden.reshape({-1, hidden_size});

        std::tuple<Tensor, Tensor> out = decoder->forward(flat_hidden);
        Tensor logits = std::get<0>(out);
        Tensor values = std::get<1>(out);

        //auto logits = decoder->forward(flat_hidden);
        //auto values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};


class PolicyLSTM : public torch::nn::Module {
private:
    int64_t input_size_;
    int64_t hidden_size_;
    int64_t num_atns_;
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Linear decoder{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::LSTM lstm{nullptr};
    std::shared_ptr<ShareableLSTMCell> cell{nullptr};

public:
    // Constructor: input_size instead of grid_size
    PolicyLSTM(int64_t input_size, int64_t num_atns, int64_t hidden_size = 128)
        : input_size_(input_size), hidden_size_(hidden_size), num_atns_(num_atns) {
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

        cell = register_module("cell", std::make_shared<ShareableLSTMCell>(
            torch::nn::LSTMCellOptions(hidden_size_, hidden_size_)));
        cell->set_shared_weights(lstm->named_parameters()["weight_ih_l0"],
            lstm->named_parameters()["weight_hh_l0"],
            lstm->named_parameters()["bias_ih_l0"],
            lstm->named_parameters()["bias_hh_l0"]);
        /*
        // Share weights between LSTM and LSTMCell. Do not register or you'll double-update during optim.
        //cell = torch::nn::LSTMCell(hidden_size_, hidden_size_);
        cell = register_module("cell", torch::nn::LSTMCell(hidden_size_, hidden_size_));
        cell->named_parameters()["weight_ih"].data() = lstm->named_parameters()["weight_ih_l0"].data();
        cell->named_parameters()["weight_hh"].data() = lstm->named_parameters()["weight_hh_l0"].data();
        cell->named_parameters()["bias_ih"].data() = lstm->named_parameters()["bias_ih_l0"].data();
        cell->named_parameters()["bias_hh"].data() = lstm->named_parameters()["bias_hh_l0"].data();
        //cell->to(torch::kCUDA);
        */
    }

    // Forward for evaluation/inference (uses LSTMCell)
    std::tuple<Tensor, Tensor, Tensor, Tensor> forward(
        Tensor observations, Tensor h, Tensor c) {
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

        Tensor hidden = encoder->forward(observations);

        std::tuple<Tensor, Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        Tensor hidden_out = std::get<0>(cell_out);
        Tensor c_out = std::get<1>(cell_out);

        //std::std::cout << std::fixed << std::setprecision(10);
        //std::std::cout << "Hidden 0 cpp: " << hidden_out[0][0].item<float>() << std::std::endl;


        Tensor logits = decoder->forward(hidden_out);
        Tensor values = value->forward(hidden_out);

        return {logits, values, hidden_out, c_out};
    }

    // Forward for training (uses LSTM)
    std::tuple<Tensor, Tensor> forward_train(
        Tensor observations, Tensor lstm_h, Tensor lstm_c) {
        Tensor x = observations;
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

        Tensor hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_size_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden_size]

        std::tuple<Tensor, std::tuple<Tensor, Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden_size]

        Tensor flat_hidden = hidden.reshape({-1, hidden_size_});
        Tensor logits = decoder->forward(flat_hidden);
        Tensor values = value->forward(flat_hidden);

        logits = logits.reshape({B, TT, num_atns_});
        values = values.reshape({B, TT, 1});

        return {logits, values};
    }
};

void sync_fp16_fp32(PolicyLSTM* policy_16, PolicyLSTM* policy_32) {
    auto params_32 = policy_32->parameters();
    auto params_16 = policy_16->parameters();
    for (size_t i = 0; i < params_32.size(); ++i) {
        params_16[i].copy_(params_32[i].to(torch::kFloat32));
    }
}
