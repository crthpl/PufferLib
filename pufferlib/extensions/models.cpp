// models.cpp - MinGRU, LSTM, and related model classes for pufferlib
// Separated from pufferlib.cpp for cleaner organization
// NOTE: This file is included directly into pufferlib.cpp inside namespace pufferlib

// Minimal interfaces for swappable components
// Inherit from torch::nn::Module so register_module works
struct Encoder : public torch::nn::Module {
    virtual Tensor forward(Tensor x) = 0;
};

struct Decoder : public torch::nn::Module {
    virtual std::tuple<Tensor, Tensor> forward(Tensor hidden) = 0;
};

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
public:
    int64_t dim;
    int64_t expansion_factor;
    bool kernels;
    torch::nn::Linear to_hidden_and_gate{nullptr};
    //Tensor to_hidden_and_gate_bf16{nullptr};
    torch::nn::Linear to_out{nullptr};
    //Tensor rmsnorm_weight{nullptr};
    //RMSNorm rmsnorm{nullptr};
    std::shared_ptr<RMSNorm> norm{nullptr};
    //std::shared_ptr<DyT> dyt{nullptr};
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
            // Training path: fully fused kernel with checkpointing for reduced memory
            if (kernels) {
                // fused_scan_checkpointed takes combined (B, T, 3*H) directly
                // output already has layout [hidden, gate, proj] from to_hidden_and_gate
                TORCH_CHECK(output.is_contiguous(), "output not contiguous before fused_scan_checkpointed");
                TORCH_CHECK(state.is_contiguous(), "state not contiguous before fused_scan_checkpointed");
                auto scan_out = fused_scan_checkpointed(output, state);
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


class DefaultEncoder : public Encoder {
    public:
        torch::nn::Linear linear{nullptr};
        int input_size;
        int hidden_size;

    DefaultEncoder(int64_t input_size, int64_t hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(input_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        return linear->forward(x);
    }
};

// Snake encoder: one-hot encode observations then linear
class SnakeEncoder : public Encoder {
    public:
        torch::nn::Linear linear{nullptr};
        int input_size;
        int hidden_size;
        int num_classes;

    SnakeEncoder(int64_t input_size, int64_t hidden_size, int64_t num_classes = 8)
        : input_size(input_size), hidden_size(hidden_size), num_classes(num_classes) {
        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(input_size * num_classes, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is [B, input_size] with values 0-7
        int64_t B = x.size(0);
        // One-hot encode: [B, input_size] -> [B, input_size, num_classes]
        Tensor onehot = torch::one_hot(x.to(torch::kLong), num_classes).to(torch::kFloat32);
        // Flatten: [B, input_size * num_classes]
        onehot = onehot.view({B, -1});
        return linear->forward(onehot);
    }
};

class DefaultDecoder : public Decoder {
    public:
        torch::nn::Linear linear{nullptr};
        int hidden_size;
        int output_size;

    DefaultDecoder(int64_t hidden_size, int64_t output_size)
        : hidden_size(hidden_size), output_size(output_size) {

        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, output_size+1).bias(false)));
        torch::nn::init::orthogonal_(linear->weight, 0.01);
    }

    std::tuple<Tensor, Tensor> forward(Tensor hidden) override {
        Tensor output = linear->forward(hidden);
        Tensor logits = output.narrow(1, 0, output_size);
        Tensor value = output.narrow(1, output_size, 1);
        return {logits, value.squeeze(1)};
    }
};

// G2048 encoder: embeddings + 3 linear layers with GELU
// Matches Python: value_embed(obs) + pos_embed -> flatten -> encoder MLP
class G2048Encoder : public Encoder {
    public:
        torch::nn::Embedding value_embed{nullptr};
        torch::nn::Embedding pos_embed{nullptr};
        torch::nn::Linear linear1{nullptr};
        torch::nn::Linear linear2{nullptr};
        torch::nn::Linear linear3{nullptr};
        int input_size;
        int hidden_size;
        static constexpr int embed_dim = 3;  // ceil(33^0.25) = 3
        static constexpr int num_grid_cells = 16;
        static constexpr int num_obs = num_grid_cells * embed_dim;  // 48

    G2048Encoder(int64_t input_size, int64_t hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        // Embeddings for tile values and positions
        value_embed = register_module("value_embed", torch::nn::Embedding(18, embed_dim));
        pos_embed = register_module("pos_embed", torch::nn::Embedding(num_grid_cells, embed_dim));

        // Encoder MLP: num_obs -> 2*hidden -> hidden -> hidden
        linear1 = register_module("linear1", torch::nn::Linear(
            torch::nn::LinearOptions(num_obs, 2*hidden_size).bias(false)));
        torch::nn::init::orthogonal_(linear1->weight, std::sqrt(2.0));

        linear2 = register_module("linear2", torch::nn::Linear(
            torch::nn::LinearOptions(2*hidden_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(linear2->weight, std::sqrt(2.0));

        linear3 = register_module("linear3", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(linear3->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        // x is (B, 16) uint8 tile values
        auto B = x.size(0);

        // value_embed(obs) -> (B, 16, embed_dim)
        auto value_obs = value_embed->forward(x.to(torch::kLong));

        // pos_embed.weight expanded to (B, 16, embed_dim)
        auto pos_obs = pos_embed->weight.unsqueeze(0).expand({B, num_grid_cells, embed_dim});

        // grid_obs = (value_obs + pos_obs).flatten(1) -> (B, 48)
        auto grid_obs = (value_obs + pos_obs).flatten(1);

        // Encoder MLP
        auto h = torch::gelu(linear1->forward(grid_obs));
        h = torch::gelu(linear2->forward(h));
        h = torch::gelu(linear3->forward(h));
        return h;
    }
};

// NMMO3 encoder: Conv2d map processing + embedding for player discrete + projection
class NMMO3Encoder : public Encoder {
    public:
        // Multi-hot encoding factors and offsets
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::Embedding player_embed{nullptr};
        torch::nn::Linear proj{nullptr};
        Tensor offsets{nullptr};
        int input_size;
        int hidden_size;

    NMMO3Encoder(int64_t input_size, int64_t hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4], sum = 59
        // Map processing: Conv2d(59, 128, 5, stride=3) -> ReLU -> Conv2d(128, 128, 3, stride=1) -> Flatten
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(59, 128, 5).stride(3).bias(true)));
        torch::nn::init::orthogonal_(conv1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv1->bias, 0.0);

        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 128, 3).stride(1).bias(true)));
        torch::nn::init::orthogonal_(conv2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(conv2->bias, 0.0);

        // Player discrete encoder: Embedding(128, 32) -> Flatten
        // Input is 47 discrete values, output is 47*32 = 1504
        player_embed = register_module("player_embed", torch::nn::Embedding(128, 32));

        // Projection: Linear(1817, hidden_size) -> ReLU
        // 1817 = conv_out (128*1*3=384) + player_embed (47*32=1504) - wait that's 1888
        // Actually from Python: 1817 = conv_out + player_discrete + player_continuous + reward
        // Let me recalculate: map_2d output size after conv layers
        // Input: (B, 59, 11, 15)
        // After conv1(5, stride=3): (11-5)/3+1=3, (15-5)/3+1=4 -> (B, 128, 3, 4)
        // After conv2(3, stride=1): (3-3)/1+1=1, (4-3)/1+1=2 -> (B, 128, 1, 2)
        // Flatten: 128*1*2 = 256
        // player_discrete: 47*32 = 1504
        // player continuous (same 47 values): 47
        // reward: 10
        // Total: 256 + 1504 + 47 + 10 = 1817
        proj = register_module("proj", torch::nn::Linear(
            torch::nn::LinearOptions(1817, hidden_size).bias(true)));
        torch::nn::init::orthogonal_(proj->weight, std::sqrt(2.0));
        torch::nn::init::constant_(proj->bias, 0.0);

        // Register offsets buffer for multi-hot encoding
        // factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
        // cumsum = [4, 8, 25, 30, 33, 38, 43, 48, 55, 59]
        // offsets = [0, 4, 8, 25, 30, 33, 38, 43, 48, 55]
        std::vector<int64_t> offset_vals = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};
        offsets = register_buffer("offsets",
            torch::tensor(offset_vals, torch::kInt64).view({1, 10, 1, 1}));
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        auto device = x.device();
        auto dtype = x.dtype();

        // Split observations: map (1650), player (47), reward (10)
        Tensor ob_map = x.narrow(1, 0, 11*15*10).view({B, 11, 15, 10});
        Tensor ob_player = x.narrow(1, 11*15*10, 47);
        Tensor ob_reward = x.narrow(1, 11*15*10 + 47, 10);

        // Multi-hot encoding for map
        // ob_map: (B, 11, 15, 10) -> permute to (B, 10, 11, 15)
        Tensor map_perm = ob_map.permute({0, 3, 1, 2}).to(torch::kInt64);
        // Add offsets: codes = map_perm + offsets
        Tensor codes = map_perm + offsets.to(device);

        // Create multi-hot buffer and scatter
        Tensor map_buf = torch::zeros({B, 59, 11, 15}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        map_buf.scatter_(1, codes.to(torch::kInt32), 1.0f);

        // Conv layers
        Tensor map_out = torch::relu(conv1->forward(map_buf));
        map_out = conv2->forward(map_out);
        map_out = map_out.flatten(1);  // (B, 256)

        // Player discrete embedding
        Tensor player_discrete = player_embed->forward(ob_player.to(torch::kInt64));
        player_discrete = player_discrete.flatten(1);  // (B, 1504)

        // Concatenate: map_out + player_discrete + player_continuous + reward
        Tensor obs = torch::cat({map_out, player_discrete, ob_player.to(torch::kFloat32), ob_reward.to(torch::kFloat32)}, 1);

        // Projection with ReLU
        obs = torch::relu(proj->forward(obs));
        return obs;
    }
};

// NMMO3 decoder: LayerNorm -> fused logits+value
class NMMO3Decoder : public Decoder {
    public:
        torch::nn::LayerNorm layer_norm{nullptr};
        torch::nn::Linear linear{nullptr};
        int hidden_size;
        int output_size;

    NMMO3Decoder(int64_t hidden_size, int64_t output_size)
        : hidden_size(hidden_size), output_size(output_size) {
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_size})));

        linear = register_module("linear", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, output_size + 1).bias(true)));
        torch::nn::init::orthogonal_(linear->weight, 0.01);
        torch::nn::init::constant_(linear->bias, 0.0);
    }

    std::tuple<Tensor, Tensor> forward(Tensor hidden) override {
        Tensor x = layer_norm->forward(hidden);
        Tensor output = linear->forward(x);
        Tensor logits = output.narrow(-1, 0, output_size);
        Tensor value = output.narrow(-1, output_size, 1);
        return {logits, value.squeeze(-1)};
    }
};

// Drive encoder: ego/partner/road encoders with max pooling
// Two modes:
//   use_fused_kernel=true:  FC -> Max (fused kernel, no intermediate layer)
//   use_fused_kernel=false: Linear -> LayerNorm -> Linear -> Max (original torch)
class DriveEncoder : public Encoder {
    public:
        // Ego encoder: Linear -> ReLU -> Linear (no max pooling, single point)
        torch::nn::Linear ego_linear1{nullptr};
        torch::nn::Linear ego_linear2{nullptr};

        // Road encoder weights - fused mode: single FC layer
        Tensor road_W{nullptr};
        Tensor road_b{nullptr};
        // Road encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear road_linear1{nullptr};
        torch::nn::LayerNorm road_ln{nullptr};
        torch::nn::Linear road_linear2{nullptr};

        // Partner encoder weights - fused mode: single FC layer
        Tensor partner_W{nullptr};
        Tensor partner_b{nullptr};
        // Partner encoder modules - torch mode: Linear -> LayerNorm -> Linear
        torch::nn::Linear partner_linear1{nullptr};
        torch::nn::LayerNorm partner_ln{nullptr};
        torch::nn::Linear partner_linear2{nullptr};

        // Shared embedding
        torch::nn::Linear shared_linear{nullptr};
        int input_size;
        int hidden_size;
        bool use_fused_kernel;

    DriveEncoder(int64_t input_size, int64_t hidden_size, bool use_fused_kernel = true)
        : input_size(128), hidden_size(hidden_size), use_fused_kernel(use_fused_kernel) {

        // Ego encoder: 7 -> 128 -> 128 (Linear -> ReLU -> Linear)
        ego_linear1 = register_module("ego_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(7, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear1->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear1->bias, 0.0);
        ego_linear2 = register_module("ego_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(128, 128).bias(true)));
        torch::nn::init::orthogonal_(ego_linear2->weight, std::sqrt(2.0));
        torch::nn::init::constant_(ego_linear2->bias, 0.0);

        if (use_fused_kernel) {
            // Fused mode: single FC -> Max (no intermediate layer)
            // Road: 13 -> 128 (6 continuous + 7 one-hot)
            road_W = register_parameter("road_W", torch::empty({128, 13}));
            road_b = register_parameter("road_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(road_W, std::sqrt(2.0));

            // Partner: 7 -> 128
            partner_W = register_parameter("partner_W", torch::empty({128, 7}));
            partner_b = register_parameter("partner_b", torch::zeros({128}));
            torch::nn::init::orthogonal_(partner_W, std::sqrt(2.0));
        } else {
            // Torch mode: Linear -> LayerNorm -> Linear -> Max
            // Road: 13 -> 128 -> 128
            road_linear1 = register_module("road_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(13, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear1->bias, 0.0);
            road_ln = register_module("road_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            road_linear2 = register_module("road_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(road_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(road_linear2->bias, 0.0);

            // Partner: 7 -> 128 -> 128
            partner_linear1 = register_module("partner_linear1", torch::nn::Linear(
                torch::nn::LinearOptions(7, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear1->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear1->bias, 0.0);
            partner_ln = register_module("partner_ln", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({128})));
            partner_linear2 = register_module("partner_linear2", torch::nn::Linear(
                torch::nn::LinearOptions(128, 128).bias(true)));
            torch::nn::init::orthogonal_(partner_linear2->weight, std::sqrt(2.0));
            torch::nn::init::constant_(partner_linear2->bias, 0.0);
        }

        // Shared embedding: 3*128 -> hidden_size
        shared_linear = register_module("shared_linear", torch::nn::Linear(
            torch::nn::LinearOptions(3*128, hidden_size).bias(true)));
        torch::nn::init::orthogonal_(shared_linear->weight, std::sqrt(2.0));
        torch::nn::init::constant_(shared_linear->bias, 0.0);
    }

    Tensor forward(Tensor x) override {
        int64_t B = x.size(0);
        x = x.to(torch::kFloat32);

        // Split observations: ego (7), partner (441), road (1400)
        Tensor ego_obs = x.narrow(1, 0, 7);
        Tensor partner_obs = x.narrow(1, 7, 63*7);
        Tensor road_obs = x.narrow(1, 7 + 63*7, 200*7);

        // Ego encoding: Linear -> ReLU -> Linear (single point, no max)
        Tensor ego_features = ego_linear2->forward(torch::relu(ego_linear1->forward(ego_obs)));

        // Partner encoding
        Tensor partner_objects = partner_obs.view({B, 63, 7}).contiguous();
        Tensor partner_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            partner_features = fc_max(partner_objects, partner_W, partner_b);
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = partner_linear1->forward(partner_objects);  // (B, 63, 128)
            h = partner_ln->forward(h);
            h = partner_linear2->forward(h);  // (B, 63, 128)
            partner_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Road encoding with one-hot
        Tensor road_objects = road_obs.view({B, 200, 7});
        Tensor road_continuous = road_objects.narrow(2, 0, 6);
        Tensor road_categorical = road_objects.narrow(2, 6, 1).squeeze(2);
        Tensor road_onehot = torch::one_hot(road_categorical.to(torch::kInt64), 7).to(torch::kFloat32);
        Tensor road_combined = torch::cat({road_continuous, road_onehot}, 2).contiguous();  // (B, 200, 13)

        Tensor road_features;
        if (use_fused_kernel) {
            // Fused FC -> Max kernel
            road_features = fc_max(road_combined, road_W, road_b);
        } else {
            // Torch: Linear -> LayerNorm -> Linear -> Max
            auto h = road_linear1->forward(road_combined);  // (B, 200, 128)
            h = road_ln->forward(h);
            h = road_linear2->forward(h);  // (B, 200, 128)
            road_features = std::get<0>(h.max(1));  // (B, 128)
        }

        // Concatenate and shared embedding: GELU -> Linear -> ReLU
        Tensor concat_features = torch::cat({ego_features, road_features, partner_features}, 1);
        Tensor embedding = torch::relu(shared_linear->forward(torch::gelu(concat_features)));
        return embedding;
    }
};

// G2048 decoder: separate policy and value heads, cat + narrow for contiguous output
class G2048Decoder : public Decoder {
    public:
        torch::nn::Linear dec_linear1{nullptr};
        torch::nn::Linear dec_linear2{nullptr};
        torch::nn::Linear val_linear1{nullptr};
        torch::nn::Linear val_linear2{nullptr};
        int hidden_size;
        int output_size;

    G2048Decoder(int64_t hidden_size, int64_t output_size)
        : hidden_size(hidden_size), output_size(output_size) {
        // Decoder head: hidden -> hidden -> num_atns
        dec_linear1 = register_module("dec_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(dec_linear1->weight, std::sqrt(2.0));

        dec_linear2 = register_module("dec_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, output_size).bias(false)));
        torch::nn::init::orthogonal_(dec_linear2->weight, 0.01);

        // Value head: hidden -> hidden -> 1
        val_linear1 = register_module("val_linear1", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, hidden_size).bias(false)));
        torch::nn::init::orthogonal_(val_linear1->weight, std::sqrt(2.0));

        val_linear2 = register_module("val_linear2", torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, 1).bias(false)));
        torch::nn::init::orthogonal_(val_linear2->weight, 1.0);
    }

    std::tuple<Tensor, Tensor> forward(Tensor hidden) override {
        // Policy head
        Tensor logits = torch::gelu(dec_linear1->forward(hidden));
        logits = dec_linear2->forward(logits);

        // Value head
        Tensor value = torch::gelu(val_linear1->forward(hidden));
        value = val_linear2->forward(value);

        // Cat and narrow for contiguous outputs
        Tensor output = torch::cat({logits, value}, 1).contiguous();
        logits = output.narrow(1, 0, output_size);
        value = output.narrow(1, output_size, 1);

        return {logits, value.squeeze(1)};
    }
};


class PolicyMinGRU : public torch::nn::Module {
public:
    int64_t input_size;
    int64_t hidden_size;
    float expansion_factor;
    int64_t num_atns;
    int64_t num_layers;
    bool kernels;

    std::shared_ptr<Encoder> encoder{nullptr};
    std::shared_ptr<Decoder> decoder{nullptr};
    torch::nn::ModuleList mingru{nullptr};
    torch::nn::Linear value{nullptr};

    PolicyMinGRU(std::shared_ptr<Encoder> enc, std::shared_ptr<Decoder> dec, int64_t input_size, int64_t num_atns, int64_t hidden_size = 128, int64_t expansion_factor = 1, int64_t num_layers = 1, bool kernels = true)
        : input_size(input_size), hidden_size(hidden_size), expansion_factor(expansion_factor),
          num_atns(num_atns), num_layers(num_layers), kernels(kernels) {
        encoder = register_module("encoder", enc);
        decoder = register_module("decoder", dec);

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
