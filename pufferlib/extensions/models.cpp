// models.cpp - MinGRU, LSTM, and related model classes for pufferlib
// Included by pufferlib.cpp and profile_kernels.cu inside namespace pufferlib

using std::tuple;
using std::vector;
using std::shared_ptr;
namespace nn = torch::nn;
typedef torch::Tensor Tensor;

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
constexpr bool USE_BF16 = false;
constexpr torch::ScalarType PRECISION_DTYPE = torch::kFloat32;
#else
constexpr bool USE_BF16 = true;
constexpr torch::ScalarType PRECISION_DTYPE = torch::kBFloat16;
#endif

// Common tensor options
auto cuda_f32 = torch::dtype(torch::kFloat32).device(torch::kCUDA);
auto cuda_f64 = torch::dtype(torch::kFloat64).device(torch::kCUDA);
auto cuda_i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
auto cuda_i64 = torch::dtype(torch::kInt64).device(torch::kCUDA);
auto cuda_t = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);

// Raw struct bundling decoder outputs: mean (logits for discrete) + logstd
struct Logits {
    Tensor mean;    // Discrete: logits. Continuous: action mean.
    Tensor logstd;  // Discrete: undefined. Continuous: log standard deviation.
};

// Minimal interfaces for swappable components
// Inherit from nn::Module so register_module works
struct Encoder : public nn::Module {
    virtual Tensor forward(Tensor x) = 0;
};

class DefaultEncoder : public Encoder {
    public:
        nn::Linear linear{nullptr};
        int input;
        int hidden;

    DefaultEncoder(int input, int hidden)
        : input(input), hidden(hidden) {
        linear = register_module("linear", nn::Linear(
            nn::LinearOptions(input, hidden).bias(false)));
        nn::init::orthogonal_(linear->weight, std::sqrt(2.0));
    }

    Tensor forward(Tensor x) override {
        return linear->forward(x.to(linear->weight.dtype()));
    }
};

struct Decoder : public nn::Module {
    virtual tuple<Logits, Tensor> forward(Tensor hidden) = 0;
};

class DefaultDecoder : public Decoder {
    public:
        nn::Linear linear{nullptr};
        Tensor logstd_param{nullptr};
        int hidden;
        int output;
        bool continuous;

    DefaultDecoder(int hidden, int output, bool continuous = false)
            : hidden(hidden), output(output), continuous(continuous) {
        linear = register_module("linear", nn::Linear(
            nn::LinearOptions(hidden, output+1).bias(false)));
        nn::init::orthogonal_(linear->weight, 0.01);
        if (continuous) {
            logstd_param = register_parameter("logstd", torch::zeros({1, output}));
        }
    }

    tuple<Logits, Tensor> forward(Tensor h) override {
        h = linear->forward(h);
        // Logits and value are fused in contiguous memory
        // This is mandatory in custom decoders for our loss kernel to work
        Logits logits = {.mean = h.narrow(-1, 0, output)};
        Tensor value = h.narrow(-1, output, 1).squeeze(-1);
        if (continuous) {
            logits.logstd = logstd_param.expand_as(logits.mean);
        }
        return {logits, value};
    }
};

// Reference implementation for mingru_gate (inference path)
// Takes combined (B, 3*H) = [hidden, gate, proj] and state (B, H)
// Returns {out, next_state} where:
//   out (B, H) = sigmoid(proj) * mingru_out
//   next_state (B, H) = mingru_out (for recurrence)
vector<Tensor> mingru_gate_cpp(Tensor state, Tensor combined) {
    auto chunks = combined.chunk(3, 1);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    auto h = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
    auto g = gate.sigmoid();
    auto mingru_out = torch::lerp(state, h, g);
    auto out = torch::sigmoid(proj) * mingru_out;
    return {out, mingru_out};
}

// Reference implementation for fused_scan (training path)
// Takes combined (B, T, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Returns {out, next_state} where:
//   out (B, T, H) = sigmoid(proj) * scan_result
//   next_state (B, 1, H) = raw scan_result at T (for recurrence)
vector<Tensor> fused_scan_cpp(Tensor combined, Tensor state) {
    auto seq_len = combined.size(1);

    // Split combined into hidden, gate, proj
    auto chunks = combined.chunk(3, 2);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    // Compute log_coeffs and log_values
    auto log_coeffs = -nn::functional::softplus(gate);
    auto log_z = -nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (nn::functional::relu(hidden) + 0.5).log(),
        -nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;

    // Cat state and pad for scan
    log_values = torch::cat({state.log(), log_values}, 1);
    log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});

    // Heinsen associative scan
    auto a_star = log_coeffs.cumsum(1);
    auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
    auto log_h = a_star + log_h0_plus_b_star;
    auto scan_result = log_h.exp();

    // Extract output and next_state
    scan_result = scan_result.narrow(1, scan_result.size(1) - seq_len, seq_len);
    auto next_state = scan_result.narrow(1, scan_result.size(1) - 1, 1);

    // Apply sigmoid(proj) * scan_result for output
    auto out = torch::sigmoid(proj) * scan_result;

    return {out, next_state};
}

// Activation buffers for encoder — separate from weights so multiple copies can exist
struct EncoderActivations {
    PufTensor saved_input;   // (B_TT, in_dim) — saved for backward
    PufTensor out;           // (B_TT, out_dim) — mm_out dest, reused as grad buffer in backward
    PufTensor wgrad_scratch; // (out_dim, in_dim) — scratch for weight grad mm_out
    PufTensor inf_out;       // (B_inf, out_dim) — inference mm_out dest
};

// Activation buffers for decoder — separate from weights so multiple copies can exist
struct DecoderActivations {
    PufTensor saved_input;   // (B_TT, hidden) — saved for backward
    PufTensor out;           // (B_TT, output+1) — mm_out dest
    PufTensor grad_out;      // (B_TT, output+1) — fused grad from PPO (assembled by kernel)
    PufTensor wgrad_scratch; // (output+1, hidden) — scratch for weight grad mm_out
    PufTensor inf_out;       // (B_inf, output+1) — inference mm_out dest
};

// Native encoder — weights only, activations created separately
struct NativeEncoder {
    PufTensor weight, weight_grad;
    int in_dim, out_dim;

    NativeEncoder() : in_dim(0), out_dim(0) {}

    NativeEncoder(Allocator& alloc, int input, int hidden)
        : in_dim(input), out_dim(hidden) {
        alloc.register_param(&weight, {hidden, input});
        alloc.register_grad(&weight_grad, {hidden, input});
    }

    void register_activations(Allocator& alloc, EncoderActivations& act, int B_TT) {
        int psz = dtype_size(PRECISION_DTYPE);
        alloc.register_puf(&act.saved_input, {B_TT, in_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, out_dim}, psz);
        alloc.register_puf(&act.wgrad_scratch, {out_dim, in_dim}, psz);
    }

    void register_inference(Allocator& alloc, EncoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, out_dim}, dtype_size(PRECISION_DTYPE));
    }

    void init_weights() {
        auto w = weight.to_torch(PRECISION_DTYPE);
        nn::init::orthogonal_(w, std::sqrt(2.0));
    }
};

// Native decoder — weights only, activations created separately
struct NativeDecoder {
    PufTensor weight, weight_grad;
    PufTensor logstd, logstd_grad;
    int hidden_dim, output_dim;
    bool continuous;

    NativeDecoder() : hidden_dim(0), output_dim(0), continuous(false) {}

    NativeDecoder(Allocator& alloc, int hidden, int output, bool continuous)
        : hidden_dim(hidden), output_dim(output), continuous(continuous) {
        alloc.register_param(&weight, {output + 1, hidden});
        alloc.register_grad(&weight_grad, {output + 1, hidden});
        if (continuous) {
            alloc.register_param(&logstd, {1, output});
            alloc.register_grad(&logstd_grad, {1, output});
        }
    }

    void register_activations(Allocator& alloc, DecoderActivations& act, int B_TT) {
        int psz = dtype_size(PRECISION_DTYPE);
        alloc.register_puf(&act.saved_input, {B_TT, hidden_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.grad_out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.wgrad_scratch, {output_dim + 1, hidden_dim}, psz);
    }

    void register_inference(Allocator& alloc, DecoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, output_dim + 1}, dtype_size(PRECISION_DTYPE));
    }

    void init_weights() {
        auto w = weight.to_torch(PRECISION_DTYPE);
        nn::init::orthogonal_(w, 0.01);
        // logstd is already zero from allocator's torch::zeros
    }
};

// Activation buffers for MinGRU — separate from weights so multiple copies can exist
struct MinGRUActivations {
    int num_layers;  // needed to size vectors

    // Training forward activations
    vector<PufTensor> saved_inputs;    // (B, TT, H) per layer
    vector<PrefixScan> scan_bufs;      // per layer
    vector<PufTensor> combined_bufs;   // (B*TT, 3*H) per layer — mm_out dest

    // Training backward buffers
    PufTensor grad_input_buf;          // (B*TT, H) — shared across layers
    PufTensor grad_next_state;         // (B, 1, H)
    PufTensor wgrad_scratch;           // (3*H, H) — scratch for weight grad mm_out

    // Inference buffers
    vector<PufTensor> inf_combined;    // (B_inf, 3*H) per layer
    PufTensor inf_out;                 // (B_inf, H) — shared across layers (written by mingru_gate)
    PufTensor inf_next_state;          // (B_inf, H) — shared across layers (written by mingru_gate)

    MinGRUActivations() : num_layers(0) {}
    MinGRUActivations(int num_layers)
        : num_layers(num_layers),
          saved_inputs(num_layers), scan_bufs(num_layers),
          combined_bufs(num_layers), inf_combined(num_layers) {}
};

// Native MinGRU — weights only, activations created separately
struct MinGRU {
    int hidden, num_layers;
    vector<PufTensor> weights;       // (3*H, H) per layer, views into allocator
    vector<PufTensor> weight_grads;  // (3*H, H) per layer, views into allocator

    MinGRU() : hidden(0), num_layers(0) {}

    MinGRU(Allocator& alloc, int hidden, int num_layers)
        : hidden(hidden), num_layers(num_layers),
          weights(num_layers), weight_grads(num_layers) {
        for (int i = 0; i < num_layers; i++) {
            alloc.register_param(&weights[i], {3 * hidden, hidden});
            alloc.register_grad(&weight_grads[i], {3 * hidden, hidden});
        }
    }

    void register_activations(Allocator& alloc, MinGRUActivations& act, int B, int TT) {
        int H = hidden;
        int B_TT = B * TT;
        int psz = dtype_size(PRECISION_DTYPE);

        // Shared backward buffers
        alloc.register_puf(&act.grad_input_buf, {B_TT, H}, psz);
        alloc.register_puf(&act.grad_next_state, {B, 1, H}, psz);
        alloc.register_puf(&act.wgrad_scratch, {3 * H, H}, psz);

        // Per-layer activations
        for (int i = 0; i < num_layers; i++) {
            alloc.register_puf(&act.saved_inputs[i], {B, TT, H}, psz);
            alloc.register_puf(&act.combined_bufs[i], {B_TT, 3 * H}, psz);
            alloc.register_puf(&act.scan_bufs[i].out, {B, TT, H}, psz);
            alloc.register_puf(&act.scan_bufs[i].next_state, {B, 1, H}, psz);
            alloc.register_puf(&act.scan_bufs[i].a_star, {B, TT + 1, H}, 4);  // float32
            alloc.register_puf(&act.scan_bufs[i].s_vals, {B, TT + 1, H}, 4);
            alloc.register_puf(&act.scan_bufs[i].log_values_buf, {B, TT + 1, H}, 4);
            alloc.register_puf(&act.scan_bufs[i].grad_combined, {B, TT, 3 * H}, psz);
            alloc.register_puf(&act.scan_bufs[i].grad_state, {B, 1, H}, psz);
        }
    }

    void register_inference(Allocator& alloc, MinGRUActivations& act, int B_inf) {
        int H = hidden;
        int dsz = dtype_size(PRECISION_DTYPE);
        for (int i = 0; i < num_layers; i++) {
            alloc.register_puf(&act.inf_combined[i], {B_inf, 3 * H}, dsz);
        }
        alloc.register_puf(&act.inf_out, {B_inf, H}, dsz);
        alloc.register_puf(&act.inf_next_state, {B_inf, H}, dsz);
    }

    void init_weights() {
        for (int i = 0; i < num_layers; i++) {
            auto w = weights[i].to_torch(PRECISION_DTYPE);
            nn::init::orthogonal_(w);
        }
    }

    Tensor initial_state(int batch_size, torch::Device device, torch::Dtype dtype) {
        return torch::zeros({num_layers, batch_size, hidden},
            torch::dtype(dtype).device(device));
    }

    // Helper: get PufTensor view of layer i from (num_layers, B, H) state
    PufTensor state_layer(PufTensor& state, int i) {
        PufTensor s;
        int64_t B = state.size(1);
        int64_t H = state.size(2);
        s.data = (char*)state.data + i * B * H * state.dtype_size;
        s.shape[0] = B; s.shape[1] = H; s.shape[2] = 1; s.shape[3] = 1;
        s.ndim = 2; s.numel = B * H; s.dtype_size = state.dtype_size;
        return s;
    }

    // Inference: x (B, H) PufTensor, state (num_layers, B, H) PufTensor -> PufTensor (B, H)
    // State is modified in-place
    PufTensor forward(PufTensor x, PufTensor state, MinGRUActivations& act) {
        for (int i = 0; i < num_layers; i++) {
            PufTensor state_i = state_layer(state, i);
            puf_mm(x, weights[i], act.inf_combined[i]);
            mingru_gate(state_i, act.inf_combined[i], act.inf_out, act.inf_next_state);
            puf_copy(state_i, act.inf_next_state);
            x = act.inf_out;
        }
        return x;
    }

    // Training forward: x (B, TT, H) PufTensor, state PufTensor -> (B, TT, H) PufTensor
    PufTensor forward_train(PufTensor x, PufTensor state, MinGRUActivations& act) {
        int B = x.size(0);
        int TT = x.size(1);

        for (int i = 0; i < num_layers; i++) {
            puf_copy(act.saved_inputs[i], x);
            PufTensor state_i = state_layer(state, i);

            // Reshape state_i (B, H) -> (B, 1, H) for prefix_scan
            PufTensor state_3d = state_i;
            state_3d.shape[0] = B; state_3d.shape[1] = 1; state_3d.shape[2] = hidden;
            state_3d.ndim = 3;

            // Flatten x from (B, TT, H) to (B*TT, H) for mm
            PufTensor x_flat = x;
            x_flat.shape[0] = B * TT; x_flat.shape[1] = hidden; x_flat.ndim = 2;
            puf_mm(x_flat, weights[i], act.combined_bufs[i]);

            // Reinterpret (B*TT, 3*H) as (B, TT, 3*H) for scan
            PufTensor combined_3d = act.combined_bufs[i];
            combined_3d.shape[0] = B; combined_3d.shape[1] = TT; combined_3d.shape[2] = 3 * hidden;
            combined_3d.ndim = 3;
            prefix_scan_forward(combined_3d, state_3d, act.scan_bufs[i]);
            x = act.scan_bufs[i].out;
        }
        return x;
    }

    // Backward: grad (B, TT, H) PufTensor -> grad_input (B, TT, H) PufTensor
    PufTensor backward(PufTensor grad, MinGRUActivations& act, MinGRU* target) {
        int B = grad.size(0);
        int TT = grad.size(1);
        int H = grad.size(2);
        for (int i = num_layers - 1; i >= 0; i--) {
            prefix_scan_backward(grad, act.grad_next_state, act.scan_bufs[i]);

            // Reinterpret grad_combined (B, TT, 3*H) as (B*TT, 3*H) for matmuls
            PufTensor gc_flat = act.scan_bufs[i].grad_combined;
            gc_flat.shape[0] = B * TT; gc_flat.shape[1] = 3 * H; gc_flat.ndim = 2;

            // Reinterpret saved_inputs (B, TT, H) as (B*TT, H)
            PufTensor inp_flat = act.saved_inputs[i];
            inp_flat.shape[0] = B * TT; inp_flat.shape[1] = H; inp_flat.ndim = 2;

            // Weight grad: gc_flat^T @ inp_flat → wgrad_scratch, then accumulate
            puf_mm_tn(gc_flat, inp_flat, act.wgrad_scratch);
            puf_add(target->weight_grads[i], act.wgrad_scratch);

            // Input grad: gc_flat @ weights[i] → grad_input_buf
            puf_mm_nn(gc_flat, weights[i], act.grad_input_buf);

            // Reshape (B*TT, H) → (B, TT, H)
            grad = act.grad_input_buf;
            grad.shape[0] = B; grad.shape[1] = TT; grad.shape[2] = H; grad.ndim = 3;
        }
        return grad;
    }

    void append_param_shapes(vector<Muon::ParamShape>& out) {
        for (int i = 0; i < num_layers; i++) {
            vector<int64_t> s(weights[i].shape, weights[i].shape + weights[i].ndim);
            out.push_back({weights[i].numel, s, weights[i].ndim});
        }
    }
};

// Native Policy — no nn::Module, no autograd. Kernels only.
// Each sub-struct self-registers with the Allocator.
// Construction order (encoder → decoder → rnn) determines param_buffer layout,
// which must match parameters() order for Muon.
// Activation buffers for the full Policy — separate from weights
struct PolicyActivations {
    EncoderActivations enc;
    DecoderActivations dec;
    MinGRUActivations rnn;

    PolicyActivations() {}
    PolicyActivations(int num_layers) : rnn(num_layers) {}
};

// Native Policy — weights only, activations created separately
struct Policy {
    NativeEncoder encoder;
    NativeDecoder decoder;
    MinGRU rnn;
    int num_atns;
    PolicyActivations act;  // owned activations (1 copy for now)

    Policy() : num_atns(0) {}

    Policy(Allocator& alloc, int input, int hidden, int output, int num_layers, int num_atns, bool continuous)
        : encoder(alloc, input, hidden),
          decoder(alloc, hidden, output, continuous),
          rnn(alloc, hidden, num_layers),
          num_atns(num_atns),
          act(num_layers) {}

    // Register training activation buffers — call after constructor, before alloc.create()
    void register_activations(Allocator& alloc, int B, int TT) {
        register_activations(alloc, act, B, TT);
    }

    void register_activations(Allocator& alloc, PolicyActivations& a, int B, int TT) {
        encoder.register_activations(alloc, a.enc, B * TT);
        decoder.register_activations(alloc, a.dec, B * TT);
        rnn.register_activations(alloc, a.rnn, B, TT);
    }

    // Register inference-only activations into a separate allocator (for per-buffer copies)
    void register_inference(Allocator& alloc, PolicyActivations& a, int B_inf) {
        encoder.register_inference(alloc, a.enc, B_inf);
        decoder.register_inference(alloc, a.dec, B_inf);
        rnn.register_inference(alloc, a.rnn, B_inf);
    }

    void init_weights() {
        encoder.init_weights();
        decoder.init_weights();
        rnn.init_weights();
    }

    Tensor initial_state(int batch_size, torch::Device device) {
        return rnn.initial_state(batch_size, device, PRECISION_DTYPE);
    }

    // Python-facing wrappers (torch in/out, uses internal act)
    tuple<Tensor, Tensor> forward(Tensor observations, Tensor state) {
        Tensor obs_cast = observations.to(PRECISION_DTYPE);
        PufTensor obs_puf = PufTensor::from_torch(obs_cast);
        PufTensor state_puf = PufTensor::from_torch(state);
        PufTensor dec_out = forward(obs_puf, state_puf, act);
        return {dec_out.to_torch(PRECISION_DTYPE), state};
    }

    Tensor forward_train(Tensor x, Tensor state) {
        PufTensor x_puf = PufTensor::from_torch(x);
        PufTensor state_puf = PufTensor::from_torch(state);
        PufTensor dec_out = forward_train(x_puf, state_puf, act);
        return dec_out.to_torch(PRECISION_DTYPE);
    }

    // Inference: obs (B, input) PufTensor, state PufTensor -> dec_out (B, output+1) PufTensor
    // State is modified in-place
    PufTensor forward(PufTensor obs, PufTensor state, PolicyActivations& a) {
        puf_mm(obs, encoder.weight, a.enc.inf_out);
        PufTensor h = rnn.forward(a.enc.inf_out, state, a.rnn);
        puf_mm(h, decoder.weight, a.dec.inf_out);
        return a.dec.inf_out;
    }

    // Training forward: x (B, TT, input) PufTensor, state PufTensor
    // Returns fused decoder output (B, TT, output+1) PufTensor
    PufTensor forward_train(PufTensor x, PufTensor state, PolicyActivations& a) {
        int B = x.size(0);
        int TT = x.size(1);

        // Flatten to (B*TT, input)
        PufTensor x_flat = x;
        x_flat.shape[0] = B * TT; x_flat.shape[1] = encoder.in_dim; x_flat.ndim = 2;

        puf_copy(a.enc.saved_input, x_flat);
        puf_mm(a.enc.saved_input, encoder.weight, a.enc.out);

        // Reshape enc output to (B, TT, H)
        PufTensor h = a.enc.out;
        h.shape[0] = B; h.shape[1] = TT; h.shape[2] = encoder.out_dim; h.ndim = 3;

        h = rnn.forward_train(h, state, a.rnn);

        // Flatten for decoder mm
        PufTensor flat_h = h;
        flat_h.shape[0] = B * TT; flat_h.shape[1] = encoder.out_dim; flat_h.ndim = 2;

        puf_copy(a.dec.saved_input, flat_h);
        puf_mm(flat_h, decoder.weight, a.dec.out);

        // Reshape to (B, TT, output+1)
        PufTensor result = a.dec.out;
        result.shape[0] = B; result.shape[1] = TT; result.shape[2] = decoder.output_dim + 1; result.ndim = 3;
        return result;
    }

    // Backward: all PufTensor grads (fp32)
    void backward(PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
                  PolicyActivations& a, Policy* target) {
        int B_TT = a.dec.saved_input.size(0);
        int B = grad_logits.size(0);
        int TT = grad_logits.size(1);

        // Assemble fused grad_out (B_TT, output+1) from separate fp32 grads → bf16
        PufTensor gl_flat = grad_logits;
        gl_flat.shape[0] = B_TT; gl_flat.shape[1] = decoder.output_dim; gl_flat.ndim = 2;
        PufTensor gv_flat = grad_value;
        gv_flat.shape[0] = B_TT; gv_flat.ndim = 1;
        puf_assemble_decoder_grad(a.dec.grad_out, gl_flat, gv_flat);

        // Decoder weight grad: bf16 matmul into scratch, then accumulate into fp32 grad
        puf_mm_tn(a.dec.grad_out, a.dec.saved_input, a.dec.wgrad_scratch);
        puf_add(target->decoder.weight_grad, a.dec.wgrad_scratch);

        // logstd grad: column-wise sum reduction into fp32 master grad
        if (decoder.continuous && grad_logstd.data != nullptr) {
            PufTensor gls_flat = grad_logstd;
            gls_flat.shape[0] = B_TT; gls_flat.shape[1] = decoder.output_dim; gls_flat.ndim = 2;
            puf_sum_rows_add(target->decoder.logstd_grad, gls_flat);
        }

        // Decoder input grad: mm into enc.out (same shape: B_TT x hidden, reused buffer)
        puf_mm_nn(a.dec.grad_out, decoder.weight, a.enc.out);

        // Reshape to (B, TT, H) for RNN backward
        PufTensor grad_h = a.enc.out;
        grad_h.shape[0] = B; grad_h.shape[1] = TT; grad_h.shape[2] = encoder.out_dim; grad_h.ndim = 3;

        // RNN backward
        grad_h = rnn.backward(grad_h, a.rnn, &target->rnn);

        // Flatten for encoder weight grad
        PufTensor grad_enc = grad_h;
        grad_enc.shape[0] = B_TT; grad_enc.shape[1] = encoder.out_dim; grad_enc.ndim = 2;
        puf_mm_tn(grad_enc, a.enc.saved_input, a.enc.wgrad_scratch);
        puf_add(target->encoder.weight_grad, a.enc.wgrad_scratch);
    }

    vector<Muon::ParamShape> param_shapes() {
        vector<Muon::ParamShape> shapes;
        auto push = [&](PufTensor& p) {
            vector<int64_t> s(p.shape, p.shape + p.ndim);
            shapes.push_back({p.numel, s, p.ndim});
        };
        push(encoder.weight);
        push(decoder.weight);
        if (decoder.continuous) push(decoder.logstd);
        rnn.append_param_shapes(shapes);
        return shapes;
    }
};

struct ShareableLSTMCell : public nn::LSTMCellImpl {
    ShareableLSTMCell(const nn::LSTMCellOptions& options) : nn::LSTMCellImpl(options) {}

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

class PolicyLSTM : public nn::Module {
private:
    int input_;
    int hidden_;
    int num_atns_;
    nn::Sequential encoder{nullptr};
    nn::Linear decoder{nullptr};
    nn::Linear value{nullptr};
    nn::LSTM lstm{nullptr};
    shared_ptr<ShareableLSTMCell> cell{nullptr};

public:
    PolicyLSTM(int input, int num_atns, int hidden = 128)
        : input_(input), hidden_(hidden), num_atns_(num_atns) {
        encoder = register_module("encoder", nn::Sequential(
            nn::Linear(input_, hidden_),
            nn::GELU()
        ));
        auto encoder_linear = (*encoder)[0]->as<nn::LinearImpl>();
        nn::init::orthogonal_(encoder_linear->weight, std::sqrt(2.0));
        nn::init::constant_(encoder_linear->bias, 0.0);

        decoder = register_module("decoder", nn::Linear(hidden_, num_atns_));
        nn::init::orthogonal_(decoder->weight, 0.01);
        nn::init::constant_(decoder->bias, 0.0);

        value = register_module("value", nn::Linear(hidden_, 1));
        nn::init::orthogonal_(value->weight, 1.0);
        nn::init::constant_(value->bias, 0.0);

        lstm = register_module("lstm", nn::LSTM(nn::LSTMOptions(hidden_, hidden_).num_layers(1)));
        nn::init::orthogonal_(lstm->named_parameters()["weight_ih_l0"], 1.0);
        nn::init::orthogonal_(lstm->named_parameters()["weight_hh_l0"], 1.0);
        lstm->named_parameters()["bias_ih_l0"].data().zero_();
        lstm->named_parameters()["bias_hh_l0"].data().zero_();

        cell = register_module("cell", std::make_shared<ShareableLSTMCell>(
            nn::LSTMCellOptions(hidden_, hidden_)));
        cell->set_shared_weights(lstm->named_parameters()["weight_ih_l0"],
            lstm->named_parameters()["weight_hh_l0"],
            lstm->named_parameters()["bias_ih_l0"],
            lstm->named_parameters()["bias_hh_l0"]);
    }

    // Forward for evaluation/inference (uses LSTMCell)
    tuple<Tensor, Tensor, Tensor, Tensor> forward(
        Tensor observations, Tensor h, Tensor c) {
        int64_t B = observations.size(0);

        TORCH_CHECK(observations.dim() == 2 && observations.size(1) == input_,
                    "Observations must be [B, input]");

        if (h.defined() && h.numel() > 0) {
            TORCH_CHECK(h.dim() == 2 && h.size(0) == B && h.size(1) == hidden_,
                        "h must be [B, hidden]");
            TORCH_CHECK(c.dim() == 2 && c.size(0) == B && c.size(1) == hidden_,
                        "c must be [B, hidden]");
        }

        Tensor hidden = encoder->forward(observations);

        tuple<Tensor, Tensor> cell_out;
        if (h.defined() && h.numel() > 0) {
            cell_out = cell->forward(hidden, std::make_optional(std::make_tuple(h, c)));
        } else {
            cell_out = cell->forward(hidden);
        }

        Tensor hidden_out = std::get<0>(cell_out);
        Tensor c_out = std::get<1>(cell_out);

        Tensor logits = decoder->forward(hidden_out);
        Tensor values = value->forward(hidden_out);

        return {logits, values, hidden_out, c_out};
    }

    // Forward for training (uses LSTM)
    tuple<Tensor, Tensor> forward_train(
        Tensor observations, Tensor lstm_h, Tensor lstm_c) {
        Tensor x = observations;
        auto x_shape = x.sizes();

        TORCH_CHECK((x.dim() == 2 || x.dim() == 3),
                    "Observations must be [B, input] or [B, TT, input]");
        TORCH_CHECK(x.size(-1) == input_,
                    "Last dimension of observations must match input");

        int64_t B = x_shape[0];
        int64_t TT = (x.dim() == 3) ? x_shape[1] : 1;

        if (lstm_h.defined() && lstm_h.numel() > 0) {
            TORCH_CHECK(lstm_h.dim() == 3 && lstm_h.size(0) == 1 && lstm_h.size(1) == B,
                        "lstm_h must be [1, B, hidden]");
            TORCH_CHECK(lstm_c.dim() == 3 && lstm_c.size(0) == 1 && lstm_c.size(1) == B,
                        "lstm_c must be [1, B, hidden]");
        }

        // Flatten time steps if needed
        if (x.dim() == 3) {
            x = x.reshape({B * TT, input_});
        } else {
            TT = 1;
        }

        Tensor hidden = encoder->forward(x);

        hidden = hidden.reshape({B, TT, hidden_});
        hidden = hidden.transpose(0, 1);  // [TT, B, hidden]

        tuple<Tensor, tuple<Tensor, Tensor>> lstm_out;
        if (lstm_h.defined() && lstm_h.numel() > 0) {
            lstm_out = lstm->forward(hidden, std::make_optional(std::make_tuple(lstm_h, lstm_c)));
        } else {
            lstm_out = lstm->forward(hidden);
        }

        hidden = std::get<0>(lstm_out);
        hidden = hidden.transpose(0, 1);  // [B, TT, hidden]

        Tensor flat_hidden = hidden.reshape({-1, hidden_});
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

// Sync bf16 working weights from fp32 master weights (for mixed-precision training)
// Uses contiguous buffer copy instead of per-param loop
void sync_policy_weights(Tensor& dst_param_buffer, const Tensor& src_param_buffer) {
    dst_param_buffer.copy_(src_param_buffer);  // copy_ handles fp32→bf16 conversion
}

// =============================================================================
// Reference/fallback implementations (pure PyTorch, no CUDA kernels)
// Moved from modules.cu for cleaner separation of CUDA vs torch-native code
// =============================================================================

torch::autograd::tensor_list log_coeffs_and_values_cpp(Tensor gate, Tensor hidden) {
    auto log_coeffs = -nn::functional::softplus(gate);
    auto log_z = -nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (nn::functional::relu(hidden) + 0.5).log(),
        -nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;
    return {log_coeffs, log_values};
}

Tensor logcumsumexp_cpp(Tensor x) {
    return x.exp().cumsum(1).log();
}

// Sample from multi-head discrete distribution
// Returns {actions (B, heads), total_logprob (B,)}
vector<Tensor> sample_discrete_cpp(Tensor logits, Tensor act_sizes_cpu, int num_heads) {
    logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);
    auto split = torch::split(logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_heads), 1);
    vector<Tensor> actions_vec, logprobs_vec;
    for (int i = 0; i < num_heads; i++) {
        auto log_probs = torch::log_softmax(split[i], 1);
        auto action = at::multinomial(log_probs.exp(), 1, true);
        actions_vec.push_back(action);
        logprobs_vec.push_back(log_probs.gather(1, action));
    }
    return {torch::cat(actions_vec, 1), torch::cat(logprobs_vec, 1).sum(1)};
}

// Sample from continuous Normal distribution
// Returns {actions (B, D), total_logprob (B,)}
vector<Tensor> sample_continuous_cpp(Tensor mean, Tensor logstd) {
    auto std = logstd.exp();
    auto actions = mean + std * torch::randn_like(mean);
    auto log_prob = -0.5 * ((actions - mean) / std).pow(2) - 0.5 * std::log(2 * M_PI) - logstd;
    return {actions, log_prob.sum(1)};
}

// Compute logprob + entropy for multi-head discrete actions
// Returns {logprob (batch,), entropy scalar}
vector<Tensor> discrete_logprob_entropy_cpp(Tensor logits, Tensor actions, Tensor act_sizes_cpu, int num_heads) {
    logits = torch::nan_to_num(logits, 1e-8, 1e-8, 1e-8);
    auto split = torch::split(logits, c10::IntArrayRef(act_sizes_cpu.data_ptr<int64_t>(), num_heads), 1);
    int batch = logits.size(0);
    vector<Tensor> logprobs_vec, entropies_vec;
    for (int h = 0; h < num_heads; h++) {
        auto log_probs = torch::log_softmax(split[h], 1);
        auto probs = log_probs.exp();
        auto head_actions = actions.select(-1, h).reshape({batch}).to(torch::kInt64);
        logprobs_vec.push_back(log_probs.gather(1, head_actions.unsqueeze(1)));
        entropies_vec.push_back(-(probs * log_probs).sum(1, true));
    }
    auto logprob = torch::cat(logprobs_vec, 1).sum(1);
    auto entropy = torch::cat(entropies_vec, 1).sum(1).mean();
    return {logprob, entropy};
}

// Compute logprob + entropy for continuous Normal actions
// Returns {logprob (batch,), entropy scalar}
vector<Tensor> continuous_logprob_entropy_cpp(Tensor mean, Tensor logstd, Tensor actions) {
    auto std = logstd.exp();
    auto normalized = (actions.to(mean.dtype()) - mean) / std;
    auto log_prob = -0.5 * normalized.pow(2) - 0.5 * std::log(2 * M_PI) - logstd;
    auto logprob = log_prob.sum(1);
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    auto entropy = (HALF_1_PLUS_LOG_2PI + logstd).sum(1).mean();
    return {logprob, entropy};
}

// PPO clipped loss with clipped value loss
Tensor ppo_loss_cpp(Tensor ratio, Tensor advantages, Tensor prio,
        Tensor newvalue, Tensor values, Tensor returns, Tensor entropy,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
        Tensor losses) {
    auto logratio = ratio.log();
    auto adv_normalized = prio * (advantages - advantages.mean()) / (advantages.std() + 1e-8);
    auto pg_loss1 = -adv_normalized * ratio;
    auto pg_loss2 = -adv_normalized * torch::clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef);
    auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

    newvalue = newvalue.view(returns.sizes());
    auto v_clipped = values + torch::clamp(newvalue - values, -vf_clip_coef, vf_clip_coef);
    auto v_loss = 0.5 * torch::max((newvalue - returns).pow(2), (v_clipped - returns).pow(2)).mean();

    auto total = pg_loss + vf_coef * v_loss - ent_coef * entropy;

    // Accumulate loss components for logging (detached, no grad)
    losses.select(0, LOSS_PG).add_(pg_loss.detach());
    losses.select(0, LOSS_VF).add_(v_loss.detach());
    losses.select(0, LOSS_ENT).add_(entropy.detach());
    losses.select(0, LOSS_TOTAL).add_(total.detach());
    losses.select(0, LOSS_OLD_APPROX_KL).add_((-logratio).mean().detach());
    losses.select(0, LOSS_APPROX_KL).add_(((ratio - 1) - logratio).mean().detach());
    losses.select(0, LOSS_CLIPFRAC).add_(((ratio - 1.0).abs() > clip_coef).to(torch::kFloat32).mean().detach());
    losses.select(0, LOSS_N).add_(1.0);

    return total;
}

// Dispatch: sample actions using kernel or cpp path, write to output buffers
void sample_actions(Logits& logits, Tensor value,
        Tensor actions_out, Tensor logprobs_out, Tensor values_out,
        Tensor act_sizes, Tensor act_sizes_cpu,
        bool is_continuous, bool kernels, uint64_t rng_seed, Tensor rng_offset) {
    if (kernels) {
        Tensor logstd = logits.logstd.defined() ? logits.logstd : Tensor();
        sample_logits(logits.mean, logstd, value, actions_out, logprobs_out,
            values_out, act_sizes, rng_seed, rng_offset);
    } else {
        vector<Tensor> result;
        if (is_continuous) {
            result = sample_continuous_cpp(logits.mean, logits.logstd);
        } else {
            result = sample_discrete_cpp(logits.mean, act_sizes_cpu, actions_out.size(1));
        }
        actions_out.copy_(result[0].to(torch::kFloat64), false);
        logprobs_out.copy_(result[1], false);
        values_out.copy_(value.flatten(), false);
    }
}

void train_select_and_copy_cpp(
        Tensor observations, Tensor actions,
        Tensor logprobs, Tensor values, Tensor advantages,
        Tensor idx, Tensor mb_prio,
        Tensor dst_obs, Tensor dst_state,
        Tensor dst_actions, Tensor dst_logprobs,
        Tensor dst_advantages, Tensor dst_prio,
        Tensor dst_values, Tensor dst_returns ){
    Tensor mb_obs = observations.index_select(0, idx);
    Tensor mb_actions = actions.index_select(0, idx);
    Tensor mb_logprobs = logprobs.index_select(0, idx);
    Tensor mb_values = values.index_select(0, idx);
    Tensor mb_advantages = advantages.index_select(0, idx);
    Tensor mb_returns = mb_advantages + mb_values;

    dst_obs.copy_(mb_obs, false);
    dst_state.zero_();
    dst_actions.copy_(mb_actions, false);
    dst_logprobs.copy_(mb_logprobs, false);
    dst_advantages.copy_(mb_advantages, false);
    dst_prio.copy_(mb_prio, false);
    dst_values.copy_(mb_values, false);
    dst_returns.copy_(mb_returns, false);
}

std::tuple<Tensor, Tensor> prio_replay_cpp(
    Tensor advantages, float prio_alpha, int minibatch_segments,
    int total_agents, float anneal_beta
) {
    Tensor adv = advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, minibatch_segments, true);
    Tensor mb_prio = torch::pow(total_agents*prio_probs.index_select(0, idx).unsqueeze(1), -anneal_beta);
    return {idx, mb_prio};
}

// Fused PPO loss (PyTorch fallback path — matches PPOLoss::apply signature)
torch::autograd::tensor_list fused_ppo_loss_cpp(
        Tensor logits, Tensor logstd, Tensor newvalue,
        Tensor actions, Tensor old_logprobs, Tensor advantages, Tensor prio,
        Tensor values, Tensor returns,
        Tensor ratio_out, Tensor newvalue_out,
        Tensor act_sizes, Tensor losses,
        float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef) {
    bool is_continuous = logstd.numel() > 0;
    int num_heads = actions.size(-1);
    int segments = actions.size(0);
    int horizon = actions.size(1);
    int batch = segments * horizon;

    vector<Tensor> result;
    if (is_continuous) {
        result = continuous_logprob_entropy_cpp(
            logits.reshape({batch, -1}), logstd.reshape({batch, -1}),
            actions.reshape({batch, -1}));
    } else {
        Tensor act_sizes_cpu = act_sizes.to(torch::kCPU).to(torch::kInt64);
        result = discrete_logprob_entropy_cpp(
            logits.reshape({batch, -1}), actions, act_sizes_cpu, num_heads);
    }
    Tensor ratio = (result[0].reshape({segments, horizon}) - old_logprobs).exp();
    ratio_out.copy_(ratio, false);
    newvalue_out.copy_(newvalue.squeeze(-1), false);

    return {ppo_loss_cpp(ratio, advantages, prio,
        newvalue, values, returns, result[1],
        clip_coef, vf_clip_coef, vf_coef, ent_coef, losses)};
}

// Fast clip_grad_norm_ for contiguous grad buffer — no fp32 copy, no scalar allocs
void clip_grad_norm_(Tensor grad_buffer, double max_norm) {
    if (!grad_buffer.defined() || grad_buffer.numel() == 0) return;
    Tensor total_norm = grad_buffer.norm(2);
    Tensor clip_coef = torch::clamp_max(max_norm / (total_norm + 1e-6), 1.0);
    grad_buffer.mul_(clip_coef);
}

float cosine_annealing(float lr_base, float lr_min, int t, int T) {
    if (T == 0) return lr_base;  // avoid division by zero
    float ratio = (float)t / (float)T;
    ratio = std::max(0.0f, std::min(1.0f, ratio));  // clamp to [0, 1]
    return lr_min + 0.5f*(lr_base - lr_min)*(1.0f + std::cos(M_PI * ratio));
}

// Reference implementation for testing
Tensor fc_relu_fc_max_cpp(
    Tensor x,      // (B, N, D_in)
    Tensor W1,     // (D_mid, D_in)
    Tensor b1,     // (D_mid)
    Tensor W2,     // (D_out, D_mid)
    Tensor b2      // (D_out)
) {
    // FC1: x @ W1.T + b1 -> (B, N, D_mid)
    auto fc1 = torch::addmm(b1, x.flatten(0, 1), W1.t()).view({x.size(0), x.size(1), -1});
    // ReLU
    auto relu_out = torch::relu(fc1);
    // FC2: relu_out @ W2.T + b2 -> (B, N, D_out)
    auto fc2 = torch::addmm(b2, relu_out.flatten(0, 1), W2.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc2.max(1));
}

// Reference implementation for testing
Tensor fc_max_cpp(Tensor x, Tensor W, Tensor b) {
    // FC: x @ W.T + b -> (B, N, D_out)
    auto fc = torch::addmm(b, x.flatten(0, 1), W.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc.max(1));
}
