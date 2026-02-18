// models.cu - Native Policy structs (torch-free internals, PufTensor only)
// Included by pufferlib.cpp inside namespace pufferlib

using std::tuple;
using std::vector;

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
constexpr bool USE_BF16 = false;
constexpr int PRECISION_SIZE = 4;   // bytes per element
#else
constexpr bool USE_BF16 = true;
constexpr int PRECISION_SIZE = 2;   // bytes per element
#endif

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
        int psz = PRECISION_SIZE;
        alloc.register_puf(&act.saved_input, {B_TT, in_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, out_dim}, psz);
        alloc.register_puf(&act.wgrad_scratch, {out_dim, in_dim}, psz);
    }

    void register_inference(Allocator& alloc, EncoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, out_dim}, PRECISION_SIZE);
    }

    void init_weights(uint64_t& seed, cudaStream_t stream) {
        PufTensor w2d;
        w2d.data = weight.data; w2d.shape[0] = out_dim; w2d.shape[1] = in_dim;
        w2d.ndim = 2; w2d.numel = weight.numel; w2d.dtype_size = weight.dtype_size;
        puf_orthogonal_init(w2d, std::sqrt(2.0f), seed++, stream);
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
        int psz = PRECISION_SIZE;
        alloc.register_puf(&act.saved_input, {B_TT, hidden_dim}, psz);
        alloc.register_puf(&act.out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.grad_out, {B_TT, output_dim + 1}, psz);
        alloc.register_puf(&act.wgrad_scratch, {output_dim + 1, hidden_dim}, psz);
    }

    void register_inference(Allocator& alloc, DecoderActivations& act, int B_inf) {
        alloc.register_puf(&act.inf_out, {B_inf, output_dim + 1}, PRECISION_SIZE);
    }

    void init_weights(uint64_t& seed, cudaStream_t stream) {
        PufTensor w2d;
        w2d.data = weight.data; w2d.shape[0] = output_dim + 1; w2d.shape[1] = hidden_dim;
        w2d.ndim = 2; w2d.numel = weight.numel; w2d.dtype_size = weight.dtype_size;
        puf_orthogonal_init(w2d, 0.01f, seed++, stream);
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
        int psz = PRECISION_SIZE;

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
        int dsz = PRECISION_SIZE;
        for (int i = 0; i < num_layers; i++) {
            alloc.register_puf(&act.inf_combined[i], {B_inf, 3 * H}, dsz);
        }
        alloc.register_puf(&act.inf_out, {B_inf, H}, dsz);
        alloc.register_puf(&act.inf_next_state, {B_inf, H}, dsz);
    }

    void init_weights(uint64_t& seed, cudaStream_t stream) {
        for (int i = 0; i < num_layers; i++) {
            PufTensor w2d;
            w2d.data = weights[i].data;
            w2d.shape[0] = 3 * hidden; w2d.shape[1] = hidden;
            w2d.ndim = 2; w2d.numel = weights[i].numel; w2d.dtype_size = weights[i].dtype_size;
            puf_orthogonal_init(w2d, 1.0f, seed++, stream);
        }
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
    PufTensor forward(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
        for (int i = 0; i < num_layers; i++) {
            PufTensor state_i = state_layer(state, i);
            puf_mm(x, weights[i], act.inf_combined[i], stream);
            mingru_gate(state_i, act.inf_combined[i], act.inf_out, act.inf_next_state, stream);
            puf_copy(state_i, act.inf_next_state, stream);
            x = act.inf_out;
        }
        return x;
    }

    // Training forward: x (B, TT, H) PufTensor, state PufTensor -> (B, TT, H) PufTensor
    PufTensor forward_train(PufTensor x, PufTensor state, MinGRUActivations& act, cudaStream_t stream) {
        int B = x.size(0);
        int TT = x.size(1);

        for (int i = 0; i < num_layers; i++) {
            puf_copy(act.saved_inputs[i], x, stream);
            PufTensor state_i = state_layer(state, i);

            // Reshape state_i (B, H) -> (B, 1, H) for prefix_scan
            PufTensor state_3d = state_i;
            state_3d.shape[0] = B; state_3d.shape[1] = 1; state_3d.shape[2] = hidden;
            state_3d.ndim = 3;

            // Flatten x from (B, TT, H) to (B*TT, H) for mm
            PufTensor x_flat = x;
            x_flat.shape[0] = B * TT; x_flat.shape[1] = hidden; x_flat.ndim = 2;
            puf_mm(x_flat, weights[i], act.combined_bufs[i], stream);

            // Reinterpret (B*TT, 3*H) as (B, TT, 3*H) for scan
            PufTensor combined_3d = act.combined_bufs[i];
            combined_3d.shape[0] = B; combined_3d.shape[1] = TT; combined_3d.shape[2] = 3 * hidden;
            combined_3d.ndim = 3;
            prefix_scan_forward(combined_3d, state_3d, act.scan_bufs[i], stream);
            x = act.scan_bufs[i].out;
        }
        return x;
    }

    // Backward: grad (B, TT, H) PufTensor -> grad_input (B, TT, H) PufTensor
    PufTensor backward(PufTensor grad, MinGRUActivations& act, MinGRU* target, cudaStream_t stream) {
        int B = grad.size(0);
        int TT = grad.size(1);
        int H = grad.size(2);
        for (int i = num_layers - 1; i >= 0; i--) {
            prefix_scan_backward(grad, act.grad_next_state, act.scan_bufs[i], stream);

            // Reinterpret grad_combined (B, TT, 3*H) as (B*TT, 3*H) for matmuls
            PufTensor gc_flat = act.scan_bufs[i].grad_combined;
            gc_flat.shape[0] = B * TT; gc_flat.shape[1] = 3 * H; gc_flat.ndim = 2;

            // Reinterpret saved_inputs (B, TT, H) as (B*TT, H)
            PufTensor inp_flat = act.saved_inputs[i];
            inp_flat.shape[0] = B * TT; inp_flat.shape[1] = H; inp_flat.ndim = 2;

            // Weight grad: gc_flat^T @ inp_flat → wgrad_scratch, then accumulate
            puf_mm_tn(gc_flat, inp_flat, act.wgrad_scratch, stream);
            puf_add(target->weight_grads[i], act.wgrad_scratch, stream);

            // Input grad: gc_flat @ weights[i] → grad_input_buf
            puf_mm_nn(gc_flat, weights[i], act.grad_input_buf, stream);

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

    void init_weights(cudaStream_t stream, uint64_t seed = 42) {
        encoder.init_weights(seed, stream);
        decoder.init_weights(seed, stream);
        rnn.init_weights(seed, stream);
    }

    // Inference: obs (B, input) PufTensor, state PufTensor -> dec_out (B, output+1) PufTensor
    // State is modified in-place
    PufTensor forward(PufTensor obs, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
        puf_mm(obs, encoder.weight, a.enc.inf_out, stream);
        PufTensor h = rnn.forward(a.enc.inf_out, state, a.rnn, stream);
        puf_mm(h, decoder.weight, a.dec.inf_out, stream);
        return a.dec.inf_out;
    }

    // Training forward: x (B, TT, input) PufTensor, state PufTensor
    // Returns fused decoder output (B, TT, output+1) PufTensor
    PufTensor forward_train(PufTensor x, PufTensor state, PolicyActivations& a, cudaStream_t stream) {
        int B = x.size(0);
        int TT = x.size(1);

        // Flatten to (B*TT, input)
        PufTensor x_flat = x;
        x_flat.shape[0] = B * TT; x_flat.shape[1] = encoder.in_dim; x_flat.ndim = 2;

        puf_copy(a.enc.saved_input, x_flat, stream);
        puf_mm(a.enc.saved_input, encoder.weight, a.enc.out, stream);

        // Reshape enc output to (B, TT, H)
        PufTensor h = a.enc.out;
        h.shape[0] = B; h.shape[1] = TT; h.shape[2] = encoder.out_dim; h.ndim = 3;

        h = rnn.forward_train(h, state, a.rnn, stream);

        // Flatten for decoder mm
        PufTensor flat_h = h;
        flat_h.shape[0] = B * TT; flat_h.shape[1] = encoder.out_dim; flat_h.ndim = 2;

        puf_copy(a.dec.saved_input, flat_h, stream);
        puf_mm(flat_h, decoder.weight, a.dec.out, stream);

        // Reshape to (B, TT, output+1)
        PufTensor result = a.dec.out;
        result.shape[0] = B; result.shape[1] = TT; result.shape[2] = decoder.output_dim + 1; result.ndim = 3;
        return result;
    }

    // Backward: all PufTensor grads (fp32)
    void backward(PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value,
                  PolicyActivations& a, Policy* target, cudaStream_t stream) {
        int B_TT = a.dec.saved_input.size(0);
        int B = grad_logits.size(0);
        int TT = grad_logits.size(1);

        // Assemble fused grad_out (B_TT, output+1) from separate fp32 grads → bf16
        PufTensor gl_flat = grad_logits;
        gl_flat.shape[0] = B_TT; gl_flat.shape[1] = decoder.output_dim; gl_flat.ndim = 2;
        PufTensor gv_flat = grad_value;
        gv_flat.shape[0] = B_TT; gv_flat.ndim = 1;
        puf_assemble_decoder_grad(a.dec.grad_out, gl_flat, gv_flat, stream);

        // Decoder weight grad: bf16 matmul into scratch, then accumulate into fp32 grad
        puf_mm_tn(a.dec.grad_out, a.dec.saved_input, a.dec.wgrad_scratch, stream);
        puf_add(target->decoder.weight_grad, a.dec.wgrad_scratch, stream);

        // logstd grad: column-wise sum reduction into fp32 master grad
        if (decoder.continuous && grad_logstd.data != nullptr) {
            PufTensor gls_flat = grad_logstd;
            gls_flat.shape[0] = B_TT; gls_flat.shape[1] = decoder.output_dim; gls_flat.ndim = 2;
            puf_sum_rows_add(target->decoder.logstd_grad, gls_flat, stream);
        }

        // Decoder input grad: mm into enc.out (same shape: B_TT x hidden, reused buffer)
        puf_mm_nn(a.dec.grad_out, decoder.weight, a.enc.out, stream);

        // Reshape to (B, TT, H) for RNN backward
        PufTensor grad_h = a.enc.out;
        grad_h.shape[0] = B; grad_h.shape[1] = TT; grad_h.shape[2] = encoder.out_dim; grad_h.ndim = 3;

        // RNN backward
        grad_h = rnn.backward(grad_h, a.rnn, &target->rnn, stream);

        // Flatten for encoder weight grad
        PufTensor grad_enc = grad_h;
        grad_enc.shape[0] = B_TT; grad_enc.shape[1] = encoder.out_dim; grad_enc.ndim = 2;
        puf_mm_tn(grad_enc, a.enc.saved_input, a.enc.wgrad_scratch, stream);
        puf_add(target->encoder.weight_grad, a.enc.wgrad_scratch, stream);
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
