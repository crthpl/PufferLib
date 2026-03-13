// Uses vector for MinGRU activations

#ifndef PUFFERLIB_MODELS_CU
#define PUFFERLIB_MODELS_CU

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

#include <stdio.h>
#include <stdlib.h>

using std::vector;

#include "kernels.cu"

// Shared function pointer types (same signature for encoder and decoder)
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc, int esz);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self, int esz);
typedef void  (*free_weights_fn)(void* weights);
typedef PufTensor (*forward_fn)(void* weights, void* activations, PufTensor input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    PufTensor grad, cudaStream_t stream);
typedef PufTensor (*decoder_backward_fn)(void* weights, void* activations,
    PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value, cudaStream_t stream);
typedef PufTensor (*network_forward_fn)(void* weights, PufTensor x,
    PufTensor state, void* activations, cudaStream_t stream);
typedef PufTensor (*network_forward_train_fn)(void* weights, PufTensor x,
    PufTensor state, void* activations, cudaStream_t stream);
typedef PufTensor (*network_backward_fn)(void* weights,
    PufTensor grad, void* activations, cudaStream_t stream);

struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    int in_dim, out_dim;
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    int hidden_dim, output_dim;
    bool continuous;
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    int hidden, num_layers, horizon;
};


struct EncoderWeights { PufTensor weight; int in_dim, out_dim; };
struct EncoderActivations { PufTensor out, saved_input, wgrad_scratch; };

static PufTensor encoder_forward(void* w, void* activations, PufTensor input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    if (a->saved_input.bytes) puf_copy(a->saved_input, input, stream);
    puf_mm(input, ew->weight, a->out, stream);
    return a->out;
}

static void encoder_backward(void* w, void* activations, PufTensor grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    puf_mm_tn(grad, a->saved_input, a->wgrad_scratch, stream);
}

static void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PufTensor wt = {
        .bytes = ew->weight.bytes,
        .shape = {ew->out_dim, ew->in_dim},
        .dtype_size = ew->weight.dtype_size
    };
    puf_kaiming_init(wt, std::sqrt(2.0f), (*seed)++, stream);
}

static void encoder_reg_params(void* w, Allocator* alloc, int esz) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}, .dtype_size = esz};
    alloc_register(alloc,&ew->weight);
}

static void encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    int p = PRECISION_SIZE;
    *a = (EncoderActivations){
        .out = {.shape = {B_TT, ew->out_dim}, .dtype_size = p},
        .saved_input = {.shape = {B_TT, ew->in_dim}, .dtype_size = p},
        .wgrad_scratch = {.shape = {ew->out_dim, ew->in_dim}, .dtype_size = p},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(grads,&a->wgrad_scratch);
}

static void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}, .dtype_size = PRECISION_SIZE};
    alloc_register(alloc,&a->out);
}

static void* encoder_create_weights(void* self, int esz) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}
static void encoder_free_weights(void* weights) { free(weights); }

#include "ocean.cu"

struct DecoderWeights { PufTensor weight, logstd; int hidden_dim, output_dim; bool continuous; };
struct DecoderActivations { PufTensor out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch; };

static PufTensor decoder_forward(void* w, void* activations, PufTensor input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->saved_input.bytes) puf_copy(a->saved_input, input, stream);
    puf_mm(input, dw->weight, a->out, stream);
    return a->out;
}

static void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    PufTensor wt = {
        .bytes = dw->weight.bytes,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
        .dtype_size = dw->weight.dtype_size
    };
    puf_kaiming_init(wt, 0.01f, (*seed)++, stream);
}

static void decoder_reg_params(void* w, Allocator* alloc, int esz) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}, .dtype_size = esz};
    alloc_register(alloc,&dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}, .dtype_size = esz};
        alloc_register(alloc,&dw->logstd);
    }
}

static void decoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int p = PRECISION_SIZE;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out = {.shape = {B_TT, od1}, .dtype_size = p},
        .grad_out = {.shape = {B_TT, od1}, .dtype_size = p},
        .saved_input = {.shape = {B_TT, dw->hidden_dim}, .dtype_size = p},
        .grad_input = {.shape = {B_TT, dw->hidden_dim}, .dtype_size = p},
        .wgrad_scratch = {.shape = {od1, dw->hidden_dim}, .dtype_size = p},
        .logstd_scratch = {.shape = {1, dw->output_dim}, .dtype_size = p},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(acts,&a->grad_out);
    alloc_register(acts,&a->grad_input);
    alloc_register(grads,&a->wgrad_scratch);
    if (dw->continuous) alloc_register(grads,&a->logstd_scratch);
}

static void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}, .dtype_size = PRECISION_SIZE};
    alloc_register(alloc,&a->out);
}

static void* decoder_create_weights(void* self, int esz) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim; dw->output_dim = d->output_dim; dw->continuous = d->continuous;
    return dw;
}
static void decoder_free_weights(void* weights) { free(weights); }

static PufTensor decoder_backward(void* w, void* activations,
    PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad_kernel<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->grad_out.bytes, (const float*)grad_logits.bytes,
        (const float*)grad_value.bytes, B_TT, od, od1);
    puf_mm_tn(a->grad_out, a->saved_input, a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.bytes != nullptr) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)a->logstd_scratch.bytes, (const float*)grad_logstd.bytes,
            B_TT, dw->output_dim);
    }
    puf_mm_nn(a->grad_out, dw->weight, a->grad_input, stream);
    return a->grad_input;
}

struct MinGRUActivations {
    int num_layers;
    // Rollout
    vector<PufTensor> combined;        // per-layer (B_inf, 3*H)
    PufTensor out;                     // (B_inf, H)
    PufTensor next_state;              // (B_inf, H)
    // Training
    vector<PufTensor> saved_inputs;    // per-layer (B, TT, H)
    vector<PrefixScan> scan_bufs;      // per-layer scan state
    vector<PufTensor> combined_bufs;   // per-layer (B_TT, 3*H)
    vector<PufTensor> wgrad_scratch;   // per-layer (3*H, H) bf16 weight grad output
    PufTensor grad_input_buf;          // (B_TT, H)
    PufTensor grad_next_state;         // (B, 1, H)
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    PufTensor* weights;  // [num_layers], malloc'd
};

static PufTensor mingru_state_layer(MinGRUWeights* m, PufTensor& state, int i) {
    long B = state.shape[1], H = state.shape[2];
    return {
        .bytes = state.bytes + i * B * H * state.dtype_size,
        .shape = {B, H},
        .dtype_size = state.dtype_size
    };
}

static void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        PufTensor w2d = {
            .bytes = m->weights[i].bytes,
            .shape = {3 * m->hidden, m->hidden},
            .dtype_size = m->weights[i].dtype_size
        };
        puf_kaiming_init(w2d, 1.0f, (*seed)++, stream);
    }
}

static void mingru_reg_params(void* w, Allocator* alloc, int esz) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}, .dtype_size = esz};
        alloc_register(alloc,&m->weights[i]);
    }
}

static void mingru_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT, p = PRECISION_SIZE;
    int f = sizeof(float);
    a->num_layers = m->num_layers;
    a->saved_inputs.resize(m->num_layers);
    a->scan_bufs.resize(m->num_layers);
    a->combined_bufs.resize(m->num_layers);
    a->wgrad_scratch.resize(m->num_layers);
    a->grad_input_buf = {.shape = {B_TT, H}, .dtype_size = p};
    a->grad_next_state = {.shape = {B, 1, H}, .dtype_size = p};
    alloc_register(acts,&a->grad_input_buf);
    alloc_register(acts,&a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {.B = B, .T = TT, .H = H,
            .a_star = {.shape = {B, TT + 1, H}, .dtype_size = f},
            .s_vals = {.shape = {B, TT + 1, H}, .dtype_size = f},
            .log_values_buf = {.shape = {B, TT + 1, H}, .dtype_size = f},
            .out = {.shape = {B, TT, H}, .dtype_size = p},
            .next_state = {.shape = {B, 1, H}, .dtype_size = p},
            .grad_combined = {.shape = {B, TT, 3 * H}, .dtype_size = p},
            .grad_state = {.shape = {B, 1, H}, .dtype_size = p},
            .grad_input = {.shape = {B, TT, H}, .dtype_size = p},
        };
        a->saved_inputs[i] = {.shape = {B, TT, H}, .dtype_size = p};
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}, .dtype_size = p};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}, .dtype_size = p};
        alloc_register(acts,&a->saved_inputs[i]);
        alloc_register(acts,&a->combined_bufs[i]);
        alloc_register(acts,&a->scan_bufs[i].out);
        alloc_register(acts,&a->scan_bufs[i].next_state);
        alloc_register(acts,&a->scan_bufs[i].a_star);
        alloc_register(acts,&a->scan_bufs[i].s_vals);
        alloc_register(acts,&a->scan_bufs[i].log_values_buf);
        alloc_register(acts,&a->scan_bufs[i].grad_combined);
        alloc_register(acts,&a->scan_bufs[i].grad_state);
        alloc_register(acts,&a->scan_bufs[i].grad_input);
        alloc_register(grads,&a->wgrad_scratch[i]);
    }
}

static void mingru_reg_rollout(void* weights, void* activations, Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden, p = PRECISION_SIZE;
    a->num_layers = w->num_layers;
    a->combined.resize(w->num_layers);
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}, .dtype_size = p};
        alloc_register(alloc,&a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}, .dtype_size = p};
    a->next_state = {.shape = {B_inf, H}, .dtype_size = p};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->next_state);
}

static void* mingru_create_weights(void* self, int esz) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden; mw->num_layers = n->num_layers; mw->horizon = n->horizon;
    mw->weights = (PufTensor*)calloc(n->num_layers, sizeof(PufTensor));
    return mw;
}
static void mingru_free_weights(void* weights) {
    MinGRUWeights* mw = (MinGRUWeights*)weights;
    free(mw->weights);
    free(mw);
}

static PufTensor mingru_forward(void* w, PufTensor x, PufTensor state, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        PufTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(x, m->weights[i], a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)a->out.bytes, (precision_t*)a->next_state.bytes,
            (const precision_t*)a->combined[i].bytes, (const precision_t*)state_i.bytes,
            (const precision_t*)x.bytes, H, B);
        puf_copy(state_i, a->next_state, stream);
        x = a->out;
    }
    return x;
}

static PufTensor mingru_forward_train(void* w, PufTensor x, PufTensor state, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = x.shape[0];
    int TT = x.shape[1];
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(a->saved_inputs[i], x, stream);
        PufTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(x, m->weights[i], a->combined_bufs[i], stream);
        a->scan_bufs[i].combined_ptr = a->combined_bufs[i].bytes;
        a->scan_bufs[i].state_ptr = state_i.bytes;
        a->scan_bufs[i].input_ptr = a->saved_inputs[i].bytes;
        fused_scan_forward<<<grid_size(B*m->hidden), BLOCK_SIZE, 0, stream>>>(a->scan_bufs[i]);
        x = a->scan_bufs[i].out;
    }
    return x;
}

static PufTensor mingru_backward(void* w, PufTensor grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        PrefixScan& scan = a->scan_bufs[i];
        fused_scan_backward<<<grid_size(scan.B*scan.H), BLOCK_SIZE, 0, stream>>>(
            scan, (const precision_t*)grad.bytes, (const precision_t*)a->grad_next_state.bytes);
        puf_mm_tn(scan.grad_combined, a->saved_inputs[i], a->wgrad_scratch[i], stream);
        puf_mm_nn(scan.grad_combined, m->weights[i], a->grad_input_buf, stream);
        // Add highway gate gradient: grad_input += grad_out * (1 - sigmoid(proj))
        int n = scan.grad_input.numel();
        add_precision_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            (precision_t*)a->grad_input_buf.bytes, (const precision_t*)scan.grad_input.bytes, n);
        grad = a->grad_input_buf;
    }
    return grad;
}

struct Policy {
    Encoder encoder;
    Decoder decoder;
    Network network;
    int input_dim, hidden_dim, output_dim;
    int num_atns;
};

struct PolicyActivations { void* encoder; void* decoder; void* network; };
struct PolicyWeights { void* encoder; void* decoder; void* network; };

static void policy_activations_free(PolicyActivations& a) {
    free(a.encoder);
    free(a.decoder);
    ((MinGRUActivations*)a.network)->~MinGRUActivations();
    free(a.network);
}

PufTensor policy_forward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PufTensor obs, PufTensor state, cudaStream_t stream) {
    PufTensor enc_out = p->encoder.forward(w.encoder, activations.encoder, obs, stream);
    PufTensor h = p->network.forward(w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

PufTensor policy_forward_train(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PufTensor x, PufTensor state, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    PufTensor h = p->encoder.forward(w.encoder, activations.encoder, x.squeeze(0), stream);
    h = p->network.forward_train(w.network, h.unsqueeze(0, B, TT), state, activations.network, stream);
    PufTensor dec_out = p->decoder.forward(w.decoder, activations.decoder, h.squeeze(0), stream);
    return dec_out.unsqueeze(0, B, TT);
}

void policy_backward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PufTensor grad_logits, PufTensor grad_logstd, PufTensor grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    PufTensor grad_h = p->decoder.backward(w.decoder, activations.decoder,
        grad_logits.squeeze(0), grad_logstd, grad_value.squeeze(0), stream);
    grad_h = p->network.backward(w.network, grad_h.unsqueeze(0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
}

PolicyActivations policy_reg_train(Policy* p, PolicyWeights& w, Allocator* acts, Allocator* grads, int B_TT) {
    PolicyActivations a;
    a.encoder = alloc_encoder_activations(p->encoder);
    a.decoder = calloc(1, sizeof(DecoderActivations));
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

PolicyActivations policy_reg_rollout(Policy* p, PolicyWeights& w, Allocator* acts, int B_inf) {
    PolicyActivations a;
    a.encoder = alloc_encoder_activations(p->encoder);
    a.decoder = calloc(1, sizeof(DecoderActivations));
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void policy_init_weights(Policy* p, PolicyWeights& w, uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

PolicyWeights policy_weights_create(Policy* p, int esz, Allocator* params) {
    PolicyWeights w;
    w.encoder = p->encoder.create_weights(&p->encoder, esz);
    w.decoder = p->decoder.create_weights(&p->decoder, esz);
    w.network = p->network.create_weights(&p->network, esz);
    p->encoder.reg_params(w.encoder, params, esz);
    p->decoder.reg_params(w.decoder, params, esz);
    p->network.reg_params(w.network, params, esz);
    return w;
}

void policy_weights_free(Policy* p, PolicyWeights* w) {
    p->encoder.free_weights(w->encoder);
    p->decoder.free_weights(w->decoder);
    p->network.free_weights(w->network);
}

#include "muon.cu"

#endif // PUFFERLIB_MODELS_CU
