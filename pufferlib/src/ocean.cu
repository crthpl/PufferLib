// NMMO3 CUDA encoder: multihot, cuDNN conv, embedding, concat, projection
// Included by models.cu — requires precision_t, PufTensor, Allocator, puf_mm, etc.

#include "cudnn_conv2d.cu"

// ---- NMMO3 constants ----

static constexpr int N3_MAP_H = 11, N3_MAP_W = 15, N3_NFEAT = 10;
static constexpr int N3_MULTIHOT = 59;
static constexpr int N3_MAP_SIZE = N3_MAP_H * N3_MAP_W * N3_NFEAT;
static constexpr int N3_PLAYER = 47, N3_REWARD = 10;
static constexpr int N3_EMBED_DIM = 32, N3_EMBED_VOCAB = 128;
static constexpr int N3_PLAYER_EMBED = N3_PLAYER * N3_EMBED_DIM;
static constexpr int N3_C1_IC = 59, N3_C1_OC = 128, N3_C1_K = 5, N3_C1_S = 3;
static constexpr int N3_C1_OH = 3, N3_C1_OW = 4;
static constexpr int N3_C2_IC = 128, N3_C2_OC = 128, N3_C2_K = 3, N3_C2_S = 1;
static constexpr int N3_C2_OH = 1, N3_C2_OW = 2;
static constexpr int N3_CONV_FLAT = N3_C2_OC * N3_C2_OH * N3_C2_OW;
static constexpr int N3_CONCAT = N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD;

__constant__ int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

static cudnnDataType_t n3_cudnn_dtype() {
    return (PRECISION_SIZE == 2) ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
}

// ---- NMMO3 kernels ----

__global__ void n3_multihot_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_MAP_H * N3_MAP_W) return;
    int b = idx / (N3_MAP_H * N3_MAP_W), rem = idx % (N3_MAP_H * N3_MAP_W);
    int h = rem / N3_MAP_W, w = rem % N3_MAP_W;
    const precision_t* src = obs + b * obs_size + (h * N3_MAP_W + w) * N3_NFEAT;
    precision_t* dst = out + b * N3_MULTIHOT * N3_MAP_H * N3_MAP_W;
    for (int f = 0; f < N3_NFEAT; f++)
        dst[(N3_OFFSETS[f] + (int)to_float(src[f])) * N3_MAP_H * N3_MAP_W + h * N3_MAP_W + w] = from_float(1.0f);
}

__global__ void n3_embedding_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs,
    const precision_t* __restrict__ embed_w, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER) return;
    int b = idx / N3_PLAYER, f = idx % N3_PLAYER;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    const precision_t* src = embed_w + val * N3_EMBED_DIM;
    precision_t* dst = out + b * N3_PLAYER_EMBED + f * N3_EMBED_DIM;
    for (int d = 0; d < N3_EMBED_DIM; d++) dst[d] = src[d];
}

__global__ void n3_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
    const precision_t* __restrict__ embed, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONCAT) return;
    int b = idx / N3_CONCAT, c = idx % N3_CONCAT;
    precision_t val;
    if (c < N3_CONV_FLAT) {
        int oc = c / (N3_C2_OH * N3_C2_OW), r = c % (N3_C2_OH * N3_C2_OW);
        int oh = r / N3_C2_OW, ow = r % N3_C2_OW;
        val = conv_flat[b * N3_CONV_FLAT + oc * N3_C2_OH * N3_C2_OW + oh * N3_C2_OW + ow];
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED)
        val = embed[b * N3_PLAYER_EMBED + (c - N3_CONV_FLAT)];
    else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER)
        val = obs[b * obs_size + N3_MAP_SIZE + (c - N3_CONV_FLAT - N3_PLAYER_EMBED)];
    else
        val = obs[b * obs_size + obs_size - N3_REWARD + (c - N3_CONV_FLAT - N3_PLAYER_EMBED - N3_PLAYER)];
    out[idx] = val;
}

__global__ void n3_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));
}

__global__ void n3_relu_backward_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (to_float(out[idx]) <= 0.0f) grad[idx] = from_float(0.0f);
}

__global__ void n3_bias_grad_kernel(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad, int N, int dim) {
    int d = blockIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += to_float(grad[i * dim + d]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[d] = from_float(sum);
    }
}

// NCHW bias grad: sum over (B, OH, OW) for each OC channel
__global__ void n3_conv_bias_grad_nchw(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad,
    int B, int OC, int spatial) {
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float sum = 0.0f;
    int total = B * spatial;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial, s = i % spatial;
        sum += to_float(grad[b * OC * spatial + oc * spatial + s]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[oc] = from_float(sum);
    }
}

__global__ void n3_concat_backward_conv_kernel(
    precision_t* __restrict__ conv_grad, const precision_t* __restrict__ concat_grad, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONV_FLAT) return;
    int b = idx / N3_CONV_FLAT, c = idx % N3_CONV_FLAT;
    conv_grad[b * N3_CONV_FLAT + c] = concat_grad[b * N3_CONCAT + c];
}

// ---- NMMO3 encoder structs ----

struct NMMO3EncoderWeights {
    ConvWeights conv1, conv2;
    PufTensor embed_w, proj_w, proj_b;
    int obs_size, hidden;
};

struct NMMO3EncoderActivations {
    ConvActivations conv1, conv2;
    PufTensor multihot, embed_out, concat, out, saved_obs;
    PufTensor embed_wgrad, proj_wgrad, proj_bgrad;
};

static NMMO3EncoderWeights* nmmo3_encoder_create(int obs_size, int hidden) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)calloc(1, sizeof(NMMO3EncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    conv_init(&ew->conv1, N3_C1_IC, N3_C1_OC, N3_C1_K, N3_C1_S, N3_MAP_H, N3_MAP_W, true);
    conv_init(&ew->conv2, N3_C2_IC, N3_C2_OC, N3_C2_K, N3_C2_S, N3_C1_OH, N3_C1_OW, false);
    return ew;
}

// ---- NMMO3 encoder interface ----

static PufTensor nmmo3_encoder_forward(void* w, void* activations, PufTensor input, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = input.shape[0];

    if (a->saved_obs.bytes) puf_copy(&a->saved_obs, &input, stream);

    cudaMemsetAsync(a->multihot.bytes, 0, (int64_t)B * N3_MULTIHOT * N3_MAP_H * N3_MAP_W * PRECISION_SIZE, stream);
    n3_multihot_kernel<<<grid_size(B * N3_MAP_H * N3_MAP_W), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->multihot.bytes, (precision_t*)input.bytes, B, ew->obs_size);

    conv_forward(&ew->conv1, &a->conv1, a->multihot.bytes, B, stream);
    conv_forward(&ew->conv2, &a->conv2, a->conv1.out.bytes, B, stream);

    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->embed_out.bytes, (precision_t*)input.bytes,
        (precision_t*)ew->embed_w.bytes, B, ew->obs_size);
    n3_concat_kernel<<<grid_size(B * N3_CONCAT), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->concat.bytes, (precision_t*)a->conv2.out.bytes,
        (precision_t*)a->embed_out.bytes, (precision_t*)input.bytes, B, ew->obs_size);

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->out.bytes, (precision_t*)ew->proj_b.bytes, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nmmo3_encoder_backward(void* w, void* activations, PufTensor grad, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden, p = PRECISION_SIZE;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)grad.bytes, (precision_t*)a->out.bytes, B * H);
    n3_bias_grad_kernel<<<H, 256, 0, stream>>>(
        (precision_t*)a->proj_bgrad.bytes, (precision_t*)grad.bytes, B, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);

    PufTensor grad_concat = {.bytes = a->concat.bytes, .shape = {B, N3_CONCAT}, .dtype_size = p};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    n3_concat_backward_conv_kernel<<<grid_size(B * N3_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->conv2.grad.bytes, (precision_t*)grad_concat.bytes, B);

    // Conv2 backward: relu mask + bias grad + cuDNN filter/data grad
    // conv2 has no relu, so skip relu backward
    n3_conv_bias_grad_nchw<<<ew->conv2.OC, 256, 0, stream>>>(
        (precision_t*)a->conv2.bgrad.bytes, (precision_t*)a->conv2.grad.bytes,
        B, ew->conv2.OC, ew->conv2.OH * ew->conv2.OW);
    conv_backward(&ew->conv2, &a->conv2, a->conv1.grad.bytes, stream);

    // Conv1 backward: relu mask + bias grad + cuDNN filter grad (no data grad)
    // Conv2's data grad lands in conv1.grad — that's conv1's upstream grad
    n3_relu_backward_kernel<<<grid_size(B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->conv1.grad.bytes, (precision_t*)a->conv1.out.bytes,
        B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW);
    n3_conv_bias_grad_nchw<<<ew->conv1.OC, 256, 0, stream>>>(
        (precision_t*)a->conv1.bgrad.bytes, (precision_t*)a->conv1.grad.bytes,
        B, ew->conv1.OC, ew->conv1.OH * ew->conv1.OW);
    conv_backward(&ew->conv1, &a->conv1, NULL, stream);

    cudaMemsetAsync(a->embed_wgrad.bytes, 0, a->embed_wgrad.numel() * p, stream);
}

static void nmmo3_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_init_weights(&ew->conv1, seed, stream);
    conv_init_weights(&ew->conv2, seed, stream);
    auto init2d = [&](PufTensor& t, int rows, int cols, float gain) {
        PufTensor wt = {.bytes = t.bytes, .shape = {rows, cols}, .dtype_size = t.dtype_size};
        puf_kaiming_init(&wt, gain, (*seed)++, stream);
    };
    init2d(ew->embed_w, N3_EMBED_VOCAB, N3_EMBED_DIM, 1.0f);
    init2d(ew->proj_w, ew->hidden, N3_CONCAT, std::sqrt(2.0f));
    cudaMemsetAsync(ew->proj_b.bytes, 0, ew->proj_b.numel() * ew->proj_b.dtype_size, stream);
}

static void nmmo3_encoder_reg_params(void* w, Allocator* alloc, int esz) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_reg_params(&ew->conv1, alloc, esz);
    conv_reg_params(&ew->conv2, alloc, esz);
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}, .dtype_size = esz};
    ew->proj_w  = {.shape = {ew->hidden, N3_CONCAT}, .dtype_size = esz};
    ew->proj_b  = {.shape = {ew->hidden}, .dtype_size = esz};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
}

static void nmmo3_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int p = PRECISION_SIZE;
    *a = {};
    a->multihot = {.shape = {B_TT, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}, .dtype_size = p};
    alloc_register(acts,&a->multihot);
    conv_reg_train(&ew->conv1, &a->conv1, acts, grads, B_TT, n3_cudnn_dtype());
    conv_reg_train(&ew->conv2, &a->conv2, acts, grads, B_TT, n3_cudnn_dtype());
    a->embed_out = {.shape = {B_TT, N3_PLAYER_EMBED}, .dtype_size = p};
    a->concat    = {.shape = {B_TT, N3_CONCAT}, .dtype_size = p};
    a->out       = {.shape = {B_TT, ew->hidden}, .dtype_size = p};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}, .dtype_size = p};
    alloc_register(acts,&a->embed_out); alloc_register(acts,&a->concat);
    alloc_register(acts,&a->out);       alloc_register(acts,&a->saved_obs);
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}, .dtype_size = p};
    a->proj_wgrad  = {.shape = {ew->hidden, N3_CONCAT}, .dtype_size = p};
    a->proj_bgrad  = {.shape = {ew->hidden}, .dtype_size = p};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
}

static void nmmo3_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int p = PRECISION_SIZE;
    a->multihot = {.shape = {B, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}, .dtype_size = p};
    alloc_register(alloc,&a->multihot);
    conv_reg_rollout(&ew->conv1, &a->conv1, alloc, B, n3_cudnn_dtype());
    conv_reg_rollout(&ew->conv2, &a->conv2, alloc, B, n3_cudnn_dtype());
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}, .dtype_size = p};
    a->concat    = {.shape = {B, N3_CONCAT}, .dtype_size = p};
    a->out       = {.shape = {B, ew->hidden}, .dtype_size = p};
    alloc_register(alloc,&a->embed_out); alloc_register(alloc,&a->concat); alloc_register(alloc,&a->out);
}

static void* nmmo3_encoder_create_weights(void* self, int esz) {
    Encoder* e = (Encoder*)self;
    return nmmo3_encoder_create(e->in_dim, e->out_dim);
}
static void nmmo3_encoder_free_weights(void* weights) { free(weights); }

// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    if (env_name == "puffer_nmmo3") {
        *enc = Encoder{
            .forward = nmmo3_encoder_forward,
            .backward = nmmo3_encoder_backward,
            .init_weights = nmmo3_encoder_init_weights,
            .reg_params = nmmo3_encoder_reg_params,
            .reg_train = nmmo3_encoder_reg_train,
            .reg_rollout = nmmo3_encoder_reg_rollout,
            .create_weights = nmmo3_encoder_create_weights,
            .free_weights = nmmo3_encoder_free_weights,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        };
    }
}

static void* alloc_encoder_activations(const Encoder& enc) {
    return (enc.forward == nmmo3_encoder_forward)
        ? calloc(1, sizeof(NMMO3EncoderActivations))
        : calloc(1, sizeof(EncoderActivations));
}
