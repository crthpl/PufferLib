// NMMO3 custom CUDA encoder: multihot scatter, im2col+GEMM conv, embedding, concat, projection
//
// Two compilation modes:
//   1. Included by models.cu (#include "ocean.cu") — uses precision_t, PufTensor, puf_mm
//   2. Standalone test build (nvcc -DOCEAN_STANDALONE) — float32 only, ctypes-callable

#ifdef OCEAN_STANDALONE
// ============================================================================
// Standalone test build — float32, uint8 obs, cuBLAS SGEMM
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)status); \
    } \
} while(0)

__constant__ int NMMO3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

static const int MAP_H = 11, MAP_W = 15, NUM_FEATURES = 10;
static const int MULTIHOT_DIM = 59, OBS_SIZE = 1707;
static const int MAP_SIZE = MAP_H * MAP_W * NUM_FEATURES;
static const int PLAYER_SIZE = 47, REWARD_SIZE = 10, EMBED_DIM = 32;
static const int PLAYER_EMBED_OUT = PLAYER_SIZE * EMBED_DIM;
static const int C1_IC = 59, C1_OC = 128, C1_K = 5, C1_S = 3, C1_OH = 3, C1_OW = 4;
static const int C2_IC = 128, C2_OC = 128, C2_K = 3, C2_S = 1, C2_OH = 1, C2_OW = 2;
static const int CONV_FLAT = C2_OC * C2_OH * C2_OW;
static const int CONCAT_DIM = CONV_FLAT + PLAYER_EMBED_OUT + PLAYER_SIZE + REWARD_SIZE;
static const int C1_K2 = C1_IC * C1_K * C1_K, C1_M = C1_OH * C1_OW;
static const int C2_K2 = C2_IC * C2_K * C2_K, C2_M = C2_OH * C2_OW;

__global__ void nmmo3_multihot_kernel(float* out, const unsigned char* obs, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * MAP_H * MAP_W) return;
    int b = idx / (MAP_H * MAP_W), rem = idx % (MAP_H * MAP_W);
    int h = rem / MAP_W, w = rem % MAP_W;
    const unsigned char* src = obs + b * OBS_SIZE + (h * MAP_W + w) * NUM_FEATURES;
    float* dst = out + b * MULTIHOT_DIM * MAP_H * MAP_W;
    for (int f = 0; f < NUM_FEATURES; f++)
        dst[(NMMO3_OFFSETS[f] + (int)src[f]) * MAP_H * MAP_W + h * MAP_W + w] = 1.0f;
}

__global__ void nmmo3_im2col_conv1(float* col, const float* in, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C1_M) return;
    int b = idx / C1_M, pos = idx % C1_M;
    int oh = pos / C1_OW, ow = pos % C1_OW;
    int h0 = oh * C1_S, w0 = ow * C1_S;
    float* dst = col + idx * C1_K2;
    const float* src = in + b * C1_IC * MAP_H * MAP_W;
    int k = 0;
    for (int ic = 0; ic < C1_IC; ic++)
        for (int kh = 0; kh < C1_K; kh++)
            for (int kw = 0; kw < C1_K; kw++)
                dst[k++] = src[ic * MAP_H * MAP_W + (h0 + kh) * MAP_W + (w0 + kw)];
}

__global__ void nmmo3_conv_bias_relu(float* data, const float* bias, int total, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = fmaxf(0.0f, data[idx] + bias[idx % OC]);
}

__global__ void nmmo3_im2col_conv2(float* col, const float* in, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C2_M) return;
    int b = idx / C2_M, pos = idx % C2_M;
    int oh = pos / C2_OW, ow = pos % C2_OW;
    int h0 = oh * C2_S, w0 = ow * C2_S;
    float* dst = col + idx * C2_K2;
    const float* src = in + b * C1_M * C2_IC;
    int k = 0;
    for (int ic = 0; ic < C2_IC; ic++)
        for (int kh = 0; kh < C2_K; kh++)
            for (int kw = 0; kw < C2_K; kw++)
                dst[k++] = src[((h0 + kh) * C1_OW + (w0 + kw)) * C2_IC + ic];
}

__global__ void nmmo3_conv_bias(float* data, const float* bias, int total, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] += bias[idx % OC];
}

__global__ void nmmo3_embedding_kernel(float* out, const unsigned char* obs, const float* embed_w, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * PLAYER_SIZE) return;
    int b = idx / PLAYER_SIZE, f = idx % PLAYER_SIZE;
    int val = (int)obs[b * OBS_SIZE + MAP_SIZE + f];
    const float* src = embed_w + val * EMBED_DIM;
    float* dst = out + b * PLAYER_EMBED_OUT + f * EMBED_DIM;
    for (int d = 0; d < EMBED_DIM; d++) dst[d] = src[d];
}

__global__ void nmmo3_concat_kernel(float* out, const float* conv_flat, const float* embed,
                                     const unsigned char* obs, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CONCAT_DIM) return;
    int b = idx / CONCAT_DIM, c = idx % CONCAT_DIM;
    float val;
    if (c < CONV_FLAT) {
        int oc = c / (C2_OH * C2_OW), rem = c % (C2_OH * C2_OW);
        int oh = rem / C2_OW, ow = rem % C2_OW;
        val = conv_flat[b * CONV_FLAT + (oh * C2_OW + ow) * C2_OC + oc];
    } else if (c < CONV_FLAT + PLAYER_EMBED_OUT)
        val = embed[b * PLAYER_EMBED_OUT + (c - CONV_FLAT)];
    else if (c < CONV_FLAT + PLAYER_EMBED_OUT + PLAYER_SIZE)
        val = (float)obs[b * OBS_SIZE + MAP_SIZE + (c - CONV_FLAT - PLAYER_EMBED_OUT)];
    else
        val = (float)obs[b * OBS_SIZE + OBS_SIZE - REWARD_SIZE + (c - CONV_FLAT - PLAYER_EMBED_OUT - PLAYER_SIZE)];
    out[idx] = val;
}

__global__ void nmmo3_bias_relu_kernel(float* data, const float* bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = fmaxf(0.0f, data[idx] + bias[idx % dim]);
}

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

static cublasHandle_t get_nmmo3_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) CHECK_CUBLAS(cublasCreate(&handle));
    return handle;
}

static void sgemm_nt(cublasHandle_t h, float* C, const float* A, const float* B,
                     int M, int N, int K, cudaStream_t s) {
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSetStream(h, s));
    CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

extern "C" {

int nmmo3_workspace_size(int B) {
    return B * (MULTIHOT_DIM * MAP_H * MAP_W + C1_M * C1_K2 + C1_M * C1_OC
              + C2_M * C2_K2 + C2_M * C2_OC + PLAYER_EMBED_OUT + CONCAT_DIM);
}

void nmmo3_encoder_forward(
    float* output, const unsigned char* obs,
    const float* conv1_w, const float* conv1_b,
    const float* conv2_w, const float* conv2_b,
    const float* embed_w, const float* proj_w, const float* proj_b,
    float* workspace, int B, cublasHandle_t handle, cudaStream_t stream)
{
    if (!handle) handle = get_nmmo3_cublas_handle();
    const int BLK = 256;
    float* multihot  = workspace;
    float* im2col1   = multihot  + B * MULTIHOT_DIM * MAP_H * MAP_W;
    float* conv1_out = im2col1   + B * C1_M * C1_K2;
    float* im2col2   = conv1_out + B * C1_M * C1_OC;
    float* conv2_out = im2col2   + B * C2_M * C2_K2;
    float* embed_out = conv2_out + B * C2_M * C2_OC;
    float* concat    = embed_out + B * PLAYER_EMBED_OUT;

    CHECK_CUDA(cudaMemsetAsync(multihot, 0, B * MULTIHOT_DIM * MAP_H * MAP_W * sizeof(float), stream));
    nmmo3_multihot_kernel<<<div_ceil(B * MAP_H * MAP_W, BLK), BLK, 0, stream>>>(multihot, obs, B);
    { int M = B * C1_M;
      nmmo3_im2col_conv1<<<div_ceil(M, BLK), BLK, 0, stream>>>(im2col1, multihot, B);
      sgemm_nt(handle, conv1_out, im2col1, conv1_w, M, C1_OC, C1_K2, stream);
      nmmo3_conv_bias_relu<<<div_ceil(M * C1_OC, BLK), BLK, 0, stream>>>(conv1_out, conv1_b, M * C1_OC, C1_OC);
    }
    { int M = B * C2_M;
      nmmo3_im2col_conv2<<<div_ceil(M, BLK), BLK, 0, stream>>>(im2col2, conv1_out, B);
      sgemm_nt(handle, conv2_out, im2col2, conv2_w, M, C2_OC, C2_K2, stream);
      nmmo3_conv_bias<<<div_ceil(M * C2_OC, BLK), BLK, 0, stream>>>(conv2_out, conv2_b, M * C2_OC, C2_OC);
    }
    nmmo3_embedding_kernel<<<div_ceil(B * PLAYER_SIZE, BLK), BLK, 0, stream>>>(embed_out, obs, embed_w, B);
    nmmo3_concat_kernel<<<div_ceil(B * CONCAT_DIM, BLK), BLK, 0, stream>>>(concat, conv2_out, embed_out, obs, B);
    sgemm_nt(handle, output, concat, proj_w, B, 512, CONCAT_DIM, stream);
    nmmo3_bias_relu_kernel<<<div_ceil(B * 512, BLK), BLK, 0, stream>>>(output, proj_b, B * 512, 512);
}

}  // extern "C"

#else
// ============================================================================
// Integrated build — included by models.cu, uses precision_t / PufTensor / puf_mm
// Requires: precision_t, to_float, from_float, PufTensor, Allocator, puf_mm,
//           puf_mm_tn, puf_copy, puf_kaiming_init, grid_size, BLOCK_SIZE,
//           PRECISION_SIZE, CHECK_CUDA (all from kernels.cu / models.cu)
// ============================================================================

// NMMO3 constants
static constexpr int N3_MAP_H = 11, N3_MAP_W = 15, N3_NFEAT = 10;
static constexpr int N3_MULTIHOT = 59;
static constexpr int N3_MAP_SIZE = N3_MAP_H * N3_MAP_W * N3_NFEAT;  // 1650
static constexpr int N3_PLAYER = 47, N3_REWARD = 10;
static constexpr int N3_EMBED_DIM = 32, N3_EMBED_VOCAB = 128;
static constexpr int N3_PLAYER_EMBED = N3_PLAYER * N3_EMBED_DIM;     // 1504
static constexpr int N3_C1_IC = 59, N3_C1_OC = 128, N3_C1_K = 5, N3_C1_S = 3;
static constexpr int N3_C1_OH = 3, N3_C1_OW = 4;
static constexpr int N3_C2_IC = 128, N3_C2_OC = 128, N3_C2_K = 3, N3_C2_S = 1;
static constexpr int N3_C2_OH = 1, N3_C2_OW = 2;
static constexpr int N3_CONV_FLAT = N3_C2_OC * N3_C2_OH * N3_C2_OW;  // 256
static constexpr int N3_CONCAT = N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD;  // 1817
static constexpr int N3_C1_K2 = N3_C1_IC * N3_C1_K * N3_C1_K;  // 1475
static constexpr int N3_C1_M  = N3_C1_OH * N3_C1_OW;            // 12
static constexpr int N3_C2_K2 = N3_C2_IC * N3_C2_K * N3_C2_K;  // 1152
static constexpr int N3_C2_M  = N3_C2_OH * N3_C2_OW;            // 2

__constant__ int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

// --- Forward kernels ---

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

__global__ void n3_im2col_conv1_kernel(
    precision_t* __restrict__ col, const precision_t* __restrict__ in, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_C1_M) return;
    int b = idx / N3_C1_M, pos = idx % N3_C1_M;
    int oh = pos / N3_C1_OW, ow = pos % N3_C1_OW;
    int h0 = oh * N3_C1_S, w0 = ow * N3_C1_S;
    precision_t* dst = col + idx * N3_C1_K2;
    const precision_t* src = in + b * N3_C1_IC * N3_MAP_H * N3_MAP_W;
    int k = 0;
    for (int ic = 0; ic < N3_C1_IC; ic++)
        for (int kh = 0; kh < N3_C1_K; kh++)
            for (int kw = 0; kw < N3_C1_K; kw++)
                dst[k++] = src[ic * N3_MAP_H * N3_MAP_W + (h0 + kh) * N3_MAP_W + (w0 + kw)];
}

__global__ void n3_im2col_conv2_kernel(
    precision_t* __restrict__ col, const precision_t* __restrict__ in, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_C2_M) return;
    int b = idx / N3_C2_M, pos = idx % N3_C2_M;
    int oh = pos / N3_C2_OW, ow = pos % N3_C2_OW;
    int h0 = oh * N3_C2_S, w0 = ow * N3_C2_S;
    precision_t* dst = col + idx * N3_C2_K2;
    // Input is NHWC from puf_mm: (B, C1_OH, C1_OW, C2_IC)
    const precision_t* src = in + b * N3_C1_M * N3_C2_IC;
    int k = 0;
    for (int ic = 0; ic < N3_C2_IC; ic++)
        for (int kh = 0; kh < N3_C2_K; kh++)
            for (int kw = 0; kw < N3_C2_K; kw++)
                dst[k++] = src[((h0 + kh) * N3_C1_OW + (w0 + kw)) * N3_C2_IC + ic];
}

__global__ void n3_conv_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % OC])));
}

__global__ void n3_conv_bias_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(to_float(data[idx]) + to_float(bias[idx % OC]));
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
        // NCHW index → read from NHWC layout
        int oc = c / (N3_C2_OH * N3_C2_OW), r = c % (N3_C2_OH * N3_C2_OW);
        int oh = r / N3_C2_OW, ow = r % N3_C2_OW;
        val = conv_flat[b * N3_CONV_FLAT + (oh * N3_C2_OW + ow) * N3_C2_OC + oc];
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

// --- Backward kernels ---

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

// --- Structs ---

struct NMMO3EncoderWeights {
    PufTensor conv1_w, conv1_b, conv2_w, conv2_b, embed_w, proj_w, proj_b;
    int obs_size, hidden;
};

struct NMMO3EncoderActivations {
    PufTensor multihot, im2col1, conv1_out, im2col2, conv2_out;
    PufTensor embed_out, concat, out;
    PufTensor saved_obs;
    PufTensor conv1_wgrad, conv1_bgrad, conv2_wgrad, conv2_bgrad;
    PufTensor embed_wgrad, proj_wgrad, proj_bgrad;
};

// --- Interface functions ---

static PufTensor nmmo3_encoder_forward(void* w, void* activations, PufTensor input, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = input.shape[0], obs_size = ew->obs_size, p = PRECISION_SIZE;
    precision_t* obs = (precision_t*)input.bytes;

    if (a->saved_obs.bytes) puf_copy(a->saved_obs, input, stream);

    CHECK_CUDA(cudaMemsetAsync(a->multihot.bytes, 0, (int64_t)B * N3_MULTIHOT * N3_MAP_H * N3_MAP_W * p, stream));
    n3_multihot_kernel<<<grid_size(B * N3_MAP_H * N3_MAP_W), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->multihot.bytes, obs, B, obs_size);

    // Conv1: im2col → puf_mm → bias+relu
    { int M = B * N3_C1_M;
      n3_im2col_conv1_kernel<<<grid_size(M), BLOCK_SIZE, 0, stream>>>(
          (precision_t*)a->im2col1.bytes, (precision_t*)a->multihot.bytes, B);
      PufTensor col = {.bytes = a->im2col1.bytes, .shape = {M, N3_C1_K2}, .dtype_size = p};
      PufTensor out = {.bytes = a->conv1_out.bytes, .shape = {M, N3_C1_OC}, .dtype_size = p};
      puf_mm(col, ew->conv1_w, out, stream);
      n3_conv_bias_relu_kernel<<<grid_size(M * N3_C1_OC), BLOCK_SIZE, 0, stream>>>(
          (precision_t*)a->conv1_out.bytes, (precision_t*)ew->conv1_b.bytes, M * N3_C1_OC, N3_C1_OC);
    }

    // Conv2: im2col → puf_mm → bias
    { int M = B * N3_C2_M;
      n3_im2col_conv2_kernel<<<grid_size(M), BLOCK_SIZE, 0, stream>>>(
          (precision_t*)a->im2col2.bytes, (precision_t*)a->conv1_out.bytes, B);
      PufTensor col = {.bytes = a->im2col2.bytes, .shape = {M, N3_C2_K2}, .dtype_size = p};
      PufTensor out = {.bytes = a->conv2_out.bytes, .shape = {M, N3_C2_OC}, .dtype_size = p};
      puf_mm(col, ew->conv2_w, out, stream);
      n3_conv_bias_kernel<<<grid_size(M * N3_C2_OC), BLOCK_SIZE, 0, stream>>>(
          (precision_t*)a->conv2_out.bytes, (precision_t*)ew->conv2_b.bytes, M * N3_C2_OC, N3_C2_OC);
    }

    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->embed_out.bytes, obs, (precision_t*)ew->embed_w.bytes, B, obs_size);

    n3_concat_kernel<<<grid_size(B * N3_CONCAT), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->concat.bytes, (precision_t*)a->conv2_out.bytes,
        (precision_t*)a->embed_out.bytes, obs, B, obs_size);

    puf_mm(a->concat, ew->proj_w, a->out, stream);

    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)a->out.bytes, (precision_t*)ew->proj_b.bytes, B * ew->hidden, ew->hidden);

    return a->out;
}

static void nmmo3_encoder_backward(void* w, void* activations, PufTensor grad, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int N = grad.shape[0], H = ew->hidden, p = PRECISION_SIZE;

    n3_relu_backward_kernel<<<grid_size(N * H), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)grad.bytes, (precision_t*)a->out.bytes, N * H);
    n3_bias_grad_kernel<<<H, 256, 0, stream>>>(
        (precision_t*)a->proj_bgrad.bytes, (precision_t*)grad.bytes, N, H);
    puf_mm_tn(grad, a->concat, a->proj_wgrad, stream);

    // Zero conv/embed gradients (backward not yet implemented)
    CHECK_CUDA(cudaMemsetAsync(a->conv1_wgrad.bytes, 0, a->conv1_wgrad.numel() * p, stream));
    CHECK_CUDA(cudaMemsetAsync(a->conv1_bgrad.bytes, 0, a->conv1_bgrad.numel() * p, stream));
    CHECK_CUDA(cudaMemsetAsync(a->conv2_wgrad.bytes, 0, a->conv2_wgrad.numel() * p, stream));
    CHECK_CUDA(cudaMemsetAsync(a->conv2_bgrad.bytes, 0, a->conv2_bgrad.numel() * p, stream));
    CHECK_CUDA(cudaMemsetAsync(a->embed_wgrad.bytes, 0, a->embed_wgrad.numel() * p, stream));
}

static void nmmo3_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    auto init2d = [&](PufTensor& t, int rows, int cols, float gain) {
        PufTensor wt = {.bytes = t.bytes, .shape = {rows, cols}, .dtype_size = t.dtype_size};
        puf_kaiming_init(wt, gain, (*seed)++, stream);
    };
    float g = std::sqrt(2.0f);
    init2d(ew->conv1_w, N3_C1_OC, N3_C1_K2, g);
    CHECK_CUDA(cudaMemsetAsync(ew->conv1_b.bytes, 0, ew->conv1_b.numel() * ew->conv1_b.dtype_size, stream));
    init2d(ew->conv2_w, N3_C2_OC, N3_C2_K2, g);
    CHECK_CUDA(cudaMemsetAsync(ew->conv2_b.bytes, 0, ew->conv2_b.numel() * ew->conv2_b.dtype_size, stream));
    init2d(ew->embed_w, N3_EMBED_VOCAB, N3_EMBED_DIM, 1.0f);
    init2d(ew->proj_w, ew->hidden, N3_CONCAT, g);
    CHECK_CUDA(cudaMemsetAsync(ew->proj_b.bytes, 0, ew->proj_b.numel() * ew->proj_b.dtype_size, stream));
}

static void nmmo3_encoder_reg_params(void* w, Allocator* alloc, int esz) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    ew->conv1_w = {.shape = {N3_C1_OC, N3_C1_K2}, .dtype_size = esz};
    ew->conv1_b = {.shape = {N3_C1_OC}, .dtype_size = esz};
    ew->conv2_w = {.shape = {N3_C2_OC, N3_C2_K2}, .dtype_size = esz};
    ew->conv2_b = {.shape = {N3_C2_OC}, .dtype_size = esz};
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}, .dtype_size = esz};
    ew->proj_w  = {.shape = {ew->hidden, N3_CONCAT}, .dtype_size = esz};
    ew->proj_b  = {.shape = {ew->hidden}, .dtype_size = esz};
    alloc->reg(&ew->conv1_w); alloc->reg(&ew->conv1_b);
    alloc->reg(&ew->conv2_w); alloc->reg(&ew->conv2_b);
    alloc->reg(&ew->embed_w);
    alloc->reg(&ew->proj_w);  alloc->reg(&ew->proj_b);
}

static void nmmo3_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int p = PRECISION_SIZE;
    *a = {};
    a->multihot  = {.shape = {B_TT, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}, .dtype_size = p};
    a->im2col1   = {.shape = {B_TT * N3_C1_M, N3_C1_K2}, .dtype_size = p};
    a->conv1_out = {.shape = {B_TT * N3_C1_M, N3_C1_OC}, .dtype_size = p};
    a->im2col2   = {.shape = {B_TT * N3_C2_M, N3_C2_K2}, .dtype_size = p};
    a->conv2_out = {.shape = {B_TT * N3_C2_M, N3_C2_OC}, .dtype_size = p};
    a->embed_out = {.shape = {B_TT, N3_PLAYER_EMBED}, .dtype_size = p};
    a->concat    = {.shape = {B_TT, N3_CONCAT}, .dtype_size = p};
    a->out       = {.shape = {B_TT, ew->hidden}, .dtype_size = p};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}, .dtype_size = p};
    acts->reg(&a->multihot);  acts->reg(&a->im2col1); acts->reg(&a->conv1_out);
    acts->reg(&a->im2col2);   acts->reg(&a->conv2_out);
    acts->reg(&a->embed_out); acts->reg(&a->concat);
    acts->reg(&a->out);       acts->reg(&a->saved_obs);
    // Weight gradients — must match reg_params order
    a->conv1_wgrad = {.shape = {N3_C1_OC, N3_C1_K2}, .dtype_size = p};
    a->conv1_bgrad = {.shape = {N3_C1_OC}, .dtype_size = p};
    a->conv2_wgrad = {.shape = {N3_C2_OC, N3_C2_K2}, .dtype_size = p};
    a->conv2_bgrad = {.shape = {N3_C2_OC}, .dtype_size = p};
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}, .dtype_size = p};
    a->proj_wgrad  = {.shape = {ew->hidden, N3_CONCAT}, .dtype_size = p};
    a->proj_bgrad  = {.shape = {ew->hidden}, .dtype_size = p};
    grads->reg(&a->conv1_wgrad); grads->reg(&a->conv1_bgrad);
    grads->reg(&a->conv2_wgrad); grads->reg(&a->conv2_bgrad);
    grads->reg(&a->embed_wgrad);
    grads->reg(&a->proj_wgrad);  grads->reg(&a->proj_bgrad);
}

static void nmmo3_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int p = PRECISION_SIZE;
    a->multihot  = {.shape = {B, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}, .dtype_size = p};
    a->im2col1   = {.shape = {B * N3_C1_M, N3_C1_K2}, .dtype_size = p};
    a->conv1_out = {.shape = {B * N3_C1_M, N3_C1_OC}, .dtype_size = p};
    a->im2col2   = {.shape = {B * N3_C2_M, N3_C2_K2}, .dtype_size = p};
    a->conv2_out = {.shape = {B * N3_C2_M, N3_C2_OC}, .dtype_size = p};
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}, .dtype_size = p};
    a->concat    = {.shape = {B, N3_CONCAT}, .dtype_size = p};
    a->out       = {.shape = {B, ew->hidden}, .dtype_size = p};
    alloc->reg(&a->multihot);  alloc->reg(&a->im2col1); alloc->reg(&a->conv1_out);
    alloc->reg(&a->im2col2);   alloc->reg(&a->conv2_out);
    alloc->reg(&a->embed_out); alloc->reg(&a->concat);  alloc->reg(&a->out);
}

#endif  // OCEAN_STANDALONE
