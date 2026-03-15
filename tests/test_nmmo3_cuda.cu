// Standalone test for NMMO3 encoder — float32, ctypes-callable.
// Build: nvcc -shared -o ocean_test.so tests/test_nmmo3_cuda.cu -I pufferlib/src -lcublas -lcudnn -Xcompiler -fPIC -O2

// Predefine PRECISION_FLOAT before including cudnn_conv2d (which includes kernels.cu)
#define PRECISION_FLOAT
#include "../pufferlib/src/cudnn_conv2d.cu"
#include <cublas_v2.h>

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

static void sgemm_nt(cublasHandle_t h, float* C, const float* A, const float* B,
                     int M, int N, int K, cudaStream_t s) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(h, s);
    cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N);
}

// NMMO3 constants
static constexpr int MAP_H = 11, MAP_W = 15, NFEAT = 10, MULTIHOT = 59, OBS_SIZE = 1707;
static constexpr int MAP_SIZE = MAP_H * MAP_W * NFEAT;
static constexpr int PLAYER = 47, REWARD = 10, EMBED_DIM = 32;
static constexpr int PLAYER_EMBED = PLAYER * EMBED_DIM;
static constexpr int C1_IC = 59, C1_OC = 128, C1_K = 5, C1_S = 3, C1_OH = 3, C1_OW = 4;
static constexpr int C2_OC = 128, C2_K = 3, C2_S = 1, C2_OH = 1, C2_OW = 2;
static constexpr int CONV_FLAT = C2_OC * C2_OH * C2_OW;
static constexpr int CONCAT = CONV_FLAT + PLAYER_EMBED + PLAYER + REWARD;

__constant__ int OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

__global__ void multihot_kernel(float* out, const unsigned char* obs, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * MAP_H * MAP_W) return;
    int b = idx / (MAP_H * MAP_W), rem = idx % (MAP_H * MAP_W);
    int h = rem / MAP_W, w = rem % MAP_W;
    const unsigned char* src = obs + b * OBS_SIZE + (h * MAP_W + w) * NFEAT;
    float* dst = out + b * MULTIHOT * MAP_H * MAP_W;
    for (int f = 0; f < NFEAT; f++)
        dst[(OFFSETS[f] + (int)src[f]) * MAP_H * MAP_W + h * MAP_W + w] = 1.0f;
}

__global__ void embedding_kernel(float* out, const unsigned char* obs, const float* embed_w, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * PLAYER) return;
    int b = idx / PLAYER, f = idx % PLAYER;
    const float* src = embed_w + (int)obs[b * OBS_SIZE + MAP_SIZE + f] * EMBED_DIM;
    float* dst = out + b * PLAYER_EMBED + f * EMBED_DIM;
    for (int d = 0; d < EMBED_DIM; d++) dst[d] = src[d];
}

__global__ void concat_kernel(float* out, const float* conv, const float* embed,
                               const unsigned char* obs, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CONCAT) return;
    int b = idx / CONCAT, c = idx % CONCAT;
    float val;
    if (c < CONV_FLAT) {
        int oc = c / (C2_OH * C2_OW), r = c % (C2_OH * C2_OW);
        val = conv[b * CONV_FLAT + oc * C2_OH * C2_OW + r];
    } else if (c < CONV_FLAT + PLAYER_EMBED)
        val = embed[b * PLAYER_EMBED + (c - CONV_FLAT)];
    else if (c < CONV_FLAT + PLAYER_EMBED + PLAYER)
        val = (float)obs[b * OBS_SIZE + MAP_SIZE + (c - CONV_FLAT - PLAYER_EMBED)];
    else
        val = (float)obs[b * OBS_SIZE + OBS_SIZE - REWARD + (c - CONV_FLAT - PLAYER_EMBED - PLAYER)];
    out[idx] = val;
}

__global__ void bias_relu_kernel(float* data, const float* bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = fmaxf(0.0f, data[idx] + bias[idx % dim]);
}

__global__ void relu_backward_kernel(float* grad, const float* out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (out[idx] <= 0.0f) grad[idx] = 0.0f;
}

__global__ void bias_grad_nchw_kernel(float* bgrad, const float* grad, int B, int OC, int spatial) {
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float sum = 0.0f;
    int total = B * spatial;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial, s = i % spatial;
        sum += grad[b * OC * spatial + oc * spatial + s];
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
        if (lane == 0) bgrad[oc] = sum;
    }
}

// ---- Global state ----

static ConvWeights g_conv1, g_conv2;

static void ensure_setup(int B) {
    if (!g_conv1.cudnn_ready) {
        conv_init(&g_conv1, C1_IC, C1_OC, C1_K, C1_S, MAP_H, MAP_W, true);
        conv_init(&g_conv2, C1_OC, C2_OC, C2_K, C2_S, C1_OH, C1_OW, false);
    }
    conv_setup(&g_conv1, B, CUDNN_DATA_FLOAT);
    conv_setup(&g_conv2, B, CUDNN_DATA_FLOAT);
}

extern "C" {

int nmmo3_workspace_size(int B) {
    return B * (MULTIHOT * MAP_H * MAP_W + C1_OC * C1_OH * C1_OW
              + C2_OC * C2_OH * C2_OW + PLAYER_EMBED + CONCAT);
}

void nmmo3_encoder_setup(int B) { ensure_setup(B); }

void nmmo3_encoder_forward(
    float* output, const unsigned char* obs,
    const float* c1w, const float* c1b, const float* c2w, const float* c2b,
    const float* ew, const float* pw, const float* pb,
    float* ws, int B, cublasHandle_t handle, cudaStream_t stream)
{
    if (!handle) handle = get_cublas_handle();
    const int BLK = 256;
    float* mh  = ws;
    float* o1  = mh + B * MULTIHOT * MAP_H * MAP_W;
    float* o2  = o1 + B * C1_OC * C1_OH * C1_OW;
    float* emb = o2 + B * C2_OC * C2_OH * C2_OW;
    float* cat = emb + B * PLAYER_EMBED;

    cudaMemsetAsync(mh, 0, B * MULTIHOT * MAP_H * MAP_W * sizeof(float), stream);
    multihot_kernel<<<div_ceil(B * MAP_H * MAP_W, BLK), BLK, 0, stream>>>(mh, obs, B);

    // Set weight pointers for conv_forward (test doesn't use Allocator)
    g_conv1.w.data = (precision_t*)c1w; g_conv1.b.data = (precision_t*)c1b;
    g_conv2.w.data = (precision_t*)c2w; g_conv2.b.data = (precision_t*)c2b;

    // Dummy activations with just out pointer
    ConvActivations a1 = {.out = {.data = (precision_t*)o1}};
    ConvActivations a2 = {.out = {.data = (precision_t*)o2}};
    conv_forward(&g_conv1, &a1, mh, B, stream);
    conv_forward(&g_conv2, &a2, o1, B, stream);

    embedding_kernel<<<div_ceil(B * PLAYER, BLK), BLK, 0, stream>>>(emb, obs, ew, B);
    concat_kernel<<<div_ceil(B * CONCAT, BLK), BLK, 0, stream>>>(cat, o2, emb, obs, B);
    sgemm_nt(handle, output, cat, pw, B, 512, CONCAT, stream);
    bias_relu_kernel<<<div_ceil(B * 512, BLK), BLK, 0, stream>>>(output, pb, B * 512, 512);
}

// Conv test helpers

void conv2d_test_forward(float* out, const float* in, const float* w, const float* b,
                           int B, int idx, cudaStream_t s) {
    ConvWeights* cw = (idx == 0) ? &g_conv1 : &g_conv2;
    cw->w.data = (precision_t*)w; cw->b.data = (precision_t*)b;
    ConvActivations ca = {.out = {.data = (precision_t*)out}};
    conv_forward(cw, &ca, (void*)in, B, s);
}

void conv2d_test_backward(float* wg, float* bg, float* ig,
                            float* go, const float* so, const float* in, const float* w,
                            int B, int idx, cudaStream_t s) {
    ConvWeights* cw = (idx == 0) ? &g_conv1 : &g_conv2;
    cw->w.data = (precision_t*)w;
    int total = B * cw->OC * cw->OH * cw->OW;
    if (cw->relu) relu_backward_kernel<<<div_ceil(total, 256), 256, 0, s>>>(go, so, total);
    bias_grad_nchw_kernel<<<cw->OC, 256, 0, s>>>(bg, go, B, cw->OC, cw->OH * cw->OW);
    ConvActivations ca = {
        .grad = {.data = (precision_t*)go},
        .saved_input = {.data = (precision_t*)in},
        .wgrad = {.data = (precision_t*)wg},
    };
    conv_backward(cw, &ca, ig, s);
}

}  // extern "C"
