// cuDNN Conv2d: forward/backward.
// Included by ocean.cu (training) and tests/test_nmmo3_cuda.cu (test).
// Algorithm search + workspace allocation done once at init for the max batch size.
// At runtime, only lightweight tensor descriptors are created per call.

#ifndef CUDNN_CONV2D_CU
#define CUDNN_CONV2D_CU

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>

#include "kernels.cu"

#ifndef CHECK_CUDNN
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t e = call; \
    if (e != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(e)); exit(1); \
    } \
} while(0)
#endif

static cudnnHandle_t get_cudnn_handle() {
    static thread_local cudnnHandle_t h = nullptr;
    if (!h) CHECK_CUDNN(cudnnCreate(&h));
    return h;
}

// ---- Bias + optional ReLU kernels (NCHW layout) ----

__global__ void conv_bias_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(to_float(data[idx]) + to_float(bias[oc]));
}

__global__ void conv_bias_relu_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[oc])));
}

// ---- ConvWeights ----

struct ConvWeights {
    PrecisionTensor w, b;  // w: (OC, IC*K*K), b: (OC)
    int IC, OC, K, S, IH, IW, OH, OW;
    bool relu;
    cudnnDataType_t dtype;
    // Batch-independent descriptors (created once)
    cudnnFilterDescriptor_t cudnn_filt;
    cudnnConvolutionDescriptor_t cudnn_conv;
    // Algorithms + workspace (found once at init for max batch size)
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filt_algo;
    size_t fwd_ws_bytes, bwd_data_ws_bytes, bwd_filt_ws_bytes;
    void* fwd_ws; void* bwd_data_ws; void* bwd_filt_ws;
    bool shared_ready;
    bool algos_ready;
};

struct ConvActivations {
    PrecisionTensor out, grad, saved_input;
    PrecisionTensor wgrad, bgrad;
};

static void conv_init(ConvWeights* cw, int IC, int OC, int K, int S, int IH, int IW, bool relu) {
    cw->IC = IC; cw->OC = OC; cw->K = K; cw->S = S; cw->IH = IH; cw->IW = IW;
    cw->OH = (IH - K) / S + 1; cw->OW = (IW - K) / S + 1;
    cw->relu = relu; cw->shared_ready = false; cw->algos_ready = false;
    cw->fwd_ws = nullptr; cw->bwd_data_ws = nullptr; cw->bwd_filt_ws = nullptr;
}

static void conv_setup_shared(ConvWeights* cw, cudnnDataType_t dt) {
    if (cw->shared_ready) return;
    cw->dtype = dt;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&cw->cudnn_filt));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(cw->cudnn_filt, dt, CUDNN_TENSOR_NCHW, cw->OC, cw->IC, cw->K, cw->K));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&cw->cudnn_conv));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(cw->cudnn_conv, 0, 0, cw->S, cw->S, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    cw->shared_ready = true;
}

// Find best algorithms and allocate workspace for batch size B.
// Called once at init for the max batch size. Workspace is reused for smaller batches.
static void conv_find_algos(ConvWeights* cw, int B) {
    if (cw->algos_ready) return;
    cudnnHandle_t h = get_cudnn_handle();
    cudnnTensorDescriptor_t in_desc, out_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, cw->dtype, B, cw->IC, cw->IH, cw->IW));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, cw->dtype, B, cw->OC, cw->OH, cw->OW));

    int returned;
    cudnnConvolutionFwdAlgoPerf_t fp;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(h, in_desc, cw->cudnn_filt, cw->cudnn_conv, out_desc, 1, &returned, &fp));
    cw->fwd_algo = fp.algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(h, in_desc, cw->cudnn_filt, cw->cudnn_conv, out_desc, cw->fwd_algo, &cw->fwd_ws_bytes));
    cw->fwd_ws = NULL; if (cw->fwd_ws_bytes > 0) cudaMalloc(&cw->fwd_ws, cw->fwd_ws_bytes);

    cudnnConvolutionBwdFilterAlgoPerf_t ffp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(h, in_desc, out_desc, cw->cudnn_conv, cw->cudnn_filt, 1, &returned, &ffp));
    cw->bwd_filt_algo = ffp.algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(h, in_desc, out_desc, cw->cudnn_conv, cw->cudnn_filt, cw->bwd_filt_algo, &cw->bwd_filt_ws_bytes));
    cw->bwd_filt_ws = NULL; if (cw->bwd_filt_ws_bytes > 0) cudaMalloc(&cw->bwd_filt_ws, cw->bwd_filt_ws_bytes);

    cudnnConvolutionBwdDataAlgoPerf_t dp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(h, cw->cudnn_filt, out_desc, cw->cudnn_conv, in_desc, 1, &returned, &dp));
    cw->bwd_data_algo = dp.algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(h, cw->cudnn_filt, out_desc, cw->cudnn_conv, in_desc, cw->bwd_data_algo, &cw->bwd_data_ws_bytes));
    cw->bwd_data_ws = NULL; if (cw->bwd_data_ws_bytes > 0) cudaMalloc(&cw->bwd_data_ws, cw->bwd_data_ws_bytes);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cw->algos_ready = true;
}

static void conv_make_io(ConvWeights* cw, int B, cudnnTensorDescriptor_t* in, cudnnTensorDescriptor_t* out) {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(in));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*in, CUDNN_TENSOR_NCHW, cw->dtype, B, cw->IC, cw->IH, cw->IW));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(out));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(*out, CUDNN_TENSOR_NCHW, cw->dtype, B, cw->OC, cw->OH, cw->OW));
}

// Legacy API for test compatibility
static void conv_setup(ConvWeights* cw, int B, cudnnDataType_t dt) {
    conv_setup_shared(cw, dt);
    conv_find_algos(cw, B);
}

static void conv_reg_params(ConvWeights* cw, Allocator* alloc) {
    cw->w = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    cw->b = {.shape = {cw->OC}};
    alloc_register(alloc,&cw->w); alloc_register(alloc,&cw->b);
}

static void conv_reg_train(ConvWeights* cw, ConvActivations* ca, Allocator* acts, Allocator* grads, int B, cudnnDataType_t dt) {
    ca->out         = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->grad        = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    ca->saved_input = {.shape = {B * cw->IC * cw->IH * cw->IW}};
    ca->wgrad       = {.shape = {cw->OC, cw->IC * cw->K * cw->K}};
    ca->bgrad       = {.shape = {cw->OC}};
    alloc_register(acts,&ca->out); alloc_register(acts,&ca->grad); alloc_register(acts,&ca->saved_input);
    alloc_register(grads,&ca->wgrad); alloc_register(grads,&ca->bgrad);
    conv_setup_shared(cw, dt);
    conv_find_algos(cw, B);  // training B is largest — find algos + alloc workspace here
}

static void conv_reg_rollout(ConvWeights* cw, ConvActivations* ca, Allocator* alloc, int B, cudnnDataType_t dt) {
    ca->out = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    alloc_register(alloc,&ca->out);
    conv_setup_shared(cw, dt);
    // algos_ready is already true from conv_reg_train — skip, reuse training workspace
}

static void conv_init_weights(ConvWeights* cw, uint64_t* seed, cudaStream_t stream) {
    PrecisionTensor wt = {.data = cw->w.data, .shape = {cw->OC, cw->IC * cw->K * cw->K}};
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(cw->b.data, 0, numel(cw->b.shape) * sizeof(precision_t), stream);
}

// ---- Forward / Backward ----

static void conv_forward(ConvWeights* cw, ConvActivations* ca, void* input, int B, cudaStream_t stream) {
    cudnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(h, stream));
    cudnnTensorDescriptor_t in_desc, out_desc;
    conv_make_io(cw, B, &in_desc, &out_desc);
    float alpha = 1.0f, beta = 0.0f;
    if (ca->saved_input.data) {
        cudaMemcpyAsync(ca->saved_input.data, input,
            (int64_t)B * cw->IC * cw->IH * cw->IW * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    }
    CHECK_CUDNN(cudnnConvolutionForward(h,
        &alpha, in_desc, input, cw->cudnn_filt, cw->w.data,
        cw->cudnn_conv, cw->fwd_algo, cw->fwd_ws, cw->fwd_ws_bytes,
        &beta, out_desc, ca->out.data));
    int spatial = cw->OH * cw->OW;
    int total = B * cw->OC * spatial;
    if (cw->relu) {
        conv_bias_relu_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
            ca->out.data, cw->b.data, B, cw->OC, spatial);
    } else {
        conv_bias_kernel<<<grid_size(total), BLOCK_SIZE, 0, stream>>>(
            ca->out.data, cw->b.data, B, cw->OC, spatial);
    }
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

static void conv_backward(ConvWeights* cw, ConvActivations* ca, void* input_grad, int B, cudaStream_t stream) {
    cudnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(h, stream));
    cudnnTensorDescriptor_t in_desc, out_desc;
    conv_make_io(cw, B, &in_desc, &out_desc);
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(h,
        &alpha, in_desc, ca->saved_input.data, out_desc, ca->grad.data,
        cw->cudnn_conv, cw->bwd_filt_algo, cw->bwd_filt_ws, cw->bwd_filt_ws_bytes,
        &beta, cw->cudnn_filt, ca->wgrad.data));

    if (input_grad) {
        CHECK_CUDNN(cudnnConvolutionBackwardData(h,
            &alpha, cw->cudnn_filt, cw->w.data, out_desc, ca->grad.data,
            cw->cudnn_conv, cw->bwd_data_algo, cw->bwd_data_ws, cw->bwd_data_ws_bytes,
            &beta, in_desc, input_grad));
    }
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

#endif // CUDNN_CONV2D_CU
