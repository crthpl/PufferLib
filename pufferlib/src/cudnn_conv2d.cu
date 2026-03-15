// cuDNN Conv2d: forward/backward with fused bias+activation.
// Included by ocean.cu (training) and tests/test_nmmo3_cuda.cu (test).

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

static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t h = nullptr;
    if (!h) CHECK_CUDNN(cudnnCreate(&h));
    return h;
}

// ---- ConvWeights: params + cuDNN descriptors ----

struct ConvWeights {
    PrecisionTensor w, b;  // w: (OC, IC*K*K), b: (OC)
    int IC, OC, K, S, IH, IW, OH, OW;
    bool relu;
    // cuDNN state
    cudnnDataType_t dtype;
    cudnnTensorDescriptor_t cudnn_in, cudnn_out, cudnn_bias;
    cudnnFilterDescriptor_t cudnn_filt;
    cudnnConvolutionDescriptor_t cudnn_conv;
    cudnnActivationDescriptor_t cudnn_act;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filt_algo;
    size_t fwd_ws_bytes, bwd_data_ws_bytes, bwd_filt_ws_bytes;
    void* fwd_ws; void* bwd_data_ws; void* bwd_filt_ws;
    bool cudnn_ready;
};

struct ConvActivations {
    PrecisionTensor out, grad, saved_input;
    PrecisionTensor wgrad, bgrad;
};

static void conv_init(ConvWeights* cw, int IC, int OC, int K, int S, int IH, int IW, bool relu) {
    cw->IC = IC; cw->OC = OC; cw->K = K; cw->S = S; cw->IH = IH; cw->IW = IW;
    cw->OH = (IH - K) / S + 1; cw->OW = (IW - K) / S + 1;
    cw->relu = relu; cw->cudnn_ready = false;
    cw->fwd_ws = nullptr; cw->bwd_data_ws = nullptr; cw->bwd_filt_ws = nullptr;
}

static void conv_setup(ConvWeights* cw, int B, cudnnDataType_t dt) {
    cudnnHandle_t h = get_cudnn_handle();
    if (cw->cudnn_ready) {
        cudnnDestroyTensorDescriptor(cw->cudnn_in);
        cudnnDestroyTensorDescriptor(cw->cudnn_out);
        cudnnDestroyTensorDescriptor(cw->cudnn_bias);
        cudnnDestroyFilterDescriptor(cw->cudnn_filt);
        cudnnDestroyConvolutionDescriptor(cw->cudnn_conv);
        cudnnDestroyActivationDescriptor(cw->cudnn_act);
        if (cw->fwd_ws) cudaFree(cw->fwd_ws);
        if (cw->bwd_data_ws) cudaFree(cw->bwd_data_ws);
        if (cw->bwd_filt_ws) cudaFree(cw->bwd_filt_ws);
    }
    cw->dtype = dt;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&cw->cudnn_in));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(cw->cudnn_in, CUDNN_TENSOR_NCHW, dt, B, cw->IC, cw->IH, cw->IW));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&cw->cudnn_filt));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(cw->cudnn_filt, dt, CUDNN_TENSOR_NCHW, cw->OC, cw->IC, cw->K, cw->K));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&cw->cudnn_conv));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(cw->cudnn_conv, 0, 0, cw->S, cw->S, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&cw->cudnn_out));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(cw->cudnn_out, CUDNN_TENSOR_NCHW, dt, B, cw->OC, cw->OH, cw->OW));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&cw->cudnn_bias));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(cw->cudnn_bias, CUDNN_TENSOR_NCHW, dt, 1, cw->OC, 1, 1));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&cw->cudnn_act));
    CHECK_CUDNN(cudnnSetActivationDescriptor(cw->cudnn_act,
        cw->relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    int returned;
    cudnnConvolutionFwdAlgoPerf_t fp;
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(h, cw->cudnn_in, cw->cudnn_filt, cw->cudnn_conv, cw->cudnn_out, 1, &returned, &fp));
    cw->fwd_algo = fp.algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(h, cw->cudnn_in, cw->cudnn_filt, cw->cudnn_conv, cw->cudnn_out, cw->fwd_algo, &cw->fwd_ws_bytes));
    cw->fwd_ws = NULL; if (cw->fwd_ws_bytes > 0) cudaMalloc(&cw->fwd_ws, cw->fwd_ws_bytes);

    cudnnConvolutionBwdDataAlgoPerf_t dp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(h, cw->cudnn_filt, cw->cudnn_out, cw->cudnn_conv, cw->cudnn_in, 1, &returned, &dp));
    cw->bwd_data_algo = dp.algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(h, cw->cudnn_filt, cw->cudnn_out, cw->cudnn_conv, cw->cudnn_in, cw->bwd_data_algo, &cw->bwd_data_ws_bytes));
    cw->bwd_data_ws = NULL; if (cw->bwd_data_ws_bytes > 0) cudaMalloc(&cw->bwd_data_ws, cw->bwd_data_ws_bytes);

    cudnnConvolutionBwdFilterAlgoPerf_t ffp;
    CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(h, cw->cudnn_in, cw->cudnn_out, cw->cudnn_conv, cw->cudnn_filt, 1, &returned, &ffp));
    cw->bwd_filt_algo = ffp.algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(h, cw->cudnn_in, cw->cudnn_out, cw->cudnn_conv, cw->cudnn_filt, cw->bwd_filt_algo, &cw->bwd_filt_ws_bytes));
    cw->bwd_filt_ws = NULL; if (cw->bwd_filt_ws_bytes > 0) cudaMalloc(&cw->bwd_filt_ws, cw->bwd_filt_ws_bytes);

    cw->cudnn_ready = true;
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
    conv_setup(cw, B, dt);
}

static void conv_reg_rollout(ConvWeights* cw, ConvActivations* ca, Allocator* alloc, int B, cudnnDataType_t dt) {
    ca->out = {.shape = {B * cw->OC * cw->OH * cw->OW}};
    alloc_register(alloc,&ca->out);
    conv_setup(cw, B, dt);
}

static void conv_init_weights(ConvWeights* cw, uint64_t* seed, cudaStream_t stream) {
    PrecisionTensor wt = {.data = cw->w.data, .shape = {cw->OC, cw->IC * cw->K * cw->K}};
    puf_kaiming_init(&wt, cw->relu ? std::sqrt(2.0f) : 1.0f, (*seed)++, stream);
    cudaMemsetAsync(cw->b.data, 0, numel(cw->b.shape) * sizeof(precision_t), stream);
}

// ---- Forward / Backward ----

// Fused conv + bias + activation. All NCHW. Saves input for backward.
static void conv_forward(ConvWeights* cw, ConvActivations* ca, void* input, int B, cudaStream_t stream) {
    cudnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;
    if (ca->saved_input.data) {
        cudaMemcpyAsync(ca->saved_input.data, input,
            (int64_t)B * cw->IC * cw->IH * cw->IW * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    }
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(h,
        &alpha, cw->cudnn_in, input, cw->cudnn_filt, cw->w.data,
        cw->cudnn_conv, cw->fwd_algo, cw->fwd_ws, cw->fwd_ws_bytes,
        &beta, cw->cudnn_out, ca->out.data, cw->cudnn_bias, cw->b.data,
        cw->cudnn_act, cw->cudnn_out, ca->out.data));
}

// Backward: upstream grad in ca->grad, relu mask in ca->out.
// Caller must apply relu backward and bias grad (dtype-specific kernels).
// This does cuDNN filter grad + optional data grad.
static void conv_backward(ConvWeights* cw, ConvActivations* ca, void* input_grad, cudaStream_t stream) {
    cudnnHandle_t h = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(h, stream));
    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(h,
        &alpha, cw->cudnn_in, ca->saved_input.data, cw->cudnn_out, ca->grad.data,
        cw->cudnn_conv, cw->bwd_filt_algo, cw->bwd_filt_ws, cw->bwd_filt_ws_bytes,
        &beta, cw->cudnn_filt, ca->wgrad.data));

    if (input_grad) {
        CHECK_CUDNN(cudnnConvolutionBackwardData(h,
            &alpha, cw->cudnn_filt, cw->w.data, cw->cudnn_out, ca->grad.data,
            cw->cudnn_conv, cw->bwd_data_algo, cw->bwd_data_ws, cw->bwd_data_ws_bytes,
            &beta, cw->cudnn_in, input_grad));
    }
}

#endif // CUDNN_CONV2D_CU
