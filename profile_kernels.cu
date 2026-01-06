// profile_kernels.cu
// Minimal standalone profiler for CUDA kernels
//
// Without torch: nvcc -O3 -arch=sm_80 profile_kernels.cu -o profile_kernels -I.
// With torch:    Build with cmake/pytorch and -DUSE_TORCH
//
// Run: ./profile_kernels

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "pufferlib/extensions/cuda/modules.cu"
#else
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

const int WARMUP_ITERS = 10;
const int TIMING_ITERS = 100;

typedef void (*kernel_fn)(void*);

float profile_kernel(kernel_fn fn, void* args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) {
        fn(args);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / TIMING_ITERS;
}

float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

typedef struct {
    float* gate;
    float* hidden;
    float* log_coeffs;
    float* log_values;
    float* grad_log_coeffs;
    float* grad_log_values;
    float* grad_gate;
    float* grad_hidden;
    int N;
} LogCoeffsAndValuesArgs;

LogCoeffsAndValuesArgs* create_logcoeffsandvaluesargs(int batch, int seq, int hidden) {
    LogCoeffsAndValuesArgs* args = (LogCoeffsAndValuesArgs*)calloc(1, sizeof(LogCoeffsAndValuesArgs));
    args->N = batch * seq * hidden;

    cudaMalloc(&args->gate, args->N * sizeof(float));
    cudaMalloc(&args->hidden, args->N * sizeof(float));
    cudaMalloc(&args->log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->log_values, args->N * sizeof(float));
    cudaMalloc(&args->grad_gate, args->N * sizeof(float));
    cudaMalloc(&args->grad_hidden, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_values, args->N * sizeof(float));

    float* gate_buf = (float*)malloc(args->N * sizeof(float));
    float* hidden_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_log_coeffs_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_log_values_buf = (float*)malloc(args->N * sizeof(float));
    for (int i = 0; i < args->N; ++i) {
        gate_buf[i] = rand1() * 5.0f;
        hidden_buf[i] = rand1() * 5.0f;
        grad_log_coeffs_buf[i] = rand1();
        grad_log_values_buf[i] = rand1();
    }

    cudaMemcpy(args->gate, gate_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->hidden, hidden_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_log_coeffs, grad_log_coeffs_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_log_values, grad_log_values_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(gate_buf);
    free(hidden_buf);
    free(grad_log_coeffs_buf);
    free(grad_log_values_buf);

    return args;
}

void free_logcoeffsandvaluesargs(LogCoeffsAndValuesArgs* args) {
    cudaFree(args->gate);
    cudaFree(args->hidden);
    cudaFree(args->log_coeffs);
    cudaFree(args->log_values);
    cudaFree(args->grad_gate);
    cudaFree(args->grad_hidden);
    cudaFree(args->grad_log_coeffs);
    cudaFree(args->grad_log_values);
    free(args);
}

void run_logcoeffsandvalues_forward(LogCoeffsAndValuesArgs* args) {
    launch_log_coeffs_and_values<float>(
        args->log_coeffs, args->log_values, args->gate, args->hidden, args->N, 0);
}

void run_logcoeffsandvalues_backward(LogCoeffsAndValuesArgs* args) {
    launch_log_coeffs_and_values_backward<float>(
        args->grad_gate, args->grad_hidden, args->grad_log_coeffs,
        args->grad_log_values, args->gate, args->hidden, args->N, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor gate;
    torch::Tensor hidden;
    torch::Tensor grad_log_coeffs;
    torch::Tensor grad_log_values;
    torch::Tensor out_log_coeffs;
    torch::Tensor out_log_values;
    int N;
} LogCoeffsAndValuesArgsTorch;

LogCoeffsAndValuesArgsTorch* create_logcoeffsandvaluesargs_torch(LogCoeffsAndValuesArgs* raw) {
    LogCoeffsAndValuesArgsTorch* args = new LogCoeffsAndValuesArgsTorch();
    args->N = raw->N;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->gate = torch::from_blob(raw->gate, {raw->N}, opts).requires_grad_(true);
    args->hidden = torch::from_blob(raw->hidden, {raw->N}, opts).requires_grad_(true);
    args->grad_log_coeffs = torch::from_blob(raw->grad_log_coeffs, {raw->N}, opts);
    args->grad_log_values = torch::from_blob(raw->grad_log_values, {raw->N}, opts);

    // Run forward once to set up outputs for backward
    auto outputs = log_coeffs_and_values(args->gate, args->hidden);
    args->out_log_coeffs = outputs[0];
    args->out_log_values = outputs[1];

    return args;
}

void run_logcoeffsandvalues_forward_torch(LogCoeffsAndValuesArgsTorch* args) {
    torch::NoGradGuard no_grad;
    log_coeffs_and_values(args->gate, args->hidden);
}

void run_logcoeffsandvalues_backward_torch(LogCoeffsAndValuesArgsTorch* args) {
    args->gate.mutable_grad() = torch::Tensor();
    args->hidden.mutable_grad() = torch::Tensor();
    torch::autograd::backward(
        {args->out_log_coeffs, args->out_log_values},
        {args->grad_log_coeffs, args->grad_log_values},
        /*retain_graph=*/true);
}

#endif

void profile_logcoeffsandvalues(int batch, int seq, int hidden) {
    LogCoeffsAndValuesArgs* args = create_logcoeffsandvaluesargs(batch, seq, hidden);

    printf("log_coeffs_and_values (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward, args);
    printf("  forward:  %.3f ms  %.2f GElem/s\n", fwd_ms, args->N / fwd_ms / 1e6);

    float bwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward, args);
    printf("  backward: %.3f ms  %.2f GElem/s\n", bwd_ms, args->N / bwd_ms / 1e6);

#ifdef USE_TORCH
    LogCoeffsAndValuesArgsTorch* args_torch = create_logcoeffsandvaluesargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward_torch, args_torch);
    printf("  forward (torch):  %.3f ms  %.2f GElem/s  (%.1f%% overhead)\n",
           fwd_torch_ms, args->N / fwd_torch_ms / 1e6, (fwd_torch_ms / fwd_ms - 1) * 100);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward_torch, args_torch);
    printf("  backward (torch): %.3f ms  %.2f GElem/s  (%.1f%% overhead)\n",
           bwd_torch_ms, args->N / bwd_torch_ms / 1e6, (bwd_torch_ms / bwd_ms - 1) * 100);

    delete args_torch;
#endif

    free_logcoeffsandvaluesargs(args);
}

int main(int argc, char** argv) {
    profile_logcoeffsandvalues(32, 1024, 2048);
    return 0;
}
