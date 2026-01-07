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
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/cuda/modules.cu"
#else
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

const int WARMUP_ITERS = 1000;
const int TIMING_ITERS = 10000;

const int BR = 4096;  // Rollout batch (no T dim)
const int BT = 512;   // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;

typedef void (*kernel_fn)(void*);

void print_timing(const char* name, float ms, int N) {
    printf("  %-18s %6.1f us  %6.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

void warmup_gpu() {
    // Warm up GPU clocks with some busy work
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);  // 64MB
    for (int i = 0; i < 100; i++) {
        cudaMemset(dummy, 0, 64 * 1024 * 1024);
    }
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

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

    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return ms / TIMING_ITERS;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
    }
    warmup_stream.synchronize();

    at::cuda::CUDAStream cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    cuda_graph.capture_begin();
    fn(args);
    cuda_graph.capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();
    at::cuda::setCurrentCUDAStream(current_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; ++i) {
        cuda_graph.replay();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / TIMING_ITERS;
}
#endif

float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

typedef struct {
    float* state;
    float* gate;
    float* hidden;
    float* out;
    int N;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->N = batch * hidden;

    cudaMalloc(&args->state, args->N * sizeof(float));
    cudaMalloc(&args->gate, args->N * sizeof(float));
    cudaMalloc(&args->hidden, args->N * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));

    float* buf = (float*)malloc(args->N * sizeof(float) * 3);
    float* state_buf = buf;
    float* gate_buf = buf + args->N;
    float* hidden_buf = buf + args->N * 2;
    for (int i = 0; i < args->N; ++i) {
        state_buf[i] = rand1() * 5.0f;
        gate_buf[i] = rand1() * 5.0f;
        hidden_buf[i] = rand1() * 5.0f;
    }

    cudaMemcpy(args->state, state_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->gate, gate_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->hidden, hidden_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(buf);
    return args;
}

void free_mingrugateargs(MingruGateArgs* args) {
    cudaFree(args->state);
    cudaFree(args->gate);
    cudaFree(args->hidden);
    cudaFree(args->out);
    free(args);
}

void run_mingrugate_forward(MingruGateArgs* args) {
    launch_mingru_gate_inference<float>(
        args->out, args->gate, args->hidden, args->state, args->N, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;
    torch::Tensor gate;
    torch::Tensor hidden;
    int N;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->N = raw->N;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->N}, opts);
    args->gate = torch::from_blob(raw->gate, {raw->N}, opts);
    args->hidden = torch::from_blob(raw->hidden, {raw->N}, opts);

    return args;
}

void run_mingrugate_forward_torch(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate(args->state, args->gate, args->hidden);
}

void run_mingrugate_forward_cpp(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate_cpp(args->state, args->gate, args->hidden);
}

#endif

void profile_mingrugate(int batch, int hidden) {
    MingruGateArgs* args = create_mingrugateargs(batch, hidden);

    printf("mingru_gate (N=%d, %dx%d)\n", args->N, batch, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_mingrugate_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_mingrugateargs(args);
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
    args->N = batch*seq * hidden;

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

void run_logcoeffsandvalues_forward_cpp(LogCoeffsAndValuesArgsTorch* args) {
    torch::NoGradGuard no_grad;
    log_coeffs_and_values_cpp(args->gate, args->hidden);
}

#endif

void profile_logcoeffsandvalues(int batch, int seq, int hidden) {
    LogCoeffsAndValuesArgs* args = create_logcoeffsandvaluesargs(batch, seq, hidden);

    printf("log_coeffs_and_values (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogCoeffsAndValuesArgsTorch* args_torch = create_logcoeffsandvaluesargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    auto kernel_outputs = log_coeffs_and_values(args_torch->gate, args_torch->hidden);
    args_torch->out_log_coeffs = kernel_outputs[0];
    args_torch->out_log_values = kernel_outputs[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    auto cpp_outputs = log_coeffs_and_values_cpp(args_torch->gate, args_torch->hidden);
    args_torch->out_log_coeffs = cpp_outputs[0];
    args_torch->out_log_values = cpp_outputs[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcoeffsandvalues_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcoeffsandvalues_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_logcoeffsandvaluesargs(args);
}

typedef struct {
    float* x;
    float* out;
    double* s_buf;
    float* grad_x;
    float* grad_out;
    int B;
    int T;
    int H;
    int N;
} LogcumsumexpArgs;

LogcumsumexpArgs* create_logcumsumexpargs(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = (LogcumsumexpArgs*)calloc(1, sizeof(LogcumsumexpArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch*seq * hidden;

    cudaMalloc(&args->x, args->N * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(float));
    cudaMalloc(&args->grad_out, args->N * sizeof(float));

    float* buf = (float*)malloc(args->N * sizeof(float) * 2);
    float* x_buf = buf;
    float* grad_out_buf = buf + args->N;
    for (int i = 0; i < args->N; ++i) {
        x_buf[i] = rand1();
        grad_out_buf[i] = rand1();
    }

    cudaMemcpy(args->x, x_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(buf);
    return args;
}

void free_logcumsumexpargs(LogcumsumexpArgs* args) {
    cudaFree(args->x);
    cudaFree(args->out);
    cudaFree(args->s_buf);
    cudaFree(args->grad_x);
    cudaFree(args->grad_out);
    free(args);
}

void run_logcumsumexp_forward(LogcumsumexpArgs* args) {
    launch_logcumsumexp_forward<float>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B, 0);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    launch_logcumsumexp_backward<float>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;
    torch::Tensor out;
    torch::Tensor grad_out;
    int N;
} LogcumsumexpArgsTorch;

LogcumsumexpArgsTorch* create_logcumsumexpargs_torch(LogcumsumexpArgs* raw) {
    LogcumsumexpArgsTorch* args = new LogcumsumexpArgsTorch();
    args->N = raw->N;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts);

    return args;
}

void run_logcumsumexp_forward_torch(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cuda(args->x);
}

void run_logcumsumexp_backward_torch(LogcumsumexpArgsTorch* args) {
    args->x.mutable_grad() = torch::Tensor();
    args->out.backward(args->grad_out, /*retain_graph=*/true);
}

void run_logcumsumexp_forward_cpp(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cpp(args->x);
}

#endif

void profile_logcumsumexp(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = create_logcumsumexpargs(batch, seq, hidden);

    printf("logcumsumexp (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogcumsumexpArgsTorch* args_torch = create_logcumsumexpargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    args_torch->out = logcumsumexp_cuda(args_torch->x);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    args_torch->out = logcumsumexp_cpp(args_torch->x);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_logcumsumexpargs(args);
}

typedef struct {
    float* log_coeffs;
    float* log_values;
    float* out;
    float* a_star;
    float* s_vals;
    float* grad_log_coeffs;
    float* grad_log_values;
    float* grad_out;
    int B;
    int T;
    int H;
    int N;
} FusedScanArgs;

FusedScanArgs* create_fusedscanargs(int batch, int seq, int hidden) {
    FusedScanArgs* args = (FusedScanArgs*)calloc(1, sizeof(FusedScanArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch*seq * hidden;

    cudaMalloc(&args->log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->log_values, args->N * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));
    cudaMalloc(&args->a_star, args->N * sizeof(float));
    cudaMalloc(&args->s_vals, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_coeffs, args->N * sizeof(float));
    cudaMalloc(&args->grad_log_values, args->N * sizeof(float));
    cudaMalloc(&args->grad_out, args->N * sizeof(float));

    float* buf = (float*)malloc(args->N * sizeof(float) * 3);
    float* log_coeffs_buf = buf;
    float* log_values_buf = buf + args->N;
    float* grad_out_buf = buf + args->N * 2;
    for (int i = 0; i < args->N; ++i) {
        log_coeffs_buf[i] = -logf(1.0f + expf(rand1() * 5.0f));
        log_values_buf[i] = -logf(1.0f + expf(rand1() * 5.0f));
        grad_out_buf[i] = rand1();
    }

    cudaMemcpy(args->log_coeffs, log_coeffs_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->log_values, log_values_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);

    free(buf);
    return args;
}

void free_fusedscanargs(FusedScanArgs* args) {
    cudaFree(args->log_coeffs);
    cudaFree(args->log_values);
    cudaFree(args->out);
    cudaFree(args->a_star);
    cudaFree(args->s_vals);
    cudaFree(args->grad_log_coeffs);
    cudaFree(args->grad_log_values);
    cudaFree(args->grad_out);
    free(args);
}

void run_fusedscan_forward(FusedScanArgs* args) {
    launch_fused_scan_forward<float>(
        args->out, args->a_star, args->s_vals,
        args->log_coeffs, args->log_values,
        args->T, args->H, args->B, 0);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    launch_fused_scan_backward<float>(
        args->grad_log_coeffs, args->grad_log_values, args->grad_out,
        args->log_coeffs, args->log_values, args->out,
        args->a_star, args->s_vals,
        args->T, args->H, args->B, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor log_coeffs;
    torch::Tensor log_values;
    torch::Tensor out;
    torch::Tensor grad_out;
    int N;
} FusedScanArgsTorch;

FusedScanArgsTorch* create_fusedscanargs_torch(FusedScanArgs* raw) {
    FusedScanArgsTorch* args = new FusedScanArgsTorch();
    args->N = raw->N;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->log_coeffs = torch::from_blob(raw->log_coeffs, {raw->B, raw->T, raw->H}, opts).requires_grad_(true);
    args->log_values = torch::from_blob(raw->log_values, {raw->B, raw->T, raw->H}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts);

    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan(args->log_coeffs, args->log_values);
}

void run_fusedscan_backward_torch(FusedScanArgsTorch* args) {
    args->log_coeffs.mutable_grad() = torch::Tensor();
    args->log_values.mutable_grad() = torch::Tensor();
    args->out.backward(args->grad_out, /*retain_graph=*/true);
}

void run_fusedscan_forward_cpp(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_cpp(args->log_coeffs, args->log_values);
}

#endif

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);

    printf("fused_scan (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    FusedScanArgsTorch* args_torch = create_fusedscanargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    args_torch->out = fused_scan(args_torch->log_coeffs, args_torch->log_values)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    args_torch->out = fused_scan_cpp(args_torch->log_coeffs, args_torch->log_values);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_fusedscanargs(args);
}

typedef struct {
    float* logits;
    float* values_pred;
    int64_t* actions;
    float* old_logprobs;
    float* advantages;
    float* prio;
    float* values;
    float* returns;
    float* adv_mean;
    float* adv_std;
    float* loss;
    double* saved_for_backward;
    float* grad_logits;
    float* grad_values_pred;
    float* grad_loss;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
} PPOLossArgs;

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions) {
    PPOLossArgs* args = (PPOLossArgs*)calloc(1, sizeof(PPOLossArgs));
    args->N = batch;
    args->T = seq;
    args->A = actions;

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(float));
    cudaMalloc(&args->values_pred, NT * sizeof(float));
    cudaMalloc(&args->actions, NT * sizeof(int64_t));
    cudaMalloc(&args->old_logprobs, NT * sizeof(float));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(float));
    cudaMalloc(&args->values, NT * sizeof(float));
    cudaMalloc(&args->returns, NT * sizeof(float));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_std, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    int64_t* actions_buf = (int64_t*)malloc(NT * sizeof(int64_t));

    float adv_sum = 0.0f, adv_sq_sum = 0.0f;
    for (int i = 0; i < NT; ++i) {
        advantages_buf[i] = rand1();
        adv_sum += advantages_buf[i];
        adv_sq_sum += advantages_buf[i] * advantages_buf[i];
    }
    float adv_mean = adv_sum / NT;
    float adv_std = sqrtf(adv_sq_sum / NT - adv_mean * adv_mean);

    for (int i = 0; i < NTA; ++i) {
        logits_buf[i] = rand1() * 2.0f;
    }
    for (int i = 0; i < NT; ++i) {
        values_pred_buf[i] = rand1();
        actions_buf[i] = rand() % actions;
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(args->logits, logits_buf, NTA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values_pred, values_pred_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(args->old_logprobs, old_logprobs_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->prio, prio_buf, batch * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->values, values_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->returns, returns_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_std, &adv_std, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    free(buf);
    free(actions_buf);
    return args;
}

void free_ppolossargs(PPOLossArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->values_pred);
    cudaFree(args->actions);
    cudaFree(args->old_logprobs);
    cudaFree(args->advantages);
    cudaFree(args->prio);
    cudaFree(args->values);
    cudaFree(args->returns);
    cudaFree(args->adv_mean);
    cudaFree(args->adv_std);
    cudaFree(args->loss);
    cudaFree(args->saved_for_backward);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    launch_ppo_loss_forward<float>(
        args->loss, args->saved_for_backward,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
}

void run_ppoloss_backward(PPOLossArgs* args) {
    launch_ppo_loss_backward<float>(
        args->grad_logits, args->grad_values_pred, args->grad_loss,
        args->logits, args->actions, args->old_logprobs,
        args->advantages, args->prio, args->values, args->returns,
        args->saved_for_backward, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;
    torch::Tensor values_pred;
    torch::Tensor actions;
    torch::Tensor old_logprobs;
    torch::Tensor advantages;
    torch::Tensor prio;
    torch::Tensor values;
    torch::Tensor returns;
    torch::Tensor adv_mean;
    torch::Tensor adv_std;
    torch::Tensor loss;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
} PPOLossArgsTorch;

PPOLossArgsTorch* create_ppolossargs_torch(PPOLossArgs* raw) {
    PPOLossArgsTorch* args = new PPOLossArgsTorch();
    args->N = raw->N;
    args->T = raw->T;
    args->A = raw->A;
    args->clip_coef = raw->clip_coef;
    args->vf_clip_coef = raw->vf_clip_coef;
    args->vf_coef = raw->vf_coef;
    args->ent_coef = raw->ent_coef;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    args->logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).requires_grad_(true);
    args->values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T}, opts).requires_grad_(true);
    args->actions = torch::from_blob(raw->actions, {raw->N, raw->T}, opts_int);
    args->old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts);
    args->advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, opts);
    args->prio = torch::from_blob(raw->prio, {raw->N}, opts);
    args->values = torch::from_blob(raw->values, {raw->N, raw->T}, opts);
    args->returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts);
    args->adv_mean = torch::from_blob(raw->adv_mean, {1}, opts);
    args->adv_std = torch::from_blob(raw->adv_std, {1}, opts);

    return args;
}

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_ppo_loss(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void run_ppoloss_forward_cpp(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_ppo_loss_cpp(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

#endif

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("\tforward", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("\tbackward", bwd_ms, NT);

#ifdef USE_TORCH
    PPOLossArgsTorch* args_torch = create_ppolossargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, NT);

    args_torch->loss = fused_ppo_loss(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, NT);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, NT);

    args_torch->loss = fused_ppo_loss_cpp(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, NT);

    delete args_torch;
#endif
    printf("\n");

    free_ppolossargs(args);
}

int main(int argc, char** argv) {
    warmup_gpu();
    profile_mingrugate(BR, H);
    profile_logcoeffsandvalues(BT, T, H);
    profile_logcumsumexp(BT, T, H);
    profile_fusedscan(BT, T, H);
    profile_ppoloss(BT, T, A);
    return 0;
}
