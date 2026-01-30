// profile_kernels.cu
// Minimal standalone profiler for CUDA kernels
//
// Without torch: nvcc -O3 -arch=sm_80 profile_kernels.cu -o profile_kernels -I.
// With torch:    Build with cmake/pytorch and -DUSE_TORCH
//
// Run: ./profile_kernels

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cmath>


#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/cuda/kernels.cu"
#include "pufferlib/extensions/modules.cpp"
#endif

#ifndef USE_TORCH
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

const int WARMUP_ITERS = 1000;
const int TIMING_ITERS = 10000;
const float TIMEOUT_SEC = 5.0f;

const int BUF = 2;
const int BR = 4096;  // Rollout batch (no T dim)
const int BT = 512;   // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;
const int INPUT_SIZE = 96;

typedef void (*kernel_fn)(void*);

void print_timing(const char* name, float ms, int N) {
    printf("  %-18s %6.1f us  %6.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

// Wall-clock time for timeout checks (only checked every BATCH_SIZE iters)
float get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9f;
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

const int BATCH_SIZE = 100;  // Check timeout every BATCH_SIZE iterations

float profile_kernel(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup with timeout
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_SIZE == 0) {
            cudaDeviceSynchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    cudaDeviceSynchronize();

    // Timed runs with timeout - check wall clock every BATCH_SIZE iters
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            fn(args);
        }
        iters += BATCH_SIZE;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
#ifdef USE_TORCH
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return ms / iters;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    // Warmup with timeout
    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_SIZE == 0) {
            warmup_stream.synchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    warmup_stream.synchronize();

    // Capture graph
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

    // Timed runs with timeout
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            cuda_graph.replay();
        }
        iters += BATCH_SIZE;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}
#endif

float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Fused mingru_gate for inference: takes combined (B, 1, 3*H) = [hidden, gate, proj]
// Outputs: out = sigmoid(proj) * mingru_out, next_state = mingru_out (for recurrence)
typedef struct {
    float* state;       // (B, 1, H) - input state
    float* combined;    // (B, 1, 3*H) = [hidden, gate, proj]
    float* out;         // (B, 1, H) - sigmoid(proj) * mingru_out
    float* next_state;  // (B, 1, H) - raw mingru_out
    int B;
    int H;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(float));
    cudaMalloc(&args->combined, N_combined * sizeof(float));
    cudaMalloc(&args->out, N_state * sizeof(float));
    cudaMalloc(&args->next_state, N_state * sizeof(float));

    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));

    // Initialize state with positive values
    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    // Initialize combined = [hidden, gate, proj]
    for (int b = 0; b < batch; ++b) {
        int base = b * 3 * hidden;
        for (int h = 0; h < hidden; ++h) {
            combined_buf[base + h] = rand1() * 5.0f;              // hidden
            combined_buf[base + hidden + h] = rand1() * 5.0f;     // gate
            combined_buf[base + 2 * hidden + h] = rand1() * 2.0f; // proj
        }
    }

    cudaMemcpy(args->state, state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->combined, combined_buf, N_combined * sizeof(float), cudaMemcpyHostToDevice);

    free(state_buf);
    free(combined_buf);
    return args;
}

void free_mingrugateargs(MingruGateArgs* args) {
    cudaFree(args->state);
    cudaFree(args->combined);
    cudaFree(args->out);
    cudaFree(args->next_state);
    free(args);
}

void run_mingrugate_forward(MingruGateArgs* args) {
    launch_mingru_gate_inference<float>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;     // (B, 1, H)
    torch::Tensor combined;  // (B, 1, 3*H)
    int B;
    int H;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->B = raw->B;
    args->H = raw->H;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts);
    args->combined = torch::from_blob(raw->combined, {raw->B, 1, 3 * raw->H}, opts);

    return args;
}

void run_mingrugate_forward_torch(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate(args->state, args->combined);
}

void run_mingrugate_forward_cpp(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate_cpp(args->state, args->combined);
}

void test_mingrugate_correct(MingruGateArgsTorch* args) {
    // Run CUDA kernel via torch wrapper
    auto kernel_outputs = mingru_gate(args->state, args->combined);
    auto kernel_out = kernel_outputs[0];
    auto kernel_next_state = kernel_outputs[1];

    // Run cpp reference
    auto cpp_outputs = mingru_gate_cpp(args->state, args->combined);
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    // Numerical comparison
    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(kernel_out, cpp_out, rtol, atol);
    float out_max_diff = (kernel_out - cpp_out).abs().max().item<float>();
    bool next_state_match = torch::allclose(kernel_next_state, cpp_next_state, rtol, atol);
    float next_state_max_diff = (kernel_next_state - cpp_next_state).abs().max().item<float>();

    printf("  correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);
}

#endif

void profile_mingrugate(int batch, int hidden) {
    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    MingruGateArgs* args = create_mingrugateargs(batch, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);

    test_mingrugate_correct(args_torch);

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

// New fused_scan takes combined (B, T, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Outputs: out (B, T, H) = sigmoid(proj) * scan_result, next_state (B, 1, H)
typedef struct {
    float* combined;       // (B, T, 3*H) = [hidden, gate, proj]
    float* state;          // (B, 1, H)
    float* out;            // (B, T, H)
    float* next_state;     // (B, 1, H)
    float* a_star;         // (B, T+1, H)
    float* s_vals;         // (B, T+1, H)
    float* log_values_buf; // (B, T+1, H)
    float* grad_combined;  // (B, T, 3*H)
    float* grad_state;     // (B, 1, H)
    float* grad_out;       // (B, T, H)
    float* grad_next_state;// (B, 1, H)
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
    args->N = batch * seq * hidden;

    int N_combined = batch * seq * 3 * hidden;
    int N_state = batch * hidden;
    int N_buf = batch * (seq + 1) * hidden;

    cudaMalloc(&args->combined, N_combined * sizeof(float));
    cudaMalloc(&args->state, N_state * sizeof(float));
    cudaMalloc(&args->out, args->N * sizeof(float));
    cudaMalloc(&args->next_state, N_state * sizeof(float));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(float));
    cudaMalloc(&args->grad_state, N_state * sizeof(float));
    cudaMalloc(&args->grad_out, args->N * sizeof(float));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(float));

    // Allocate and initialize host buffers
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));
    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* grad_out_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_next_state_buf = (float*)malloc(N_state * sizeof(float));

    // Initialize combined = [hidden, gate, proj] with reasonable values
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < hidden; ++h) {
                int base = b * seq * 3 * hidden + t * 3 * hidden;
                combined_buf[base + h] = rand1() * 5.0f;             // hidden
                combined_buf[base + hidden + h] = rand1() * 5.0f;    // gate
                combined_buf[base + 2 * hidden + h] = rand1() * 2.0f; // proj
            }
        }
    }
    // Initialize state with positive values (will be log'd)
    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    // Initialize gradients
    for (int i = 0; i < args->N; ++i) {
        grad_out_buf[i] = rand1();
    }
    for (int i = 0; i < N_state; ++i) {
        grad_next_state_buf[i] = rand1();
    }

    cudaMemcpy(args->combined, combined_buf, N_combined * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->state, state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, args->N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_next_state, grad_next_state_buf, N_state * sizeof(float), cudaMemcpyHostToDevice);

    free(combined_buf);
    free(state_buf);
    free(grad_out_buf);
    free(grad_next_state_buf);
    return args;
}

void free_fusedscanargs(FusedScanArgs* args) {
    cudaFree(args->combined);
    cudaFree(args->state);
    cudaFree(args->out);
    cudaFree(args->next_state);
    cudaFree(args->a_star);
    cudaFree(args->s_vals);
    cudaFree(args->log_values_buf);
    cudaFree(args->grad_combined);
    cudaFree(args->grad_state);
    cudaFree(args->grad_out);
    cudaFree(args->grad_next_state);
    free(args);
}

void run_fusedscan_forward(FusedScanArgs* args) {
    launch_fused_scan_forward<float>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B, 0);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    launch_fused_scan_backward<float>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B, 0);
}

void run_fusedscan_forward_checkpointed(FusedScanArgs* args) {
    launch_fused_scan_forward_checkpointed<float>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B, 0);
}

void run_fusedscan_backward_checkpointed(FusedScanArgs* args) {
    launch_fused_scan_backward_checkpointed<float>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor combined;    // (B, T, 3*H)
    torch::Tensor state;       // (B, 1, H)
    torch::Tensor out;         // (B, T, H)
    torch::Tensor next_state;  // (B, 1, H)
    torch::Tensor grad_out;    // (B, T, H)
    torch::Tensor grad_next_state; // (B, 1, H)
    int B;
    int T;
    int H;
} FusedScanArgsTorch;

FusedScanArgsTorch* create_fusedscanargs_torch(FusedScanArgs* raw) {
    FusedScanArgsTorch* args = new FusedScanArgsTorch();
    args->B = raw->B;
    args->T = raw->T;
    args->H = raw->H;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3 * raw->H}, opts).requires_grad_(true);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts);
    args->grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts);

    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan(args->combined, args->state);
}

void run_fusedscan_backward_torch(FusedScanArgsTorch* args) {
    args->combined.mutable_grad() = torch::Tensor();
    args->state.mutable_grad() = torch::Tensor();
    torch::autograd::backward(
        {args->out, args->next_state},
        {args->grad_out, args->grad_next_state},
        /*retain_graph=*/true);
}

void run_fusedscan_forward_cpp(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_cpp(args->combined, args->state);
}

void test_fusedscan_checkpointed_correct(FusedScanArgsTorch* args) {
    // Run reference (non-checkpointed) kernel forward
    auto combined_ref = args->combined.clone().requires_grad_(true);
    auto state_ref = args->state.clone().requires_grad_(true);
    combined_ref.retain_grad();
    state_ref.retain_grad();
    auto ref_outputs = fused_scan(combined_ref, state_ref);
    auto ref_out = ref_outputs[0];
    auto ref_next_state = ref_outputs[1];

    // Run checkpointed forward kernel via raw launch
    auto opts = torch::TensorOptions().dtype(args->combined.dtype()).device(torch::kCUDA);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    
    auto out_ckpt = torch::empty({args->B, args->T, args->H}, opts);
    auto next_state_ckpt = torch::empty({args->B, 1, args->H}, opts);
    auto a_star = torch::empty({args->B, args->T + 1, args->H}, opts_float);
    auto s_vals = torch::empty({args->B, args->T + 1, args->H}, opts_float);
    auto log_values_buf = torch::empty({args->B, args->T + 1, args->H}, opts_float);
    
    launch_fused_scan_forward_checkpointed<float>(
        out_ckpt.data_ptr<float>(),
        next_state_ckpt.data_ptr<float>(),
        a_star.data_ptr<float>(),
        s_vals.data_ptr<float>(),
        log_values_buf.data_ptr<float>(),
        args->combined.data_ptr<float>(),
        args->state.data_ptr<float>(),
        args->T, args->H, args->B,
        at::cuda::getCurrentCUDAStream());
    cudaDeviceSynchronize();

    // Numerical comparison - use same tolerances as other tests
    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(out_ckpt, ref_out, rtol, atol);
    float out_max_diff = (out_ckpt - ref_out).abs().max().item<float>();
    bool next_state_match = torch::allclose(next_state_ckpt, ref_next_state, rtol, atol);
    float next_state_max_diff = (next_state_ckpt - ref_next_state).abs().max().item<float>();

    printf("  checkpointed forward correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);

    // Test backward pass - run reference backward
    torch::autograd::backward({ref_out, ref_next_state}, {args->grad_out, args->grad_next_state});
    auto grad_combined_ref = combined_ref.grad().clone();
    auto grad_state_ref = state_ref.grad().clone();

    // Run checkpointed backward
    auto grad_combined_ckpt = torch::empty_like(args->combined);
    auto grad_state_ckpt = torch::empty_like(args->state);

    launch_fused_scan_backward_checkpointed<float>(
        grad_combined_ckpt.data_ptr<float>(),
        grad_state_ckpt.data_ptr<float>(),
        args->grad_out.data_ptr<float>(),
        args->grad_next_state.data_ptr<float>(),
        args->combined.data_ptr<float>(),
        args->state.data_ptr<float>(),
        a_star.data_ptr<float>(),
        s_vals.data_ptr<float>(),
        log_values_buf.data_ptr<float>(),
        args->T, args->H, args->B,
        at::cuda::getCurrentCUDAStream());
    cudaDeviceSynchronize();

    bool grad_combined_match = torch::allclose(grad_combined_ckpt, grad_combined_ref, rtol, atol);
    float grad_combined_max_diff = (grad_combined_ckpt - grad_combined_ref).abs().max().item<float>();
    bool grad_state_match = torch::allclose(grad_state_ckpt, grad_state_ref, rtol, atol);
    float grad_state_max_diff = (grad_state_ckpt - grad_state_ref).abs().max().item<float>();

    printf("  checkpointed backward correctness: grad_combined=%s(%.2e) grad_state=%s(%.2e)\n",
           grad_combined_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_combined_max_diff,
           grad_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_state_max_diff);
}

#endif

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);

    printf("fused_scan (N=%d, %dx%dx%d, combined=%dx%dx%d)\n",
           args->N, batch, seq, hidden, batch, seq, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("\tforward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("\tbackward", bwd_ms, batch*seq);

    float fwd_ckpt_ms = profile_kernel((kernel_fn)run_fusedscan_forward_checkpointed, args);
    print_timing("\tforward (checkpointed)", fwd_ckpt_ms, batch*seq);

    float bwd_ckpt_ms = profile_kernel((kernel_fn)run_fusedscan_backward_checkpointed, args);
    print_timing("\tbackward (checkpointed)", bwd_ckpt_ms, batch*seq);

#ifdef USE_TORCH
    FusedScanArgsTorch* args_torch = create_fusedscanargs_torch(args);

    test_fusedscan_checkpointed_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch*seq);

    auto scan_out = fused_scan(args_torch->combined, args_torch->state);
    args_torch->out = scan_out[0];
    args_torch->next_state = scan_out[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch*seq);

    auto scan_out_cpp = fused_scan_cpp(args_torch->combined, args_torch->state);
    args_torch->out = scan_out_cpp[0];
    args_torch->next_state = scan_out_cpp[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");

    free_fusedscanargs(args);
}

// =============================================================================
// FCMax: Simple FC -> Max kernel (no intermediate ReLU layer)
// Input: x (B, N, D_in), W (D_out, D_in), b (D_out)
// Output: (B, D_out) = max_over_N(x @ W.T + b)
// =============================================================================

typedef struct {
    float* x;              // (B, N, D_in)
    float* W;              // (D_out, D_in)
    float* b;              // (D_out)
    float* out;            // (B, D_out)
    int* argmax_indices;   // (B, D_out)
    float* grad_x;         // (B, N, D_in)
    float* grad_W;         // (D_out, D_in)
    float* grad_b;         // (D_out)
    float* grad_out;       // (B, D_out)
    int B;
    int N;
    int D_in;
    int D_out;
} FCMaxArgs;

FCMaxArgs* create_fcmaxargs(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = (FCMaxArgs*)calloc(1, sizeof(FCMaxArgs));
    args->B = batch;
    args->N = num_points;
    args->D_in = d_in;
    args->D_out = d_out;

    int N_x = batch * num_points * d_in;
    int N_W = d_out * d_in;
    int N_out = batch * d_out;

    cudaMalloc(&args->x, N_x * sizeof(float));
    cudaMalloc(&args->W, N_W * sizeof(float));
    cudaMalloc(&args->b, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(float));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W, N_W * sizeof(float));
    cudaMalloc(&args->grad_b, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(float));

    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W_buf = (float*)malloc(N_W * sizeof(float));
    float* b_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W; ++i) W_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    cudaMemcpy(args->x, x_buf, N_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->W, W_buf, N_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b, b_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, N_out * sizeof(float), cudaMemcpyHostToDevice);

    free(x_buf);
    free(W_buf);
    free(b_buf);
    free(grad_out_buf);
    return args;
}

void free_fcmaxargs(FCMaxArgs* args) {
    cudaFree(args->x);
    cudaFree(args->W);
    cudaFree(args->b);
    cudaFree(args->out);
    cudaFree(args->argmax_indices);
    cudaFree(args->grad_x);
    cudaFree(args->grad_W);
    cudaFree(args->grad_b);
    cudaFree(args->grad_out);
    free(args);
}

void run_fcmax_forward(FCMaxArgs* args) {
    launch_fc_max_forward_float(
        args->out, args->argmax_indices,
        args->x, args->W, args->b,
        args->B, args->N, args->D_in, args->D_out, 0);
}

void run_fcmax_backward(FCMaxArgs* args) {
    cudaMemset(args->grad_x, 0, args->B * args->N * args->D_in * sizeof(float));
    cudaMemset(args->grad_W, 0, args->D_out * args->D_in * sizeof(float));
    cudaMemset(args->grad_b, 0, args->D_out * sizeof(float));

    launch_fc_max_backward_float(
        args->grad_x, args->grad_W, args->grad_b,
        args->grad_out, args->x, args->W,
        args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;       // (B, N, D_in)
    torch::Tensor W;       // (D_out, D_in)
    torch::Tensor b;       // (D_out)
    torch::Tensor out;     // (B, D_out)
    torch::Tensor grad_out;// (B, D_out)
    int B, N, D_in, D_out;
} FCMaxArgsTorch;

FCMaxArgsTorch* create_fcmaxargs_torch(FCMaxArgs* raw) {
    FCMaxArgsTorch* args = new FCMaxArgsTorch();
    args->B = raw->B;
    args->N = raw->N;
    args->D_in = raw->D_in;
    args->D_out = raw->D_out;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->x = torch::from_blob(raw->x, {raw->B, raw->N, raw->D_in}, opts).clone().requires_grad_(true);
    args->W = torch::from_blob(raw->W, {raw->D_out, raw->D_in}, opts).clone().requires_grad_(true);
    args->b = torch::from_blob(raw->b, {raw->D_out}, opts).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->D_out}, opts).clone();

    return args;
}

void run_fcmax_forward_torch(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max(args->x, args->W, args->b);
}

void run_fcmax_backward_torch(FCMaxArgsTorch* args) {
    // Recompute forward each time since backward frees the graph
    auto out = fc_max(args->x, args->W, args->b);
    if (args->x.grad().defined()) args->x.grad().zero_();
    if (args->W.grad().defined()) args->W.grad().zero_();
    if (args->b.grad().defined()) args->b.grad().zero_();
    out.backward(args->grad_out);
}

void run_fcmax_forward_cpp(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max_cpp(args->x, args->W, args->b);
}

void test_fcmax_correct(FCMaxArgsTorch* args) {
    auto x_fused = args->x.detach().clone().requires_grad_(true);
    auto W_fused = args->W.detach().clone().requires_grad_(true);
    auto b_fused = args->b.detach().clone().requires_grad_(true);
    auto fused_out = fc_max(x_fused, W_fused, b_fused);

    auto x_ref = args->x.detach().clone().requires_grad_(true);
    auto W_ref = args->W.detach().clone().requires_grad_(true);
    auto b_ref = args->b.detach().clone().requires_grad_(true);
    auto ref_out = fc_max_cpp(x_ref, W_ref, b_ref);

    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(fused_out, ref_out, rtol, atol);
    float out_max_diff = (fused_out - ref_out).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Backward
    fused_out.backward(args->grad_out);
    ref_out.backward(args->grad_out);

    bool grad_x_match = torch::allclose(x_fused.grad(), x_ref.grad(), rtol, atol);
    float grad_x_max_diff = (x_fused.grad() - x_ref.grad()).abs().max().item<float>();
    bool grad_W_match = torch::allclose(W_fused.grad(), W_ref.grad(), rtol, atol);
    float grad_W_max_diff = (W_fused.grad() - W_ref.grad()).abs().max().item<float>();

    printf("  backward correctness: grad_x=%s(%.2e) grad_W=%s(%.2e)\n",
           grad_x_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_x_max_diff,
           grad_W_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_W_max_diff);
}

#endif

void profile_fcmax(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = create_fcmaxargs(batch, num_points, d_in, d_out);

    printf("fc_max (B=%d, N=%d, D_in=%d, D_out=%d)\n", batch, num_points, d_in, d_out);

    float fwd_ms = profile_kernel((kernel_fn)run_fcmax_forward, args);
    print_timing("\tforward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcmax_backward, args);
    print_timing("\tbackward", bwd_ms, batch);

#ifdef USE_TORCH
    FCMaxArgsTorch* args_torch = create_fcmaxargs_torch(args);

    test_fcmax_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    args_torch->out = fc_max(args_torch->x, args_torch->W, args_torch->b);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_fcmaxargs(args);
}

// FCReluFCMax: Fused FC -> ReLU -> FC -> Max kernel
// Input: x (B, N, D_in), W1 (D_mid, D_in), b1 (D_mid), W2 (D_out, D_mid), b2 (D_out)
// Output: (B, D_out) = max_over_N(FC2(ReLU(FC1(x))))
typedef struct {
    float* x;              // (B, N, D_in)
    float* W1;             // (D_mid, D_in)
    float* b1;             // (D_mid)
    float* W2;             // (D_out, D_mid)
    float* b2;             // (D_out)
    float* out;            // (B, D_out)
    int* argmax_indices;   // (B, D_out)
    float* fc1_at_argmax;  // (B, D_out, D_mid)
    float* grad_x;         // (B, N, D_in)
    float* grad_W1;        // (D_mid, D_in)
    float* grad_b1;        // (D_mid)
    float* grad_W2;        // (D_out, D_mid)
    float* grad_b2;        // (D_out)
    float* grad_out;       // (B, D_out)
    int B;
    int N;
    int D_in;
    int D_mid;
    int D_out;
} FCReluFCMaxArgs;

FCReluFCMaxArgs* create_fcrelufcmaxargs(int batch, int num_points, int d_in, int d_mid, int d_out) {
    FCReluFCMaxArgs* args = (FCReluFCMaxArgs*)calloc(1, sizeof(FCReluFCMaxArgs));
    args->B = batch;
    args->N = num_points;
    args->D_in = d_in;
    args->D_mid = d_mid;
    args->D_out = d_out;

    int N_x = batch * num_points * d_in;
    int N_W1 = d_mid * d_in;
    int N_W2 = d_out * d_mid;
    int N_out = batch * d_out;
    int N_fc1_at_argmax = batch * d_out * d_mid;

    cudaMalloc(&args->x, N_x * sizeof(float));
    cudaMalloc(&args->W1, N_W1 * sizeof(float));
    cudaMalloc(&args->b1, d_mid * sizeof(float));
    cudaMalloc(&args->W2, N_W2 * sizeof(float));
    cudaMalloc(&args->b2, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(float));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->fc1_at_argmax, N_fc1_at_argmax * sizeof(float));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W1, N_W1 * sizeof(float));
    cudaMalloc(&args->grad_b1, d_mid * sizeof(float));
    cudaMalloc(&args->grad_W2, N_W2 * sizeof(float));
    cudaMalloc(&args->grad_b2, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(float));

    // Allocate and initialize host buffers
    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W1_buf = (float*)malloc(N_W1 * sizeof(float));
    float* b1_buf = (float*)malloc(d_mid * sizeof(float));
    float* W2_buf = (float*)malloc(N_W2 * sizeof(float));
    float* b2_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W1; ++i) W1_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_mid; ++i) b1_buf[i] = 0.0f;
    for (int i = 0; i < N_W2; ++i) W2_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b2_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    cudaMemcpy(args->x, x_buf, N_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->W1, W1_buf, N_W1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b1, b1_buf, d_mid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->W2, W2_buf, N_W2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b2, b2_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->grad_out, grad_out_buf, N_out * sizeof(float), cudaMemcpyHostToDevice);

    free(x_buf);
    free(W1_buf);
    free(b1_buf);
    free(W2_buf);
    free(b2_buf);
    free(grad_out_buf);
    return args;
}

void free_fcrelufcmaxargs(FCReluFCMaxArgs* args) {
    cudaFree(args->x);
    cudaFree(args->W1);
    cudaFree(args->b1);
    cudaFree(args->W2);
    cudaFree(args->b2);
    cudaFree(args->out);
    cudaFree(args->argmax_indices);
    cudaFree(args->fc1_at_argmax);
    cudaFree(args->grad_x);
    cudaFree(args->grad_W1);
    cudaFree(args->grad_b1);
    cudaFree(args->grad_W2);
    cudaFree(args->grad_b2);
    cudaFree(args->grad_out);
    free(args);
}

void run_fcrelufcmax_forward(FCReluFCMaxArgs* args) {
    launch_fc_relu_fc_max_forward<float>(
        args->out, args->argmax_indices, args->fc1_at_argmax,
        args->x, args->W1, args->b1, args->W2, args->b2,
        args->B, args->N, args->D_in, args->D_mid, args->D_out, 0);
}

void run_fcrelufcmax_backward(FCReluFCMaxArgs* args) {
    // Zero gradients before backward (kernel uses atomicAdd)
    cudaMemset(args->grad_x, 0, args->B * args->N * args->D_in * sizeof(float));
    cudaMemset(args->grad_W1, 0, args->D_mid * args->D_in * sizeof(float));
    cudaMemset(args->grad_b1, 0, args->D_mid * sizeof(float));
    cudaMemset(args->grad_W2, 0, args->D_out * args->D_mid * sizeof(float));
    cudaMemset(args->grad_b2, 0, args->D_out * sizeof(float));

    launch_fc_relu_fc_max_backward<float>(
        args->grad_x, args->grad_W1, args->grad_b1, args->grad_W2, args->grad_b2,
        args->grad_out, args->x, args->W1, args->W2,
        args->argmax_indices, args->fc1_at_argmax,
        args->B, args->N, args->D_in, args->D_mid, args->D_out, 0);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;       // (B, N, D_in)
    torch::Tensor W1;      // (D_mid, D_in)
    torch::Tensor b1;      // (D_mid)
    torch::Tensor W2;      // (D_out, D_mid)
    torch::Tensor b2;      // (D_out)
    torch::Tensor out;     // (B, D_out)
    torch::Tensor grad_out;// (B, D_out)
    int B;
    int N;
    int D_in;
    int D_mid;
    int D_out;
} FCReluFCMaxArgsTorch;

FCReluFCMaxArgsTorch* create_fcrelufcmaxargs_torch(FCReluFCMaxArgs* raw) {
    FCReluFCMaxArgsTorch* args = new FCReluFCMaxArgsTorch();
    args->B = raw->B;
    args->N = raw->N;
    args->D_in = raw->D_in;
    args->D_mid = raw->D_mid;
    args->D_out = raw->D_out;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->x = torch::from_blob(raw->x, {raw->B, raw->N, raw->D_in}, opts).requires_grad_(true);
    args->W1 = torch::from_blob(raw->W1, {raw->D_mid, raw->D_in}, opts).requires_grad_(true);
    args->b1 = torch::from_blob(raw->b1, {raw->D_mid}, opts).requires_grad_(true);
    args->W2 = torch::from_blob(raw->W2, {raw->D_out, raw->D_mid}, opts).requires_grad_(true);
    args->b2 = torch::from_blob(raw->b2, {raw->D_out}, opts).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->D_out}, opts);

    return args;
}

void run_fcrelufcmax_forward_torch(FCReluFCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_relu_fc_max(args->x, args->W1, args->b1, args->W2, args->b2);
}

void run_fcrelufcmax_backward_torch(FCReluFCMaxArgsTorch* args) {
    // Recompute forward each time since backward frees the graph
    auto out = fc_relu_fc_max(args->x, args->W1, args->b1, args->W2, args->b2);
    args->x.mutable_grad() = torch::Tensor();
    args->W1.mutable_grad() = torch::Tensor();
    args->b1.mutable_grad() = torch::Tensor();
    args->W2.mutable_grad() = torch::Tensor();
    args->b2.mutable_grad() = torch::Tensor();
    out.backward(args->grad_out);
}

void run_fcrelufcmax_forward_cpp(FCReluFCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_relu_fc_max_cpp(args->x, args->W1, args->b1, args->W2, args->b2);
}

void test_fcrelufcmax_correct(FCReluFCMaxArgsTorch* args) {
    // Run fused kernel forward
    auto x_fused = args->x.detach().clone().requires_grad_(true);
    auto W1_fused = args->W1.detach().clone().requires_grad_(true);
    auto b1_fused = args->b1.detach().clone().requires_grad_(true);
    auto W2_fused = args->W2.detach().clone().requires_grad_(true);
    auto b2_fused = args->b2.detach().clone().requires_grad_(true);
    auto fused_out = fc_relu_fc_max(x_fused, W1_fused, b1_fused, W2_fused, b2_fused);

    // Run reference (unfused) forward
    auto x_ref = args->x.detach().clone().requires_grad_(true);
    auto W1_ref = args->W1.detach().clone().requires_grad_(true);
    auto b1_ref = args->b1.detach().clone().requires_grad_(true);
    auto W2_ref = args->W2.detach().clone().requires_grad_(true);
    auto b2_ref = args->b2.detach().clone().requires_grad_(true);
    auto ref_out = fc_relu_fc_max_cpp(x_ref, W1_ref, b1_ref, W2_ref, b2_ref);

    // Numerical comparison
    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(fused_out, ref_out, rtol, atol);
    float out_max_diff = (fused_out - ref_out).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Test backward pass
    torch::autograd::backward({fused_out}, {args->grad_out});
    torch::autograd::backward({ref_out}, {args->grad_out});

    bool grad_x_match = torch::allclose(x_fused.grad(), x_ref.grad(), rtol, atol);
    float grad_x_max_diff = (x_fused.grad() - x_ref.grad()).abs().max().item<float>();
    bool grad_W1_match = torch::allclose(W1_fused.grad(), W1_ref.grad(), rtol, atol);
    float grad_W1_max_diff = (W1_fused.grad() - W1_ref.grad()).abs().max().item<float>();
    bool grad_W2_match = torch::allclose(W2_fused.grad(), W2_ref.grad(), rtol, atol);
    float grad_W2_max_diff = (W2_fused.grad() - W2_ref.grad()).abs().max().item<float>();

    printf("  backward correctness: grad_x=%s(%.2e) grad_W1=%s(%.2e) grad_W2=%s(%.2e)\n",
           grad_x_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_x_max_diff,
           grad_W1_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_W1_max_diff,
           grad_W2_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_W2_max_diff);
}

#endif

void profile_fcrelufcmax(int batch, int num_points, int d_in, int d_mid, int d_out) {
    FCReluFCMaxArgs* args = create_fcrelufcmaxargs(batch, num_points, d_in, d_mid, d_out);

    printf("fc_relu_fc_max (B=%d, N=%d, D_in=%d, D_mid=%d, D_out=%d)\n",
           batch, num_points, d_in, d_mid, d_out);

    float fwd_ms = profile_kernel((kernel_fn)run_fcrelufcmax_forward, args);
    print_timing("\tforward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcrelufcmax_backward, args);
    print_timing("\tbackward", bwd_ms, batch);

#ifdef USE_TORCH
    FCReluFCMaxArgsTorch* args_torch = create_fcrelufcmaxargs_torch(args);

    test_fcrelufcmax_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fcrelufcmax_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    args_torch->out = fc_relu_fc_max(args_torch->x, args_torch->W1, args_torch->b1, args_torch->W2, args_torch->b2);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fcrelufcmax_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fcrelufcmax_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    args_torch->out = fc_relu_fc_max_cpp(args_torch->x, args_torch->W1, args_torch->b1, args_torch->W2, args_torch->b2);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fcrelufcmax_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fcrelufcmax_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_fcrelufcmaxargs(args);
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
    float* adv_var;  // variance, kernel does sqrt
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
    int logits_stride_n;
    int logits_stride_t;
    int logits_stride_a;
    int values_stride_n;
    int values_stride_t;
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
    cudaMalloc(&args->adv_var, sizeof(float));
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
    float adv_var = adv_sq_sum / NT - adv_mean * adv_mean;

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
    cudaMemcpy(args->adv_var, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    args->logits_stride_n = seq * actions;  // T * A
    args->logits_stride_t = actions;        // A
    args->logits_stride_a = 1;
    args->values_stride_n = seq;            // T
    args->values_stride_t = 1;

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
    cudaFree(args->adv_var);
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
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
}

void run_ppoloss_backward(PPOLossArgs* args) {
    launch_ppo_loss_backward<float>(
        args->grad_logits, args->grad_values_pred, args->grad_loss,
        args->logits, args->actions, args->old_logprobs,
        args->advantages, args->prio, args->values, args->returns,
        args->saved_for_backward, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);
}

void run_ppoloss_forward_opt(PPOLossArgs* args) {
    launch_ppo_loss_forward_optimized<float>(
        args->loss, args->saved_for_backward,
        nullptr, nullptr,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        0);
}

void run_ppoloss_backward_opt(PPOLossArgs* args) {
    launch_ppo_loss_backward_optimized<float>(
        args->grad_logits, args->grad_values_pred, args->grad_loss,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        0);
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
    torch::Tensor adv_var;  // variance, kernel computes sqrt
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
    args->adv_var = torch::from_blob(raw->adv_var, {1}, opts);

    return args;
}

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto adv_std = args->adv_var.sqrt();  // fused_ppo_loss expects std, not var
    fused_ppo_loss(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void run_ppoloss_forward_cpp(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto adv_std = args->adv_var.sqrt();  // fused_ppo_loss_cpp expects std, not var
    fused_ppo_loss_cpp(
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, adv_std,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

#endif

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    int NTA = batch*seq*actions;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("\tforward (original)", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("\tbackward (original)", bwd_ms, NT);

    float fwd_opt_ms = profile_kernel((kernel_fn)run_ppoloss_forward_opt, args);
    print_timing("\tforward (optimized)", fwd_opt_ms, NT);

    float bwd_opt_ms = profile_kernel((kernel_fn)run_ppoloss_backward_opt, args);
    print_timing("\tbackward (optimized)", bwd_opt_ms, NT);

#ifdef USE_TORCH
    PPOLossArgsTorch* args_torch = create_ppolossargs_torch(args);
    auto adv_std = args_torch->adv_var.sqrt();  // fused_ppo_loss/cpp expect std, not var

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, NT);

    args_torch->loss = fused_ppo_loss(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (torch)", bwd_torch_ms, NT);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, NT);

    args_torch->loss = fused_ppo_loss_cpp(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, adv_std,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("\tbackward (cpp)", bwd_cpp_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_cpp, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, NT);

    // ========================================================================
    // Numerical Stability Comparison: Torch vs Original vs Optimized
    // ========================================================================
    printf("\n\tNumerical Stability Comparison:\n");

    // Allocate separate output buffers for each implementation
    float* loss_orig = nullptr;
    float* loss_opt = nullptr;
    double* saved_orig = nullptr;
    double* saved_opt = nullptr;
    float* grad_logits_orig = nullptr;
    float* grad_logits_opt = nullptr;
    float* grad_values_orig = nullptr;
    float* grad_values_opt = nullptr;

    cudaMalloc(&loss_orig, sizeof(float));
    cudaMalloc(&loss_opt, sizeof(float));
    cudaMalloc(&saved_orig, NT * 5 * sizeof(double));
    cudaMalloc(&saved_opt, NT * 5 * sizeof(double));
    cudaMalloc(&grad_logits_orig, NTA * sizeof(float));
    cudaMalloc(&grad_logits_opt, NTA * sizeof(float));
    cudaMalloc(&grad_values_orig, NT * sizeof(float));
    cudaMalloc(&grad_values_opt, NT * sizeof(float));

    // Zero loss buffers before kernel calls (they use atomicAdd)
    cudaMemset(loss_orig, 0, sizeof(float));
    cudaMemset(loss_opt, 0, sizeof(float));

    // Run original forward
    launch_ppo_loss_forward<float>(
        loss_orig, saved_orig,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);

    // Run optimized forward
    launch_ppo_loss_forward_optimized<float>(
        loss_opt, saved_opt,
        nullptr, nullptr,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        0);

    // Run pure PyTorch reference (ground truth) - fused_ppo_loss_cpp is the correct implementation
    auto adv_std_torch = args_torch->adv_var.sqrt();  // fused_ppo_loss_cpp expects std
    torch::Tensor torch_loss = fused_ppo_loss_cpp(
        args_torch->logits, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, adv_std_torch,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef);

    cudaDeviceSynchronize();

    // Copy results to host
    float h_loss_orig, h_loss_opt;
    cudaMemcpy(&h_loss_orig, loss_orig, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_loss_opt, loss_opt, sizeof(float), cudaMemcpyDeviceToHost);
    float h_loss_torch = torch_loss.item<float>();

    // Check for NaN/Inf
    bool orig_nan = std::isnan(h_loss_orig) || std::isinf(h_loss_orig);
    bool opt_nan = std::isnan(h_loss_opt) || std::isinf(h_loss_opt);
    bool torch_nan = std::isnan(h_loss_torch) || std::isinf(h_loss_torch);

    printf("\t  Forward Loss Values:\n");
    printf("\t    PyTorch:   %.8f %s\n", h_loss_torch, torch_nan ? "(NaN/Inf!)" : "");
    printf("\t    Original:  %.8f %s\n", h_loss_orig, orig_nan ? "(NaN/Inf!)" : "");
    printf("\t    Optimized: %.8f %s\n", h_loss_opt, opt_nan ? "(NaN/Inf!)" : "");

    // Compute relative differences
    float diff_orig_torch = fabsf(h_loss_orig - h_loss_torch) / (fabsf(h_loss_torch) + 1e-8f);
    float diff_opt_torch = fabsf(h_loss_opt - h_loss_torch) / (fabsf(h_loss_torch) + 1e-8f);
    float diff_orig_opt = fabsf(h_loss_orig - h_loss_opt) / (fabsf(h_loss_orig) + 1e-8f);

    printf("\t  Relative Differences:\n");
    printf("\t    Original vs PyTorch:   %.2e %s\n", diff_orig_torch, diff_orig_torch > 0.01f ? "(>1%% MISMATCH)" : "(OK)");
    printf("\t    Optimized vs PyTorch:  %.2e %s\n", diff_opt_torch, diff_opt_torch > 0.01f ? "(>1%% MISMATCH)" : "(OK)");
    printf("\t    Original vs Optimized: %.2e %s\n", diff_orig_opt, diff_orig_opt > 1e-5f ? "(DIFF)" : "(OK)");
    fflush(stdout);

    // Run backward passes
    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    launch_ppo_loss_backward<float>(
        grad_logits_orig, grad_values_orig, args->grad_loss,
        args->logits, args->actions, args->old_logprobs,
        args->advantages, args->prio, args->values, args->returns,
        saved_orig, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N, 0);

    launch_ppo_loss_backward_optimized<float>(
        grad_logits_opt, grad_values_opt, args->grad_loss,
        args->logits, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        0);

    // Run torch backward using fused_ppo_loss_cpp (pure PyTorch, has proper autograd)
    bool torch_backward_ok = false;
    try {
        args_torch->logits.mutable_grad() = torch::Tensor();
        args_torch->values_pred.mutable_grad() = torch::Tensor();
        torch_loss.backward();
        torch_backward_ok = args_torch->logits.grad().defined() && args_torch->values_pred.grad().defined();
    } catch (...) {
        torch_backward_ok = false;
    }

    cudaDeviceSynchronize();

    // Copy gradients to host for comparison
    float* h_grad_logits_orig = (float*)malloc(NTA * sizeof(float));
    float* h_grad_logits_opt = (float*)malloc(NTA * sizeof(float));
    float* h_grad_values_orig = (float*)malloc(NT * sizeof(float));
    float* h_grad_values_opt = (float*)malloc(NT * sizeof(float));

    cudaMemcpy(h_grad_logits_orig, grad_logits_orig, NTA * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_logits_opt, grad_logits_opt, NTA * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_values_orig, grad_values_orig, NT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_values_opt, grad_values_opt, NT * sizeof(float), cudaMemcpyDeviceToHost);

    // Get torch gradients - check if backward succeeded
    float* h_grad_logits_torch = nullptr;
    float* h_grad_values_torch = nullptr;
    bool has_torch_grads = torch_backward_ok;
    
    torch::Tensor torch_grad_logits, torch_grad_values;
    if (has_torch_grads) {
        torch_grad_logits = args_torch->logits.grad().contiguous().cpu();
        torch_grad_values = args_torch->values_pred.grad().contiguous().cpu();
        h_grad_logits_torch = torch_grad_logits.data_ptr<float>();
        h_grad_values_torch = torch_grad_values.data_ptr<float>();
    }
    
    // Debug: check torch gradient sizes
    if (has_torch_grads) {
        // Safety check - if sizes don't match, skip torch comparison
        if (torch_grad_logits.numel() != NTA || torch_grad_values.numel() != NT) {
            has_torch_grads = false;
        }
    }

    // Compute gradient statistics
    double grad_logits_orig_torch_diff = 0.0, grad_logits_opt_torch_diff = 0.0, grad_logits_orig_opt_diff = 0.0;
    double grad_logits_orig_torch_max = 0.0, grad_logits_opt_torch_max = 0.0, grad_logits_orig_opt_max = 0.0;
    int grad_logits_nan_orig = 0, grad_logits_nan_opt = 0, grad_logits_nan_torch = 0;

    for (int i = 0; i < NTA; ++i) {
        if (std::isnan(h_grad_logits_orig[i]) || std::isinf(h_grad_logits_orig[i])) grad_logits_nan_orig++;
        if (std::isnan(h_grad_logits_opt[i]) || std::isinf(h_grad_logits_opt[i])) grad_logits_nan_opt++;

        double d3 = fabs((double)h_grad_logits_orig[i] - (double)h_grad_logits_opt[i]);
        grad_logits_orig_opt_diff += d3;
        grad_logits_orig_opt_max = fmax(grad_logits_orig_opt_max, d3);

        if (has_torch_grads) {
            if (std::isnan(h_grad_logits_torch[i]) || std::isinf(h_grad_logits_torch[i])) grad_logits_nan_torch++;
            double d1 = fabs((double)h_grad_logits_orig[i] - (double)h_grad_logits_torch[i]);
            double d2 = fabs((double)h_grad_logits_opt[i] - (double)h_grad_logits_torch[i]);
            grad_logits_orig_torch_diff += d1;
            grad_logits_opt_torch_diff += d2;
            grad_logits_orig_torch_max = fmax(grad_logits_orig_torch_max, d1);
            grad_logits_opt_torch_max = fmax(grad_logits_opt_torch_max, d2);
        }
    }

    double grad_values_orig_torch_diff = 0.0, grad_values_opt_torch_diff = 0.0, grad_values_orig_opt_diff = 0.0;
    double grad_values_orig_torch_max = 0.0, grad_values_opt_torch_max = 0.0, grad_values_orig_opt_max = 0.0;
    int grad_values_nan_orig = 0, grad_values_nan_opt = 0, grad_values_nan_torch = 0;

    for (int i = 0; i < NT; ++i) {
        if (std::isnan(h_grad_values_orig[i]) || std::isinf(h_grad_values_orig[i])) grad_values_nan_orig++;
        if (std::isnan(h_grad_values_opt[i]) || std::isinf(h_grad_values_opt[i])) grad_values_nan_opt++;

        double d3 = fabs((double)h_grad_values_orig[i] - (double)h_grad_values_opt[i]);
        grad_values_orig_opt_diff += d3;
        grad_values_orig_opt_max = fmax(grad_values_orig_opt_max, d3);

        if (has_torch_grads) {
            if (std::isnan(h_grad_values_torch[i]) || std::isinf(h_grad_values_torch[i])) grad_values_nan_torch++;
            double d1 = fabs((double)h_grad_values_orig[i] - (double)h_grad_values_torch[i]);
            double d2 = fabs((double)h_grad_values_opt[i] - (double)h_grad_values_torch[i]);
            grad_values_orig_torch_diff += d1;
            grad_values_opt_torch_diff += d2;
            grad_values_orig_torch_max = fmax(grad_values_orig_torch_max, d1);
            grad_values_opt_torch_max = fmax(grad_values_opt_torch_max, d2);
        }
    }

    printf("\t  Backward grad_logits (NTA=%d):\n", NTA);
    if (has_torch_grads) {
        printf("\t    NaN/Inf counts: torch=%d, orig=%d, opt=%d\n", grad_logits_nan_torch, grad_logits_nan_orig, grad_logits_nan_opt);
        printf("\t    Mean abs diff: orig-torch=%.2e, opt-torch=%.2e, orig-opt=%.2e\n",
               grad_logits_orig_torch_diff/NTA, grad_logits_opt_torch_diff/NTA, grad_logits_orig_opt_diff/NTA);
        printf("\t    Max abs diff:  orig-torch=%.2e, opt-torch=%.2e, orig-opt=%.2e\n",
               grad_logits_orig_torch_max, grad_logits_opt_torch_max, grad_logits_orig_opt_max);
    } else {
        printf("\t    NaN/Inf counts: orig=%d, opt=%d (torch grads unavailable)\n", grad_logits_nan_orig, grad_logits_nan_opt);
        printf("\t    Mean abs diff: orig-opt=%.2e\n", grad_logits_orig_opt_diff/NTA);
        printf("\t    Max abs diff:  orig-opt=%.2e\n", grad_logits_orig_opt_max);
    }

    printf("\t  Backward grad_values (NT=%d):\n", NT);
    if (has_torch_grads) {
        printf("\t    NaN/Inf counts: torch=%d, orig=%d, opt=%d\n", grad_values_nan_torch, grad_values_nan_orig, grad_values_nan_opt);
        printf("\t    Mean abs diff: orig-torch=%.2e, opt-torch=%.2e, orig-opt=%.2e\n",
               grad_values_orig_torch_diff/NT, grad_values_opt_torch_diff/NT, grad_values_orig_opt_diff/NT);
        printf("\t    Max abs diff:  orig-torch=%.2e, opt-torch=%.2e, orig-opt=%.2e\n",
               grad_values_orig_torch_max, grad_values_opt_torch_max, grad_values_orig_opt_max);
    } else {
        printf("\t    NaN/Inf counts: orig=%d, opt=%d (torch grads unavailable)\n", grad_values_nan_orig, grad_values_nan_opt);
        printf("\t    Mean abs diff: orig-opt=%.2e\n", grad_values_orig_opt_diff/NT);
        printf("\t    Max abs diff:  orig-opt=%.2e\n", grad_values_orig_opt_max);
    }

    // Cleanup
    free(h_grad_logits_orig);
    free(h_grad_logits_opt);
    free(h_grad_values_orig);
    free(h_grad_values_opt);
    cudaFree(loss_orig);
    cudaFree(loss_opt);
    cudaFree(saved_orig);
    cudaFree(saved_opt);
    cudaFree(grad_logits_orig);
    cudaFree(grad_logits_opt);
    cudaFree(grad_values_orig);
    cudaFree(grad_values_opt);

    delete args_torch;
#endif
    printf("\n");

    free_ppolossargs(args);
}

/*
// ============================================================================
// sample_logits profiling
// ============================================================================

typedef struct {
    float* logits;        // (B, A)
    float* value;         // (B, 1)
    double* actions;      // (B,) - float64 for discrete/continuous compatibility
    float* logprobs;      // (B,)
    float* value_out;     // (B,)
    int64_t* offset;      // RNG offset (on device for CUDA graph support)
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgs;

SampleLogitsArgs* create_samplelogitsargs(int batch, int num_actions) {
    SampleLogitsArgs* args = (SampleLogitsArgs*)calloc(1, sizeof(SampleLogitsArgs));
    args->B = batch;
    args->A = num_actions;
    args->seed = 42;

    int N_logits = batch * num_actions;
    int N_batch = batch;

    cudaMalloc(&args->logits, N_logits * sizeof(float));
    cudaMalloc(&args->value, N_batch * sizeof(float));  // (B, 1) flattened
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(float));
    cudaMalloc(&args->value_out, N_batch * sizeof(float));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMemset(args->offset, 0, sizeof(int64_t));  // Initialize offset to 0

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    // Initialize logits and value with random values
    for (int i = 0; i < N_logits; ++i) {
        logits_buf[i] = rand1() * 5.0f;
    }
    for (int i = 0; i < N_batch; ++i) {
        value_buf[i] = rand1();
    }

    cudaMemcpy(args->logits, logits_buf, N_logits * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->value, value_buf, N_batch * sizeof(float), cudaMemcpyHostToDevice);

    free(logits_buf);
    free(value_buf);
    return args;
}

void free_samplelogitsargs(SampleLogitsArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->value);
    cudaFree(args->actions);
    cudaFree(args->logprobs);
    cudaFree(args->value_out);
    cudaFree(args->offset);
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    launch_sample_logits<float>(
        args->actions, args->logprobs, args->value_out,
        args->logits, args->value,
        args->seed, args->offset,
        args->A, args->B,
        args->A,  // logits_stride = A (contiguous)
        1,        // value_stride = 1 (contiguous, 1D)
        0);
    // Note: not incrementing offset here since this is just for profiling
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;       // (B, A)
    torch::Tensor value;        // (B, 1) - input
    torch::Tensor actions;      // (B,) float64 - output
    torch::Tensor logprobs;     // (B,) - output
    torch::Tensor value_out;    // (B,) - output
    torch::Tensor offset;       // (1,) int64 - RNG offset tensor
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgsTorch;

SampleLogitsArgsTorch* create_samplelogitsargs_torch(SampleLogitsArgs* raw) {
    SampleLogitsArgsTorch* args = new SampleLogitsArgsTorch();
    args->B = raw->B;
    args->A = raw->A;
    args->seed = 42;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    args->logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    args->value = torch::from_blob(raw->value, {raw->B, 1}, opts);
    args->actions = torch::empty({raw->B}, opts.dtype(torch::kFloat64));
    args->logprobs = torch::empty({raw->B}, opts);
    args->value_out = torch::empty({raw->B}, opts);
    args->offset = torch::zeros({1}, opts.dtype(torch::kInt64));

    return args;
}

void run_samplelogits_forward_torch(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_logits(args->logits, args->value, args->actions, args->logprobs, args->value_out, args->seed, args->offset);
    args->offset.add_(1);  // Increment with CUDA op
}

void run_samplelogits_forward_cpp(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_logits_cpp(args->logits);
}

#endif

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("\tforward", fwd_ms, batch);

#ifdef USE_TORCH
    SampleLogitsArgsTorch* args_torch = create_samplelogitsargs_torch(args);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("\tforward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_samplelogits_forward_cpp, args_torch);
    print_timing("\tforward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("\tforward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_samplelogitsargs(args);
}
*/

// ============================================================================
// OUTDATED TESTS BELOW - GraphBuf and DLL loading no longer used
// Uncomment and update when needed
// ============================================================================

#if 0  // Disabled - uses outdated GraphBuf/DLL patterns

// ============================================================================
// forward_call profiling (inference forward pass) - using GraphBuf
// ============================================================================

#ifdef USE_TORCH

typedef struct {
    std::shared_ptr<PolicyMinGRU> policy;
    GraphBuf graph;
    Tensor rng_offset;
    Tensor act_sizes;
    Tensor act_sizes_cpu;
    uint64_t seed;
    bool use_kernels;
} ForwardCallArgs;

ForwardCallArgs* create_forwardcallargs(int batch, int input_size, int hidden_size,
                                        int act_n, int num_layers, bool use_kernels) {
    // act_n = total action space size (decoder output size)
    // For discrete: num_action_heads=1, act_sizes=[act_n]
    // For multidiscrete: num_action_heads=len(act_sizes), act_n=sum(act_sizes)
    int num_action_heads = 1;  // Using discrete for profiling

    ForwardCallArgs* args = new ForwardCallArgs();
    args->use_kernels = use_kernels;
    args->seed = 42;

    // Create policy with default encoder/decoder
    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
    auto dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    args->policy = std::make_shared<PolicyMinGRU>(enc, dec, input_size, act_n, hidden_size, 1, num_layers, use_kernels);
    args->policy->to(torch::kCUDA);

    // Use create_graph factory (minibatch_segments=0 since not used for inference)
    // num_action_heads=1 for discrete action space
    args->graph = create_graph(batch, input_size, 0, 0, num_layers, hidden_size, 1, num_action_heads, args->policy.get());
    args->graph.obs = torch::randn({batch, input_size}, torch::dtype(DTYPE).device(torch::kCUDA));
    args->rng_offset = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));

    // Create act_sizes tensor: for discrete, single entry with total action count
    args->act_sizes = torch::tensor({act_n}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_forwardcallargs(ForwardCallArgs* args) {
    delete args;
}

void run_forward_call(ForwardCallArgs* args) {
    forward_call(args->graph, args->policy.get(), args->use_kernels, args->seed, args->rng_offset, args->act_sizes, args->act_sizes_cpu);
}

#endif

void profile_forwardcall(int batch, int input_size, int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    printf("forward_call (B=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           batch, input_size, hidden_size, num_atns, num_layers);

    // Profile with kernels
    ForwardCallArgs* args_kernel = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, true);
    float fwd_kernel_ms = profile_kernel((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel");
    print_timing("\tkernel path", fwd_kernel_ms, batch);

    float fwd_kernel_graph_ms = profile_graph((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel_graph");
    print_timing("\tkernel (graph)", fwd_kernel_graph_ms, batch);

    free_forwardcallargs(args_kernel);

    // Profile without kernels (torch ops path)
    ForwardCallArgs* args_torch = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, false);
    float fwd_torch_ms = profile_kernel((kernel_fn)run_forward_call, args_torch, "forward_call_torch");
    print_timing("\ttorch path", fwd_torch_ms, batch);

    float fwd_torch_graph_ms = profile_graph((kernel_fn)run_forward_call, args_torch, "forward_call_torch_graph");
    print_timing("\ttorch (graph)", fwd_torch_graph_ms, batch);

    free_forwardcallargs(args_torch);

    printf("\n");
#else
    printf("forward_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// rollout_copy_call profiling - using RolloutBuf, GraphBuf, EnvBuf
// ============================================================================

#ifdef USE_TORCH

typedef struct {
    RolloutBuf rollouts;
    GraphBuf graph;
    EnvBuf env;
    int horizon;
    int num_envs;
    int num_buffers;
    int h;   // current timestep
    int buf; // current buffer index
} RolloutCopyArgs;

RolloutCopyArgs* create_rolloutcopyargs(int horizon, int num_envs, int num_buffers, int input_size, int num_action_heads = 1) {
    // num_action_heads = number of action heads (1 for discrete, >1 for multidiscrete)
    RolloutCopyArgs* args = new RolloutCopyArgs();
    args->horizon = horizon;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->h = 0;
    args->buf = 0;

    int block_size = num_envs / num_buffers;

    // Use factory functions
    args->rollouts = create_rollouts(horizon, num_envs, input_size, num_action_heads);
    args->env = create_env(num_envs, input_size);

    // Create minimal graph for rollout (only rollout tensors needed, use dummy policy for state)
    auto opts = torch::TensorOptions().dtype(DTYPE).device(torch::kCUDA);
    args->graph.obs = torch::randn({block_size, input_size}, opts);
    args->graph.actions = torch::randint(0, 4, {block_size, num_action_heads}, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    args->graph.logprobs = torch::randn({block_size}, opts);
    args->graph.value = torch::randn({block_size}, opts);

    return args;
}

void free_rolloutcopyargs(RolloutCopyArgs* args) {
    delete args;
}

void run_rollout_copy_call(RolloutCopyArgs* args) {
    rollout_copy_call(args->rollouts, args->env, args->graph, args->num_envs, args->num_buffers, args->h, args->buf);
}

#endif

void profile_rolloutcopycall(int horizon, int num_envs, int num_buffers, int input_size) {
#ifdef USE_TORCH
    int block_size = num_envs / num_buffers;
    printf("rollout_copy_call (H=%d, envs=%d, buffers=%d, block=%d)\n",
           horizon, num_envs, num_buffers, block_size);

    RolloutCopyArgs* args = create_rolloutcopyargs(horizon, num_envs, num_buffers, input_size);

    float copy_ms = profile_kernel((kernel_fn)run_rollout_copy_call, args, "rollout_copy");
    print_timing("\tcopy ops", copy_ms, block_size);

    float copy_graph_ms = profile_graph((kernel_fn)run_rollout_copy_call, args, "rollout_copy_graph");
    print_timing("\tcopy (graph)", copy_graph_ms, block_size);

    free_rolloutcopyargs(args);

    printf("\n");
#else
    printf("rollout_copy_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// train_forward_call profiling - using GraphBuf
// ============================================================================

#ifdef USE_TORCH

typedef struct {
    std::shared_ptr<PolicyMinGRU> policy;
    torch::optim::Muon* muon;
    GraphBuf graph;
    HypersT hypers;
    Tensor adv_mean;
    Tensor adv_std;
    Tensor act_sizes_cpu;
    bool use_kernels;
} TrainForwardArgs;

TrainForwardArgs* create_trainforwardargs(int segments, int horizon, int input_size,
                                          int hidden_size, int act_n, int num_layers,
                                          bool use_kernels) {
    // act_n = total action space size (decoder output size)
    // For discrete: num_action_heads=1, act_sizes=[act_n]
    int num_action_heads = 1;  // Using discrete for profiling

    TrainForwardArgs* args = new TrainForwardArgs();
    args->use_kernels = use_kernels;

    // Setup hypers
    args->hypers.minibatch_segments = segments;
    args->hypers.horizon = horizon;
    args->hypers.clip_coef = 0.1f;
    args->hypers.vf_clip_coef = 0.1f;
    args->hypers.vf_coef = 0.5f;
    args->hypers.ent_coef = 0.01f;
    args->hypers.max_grad_norm = 0.5f;

    // Create policy with default encoder/decoder
    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
    auto dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    args->policy = std::make_shared<PolicyMinGRU>(enc, dec, input_size, act_n, hidden_size, 1, num_layers, use_kernels);
    args->policy->to(torch::kCUDA);

    // Create Muon optimizer
    args->muon = new torch::optim::Muon(args->policy->parameters(),
        torch::optim::MuonOptions(0.0003).momentum(0.95).eps(1e-8));

    // Use create_graph factory (batch=0 since not used for training)
    // num_action_heads=1 for discrete action space
    args->graph = create_graph(0, input_size, segments, horizon, num_layers, hidden_size, 1, num_action_heads, args->policy.get());

    // Initialize mb_* tensors with test data for training
    args->graph.mb_obs = torch::randn({segments, horizon, input_size}, torch::dtype(DTYPE).device(torch::kCUDA));
    args->graph.mb_actions = torch::randint(0, act_n, {segments, horizon, num_action_heads}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    args->graph.mb_logprobs = torch::randn({segments, horizon}, torch::dtype(DTYPE).device(torch::kCUDA)) * 0.1f - 2.0f;  // ~log probs
    args->graph.mb_advantages = torch::randn({segments, horizon}, torch::dtype(DTYPE).device(torch::kCUDA));
    args->graph.mb_values = torch::randn({segments, horizon}, torch::dtype(DTYPE).device(torch::kCUDA));
    args->graph.mb_returns = args->graph.mb_advantages + args->graph.mb_values;
    args->graph.mb_prio = torch::ones({segments, 1}, torch::dtype(DTYPE).device(torch::kCUDA));

    // Adv normalization tensors
    args->adv_mean = torch::zeros({1}, torch::dtype(DTYPE).device(torch::kCUDA));
    args->adv_std = torch::ones({1}, torch::dtype(DTYPE).device(torch::kCUDA));

    // Create act_sizes_cpu tensor: for discrete, single entry with total action count
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_trainforwardargs(TrainForwardArgs* args) {
    delete args->muon;
    delete args;
}

void run_train_forward_call(TrainForwardArgs* args) {
    train_forward_call(args->graph, args->policy.get(), args->muon, args->hypers, args->adv_mean, args->adv_std, args->act_sizes_cpu, args->use_kernels);
}

#endif

void profile_trainforwardcall(int segments, int horizon, int input_size,
                              int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    printf("train_forward_call (seg=%d, H=%d, in=%d, hid=%d, A=%d, layers=%d)\n",
           segments, horizon, input_size, hidden_size, num_atns, num_layers);

    // Profile with kernels
    TrainForwardArgs* args_kernel = create_trainforwardargs(segments, horizon, input_size,
                                                            hidden_size, num_atns, num_layers, true);
    float train_kernel_ms = profile_kernel((kernel_fn)run_train_forward_call, args_kernel, "train_forward_kernel");
    print_timing("\tkernel path", train_kernel_ms, segments * horizon);

    //float train_kernel_graph_ms = profile_graph((kernel_fn)run_train_forward_call, args_kernel, "train_forward_kernel_graph");
    //print_timing("\tkernel (graph)", train_kernel_graph_ms, segments * horizon);

    free_trainforwardargs(args_kernel);

    // Profile without kernels (torch ops path)
    TrainForwardArgs* args_torch = create_trainforwardargs(segments, horizon, input_size,
                                                           hidden_size, num_atns, num_layers, false);
    float train_torch_ms = profile_kernel((kernel_fn)run_train_forward_call, args_torch, "train_forward_torch");
    print_timing("\ttorch path", train_torch_ms, segments * horizon);

    //float train_torch_graph_ms = profile_graph((kernel_fn)run_train_forward_call, args_torch, "train_forward_torch_graph");
    //print_timing("\ttorch (graph)", train_torch_graph_ms, segments * horizon);

    free_trainforwardargs(args_torch);

    printf("\n");
#else
    printf("train_forward_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// Environment speed test (breakout)
// ============================================================================

// Function pointers for env interface (loaded dynamically)
static create_environments_fn profile_create_envs = nullptr;
static create_threads_fn profile_create_threads = nullptr;
static vec_reset_fn profile_vec_reset = nullptr;
static vec_send_fn profile_vec_send = nullptr;
static vec_recv_fn profile_vec_recv = nullptr;
static vec_close_fn profile_vec_close = nullptr;

typedef struct {
    VecEnv* vec;
    int num_envs;
    int num_buffers;
    int num_threads;
    int horizon;
    int obs_n;
    int act_n;
} EnvSpeedArgs;

EnvSpeedArgs* create_envspeedargs(int total_agents, int num_buffers, int num_threads, int horizon) {
    // Load drive.so dynamically
    void* handle = dlopen("./drive.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
        fprintf(stderr, "Make sure to build drive first: ./scripts/build_vec.sh drive\n");
        return nullptr;
    }
    dlerror();

    // Load function pointers
    profile_create_envs = (create_environments_fn)dlsym(handle, "create_environments");
    profile_create_threads = (create_threads_fn)dlsym(handle, "create_threads");
    profile_vec_reset = (vec_reset_fn)dlsym(handle, "vec_reset");
    profile_vec_send = (vec_send_fn)dlsym(handle, "vec_send");
    profile_vec_recv = (vec_recv_fn)dlsym(handle, "vec_recv");
    profile_vec_close = (vec_close_fn)dlsym(handle, "vec_close");
    int obs_n = *(int*)dlsym(handle, "OBS_N");
    int num_atns = *(int*)dlsym(handle, "NUM_ATNS_EXPORT");

    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        return nullptr;
    }

    // Create vec_kwargs with total_agents and num_buffers
    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    // Create env_kwargs for drive
    Dict* env_kwargs = create_dict(32);
    dict_set(env_kwargs, "human_agent_idx", 0);
    dict_set(env_kwargs, "reward_vehicle_collision", -0.5);
    dict_set(env_kwargs, "reward_offroad_collision", -0.2);
    dict_set(env_kwargs, "spawn_immunity_timer", 50);
    dict_set(env_kwargs, "reward_goal_post_respawn", 0.25);
    dict_set(env_kwargs, "reward_vehicle_collision_post_respawn", -0.5);
    dict_set(env_kwargs, "num_maps", 10000);

    // Create env_kwargs for breakout
    /*
    Dict* env_kwargs = create_dict(32);
    dict_set(env_kwargs, "frameskip", 4);
    dict_set(env_kwargs, "width", 576);
    dict_set(env_kwargs, "height", 330);
    dict_set(env_kwargs, "paddle_width", 62);
    dict_set(env_kwargs, "paddle_height", 8);
    dict_set(env_kwargs, "ball_width", 32);
    dict_set(env_kwargs, "ball_height", 32);
    dict_set(env_kwargs, "brick_width", 32);
    dict_set(env_kwargs, "brick_height", 12);
    dict_set(env_kwargs, "brick_rows", 6);
    dict_set(env_kwargs, "brick_cols", 18);
    dict_set(env_kwargs, "initial_ball_speed", 256);
    dict_set(env_kwargs, "max_ball_speed", 448);
    dict_set(env_kwargs, "paddle_speed", 620);
    dict_set(env_kwargs, "continuous", 0);
    */

    // Create environments with new signature
    VecEnv* vec = profile_create_envs(num_buffers, true, 0, vec_kwargs, env_kwargs);
    if (!vec) {
        fprintf(stderr, "Failed to create environments\n");
        return nullptr;
    }

    // Create threads
    int num_envs = vec->size;
    int block_size = num_envs / num_threads;
    if (block_size < 1) block_size = 1;
    profile_create_threads(vec, num_threads, block_size);

    // Reset
    profile_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_n = obs_n;
    args->act_n = num_atns;

    return args;
}

void free_envspeedargs(EnvSpeedArgs* args) {
    if (args && args->vec) {
        profile_vec_close(args->vec);
    }
    free(args);
}

// Run full rollout iteration: iterate through all buffers * horizon
void run_env_rollout(EnvSpeedArgs* args) {
    int num_buffers = args->num_buffers;
    int horizon = args->horizon;
    VecEnv* vec = args->vec;

    for (int i = 0; i < num_buffers * horizon; ++i) {
        int buf = i % num_buffers;
        profile_vec_recv(vec, buf);
        // In real usage, policy forward would happen here (async on GPU)
        profile_vec_send(vec, buf);
    }
}

float profile_env_rollout(EnvSpeedArgs* args, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) {
        run_env_rollout(args);
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > TIMEOUT_SEC) break;
    }

    start_time = std::chrono::steady_clock::now();
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        run_env_rollout(args);
        completed += 1;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > TIMEOUT_SEC) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / completed;  // per rollout
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);

    EnvSpeedArgs* args = create_envspeedargs(total_agents, num_buffers, num_threads, horizon);
    if (!args) {
        printf("\tFailed to create env - skipping\n\n");
        return;
    }

    printf("\tnum_envs=%d, obs_n=%d, num_atns=%d\n", args->num_envs, args->obs_n, args->act_n);

    // Profile full rollout (num_buffers * horizon steps)
    float rollout_ms = profile_env_rollout(args, "env_rollout");
    int total_steps = total_agents * horizon;
    printf("\trollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);

    // Compute throughput
    float sps = total_steps / rollout_ms * 1000.0f;
    printf("\tthroughput: %.2f M steps/s\n", sps / 1e6);

    free_envspeedargs(args);
    printf("\n");
}

#endif  // Disabled outdated tests

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("  kernels        - Individual kernel profiling (no nsys needed)\n");
    // Disabled - these tests use outdated GraphBuf/DLL patterns
    // printf("  forwardcall    - Inference forward pass\n");
    // printf("  trainforward   - Training forward + backward + optimizer\n");
    // printf("  rolloutcopy    - Rollout buffer copy operations\n");
    // printf("  envspeed       - Environment step throughput\n");
    printf("  all            - Run all profiles\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* profile = argv[1];
    warmup_gpu();

    // Using typical breakout settings: INPUT_SIZE=96, H=128, A=4

    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "all") == 0) {
        // profile_mingrugate(BR, H);
        // profile_logcoeffsandvalues(BT, T, H);
        // profile_logcumsumexp(BT, T, H);
        // profile_fusedscan(BT, T, H);
        //profile_samplelogits(BR, A);
        //profile_ppoloss(BT, T, A);

        // FCMax: simple FC -> Max (no intermediate layer)
        // Drive encoder dimensions: partner (B, 63, 7) -> 128, road (B, 200, 13) -> 128
        profile_fcmax(BR, 63, 7, 128);    // partner encoder
        profile_fcmax(BR, 200, 13, 128);  // road encoder

        // FCReluFCMax: FC -> ReLU -> FC -> Max (for comparison)
        profile_fcrelufcmax(BR, 63, 7, 128, 128);   // partner encoder
        profile_fcrelufcmax(BR, 200, 13, 128, 128); // road encoder
    }

    // Disabled - these tests use outdated GraphBuf/DLL patterns
    // if (strcmp(profile, "forwardcall") == 0 || strcmp(profile, "all") == 0) {
    //     profile_forwardcall(BR, INPUT_SIZE, H, A, 1);
    // }
    // if (strcmp(profile, "trainforward") == 0 || strcmp(profile, "all") == 0) {
    //     profile_trainforwardcall(BT, T, INPUT_SIZE, H, A, 1);
    // }
    // if (strcmp(profile, "rolloutcopy") == 0 || strcmp(profile, "all") == 0) {
    //     profile_rolloutcopycall(T, BR, 1, INPUT_SIZE);
    // }
    // if (strcmp(profile, "envspeed") == 0 || strcmp(profile, "all") == 0) {
    //     profile_envspeed(BUF*BR, BUF, 8, T);
    // }

    return 0;
}
