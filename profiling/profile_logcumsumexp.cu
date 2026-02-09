// profiling/profile_logcumsumexp.cu
// Logcumsumexp kernel profiling
#pragma once
#include "profile.h"

typedef struct {
    precision_t* x;
    precision_t* out;
    double* s_buf;
    precision_t* grad_x;
    precision_t* grad_out;
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
    args->N = batch * seq * hidden;

    cudaMalloc(&args->x, args->N * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));

    float* buf = (float*)malloc(args->N * sizeof(float) * 2);
    float* x_buf = buf;
    float* grad_out_buf = buf + args->N;
    for (int i = 0; i < args->N; ++i) {
        x_buf[i] = rand1();
        grad_out_buf[i] = rand1();
    }

    float_to_device(args->x, x_buf, args->N);
    float_to_device(args->grad_out, grad_out_buf, args->N);

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
    logcumsumexp_forward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    logcumsumexp_backward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B);
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
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    // Keep in native precision — kernel wrappers cast to precision_t*
    args->x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone();
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

void test_logcumsumexp_correct(LogcumsumexpArgsTorch* args) {
    // Kernel path: native precision (bf16 or f32) — kernel casts to precision_t*
    auto x_k = args->x.detach().clone().requires_grad_(true);
    auto out_k = logcumsumexp_cuda(x_k);

    // Cpp path: float32 for higher-precision reference
    auto x_c = args->x.detach().to(torch::kFloat32).requires_grad_(true);
    auto out_c = logcumsumexp_cpp(x_c);

    float rtol = USE_BF16 ? 5e-2f : 1e-3f;
    float atol = USE_BF16 ? 1e-2f : 1e-4f;
    float out_max_diff = (out_k.to(torch::kFloat32) - out_c).abs().max().item<float>();
    bool out_match = torch::allclose(out_k.to(torch::kFloat32), out_c, rtol, atol);
    printf("  forward correctness: %s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Backward: kernel in native precision
    auto grad_k = args->grad_out.to(x_k.dtype());
    out_k.backward(grad_k, /*retain_graph=*/false);

    // Backward: cpp in float32
    auto grad_c = args->grad_out.to(torch::kFloat32);
    out_c.backward(grad_c, /*retain_graph=*/false);

    float grad_max_diff = (x_k.grad().to(torch::kFloat32) - x_c.grad()).abs().max().item<float>();
    bool grad_match = torch::allclose(x_k.grad().to(torch::kFloat32), x_c.grad(), rtol, atol);
    printf("  backward correctness: %s(%.2e)\n",
           grad_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_max_diff);
}

#endif

void profile_logcumsumexp(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = create_logcumsumexpargs(batch, seq, hidden);
    printf("logcumsumexp (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogcumsumexpArgsTorch* args_torch = create_logcumsumexpargs_torch(args);
    test_logcumsumexp_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch*seq);

    args_torch->out = logcumsumexp_cuda(args_torch->x);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch*seq);

    args_torch->out = logcumsumexp_cpp(args_torch->x);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("backward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");
    free_logcumsumexpargs(args);
}
