// profiling/profile_fcmax.cu
// FCMax kernel profiling: FC -> Max reduction
#pragma once
#include "profile.h"

typedef struct {
    precision_t* x;              // (B, N, D_in)
    float* W;              // (D_out, D_in) - always float32
    float* b;              // (D_out) - always float32
    precision_t* out;            // (B, D_out)
    int* argmax_indices;   // (B, D_out)
    float* grad_x;         // (B, N, D_in) - always float32 for atomicAdd
    float* grad_W;         // (D_out, D_in)
    float* grad_b;         // (D_out)
    precision_t* grad_out;       // (B, D_out)
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

    cudaMalloc(&args->x, N_x * sizeof(precision_t));
    cudaMalloc(&args->W, N_W * sizeof(float));
    cudaMalloc(&args->b, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(precision_t));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W, N_W * sizeof(float));
    cudaMalloc(&args->grad_b, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(precision_t));

    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W_buf = (float*)malloc(N_W * sizeof(float));
    float* b_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W; ++i) W_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    float_to_device(args->x, x_buf, N_x);
    cudaMemcpy(args->W, W_buf, N_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b, b_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->grad_out, grad_out_buf, N_out);

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
    fc_max_forward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->out, args->argmax_indices,
        args->x, args->W, args->b,
        args->B, args->N, args->D_in, args->D_out);
}

void run_fcmax_backward(FCMaxArgs* args) {
    cudaMemset(args->grad_x, 0, args->B * args->N * args->D_in * sizeof(float));
    cudaMemset(args->grad_W, 0, args->D_out * args->D_in * sizeof(float));
    cudaMemset(args->grad_b, 0, args->D_out * sizeof(float));

    fc_max_backward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->grad_x, args->grad_W, args->grad_b,
        args->grad_out, args->x, args->W,
        args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out);
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

    auto prec_opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    // Keep x and grad_out in native precision — kernel casts to precision_t*
    // W and b are always float32 (kernel reads them as float directly)
    args->x = torch::from_blob(raw->x, {raw->B, raw->N, raw->D_in}, prec_opts).clone().requires_grad_(true);
    args->W = torch::from_blob(raw->W, {raw->D_out, raw->D_in}, cuda_f32).clone().requires_grad_(true);
    args->b = torch::from_blob(raw->b, {raw->D_out}, cuda_f32).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->D_out}, prec_opts).clone();

    return args;
}

void run_fcmax_forward_torch(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max(args->x, args->W, args->b);
}

void run_fcmax_backward_torch(FCMaxArgsTorch* args) {
    if (args->x.grad().defined()) args->x.grad().zero_();
    if (args->W.grad().defined()) args->W.grad().zero_();
    if (args->b.grad().defined()) args->b.grad().zero_();
    args->out.backward(args->grad_out.to(args->out.dtype()), /*retain_graph=*/true);
}

void run_fcmax_forward_cpp(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    // fc_max_cpp uses addmm which requires matching dtypes — upcast x to float32
    fc_max_cpp(args->x.to(torch::kFloat32), args->W, args->b);
}

void test_fcmax_correct(FCMaxArgsTorch* args) {
    // Kernel path: x in native precision (kernel casts to precision_t*)
    auto x_fused = args->x.detach().clone().requires_grad_(true);
    auto W_fused = args->W.detach().clone().requires_grad_(true);
    auto b_fused = args->b.detach().clone().requires_grad_(true);
    auto fused_out = fc_max(x_fused, W_fused, b_fused);

    // Cpp path: x in float32 (addmm requires matching dtypes)
    auto x_ref = args->x.detach().to(torch::kFloat32).requires_grad_(true);
    auto W_ref = args->W.detach().clone().requires_grad_(true);
    auto b_ref = args->b.detach().clone().requires_grad_(true);
    auto ref_out = fc_max_cpp(x_ref, W_ref, b_ref);

    float rtol = USE_BF16 ? 5e-2f : 1e-3f;
    float atol = USE_BF16 ? 1e-2f : 1e-4f;
    bool out_match = torch::allclose(fused_out.to(torch::kFloat32), ref_out, rtol, atol);
    float out_max_diff = (fused_out.to(torch::kFloat32) - ref_out).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    auto grad_fused = args->grad_out.to(fused_out.dtype());
    auto grad_ref = args->grad_out.to(torch::kFloat32);
    fused_out.backward(grad_fused);
    ref_out.backward(grad_ref);

    bool grad_x_match = torch::allclose(x_fused.grad().to(torch::kFloat32), x_ref.grad(), rtol, atol);
    float grad_x_max_diff = (x_fused.grad().to(torch::kFloat32) - x_ref.grad()).abs().max().item<float>();
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
    print_timing("forward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcmax_backward, args);
    print_timing("backward", bwd_ms, batch);

#ifdef USE_TORCH
    FCMaxArgsTorch* args_torch = create_fcmaxargs_torch(args);

    test_fcmax_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    args_torch->out = fc_max(args_torch->x, args_torch->W, args_torch->b);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_fcmaxargs(args);
}
