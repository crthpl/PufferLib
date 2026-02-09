// profiling/profile_fusedscan.cu
// Fused scan (checkpointed) kernel profiling
#pragma once
#include "profile.h"

typedef struct {
    precision_t* combined;       // (B, T, 3*H)
    precision_t* state;          // (B, 1, H)
    precision_t* out;            // (B, T, H)
    precision_t* next_state;     // (B, 1, H)
    float* a_star;               // (B, T+1, H)
    float* s_vals;               // (B, T+1, H)
    float* log_values_buf;       // (B, T+1, H)
    precision_t* grad_combined;  // (B, T, 3*H)
    precision_t* grad_state;     // (B, 1, H)
    precision_t* grad_out;       // (B, T, H)
    precision_t* grad_next_state;// (B, 1, H)
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

    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->grad_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(precision_t));

    float* combined_buf = (float*)malloc(N_combined * sizeof(float));
    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* grad_out_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_next_state_buf = (float*)malloc(N_state * sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < hidden; ++h) {
                int base = b * seq * 3 * hidden + t * 3 * hidden;
                combined_buf[base + h] = rand1() * 5.0f;
                combined_buf[base + hidden + h] = rand1() * 5.0f;
                combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
            }
        }
    }
    for (int i = 0; i < N_state; ++i) state_buf[i] = fabsf(rand1()) + 0.1f;
    for (int i = 0; i < args->N; ++i) grad_out_buf[i] = rand1();
    for (int i = 0; i < N_state; ++i) grad_next_state_buf[i] = rand1();

    float_to_device(args->combined, combined_buf, N_combined);
    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->grad_out, grad_out_buf, args->N);
    float_to_device(args->grad_next_state, grad_next_state_buf, N_state);

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
    fused_scan_forward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    fused_scan_backward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor combined;
    torch::Tensor state;
    torch::Tensor out;
    torch::Tensor next_state;
    torch::Tensor grad_out;
    torch::Tensor grad_next_state;
    int B;
    int T;
    int H;
} FusedScanArgsTorch;

FusedScanArgsTorch* create_fusedscanargs_torch(FusedScanArgs* raw) {
    FusedScanArgsTorch* args = new FusedScanArgsTorch();
    args->B = raw->B;
    args->T = raw->T;
    args->H = raw->H;
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3 * raw->H}, opts).clone().to(torch::kFloat32).requires_grad_(true);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).clone().to(torch::kFloat32).requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone().to(torch::kFloat32);
    args->grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts).clone().to(torch::kFloat32);
    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_checkpointed(args->combined, args->state);
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

void test_fusedscan_correct(FusedScanArgsTorch* args) {
    auto combined_ref = args->combined.detach().clone().requires_grad_(true);
    auto state_ref = args->state.detach().clone().requires_grad_(true);
    combined_ref.retain_grad();
    state_ref.retain_grad();
    auto ref_outputs = fused_scan_checkpointed(combined_ref, state_ref);
    auto ref_out = ref_outputs[0];
    auto ref_next_state = ref_outputs[1];

    auto cpp_outputs = fused_scan_cpp(args->combined.detach(), args->state.detach());
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    float rtol = 1e-3f, atol = 3e-4f;
    bool out_match = torch::allclose(ref_out, cpp_out, rtol, atol);
    float out_max_diff = (ref_out - cpp_out).abs().max().item<float>();
    bool ns_match = torch::allclose(ref_next_state, cpp_next_state, rtol, atol);
    float ns_max_diff = (ref_next_state - cpp_next_state).abs().max().item<float>();
    printf("  forward correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           ns_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", ns_max_diff);

    torch::autograd::backward({ref_out, ref_next_state}, {args->grad_out, args->grad_next_state});
    auto grad_combined_ref = combined_ref.grad().clone();
    auto grad_state_ref = state_ref.grad().clone();

    auto combined_cpp = args->combined.detach().clone().requires_grad_(true);
    auto state_cpp = args->state.detach().clone().requires_grad_(true);
    auto cpp_out2 = fused_scan_cpp(combined_cpp, state_cpp);
    torch::autograd::backward({cpp_out2[0], cpp_out2[1]}, {args->grad_out, args->grad_next_state});

    bool gc_match = torch::allclose(grad_combined_ref, combined_cpp.grad(), rtol, atol);
    float gc_diff = (grad_combined_ref - combined_cpp.grad()).abs().max().item<float>();
    bool gs_match = torch::allclose(grad_state_ref, state_cpp.grad(), rtol, atol);
    float gs_diff = (grad_state_ref - state_cpp.grad()).abs().max().item<float>();
    printf("  backward correctness: grad_combined=%s(%.2e) grad_state=%s(%.2e)\n",
           gc_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gc_diff,
           gs_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gs_diff);
}

#endif

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);
    printf("fused_scan (N=%d, %dx%dx%d, combined=%dx%dx%d)\n",
           args->N, batch, seq, hidden, batch, seq, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    FusedScanArgsTorch* args_torch = create_fusedscanargs_torch(args);
    test_fusedscan_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch*seq);

    auto scan_out = fused_scan_checkpointed(args_torch->combined, args_torch->state);
    args_torch->out = scan_out[0];
    args_torch->next_state = scan_out[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch*seq);

    auto scan_out_cpp = fused_scan_cpp(args_torch->combined, args_torch->state);
    args_torch->out = scan_out_cpp[0];
    args_torch->next_state = scan_out_cpp[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("backward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");
    free_fusedscanargs(args);
}
