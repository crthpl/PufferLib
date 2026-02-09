// profiling/profile_mingru.cu
// MinGRU gate inference kernel profiling
#pragma once
#include "profile.h"

typedef struct {
    precision_t* state;       // (B, H)
    precision_t* combined;    // (B, 3*H) = [hidden, gate, proj]
    precision_t* out;         // (B, H)
    precision_t* next_state;  // (B, H)
    int B;
    int H;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->out, N_state * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));

    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));

    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    for (int b = 0; b < batch; ++b) {
        int base = b * 3 * hidden;
        for (int h = 0; h < hidden; ++h) {
            combined_buf[base + h] = rand1() * 5.0f;
            combined_buf[base + hidden + h] = rand1() * 5.0f;
            combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
        }
    }

    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->combined, combined_buf, N_combined);

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
    mingru_gate_inference_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;
    torch::Tensor combined;
    int B;
    int H;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->B = raw->B;
    args->H = raw->H;
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->B, raw->H}, opts);
    args->combined = torch::from_blob(raw->combined, {raw->B, 3 * raw->H}, opts);
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
    auto kernel_outputs = mingru_gate(args->state, args->combined);
    auto kernel_out = kernel_outputs[0];
    auto kernel_next_state = kernel_outputs[1];

    auto cpp_outputs = mingru_gate_cpp(args->state, args->combined);
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    float rtol = 1e-3f, atol = 1e-4f;
    bool out_match = torch::allclose(kernel_out.to(torch::kFloat32), cpp_out.to(torch::kFloat32), rtol, atol);
    float out_max_diff = (kernel_out.to(torch::kFloat32) - cpp_out.to(torch::kFloat32)).abs().max().item<float>();
    bool next_state_match = torch::allclose(kernel_next_state.to(torch::kFloat32), cpp_next_state.to(torch::kFloat32), rtol, atol);
    float next_state_max_diff = (kernel_next_state.to(torch::kFloat32) - cpp_next_state.to(torch::kFloat32)).abs().max().item<float>();

    printf("  correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);
}

#endif

void profile_mingrugate(int batch, int hidden) {
    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    MingruGateArgs* args = create_mingrugateargs(batch, hidden);
    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);
    test_mingrugate_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_mingrugate_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");
    free_mingrugateargs(args);
}
