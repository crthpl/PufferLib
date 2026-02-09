// profiling/profile_sample.cu
// Sample logits kernel profiling
#pragma once
#include "profile.h"

typedef struct {
    precision_t* logits;        // (B, A)
    precision_t* value;         // (B, 1)
    double* actions;      // (B, 1) - float64 for discrete/continuous compatibility
    precision_t* logprobs;      // (B,)
    precision_t* value_out;     // (B,)
    int64_t* offset;      // RNG offset (on device for CUDA graph support)
    int* act_sizes;       // (1,) - single action head
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

    cudaMalloc(&args->logits, N_logits * sizeof(precision_t));
    cudaMalloc(&args->value, N_batch * sizeof(precision_t));
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(precision_t));
    cudaMalloc(&args->value_out, N_batch * sizeof(precision_t));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMalloc(&args->act_sizes, sizeof(int));
    cudaMemset(args->offset, 0, sizeof(int64_t));
    cudaMemcpy(args->act_sizes, &num_actions, sizeof(int), cudaMemcpyHostToDevice);

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    for (int i = 0; i < N_logits; ++i) {
        logits_buf[i] = rand1() * 5.0f;
    }
    for (int i = 0; i < N_batch; ++i) {
        value_buf[i] = rand1();
    }

    float_to_device(args->logits, logits_buf, N_logits);
    float_to_device(args->value, value_buf, N_batch);

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
    cudaFree(args->act_sizes);
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    sample_logits_kernel<<<grid_size(args->B), BLOCK_SIZE>>>(
        args->actions, args->logprobs, args->value_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->value,
        args->act_sizes, args->seed, args->offset,
        1,  // num_atns
        args->B,
        args->A,  // logits_stride
        0,        // logstd_stride (unused for discrete)
        1,        // value_stride
        false);   // is_continuous
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;       // (B, A)
    torch::Tensor value;        // (B, 1) - input
    torch::Tensor actions;      // (B, 1) float64 - output
    torch::Tensor logprobs;     // (B,) - output
    torch::Tensor value_out;    // (B,) - output
    torch::Tensor offset;       // (1,) int64 - RNG offset tensor
    torch::Tensor act_sizes;    // (1,) int32 - action sizes
    torch::Tensor act_sizes_cpu;// (1,) int64 - CPU version for cpp path
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgsTorch;

SampleLogitsArgsTorch* create_samplelogitsargs_torch(SampleLogitsArgs* raw) {
    SampleLogitsArgsTorch* args = new SampleLogitsArgsTorch();
    args->B = raw->B;
    args->A = raw->A;
    args->seed = 42;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    args->value = torch::from_blob(raw->value, {raw->B, 1}, opts);
    args->actions = torch::empty({raw->B, 1}, opts.dtype(torch::kFloat64));
    args->logprobs = torch::empty({raw->B}, opts);
    args->value_out = torch::empty({raw->B}, opts);
    args->offset = torch::zeros({1}, opts.dtype(torch::kInt64));
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)raw->A}, torch::dtype(torch::kInt64));

    return args;
}

void run_samplelogits_forward_torch(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto logstd = torch::Tensor();  // empty/undefined for discrete
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);
}

void run_samplelogits_forward_cpp(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_discrete_cpp(args->logits, args->act_sizes_cpu, 1);
}

void test_samplelogits_correct(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;

    auto logstd = torch::Tensor();
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);

    auto log_probs = torch::log_softmax(args->logits.to(torch::kFloat32), 1);
    auto actions_i64 = args->actions.to(torch::kInt64);
    auto expected_logprobs = log_probs.gather(1, actions_i64).squeeze(1);
    auto actual_logprobs = args->logprobs.to(torch::kFloat32);

    float rtol = 1e-2f, atol = 1e-3f;
    float logprob_max_diff = (actual_logprobs - expected_logprobs).abs().max().item<float>();
    bool logprob_match = torch::allclose(actual_logprobs, expected_logprobs, rtol, atol);
    printf("  logprob consistency: %s(%.2e)\n",
           logprob_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", logprob_max_diff);

    auto expected_values = args->value.squeeze(1).to(torch::kFloat32);
    auto actual_values = args->value_out.to(torch::kFloat32);
    float value_max_diff = (actual_values - expected_values).abs().max().item<float>();
    bool value_match = torch::allclose(actual_values, expected_values, rtol, atol);
    printf("  value passthrough: %s(%.2e)\n",
           value_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", value_max_diff);

    auto actions_flat = args->actions.to(torch::kInt64).flatten();
    bool valid_actions = (actions_flat.ge(0) & actions_flat.lt(args->A)).all().item<bool>();
    printf("  action validity: %s\n",
           valid_actions ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m");
}

#endif

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    SampleLogitsArgsTorch* args_torch = create_samplelogitsargs_torch(args);

    test_samplelogits_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_samplelogits_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_samplelogitsargs(args);
}
