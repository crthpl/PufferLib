// profiling/profile_forward.cu
// Inference forward pass profiling (Policy.forward + sample_actions)
#pragma once
#include "profile.h"

#ifdef USE_TORCH

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor obs;
    Tensor state;
    Tensor actions;
    Tensor logprobs;
    Tensor values;
    Tensor rng_offset;
    Tensor act_sizes;
    Tensor act_sizes_cpu;
    uint64_t seed;
    bool use_kernels;
    int batch;
} ForwardCallArgs;

ForwardCallArgs* create_forwardcallargs(int batch, int input_size, int hidden_size,
                                        int act_n, int num_layers, bool use_kernels) {
    int num_action_heads = 1;

    ForwardCallArgs* args = new ForwardCallArgs();
    args->use_kernels = use_kernels;
    args->seed = 42;
    args->batch = batch;

    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
    auto dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, use_kernels);
    args->policy = std::make_shared<Policy>(enc, dec, rnn, input_size, act_n, hidden_size);
    args->policy->to(torch::kCUDA);

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->obs = torch::randn({batch, input_size}, opts);
    args->state = args->policy->initial_state(batch, torch::kCUDA);
    args->actions = torch::zeros({batch, num_action_heads}, cuda_f64);
    args->logprobs = torch::zeros({batch}, opts);
    args->values = torch::zeros({batch}, opts);
    args->rng_offset = torch::zeros({1}, cuda_i64);
    args->act_sizes = torch::tensor({act_n}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_forwardcallargs(ForwardCallArgs* args) {
    delete args;
}

void run_forward_call(ForwardCallArgs* args) {
    torch::NoGradGuard no_grad;

    auto [logits_out, value, state_out] = args->policy->forward(args->obs, args->state);

    sample_actions(logits_out, value, args->actions, args->logprobs, args->values,
        args->act_sizes, args->act_sizes_cpu,
        /*is_continuous=*/false, args->use_kernels, args->seed, args->rng_offset);

    args->state.copy_(state_out, false);
}

#endif

void profile_forwardcall(int batch, int input_size, int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    printf("forward_call (B=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           batch, input_size, hidden_size, num_atns, num_layers);

    ForwardCallArgs* args_kernel = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, true);
    float fwd_kernel_ms = profile_kernel((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel");
    print_timing("kernel path", fwd_kernel_ms, batch);

    float fwd_kernel_graph_ms = profile_graph((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel_graph");
    print_timing("kernel (graph)", fwd_kernel_graph_ms, batch);

    free_forwardcallargs(args_kernel);

    ForwardCallArgs* args_torch = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, false);
    float fwd_torch_ms = profile_kernel((kernel_fn)run_forward_call, args_torch, "forward_call_torch");
    print_timing("torch path", fwd_torch_ms, batch);

    float fwd_torch_graph_ms = profile_graph((kernel_fn)run_forward_call, args_torch, "forward_call_torch_graph");
    print_timing("torch (graph)", fwd_torch_graph_ms, batch);

    free_forwardcallargs(args_torch);

    printf("\n");
#else
    printf("forward_call: requires USE_TORCH\n\n");
#endif
}
