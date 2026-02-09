// profiling/profile_ppoloss.cu
// PPO loss (optimized fused kernel) profiling
#pragma once
#include "profile.h"

typedef struct {
    precision_t* logits;
    precision_t* values_pred;
    double* actions;          // float64 for both continuous and discrete
    precision_t* old_logprobs;
    float* advantages;        // always fp32 for precision
    precision_t* prio;
    precision_t* values;
    precision_t* returns;
    float* adv_mean;
    float* adv_var;           // variance, kernel does sqrt
    float* loss;
    double* saved_for_backward;
    precision_t* ratio_out;
    precision_t* newvalue_out;
    float* grad_logits;
    float* grad_values_pred;
    float* grad_loss;
    int* act_sizes;           // (num_atns,) action head sizes
    int num_atns;
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
    args->num_atns = 1;  // single action head for profiling

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(precision_t));
    cudaMalloc(&args->values_pred, NT * sizeof(precision_t));
    cudaMalloc(&args->actions, NT * sizeof(double));
    cudaMalloc(&args->old_logprobs, NT * sizeof(precision_t));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(precision_t));
    cudaMalloc(&args->values, NT * sizeof(precision_t));
    cudaMalloc(&args->returns, NT * sizeof(precision_t));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_var, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->ratio_out, NT * sizeof(precision_t));
    cudaMalloc(&args->newvalue_out, NT * sizeof(precision_t));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));
    cudaMalloc(&args->act_sizes, sizeof(int));

    cudaMemcpy(args->act_sizes, &actions, sizeof(int), cudaMemcpyHostToDevice);

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    double* actions_buf = (double*)malloc(NT * sizeof(double));

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
        actions_buf[i] = (double)(rand() % actions);
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    float_to_device(args->logits, logits_buf, NTA);
    float_to_device(args->values_pred, values_pred_buf, NT);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(double), cudaMemcpyHostToDevice);
    float_to_device(args->old_logprobs, old_logprobs_buf, NT);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->prio, prio_buf, batch);
    float_to_device(args->values, values_buf, NT);
    float_to_device(args->returns, returns_buf, NT);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_var, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    args->logits_stride_n = seq * actions;
    args->logits_stride_t = actions;
    args->logits_stride_a = 1;
    args->values_stride_n = seq;
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
    cudaFree(args->ratio_out);
    cudaFree(args->newvalue_out);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    cudaFree(args->act_sizes);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    cudaMemset(args->loss, 0, sizeof(float));
    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->loss, args->saved_for_backward,
        args->ratio_out, args->newvalue_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

void run_ppoloss_backward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->grad_logits,
        nullptr,  // grad_logstd (nullptr for discrete)
        args->grad_values_pred, args->grad_loss,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
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
    torch::Tensor ratio_out;
    torch::Tensor newvalue_out;
    torch::Tensor act_sizes;
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

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);

    args->logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).clone().requires_grad_(true);
    args->values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T, 1}, opts).clone().requires_grad_(true);
    args->actions = torch::from_blob(raw->actions, {raw->N, raw->T, 1}, cuda_f64).clone();
    args->old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts).clone();
    args->advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, cuda_f32).clone();
    args->prio = torch::from_blob(raw->prio, {raw->N, 1}, opts).clone();
    args->values = torch::from_blob(raw->values, {raw->N, raw->T}, opts).clone();
    args->returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts).clone();
    args->adv_mean = torch::from_blob(raw->adv_mean, {1}, cuda_f32).clone();
    args->adv_var = torch::from_blob(raw->adv_var, {1}, cuda_f32).clone();
    args->ratio_out = torch::zeros({raw->N, raw->T}, opts);
    args->newvalue_out = torch::zeros({raw->N, raw->T}, opts);
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);

    return args;
}

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto logstd = torch::empty({0}, args->logits.options());
    fused_ppo_loss_optimized(
        args->logits, logstd, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->ratio_out, args->newvalue_out, args->act_sizes,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void test_ppoloss_correct(PPOLossArgsTorch* args) {
    int N = args->N;
    int T = args->T;
    int A = args->A;
    int minibatch_size = N * T;

    // Kernel path via compute_train_loss
    auto logits_k = args->logits.detach().clone().requires_grad_(true);
    auto values_pred_k = args->values_pred.detach().clone().requires_grad_(true);
    auto ratio_out_k = torch::zeros({N, T}, logits_k.options());
    auto newvalue_out_k = torch::zeros({N, T}, logits_k.options());
    Logits logits_struct_k = {.mean = logits_k};
    auto loss_k = compute_train_loss(
        logits_struct_k, values_pred_k,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, ratio_out_k, newvalue_out_k,
        args->act_sizes, torch::tensor({(int64_t)A}, torch::dtype(torch::kInt64)),
        minibatch_size, T,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, /*kernels=*/true);

    // Cpp reference path via compute_train_loss
    auto logits_c = args->logits.detach().clone().requires_grad_(true);
    auto values_pred_c = args->values_pred.detach().clone().requires_grad_(true);
    auto ratio_out_c = torch::zeros({N, T}, logits_c.options());
    auto newvalue_out_c = torch::zeros({N, T}, logits_c.options());
    Logits logits_struct_c = {.mean = logits_c};
    auto loss_c = compute_train_loss(
        logits_struct_c, values_pred_c,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, ratio_out_c, newvalue_out_c,
        args->act_sizes, torch::tensor({(int64_t)A}, torch::dtype(torch::kInt64)),
        minibatch_size, T,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, /*kernels=*/false);

    float rtol = 1e-2f, atol = 1e-3f;
    float loss_diff = (loss_k - loss_c).abs().item<float>();
    bool loss_match = loss_diff < atol;
    printf("  forward correctness: loss=%s(%.2e)\n",
           loss_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", loss_diff);

    loss_k.backward();
    loss_c.backward();

    auto grad_logits_k = logits_k.grad();
    auto grad_logits_c = logits_c.grad();
    float grad_logits_diff = (grad_logits_k - grad_logits_c).abs().max().item<float>();
    bool grad_logits_match = torch::allclose(grad_logits_k, grad_logits_c, rtol, atol);
    printf("  backward correctness: grad_logits=%s(%.2e)\n",
           grad_logits_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_logits_diff);

    auto grad_values_k = values_pred_k.grad();
    auto grad_values_c = values_pred_c.grad();
    float grad_values_diff = (grad_values_k - grad_values_c).abs().max().item<float>();
    bool grad_values_match = torch::allclose(grad_values_k, grad_values_c, rtol, atol);
    printf("  backward correctness: grad_values=%s(%.2e)\n",
           grad_values_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_values_diff);
}

#endif

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("forward", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("backward", bwd_ms, NT);

#ifdef USE_TORCH
    PPOLossArgsTorch* args_torch = create_ppolossargs_torch(args);

    test_ppoloss_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, NT);

    auto logstd_empty = torch::empty({0}, args_torch->logits.options());
    args_torch->loss = fused_ppo_loss_optimized(
        args_torch->logits, logstd_empty, args_torch->values_pred, args_torch->actions,
        args_torch->old_logprobs, args_torch->advantages, args_torch->prio,
        args_torch->values, args_torch->returns, args_torch->adv_mean, args_torch->adv_var,
        args_torch->ratio_out, args_torch->newvalue_out, args_torch->act_sizes,
        args_torch->clip_coef, args_torch->vf_clip_coef, args_torch->vf_coef, args_torch->ent_coef)[0];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, NT);

    delete args_torch;
#endif
    printf("\n");

    free_ppolossargs(args);
}
