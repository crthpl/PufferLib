// profiling/profile_rolloutcopy.cu
// Per-minibatch data preparation profiling: advantage, priority sampling, select+copy
// Mirrors the per-minibatch loop in train_impl (pufferlib.cpp lines 385-419)
// Requires USE_TORCH
#pragma once
#include "profile.h"

#ifdef USE_TORCH

// ============================================================================
// RolloutCopyArgs: synthetic rollout + TrainGraph destination buffers
// ============================================================================

typedef struct {
    // Rollout buffers (shaped after transpose: [segments, horizon, ...])
    Tensor values;        // (S, T) bf16
    Tensor rewards;       // (S, T) bf16
    Tensor terminals;     // (S, T) bf16
    Tensor ratio;         // (S, T) bf16
    Tensor observations;  // (S, T, input_size) bf16
    Tensor actions;       // (S, T, num_atns) f64
    Tensor logprobs;      // (S, T) bf16
    Tensor advantages;    // (S, T) f32

    // TrainGraph destination buffers (minibatch sized: [mb_segs, T, ...])
    Tensor mb_obs;
    Tensor mb_state;
    Tensor mb_actions;
    Tensor mb_logprobs;
    Tensor mb_advantages;
    Tensor mb_prio;
    Tensor mb_values;
    Tensor mb_returns;

    // Config
    int num_segments;     // S = total_agents (full rollout rows)
    int horizon;          // T = sequence length
    int minibatch_segs;   // how many segments per minibatch
    int input_size;
    int num_atns;
    int num_layers;
    int hidden_size;
    float gamma;
    float gae_lambda;
    float rho_clip;
    float c_clip;
    float prio_alpha;
    float anneal_beta;
    int total_agents;
} RolloutCopyArgs;

RolloutCopyArgs* create_rolloutcopyargs(int num_segments, int horizon, int minibatch_segs,
                                         int input_size, int num_atns, int num_layers, int hidden) {
    auto* args = new RolloutCopyArgs();
    args->num_segments = num_segments;
    args->horizon = horizon;
    args->minibatch_segs = minibatch_segs;
    args->input_size = input_size;
    args->num_atns = num_atns;
    args->num_layers = num_layers;
    args->hidden_size = hidden;
    args->gamma = 0.99f;
    args->gae_lambda = 0.95f;
    args->rho_clip = 1.0f;
    args->c_clip = 1.0f;
    args->prio_alpha = 0.6f;
    args->anneal_beta = 0.4f;
    args->total_agents = num_segments;

    // Create synthetic rollout data (post-transpose shape: [segments, horizon, ...])
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->values = torch::randn({num_segments, horizon}, opts);
    args->rewards = torch::randn({num_segments, horizon}, opts) * 0.1f;
    args->terminals = torch::zeros({num_segments, horizon}, opts);
    // Sprinkle some terminals
    for (int i = 0; i < num_segments / 10; i++) {
        int row = rand() % num_segments;
        int col = rand() % horizon;
        args->terminals[row][col] = 1.0f;
    }
    args->ratio = torch::ones({num_segments, horizon}, opts);
    args->observations = torch::randn({num_segments, horizon, input_size}, opts);
    args->actions = torch::randint(0, num_atns, {num_segments, horizon, num_atns}, cuda_f64);
    args->logprobs = torch::randn({num_segments, horizon}, opts) * 0.5f;
    args->advantages = torch::zeros({num_segments, horizon}, cuda_f32);

    // TrainGraph destination buffers (minibatch sized)
    args->mb_obs = torch::zeros({minibatch_segs, horizon, input_size}, opts);
    args->mb_state = torch::zeros({num_layers, minibatch_segs, 1, hidden}, opts);
    args->mb_actions = torch::zeros({minibatch_segs, horizon, num_atns}, cuda_f64);
    args->mb_logprobs = torch::zeros({minibatch_segs, horizon}, opts);
    args->mb_advantages = torch::zeros({minibatch_segs, horizon}, cuda_f32);
    args->mb_prio = torch::zeros({minibatch_segs, 1}, opts);
    args->mb_values = torch::zeros({minibatch_segs, horizon}, opts);
    args->mb_returns = torch::zeros({minibatch_segs, horizon}, opts);

    return args;
}

void free_rolloutcopyargs(RolloutCopyArgs* args) {
    delete args;
}

// ============================================================================
// Phase runners (mirror train_impl per-minibatch loop)
// ============================================================================

// Phase 1: compute_advantage — calls puff_advantage CUDA kernel
void run_compute_advantage(RolloutCopyArgs* args) {
    args->advantages.fill_(0.0);
    compute_puff_advantage_cuda(
        args->values, args->rewards, args->terminals, args->ratio,
        args->advantages, args->gamma, args->gae_lambda,
        args->rho_clip, args->c_clip);
}

// Phase 2: compute_prio — priority-weighted sampling (all PyTorch ops)
void run_compute_prio(RolloutCopyArgs* args) {
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, args->minibatch_segs, true);
    Tensor mb_prio = torch::pow(
        args->total_agents * prio_probs.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);
}

// Phase 3: train_select_and_copy — index_select + copy_ into graph buffers
void run_select_and_copy(RolloutCopyArgs* args) {
    // Recompute idx (multinomial is stochastic, needs recompute each call)
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, args->minibatch_segs, true);
    Tensor mb_prio = torch::pow(
        args->total_agents * prio_probs.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);

    Tensor mb_obs = args->observations.index_select(0, idx);
    Tensor mb_actions = args->actions.index_select(0, idx);
    Tensor mb_logprobs = args->logprobs.index_select(0, idx);
    Tensor mb_values = args->values.index_select(0, idx);
    Tensor mb_advantages = args->advantages.index_select(0, idx);
    Tensor mb_returns = mb_advantages + mb_values;

    args->mb_state.zero_();
    args->mb_obs.copy_(mb_obs, false);
    args->mb_actions.copy_(mb_actions, false);
    args->mb_logprobs.copy_(mb_logprobs, false);
    args->mb_advantages.copy_(mb_advantages, false);
    args->mb_prio.copy_(mb_prio, false);
    args->mb_values.copy_(mb_values, false);
    args->mb_returns.copy_(mb_returns, false);
}

// Full rollout copy: advantage + prio + select_and_copy (one minibatch iteration)
void run_full_rolloutcopy(RolloutCopyArgs* args) {
    nvtxRangePushA("compute_advantage");
    run_compute_advantage(args);
    nvtxRangePop();

    // Recompute idx inline (matches train_impl flow)
    nvtxRangePushA("compute_prio");
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, args->minibatch_segs, true);
    Tensor mb_prio = torch::pow(
        args->total_agents * prio_probs.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);
    nvtxRangePop();

    nvtxRangePushA("train_select_and_copy");
    Tensor mb_obs = args->observations.index_select(0, idx);
    Tensor mb_actions = args->actions.index_select(0, idx);
    Tensor mb_logprobs = args->logprobs.index_select(0, idx);
    Tensor mb_values = args->values.index_select(0, idx);
    Tensor mb_advantages = args->advantages.index_select(0, idx);
    Tensor mb_returns = mb_advantages + mb_values;

    args->mb_state.zero_();
    args->mb_obs.copy_(mb_obs, false);
    args->mb_actions.copy_(mb_actions, false);
    args->mb_logprobs.copy_(mb_logprobs, false);
    args->mb_advantages.copy_(mb_advantages, false);
    args->mb_prio.copy_(mb_prio, false);
    args->mb_values.copy_(mb_values, false);
    args->mb_returns.copy_(mb_returns, false);
    nvtxRangePop();
}

// ============================================================================
// Profile function
// ============================================================================

void profile_rolloutcopy(int num_segments, int horizon, int minibatch_segs,
                          int input_size, int num_atns, int num_layers, int hidden) {
    printf("========================================\n");
    printf("rolloutcopy (S=%d, T=%d, mb_segs=%d, in=%d, A=%d, H=%d)\n",
           num_segments, horizon, minibatch_segs, input_size, num_atns, hidden);
    printf("  rollout_rows=%d, minibatch=%d, using %s\n",
           num_segments, minibatch_segs * horizon, USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    auto* args = create_rolloutcopyargs(num_segments, horizon, minibatch_segs,
                                         input_size, num_atns, num_layers, hidden);

    // --- Individual phase timing ---
    printf("--- Per-Phase Timing ---\n");

    float adv_ms = profile_kernel((kernel_fn)run_compute_advantage, args, "compute_advantage");
    print_timing("compute_advantage", adv_ms, num_segments);

    // Ensure advantages are populated for prio/select phases
    run_compute_advantage(args);

    float prio_ms = profile_kernel((kernel_fn)run_compute_prio, args, "compute_prio");
    print_timing("compute_prio", prio_ms, num_segments);

    float copy_ms = profile_kernel((kernel_fn)run_select_and_copy, args, "train_select_and_copy");
    print_timing("train_select_and_copy", copy_ms, minibatch_segs);
    printf("\n");

    // --- Full rollout copy (all 3 phases) ---
    printf("--- Full Rollout Copy (one minibatch iteration) ---\n");

    float full_ms = profile_kernel((kernel_fn)run_full_rolloutcopy, args, "rolloutcopy_full");
    print_timing("rolloutcopy (full)", full_ms, num_segments);
    printf("\n");

    // --- Proportional breakdown ---
    printf("--- Proportional Breakdown ---\n");
    float total_phases = adv_ms + prio_ms + copy_ms;
    print_timing_pct("compute_advantage", adv_ms, num_segments, total_phases);
    print_timing_pct("compute_prio", prio_ms, num_segments, total_phases);
    print_timing_pct("train_select_and_copy", copy_ms, minibatch_segs, total_phases);
    printf("  %-28s %8.1f us  100.0%%\n", "total (sum of phases)", total_phases * 1000);
    printf("  %-28s %8.1f us  (measured)\n", "full rolloutcopy actual", full_ms * 1000);
    printf("\n");

    free_rolloutcopyargs(args);
}

#endif  // USE_TORCH
