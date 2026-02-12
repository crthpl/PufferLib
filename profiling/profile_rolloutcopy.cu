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

    // Pre-computed prio results (for isolated select+copy profiling)
    Tensor cached_idx;
    Tensor cached_mb_prio;

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
// Shared helpers (single source of truth for prio + select logic)
// ============================================================================

struct PrioResult {
    Tensor idx;
    Tensor mb_prio;
};

// Priority-weighted sampling: advantages -> probabilities -> multinomial -> importance weights
PrioResult compute_prio_impl(RolloutCopyArgs* args) {
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, args->minibatch_segs, true);
    Tensor mb_prio = torch::pow(
        args->total_agents * prio_probs.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);
    return {idx, mb_prio};
}

// index_select sampled segments + copy_ into graph buffers
void select_and_copy_impl(RolloutCopyArgs* args, const Tensor& idx, const Tensor& mb_prio) {
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

// ============================================================================
// Phase runners (for individual profile_kernel calls)
// ============================================================================

// Helper: run an advantage dispatch function with args unpacked
typedef void (*adv_dispatch_fn)(Tensor, Tensor, Tensor, Tensor, Tensor,
                                double, double, double, double);

void run_advantage_impl(RolloutCopyArgs* args, adv_dispatch_fn fn) {
    args->advantages.fill_(0.0);
    fn(args->values, args->rewards, args->terminals, args->ratio,
       args->advantages, args->gamma, args->gae_lambda,
       args->rho_clip, args->c_clip);
}

// Phase 1: compute_advantage — vectorized CUDA kernel (production)
void run_compute_advantage(RolloutCopyArgs* args) {
    run_advantage_impl(args, compute_puff_advantage_cuda);
}

// Phase 1 (scalar baseline): scalar-only CUDA kernel (for benchmarking)
void run_compute_advantage_scalar(RolloutCopyArgs* args) {
    run_advantage_impl(args, compute_puff_advantage_cuda_scalar);
}

// Phase 1 (PyTorch reference): GAE in pure PyTorch ops
// Sequential loop over time, but vectorized across all rows per timestep
void run_compute_advantage_torch(RolloutCopyArgs* args) {
    auto& vals = args->values;
    auto& rews = args->rewards;
    auto& terms = args->terminals;
    auto& ratio = args->ratio;
    auto& advantages = args->advantages;
    float gamma = args->gamma;
    float gae_lambda = args->gae_lambda;
    float rho_clip = args->rho_clip;
    float c_clip = args->c_clip;
    int T = args->horizon;

    advantages.zero_();
    auto lastgaelam = torch::zeros({args->num_segments}, cuda_f32);

    for (int t = T - 2; t >= 0; t--) {
        auto nextnonterminal = 1.0f - terms.select(1, t + 1).to(torch::kFloat32);
        auto imp = ratio.select(1, t).to(torch::kFloat32);
        auto rho_t = imp.clamp_max(rho_clip);
        auto c_t = imp.clamp_max(c_clip);
        auto next_val = vals.select(1, t + 1).to(torch::kFloat32);
        auto cur_val = vals.select(1, t).to(torch::kFloat32);
        auto next_rew = rews.select(1, t + 1).to(torch::kFloat32);

        auto delta = rho_t * (next_rew + gamma * next_val * nextnonterminal - cur_val);
        lastgaelam = delta + gamma * gae_lambda * c_t * lastgaelam * nextnonterminal;
        advantages.select(1, t).copy_(lastgaelam);
    }
}

// Phase 2 (PyTorch reference): compute_prio — priority-weighted sampling (all PyTorch ops)
void run_compute_prio_torch(RolloutCopyArgs* args) {
    compute_prio_impl(args);
}

// Phase 2 (CUDA kernel): fused 3-kernel compute_prio (production path)
void run_compute_prio_kernel(RolloutCopyArgs* args) {
    compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
}

// Phase 3 (torch): train_select_and_copy — pure PyTorch select+copy using cached prio results
void run_select_and_copy_torch(RolloutCopyArgs* args) {
    select_and_copy_impl(args, args->cached_idx, args->cached_mb_prio);
}

// Phase 3 (kernel): train_select_and_copy — fused CUDA kernel using cached prio results
void run_select_and_copy_kernel(RolloutCopyArgs* args) {
    train_select_and_copy_cuda(
        args->observations, args->actions, args->logprobs,
        args->values, args->advantages,
        args->cached_idx, args->cached_mb_prio,
        args->mb_obs, args->mb_state, args->mb_actions,
        args->mb_logprobs, args->mb_advantages, args->mb_prio,
        args->mb_values, args->mb_returns);
}

// Full rollout copy: advantage + kernel prio + kernel select_and_copy (one minibatch iteration)
void run_full_rolloutcopy(RolloutCopyArgs* args) {
    nvtxRangePushA("compute_advantage");
    run_compute_advantage(args);
    nvtxRangePop();

    nvtxRangePushA("compute_prio");
    auto [idx, mb_prio] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
    nvtxRangePop();

    nvtxRangePushA("train_select_and_copy");
    args->cached_idx = idx;
    args->cached_mb_prio = mb_prio;
    run_select_and_copy_kernel(args);
    nvtxRangePop();
}

// ============================================================================
// Correctness check: vectorized vs scalar (must match exactly)
// ============================================================================

void test_advantage_correct(RolloutCopyArgs* args) {
    int S = args->num_segments;
    int T = args->horizon;

    // Run vectorized kernel
    auto adv_vec = torch::zeros({S, T}, cuda_f32);
    compute_puff_advantage_cuda(
        args->values, args->rewards, args->terminals, args->ratio,
        adv_vec, args->gamma, args->gae_lambda, args->rho_clip, args->c_clip);

    // Run scalar kernel
    auto adv_scalar = torch::zeros({S, T}, cuda_f32);
    compute_puff_advantage_cuda_scalar(
        args->values, args->rewards, args->terminals, args->ratio,
        adv_scalar, args->gamma, args->gae_lambda, args->rho_clip, args->c_clip);

    cudaDeviceSynchronize();

    // Compare: should be bit-identical (same float arithmetic, just different load pattern)
    float max_diff = (adv_vec - adv_scalar).abs().max().item<float>();
    const char* status = (max_diff < 1e-6f) ? "PASS" : "FAIL";
    printf("  correctness (vec vs scalar): %s (max_diff=%.2e)\n", status, max_diff);

    // Also compare against PyTorch reference (may have small fp differences)
    auto saved_adv = args->advantages;
    args->advantages = torch::zeros({S, T}, cuda_f32);
    run_compute_advantage_torch(args);
    auto adv_torch = args->advantages;
    args->advantages = saved_adv;
    cudaDeviceSynchronize();

    float max_diff_torch = (adv_vec - adv_torch).abs().max().item<float>();
    float atol = USE_BF16 ? 5e-2f : 1e-5f;  // bf16 inputs lose precision in .to() conversions
    const char* torch_status = (max_diff_torch < atol) ? "PASS" : "FAIL";
    printf("  correctness (vec vs torch):  %s (max_diff=%.2e)\n", torch_status, max_diff_torch);
}

// ============================================================================
// Correctness check: kernel prio vs PyTorch prio
// ============================================================================

void test_prio_correct(RolloutCopyArgs* args) {
    // Run the full kernel pipeline
    auto [idx, mb_prio_kernel] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);

    // PyTorch reference probs
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor probs_torch = (prio_weights + 1e-6f) / (prio_weights.sum() + 1e-6f);

    // Check full pipeline: kernel's mb_prio vs torch mb_prio (using kernel's idx)
    Tensor mb_prio_torch = torch::pow(
        (float)args->total_agents * probs_torch.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);

    cudaDeviceSynchronize();

    float prio_diff = (mb_prio_kernel - mb_prio_torch).abs().max().item<float>();
    const char* prio_status = (prio_diff < 1e-2f) ? "PASS" : "FAIL";
    printf("  correctness (mb_prio):  %s (max_diff=%.2e)\n", prio_status, prio_diff);
}

// ============================================================================
// Correctness check: kernel select+copy vs PyTorch select+copy
// ============================================================================

void test_select_copy_correct(RolloutCopyArgs* args) {
    // Pre-compute prio results for both paths
    auto [idx, mb_prio] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
    args->cached_idx = idx;
    args->cached_mb_prio = mb_prio;

    // Run PyTorch reference path
    run_select_and_copy_torch(args);
    auto ref_obs = args->mb_obs.clone();
    auto ref_actions = args->mb_actions.clone();
    auto ref_logprobs = args->mb_logprobs.clone();
    auto ref_advantages = args->mb_advantages.clone();
    auto ref_values = args->mb_values.clone();
    auto ref_returns = args->mb_returns.clone();

    // Zero destination buffers and run kernel path
    args->mb_obs.zero_();
    args->mb_actions.zero_();
    args->mb_logprobs.zero_();
    args->mb_advantages.zero_();
    args->mb_values.zero_();
    args->mb_returns.zero_();
    run_select_and_copy_kernel(args);

    cudaDeviceSynchronize();

    // Compare all outputs
    auto check = [](const char* name, Tensor& kernel, Tensor& ref) {
        float diff = (kernel.to(torch::kFloat32) - ref.to(torch::kFloat32)).abs().max().item<float>();
        const char* status = (diff < 1e-5f) ? "PASS" : "FAIL";
        printf("  %-16s %s (max_diff=%.2e)\n", name, status, diff);
    };

    check("obs", args->mb_obs, ref_obs);
    check("actions", args->mb_actions, ref_actions);
    check("logprobs", args->mb_logprobs, ref_logprobs);
    check("advantages", args->mb_advantages, ref_advantages);
    check("values", args->mb_values, ref_values);
    check("returns", args->mb_returns, ref_returns);
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

    // === Advantage Kernel Comparison ===
    printf("--- Advantage Kernel Comparison (S=%d, T=%d) ---\n", num_segments, horizon);

    float adv_vec_ms = profile_kernel((kernel_fn)run_compute_advantage, args, "advantage_vectorized");
    print_timing("advantage (vectorized)", adv_vec_ms, num_segments);

    float adv_scalar_ms = profile_kernel((kernel_fn)run_compute_advantage_scalar, args, "advantage_scalar");
    print_timing("advantage (scalar)", adv_scalar_ms, num_segments);

    float adv_torch_ms = profile_kernel((kernel_fn)run_compute_advantage_torch, args, "advantage_torch");
    print_timing("advantage (torch)", adv_torch_ms, num_segments);

    printf("  vectorized vs scalar:  %.2fx\n", adv_scalar_ms / adv_vec_ms);
    printf("  vectorized vs torch:   %.2fx\n", adv_torch_ms / adv_vec_ms);

    test_advantage_correct(args);
    printf("\n");

    // === Prio Kernel Comparison ===
    printf("--- Prio Kernel Comparison (S=%d, T=%d, mb=%d) ---\n",
           num_segments, horizon, minibatch_segs);

    // Ensure advantages are populated for prio
    run_compute_advantage(args);

    float prio_torch_ms = profile_kernel((kernel_fn)run_compute_prio_torch, args, "prio_torch");
    print_timing("prio (torch)", prio_torch_ms, num_segments);

    float prio_kernel_ms = profile_kernel((kernel_fn)run_compute_prio_kernel, args, "prio_kernel");
    print_timing("prio (kernel)", prio_kernel_ms, num_segments);

    printf("  kernel vs torch:       %.2fx\n", prio_torch_ms / prio_kernel_ms);

    test_prio_correct(args);
    printf("\n");

    // === Select+Copy Kernel Comparison ===
    printf("--- Select+Copy Kernel Comparison (mb=%d) ---\n", minibatch_segs);

    // Ensure advantages + cached prio are populated
    run_compute_advantage(args);
    {
        auto [idx, mb_prio] = compute_prio_cuda(
            args->advantages, args->prio_alpha, args->minibatch_segs,
            args->total_agents, args->anneal_beta);
        args->cached_idx = idx;
        args->cached_mb_prio = mb_prio;
    }

    float copy_torch_ms = profile_kernel((kernel_fn)run_select_and_copy_torch, args, "select_copy_torch");
    print_timing("select+copy (torch)", copy_torch_ms, minibatch_segs);

    float copy_kernel_ms = profile_kernel((kernel_fn)run_select_and_copy_kernel, args, "select_copy_kernel");
    print_timing("select+copy (kernel)", copy_kernel_ms, minibatch_segs);

    printf("  kernel vs torch:       %.2fx\n", copy_torch_ms / copy_kernel_ms);

    test_select_copy_correct(args);
    printf("\n");

    // === Per-Phase Timing (rollout copy pipeline, using kernel paths) ===
    printf("--- Per-Phase Timing ---\n");

    float adv_ms = adv_vec_ms;
    print_timing("compute_advantage", adv_ms, num_segments);

    run_compute_advantage(args);

    float prio_ms = prio_kernel_ms;  // use kernel prio (production path)
    print_timing("compute_prio", prio_ms, num_segments);

    float copy_ms = copy_kernel_ms;  // use kernel select+copy (production path)
    print_timing("train_select_and_copy", copy_ms, minibatch_segs);
    printf("\n");

    // --- Full rollout copy (all 3 phases, fresh args) ---
    printf("--- Full Rollout Copy (one minibatch iteration) ---\n");

    auto* full_args = create_rolloutcopyargs(num_segments, horizon, minibatch_segs,
                                         input_size, num_atns, num_layers, hidden);
    float full_ms = profile_kernel((kernel_fn)run_full_rolloutcopy, full_args, "rolloutcopy_full");
    print_timing("rolloutcopy (full)", full_ms, num_segments);
    free_rolloutcopyargs(full_args);
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
