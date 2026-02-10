// profiling/profile_train.cu
// Training forward + backward profiling with per-phase breakdown
// Profiles: encoder, RNN scan, decoder, loss, backward, optimizer step
// Uses Muon optimizer (matching production) and supports CUDA graph profiling
// Requires USE_TORCH
#pragma once
#include "profile.h"

#ifdef USE_TORCH

// ============================================================================
// TrainArgs: shared state for training profiles
// ============================================================================

typedef struct {
    // Model
    std::shared_ptr<Policy> policy_bf16;
    std::shared_ptr<Policy> policy_fp32;  // fp32 master weights for mixed precision

    // Training data (shaped for minibatch)
    Tensor obs;           // (N, T, input_size) - training observations
    Tensor state;         // (num_layers, N, 1, H) - initial RNN state for training
    Tensor actions;       // (N, T, 1) float64
    Tensor old_logprobs;  // (N, T)
    Tensor advantages;    // (N, T)
    Tensor prio;          // (N, 1)
    Tensor values;        // (N, T)
    Tensor returns;       // (N, T)
    Tensor ratio_out;     // (N, T) - side-effect output
    Tensor newvalue_out;  // (N, T) - side-effect output
    Tensor act_sizes;     // (1,) int32 cuda
    Tensor act_sizes_cpu; // (1,) int64 cpu

    // Muon optimizer (matches production — NOT Adam)
    std::shared_ptr<torch::optim::Muon> muon;

    // Config
    bool use_kernels;
    int N;                // number of segments (batch dim before T)
    int T_seq;            // sequence length
    int H;
    int A;
    int input_size;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float max_grad_norm;
} TrainArgs;

TrainArgs* create_trainargs(int N, int T_seq, int input_size, int hidden, int act_n,
                            int num_layers, bool use_kernels) {
    TrainArgs* args = new TrainArgs();
    args->use_kernels = use_kernels;
    args->N = N;
    args->T_seq = T_seq;
    args->H = hidden;
    args->A = act_n;
    args->input_size = input_size;
    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;
    args->max_grad_norm = 0.5f;

    // Create primary policy (bf16 when mixed-precision, fp32 otherwise)
    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden);
    auto dec = std::make_shared<DefaultDecoder>(hidden, act_n);
    auto rnn = std::make_shared<MinGRU>(hidden, num_layers, use_kernels);
    args->policy_bf16 = std::make_shared<Policy>(enc, dec, rnn, input_size, act_n, hidden);
    args->policy_bf16->to(torch::kCUDA);
    if (USE_BF16) {
        args->policy_bf16->to(torch::kBFloat16);
    }

    // Create fp32 master weights (for mixed-precision training)
    auto enc32 = std::make_shared<DefaultEncoder>(input_size, hidden);
    auto dec32 = std::make_shared<DefaultDecoder>(hidden, act_n);
    auto rnn32 = std::make_shared<MinGRU>(hidden, num_layers, use_kernels);
    args->policy_fp32 = std::make_shared<Policy>(enc32, dec32, rnn32, input_size, act_n, hidden);
    args->policy_fp32->to(torch::kCUDA);

    // Sync bf16 from fp32 (only needed in mixed precision)
    if (USE_BF16) {
        sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
    }

    // Create Muon optimizer over fp32 master weights (matches production)
    args->muon = std::make_shared<torch::optim::Muon>(
        args->policy_fp32->parameters(),
        torch::optim::MuonOptions(0.0025));
    args->muon->init_contiguous_weights();

    // Create synthetic training data
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->obs = torch::randn({N, T_seq, input_size}, opts);
    args->state = torch::zeros({num_layers, N, 1, hidden}, opts);
    args->actions = torch::randint(0, act_n, {N, T_seq, 1}, cuda_f64);
    args->old_logprobs = torch::randn({N, T_seq}, opts) * 0.5f;
    args->advantages = torch::randn({N, T_seq}, cuda_f32);
    args->prio = torch::ones({N, 1}, opts);
    args->values = torch::randn({N, T_seq}, opts);
    args->returns = torch::randn({N, T_seq}, opts);
    args->ratio_out = torch::zeros({N, T_seq}, opts);
    args->newvalue_out = torch::zeros({N, T_seq}, opts);
    args->act_sizes = torch::tensor({act_n}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_trainargs(TrainArgs* args) {
    delete args;
}

// ============================================================================
// Helper: compute loss from forward outputs (eliminates 5x duplicated arg list)
// ============================================================================

Tensor compute_loss_impl(TrainArgs* args, Logits& raw_logits, Tensor& newvalue) {
    Logits ls = {.mean = raw_logits.mean};
    if (raw_logits.logstd.defined()) ls.logstd = raw_logits.logstd;
    int mb = args->N * args->T_seq;
    return compute_train_loss(
        ls, newvalue,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->ratio_out, args->newvalue_out,
        args->act_sizes, args->act_sizes_cpu, mb, args->T_seq,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, args->use_kernels);
}

// ============================================================================
// Run functions for individual phases
// ============================================================================

// Full training forward + loss (no backward)
// NoGradGuard: we only measure forward time, no backward will follow.
// CUDA kernels execute identically — autograd tracking is CPU-side only.
void run_train_forward(TrainArgs* args) {
    torch::NoGradGuard no_grad;
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    compute_loss_impl(args, logits, newvalue);
}

// Full training forward + loss + backward
void run_train_forward_backward(TrainArgs* args) {
    args->policy_bf16->zero_grad();

    nvtxRangePushA("forward_train");
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    nvtxRangePop();

    nvtxRangePushA("compute_loss");
    auto loss = compute_loss_impl(args, logits, newvalue);
    nvtxRangePop();

    nvtxRangePushA("backward");
    loss.backward();
    nvtxRangePop();
}

// Full training step: forward + loss + backward + Muon optimizer (matches production)
void run_train_step(TrainArgs* args) {
    args->muon->zero_grad();
    args->policy_bf16->zero_grad();

    nvtxRangePushA("forward_train");
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    nvtxRangePop();

    nvtxRangePushA("compute_loss");
    auto loss = compute_loss_impl(args, logits, newvalue);
    nvtxRangePop();

    nvtxRangePushA("backward");
    loss.backward();
    nvtxRangePop();

    nvtxRangePushA("grad_sync");
    if (USE_BF16) {
        copy_gradients_to_fp32(args->policy_bf16.get(), args->policy_fp32.get());
    }
    clip_grad_norm_(args->policy_fp32->parameters(), args->max_grad_norm);
    nvtxRangePop();

    nvtxRangePushA("muon_step");
    args->muon->step();
    nvtxRangePop();

    nvtxRangePushA("weight_sync");
    args->muon->zero_grad();
    args->policy_bf16->zero_grad();
    if (USE_BF16) {
        sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
    }
    nvtxRangePop();
}

// ============================================================================
// Per-section isolated profiling (for forward breakdown)
// ============================================================================

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor obs;       // (N, T, input)
    Tensor state;     // (layers, N, 1, H)
    int N, T_seq, H, input_size;
} EncoderIsolatedArgs;

void run_encoder_isolated(EncoderIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    int B = args->N;
    int TT = args->T_seq;
    auto x = args->obs.reshape({B * TT, args->input_size});
    args->policy->encoder->forward(x);
}

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor h_encoded;  // (N, T, H)
    Tensor state;      // (layers, N, 1, H)
} RNNIsolatedArgs;

void run_rnn_isolated(RNNIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    args->policy->rnn->forward_train(args->h_encoded, args->state);
}

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor flat_h;  // (N*T, H)
} DecoderIsolatedArgs;

void run_decoder_isolated(DecoderIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    args->policy->decoder->forward(args->flat_h);
}

// ============================================================================
// Instrumented breakdowns: CUDA events at phase boundaries within a single run
// (Avoids the error of subtracting independently-measured profile_kernel calls)
// ============================================================================

struct StepTimings {
    float forward_ms;
    float loss_ms;
    float backward_ms;
    float grad_sync_ms;
    float muon_ms;
    float weight_sync_ms;
};

StepTimings profile_step_instrumented(TrainArgs* args, int num_iters = 200) {
    // Warmup
    for (int i = 0; i < 10; i++) run_train_step(args);
    cudaDeviceSynchronize();

    // Pre-allocate all events — NO per-iteration sync keeps GPU pipeline full
    const int P = 7;  // 7 boundary markers per iteration
    std::vector<cudaEvent_t> events(num_iters * P);
    for (auto& e : events) cudaEventCreate(&e);

    for (int i = 0; i < num_iters; i++) {
        args->muon->zero_grad();
        args->policy_bf16->zero_grad();
        int b = i * P;

        cudaEventRecord(events[b + 0]);
        auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
        cudaEventRecord(events[b + 1]);

        auto loss = compute_loss_impl(args, logits, newvalue);
        cudaEventRecord(events[b + 2]);

        loss.backward();
        cudaEventRecord(events[b + 3]);

        if (USE_BF16) copy_gradients_to_fp32(args->policy_bf16.get(), args->policy_fp32.get());
        clip_grad_norm_(args->policy_fp32->parameters(), args->max_grad_norm);
        cudaEventRecord(events[b + 4]);

        args->muon->step();
        cudaEventRecord(events[b + 5]);

        args->muon->zero_grad();
        args->policy_bf16->zero_grad();
        if (USE_BF16) sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
        cudaEventRecord(events[b + 6]);
    }

    cudaDeviceSynchronize();  // single sync at end

    StepTimings sum = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < num_iters; i++) {
        int b = i * P;
        float ms;
        cudaEventElapsedTime(&ms, events[b+0], events[b+1]); sum.forward_ms += ms;
        cudaEventElapsedTime(&ms, events[b+1], events[b+2]); sum.loss_ms += ms;
        cudaEventElapsedTime(&ms, events[b+2], events[b+3]); sum.backward_ms += ms;
        cudaEventElapsedTime(&ms, events[b+3], events[b+4]); sum.grad_sync_ms += ms;
        cudaEventElapsedTime(&ms, events[b+4], events[b+5]); sum.muon_ms += ms;
        cudaEventElapsedTime(&ms, events[b+5], events[b+6]); sum.weight_sync_ms += ms;
    }

    for (auto& e : events) cudaEventDestroy(e);

    float n = (float)num_iters;
    return { sum.forward_ms/n, sum.loss_ms/n, sum.backward_ms/n,
             sum.grad_sync_ms/n, sum.muon_ms/n, sum.weight_sync_ms/n };
}

// ============================================================================
// Main profile functions
// ============================================================================

void profile_trainforward(int N, int T_seq, int input_size, int hidden, int act_n, int num_layers) {
    printf("========================================\n");
    printf("trainforward (N=%d, T=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           N, T_seq, input_size, hidden, act_n, num_layers);
    printf("  minibatch=%d, using %s\n", N * T_seq,
           USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    bool use_kernels = true;

    // ----- Full forward + loss (no backward) -----
    printf("--- Forward + Loss (no backward) ---\n");
    TrainArgs* args_fwd = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
    float fwd_ms = profile_kernel((kernel_fn)run_train_forward, args_fwd, "trainforward");
    print_timing("forward+loss (kernel)", fwd_ms, N * T_seq);
    // args_fwd kept alive — reused by isolated phase breakdown below

    {
        TrainArgs* args_fwd_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        float fwd_cpp_ms = profile_kernel((kernel_fn)run_train_forward, args_fwd_cpp, "trainforward_cpp");
        print_timing("forward+loss (cpp)", fwd_cpp_ms, N * T_seq);
        free_trainargs(args_fwd_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    printf("\n");

    // ----- Full forward + loss + backward -----
    printf("--- Forward + Loss + Backward ---\n");
    float fb_ms, fb_cpp_ms;
    {
        TrainArgs* args_fb = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        fb_ms = profile_kernel((kernel_fn)run_train_forward_backward, args_fb, "train_fwd_bwd");
        print_timing("fwd+loss+bwd (kernel)", fb_ms, N * T_seq);
        free_trainargs(args_fb);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    {
        TrainArgs* args_fb_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        fb_cpp_ms = profile_kernel((kernel_fn)run_train_forward_backward, args_fb_cpp, "train_fwd_bwd_cpp");
        print_timing("fwd+loss+bwd (cpp)", fb_cpp_ms, N * T_seq);
        free_trainargs(args_fb_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    printf("\n");

    // ----- Per-section breakdown (forward only, no autograd) -----
    printf("--- Phase Breakdown (forward only, no autograd) ---\n");

    // Encoder
    auto enc_args = new EncoderIsolatedArgs();
    enc_args->policy = args_fwd->policy_bf16;
    enc_args->obs = args_fwd->obs;
    enc_args->state = args_fwd->state;
    enc_args->N = N;
    enc_args->T_seq = T_seq;
    enc_args->H = hidden;
    enc_args->input_size = input_size;
    float enc_ms = profile_kernel((kernel_fn)run_encoder_isolated, enc_args, "encoder");
    delete enc_args;
    c10::cuda::CUDACachingAllocator::emptyCache();

    // RNN
    auto rnn_args = new RNNIsolatedArgs();
    rnn_args->policy = args_fwd->policy_bf16;
    {
        torch::NoGradGuard no_grad;
        auto x = args_fwd->obs.reshape({N * T_seq, input_size});
        auto h = args_fwd->policy_bf16->encoder->forward(x);
        rnn_args->h_encoded = h.reshape({N, T_seq, hidden});
    }
    rnn_args->state = args_fwd->state;
    float rnn_ms = profile_kernel((kernel_fn)run_rnn_isolated, rnn_args, "rnn_scan");
    delete rnn_args;
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Decoder
    auto dec_args = new DecoderIsolatedArgs();
    dec_args->policy = args_fwd->policy_bf16;
    {
        torch::NoGradGuard no_grad;
        auto x = args_fwd->obs.reshape({N * T_seq, input_size});
        auto h = args_fwd->policy_bf16->encoder->forward(x);
        h = h.reshape({N, T_seq, hidden});
        h = args_fwd->policy_bf16->rnn->forward_train(h, args_fwd->state);
        dec_args->flat_h = h.reshape({-1, hidden});
    }
    float dec_ms = profile_kernel((kernel_fn)run_decoder_isolated, dec_args, "decoder");
    delete dec_args;

    float total_phases = enc_ms + rnn_ms + dec_ms;
    print_timing_pct("encoder (linear)", enc_ms, N * T_seq, total_phases);
    print_timing_pct("rnn (fused_scan)", rnn_ms, N * T_seq, total_phases);
    print_timing_pct("decoder (linear)", dec_ms, N * T_seq, total_phases);
    printf("  %-28s %8.1f us  100.0%%\n", "total (sum of phases)", total_phases * 1000);
    printf("  %-28s %8.1f us  (measured)\n", "forward+loss actual", fwd_ms * 1000);
    printf("\n");

    free_trainargs(args_fwd);
}

void profile_trainstep(int N, int T_seq, int input_size, int hidden, int act_n, int num_layers) {
    printf("========================================\n");
    printf("trainstep (N=%d, T=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           N, T_seq, input_size, hidden, act_n, num_layers);
    printf("  minibatch=%d, using %s, optimizer=Muon\n", N * T_seq,
           USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    bool use_kernels = true;

    // ----- Full training step (eager) -----
    printf("--- Full Training Step: fwd + loss + bwd + clip + Muon + sync ---\n");
    float step_ms;
    {
        TrainArgs* args = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        step_ms = profile_kernel((kernel_fn)run_train_step, args, "trainstep");
        print_timing("trainstep (kernel)", step_ms, N * T_seq);
        free_trainargs(args);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    {
        TrainArgs* args_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        float step_cpp_ms = profile_kernel((kernel_fn)run_train_step, args_cpp, "trainstep_cpp");
        print_timing("trainstep (cpp)", step_cpp_ms, N * T_seq);
        free_trainargs(args_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // ----- Instrumented breakdown (CUDA events at phase boundaries) -----
    printf("\n--- Training Step Breakdown (instrumented) ---\n");
    {
        TrainArgs* args_bd = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        auto t = profile_step_instrumented(args_bd);
        float total = t.forward_ms + t.loss_ms + t.backward_ms
                    + t.grad_sync_ms + t.muon_ms + t.weight_sync_ms;
        print_timing_pct("forward", t.forward_ms, N * T_seq, total);
        print_timing_pct("loss", t.loss_ms, N * T_seq, total);
        print_timing_pct("backward", t.backward_ms, N * T_seq, total);
        print_timing_pct("grad_sync+clip", t.grad_sync_ms, N * T_seq, total);
        print_timing_pct("Muon step", t.muon_ms, N * T_seq, total);
        print_timing_pct("weight_sync", t.weight_sync_ms, N * T_seq, total);
        printf("  %-28s %8.1f us  100.0%%\n", "total step", total * 1000);
        printf("  %-28s %8.1f us  (profile_kernel)\n", "trainstep measured", step_ms * 1000);
        printf("\n");
        free_trainargs(args_bd);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // ----- CUDA Graph training step -----
    // Production captures fwd+loss+bwd+optim+sync as a single graph.
    // profile_graph does warmup → capture → timed replay, matching production.
    printf("--- CUDA Graph Training Step ---\n");
    {
        TrainArgs* args_graph = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        float graph_ms = profile_graph((kernel_fn)run_train_step, args_graph, "trainstep_graph");
        print_timing("trainstep (graph)", graph_ms, N * T_seq);

        float graph_speedup = step_ms / graph_ms;
        printf("  graph speedup vs eager:    %.2fx\n", graph_speedup);
        printf("\n");
        free_trainargs(args_graph);
    }
}

#endif  // USE_TORCH
