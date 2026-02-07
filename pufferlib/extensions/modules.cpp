#ifndef PUFFERLIB_MODULES_CPP
#define PUFFERLIB_MODULES_CPP

#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "cuda/kernels.h"

#include <stdio.h>
#include <stdlib.h>

// Fused: chunk + mingru_gate + sigmoid(proj) * out
// combined is (B, 1, 3*H) = [hidden, gate, proj]
// state is (B, 1, H)
// returns {out, next_state} where:
//   out (B, 1, H) = sigmoid(proj) * mingru_out
//   next_state (B, 1, H) = mingru_out (for recurrence)
std::vector<torch::Tensor> mingru_gate(
    torch::Tensor state,
    torch::Tensor combined
) {
    TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
    TORCH_CHECK(combined.is_cuda(), "combined must be on CUDA");
    TORCH_CHECK(state.dtype() == combined.dtype(), "dtypes must match");
    TORCH_CHECK(state.dim() == 3 && combined.dim() == 3, "must be 3D tensors");
    TORCH_CHECK(combined.size(2) == 3 * state.size(2), "combined must be 3*H");
    TORCH_CHECK(state.size(0) == combined.size(0), "batch size must match");
    TORCH_CHECK(state.is_contiguous() && combined.is_contiguous(), "must be contiguous");

    auto B = state.size(0);
    auto H = state.size(2);

    auto out = torch::empty_like(state);
    auto next_state = torch::empty_like(state);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_mingru_gate_inference(
        (precision_t*)out.data_ptr(),
        (precision_t*)next_state.data_ptr(),
        (const precision_t*)combined.data_ptr(),
        (const precision_t*)state.data_ptr(),
        static_cast<int>(H),
        static_cast<int>(B),
        stream
    );
    return {out, next_state};
}

// Checkpointed version: uses sparse checkpoints to reduce memory, recomputes in backward
class FusedScanCheckpointedFunction : public torch::autograd::Function<FusedScanCheckpointedFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor combined,  // (B, T, 3*H) = [hidden, gate, proj]
        torch::Tensor state      // (B, 1, H)
    ) {
        TORCH_CHECK(combined.is_cuda(), "combined must be on CUDA");
        TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
        TORCH_CHECK(combined.dtype() == state.dtype(), "dtypes must match");
        TORCH_CHECK(combined.dim() == 3, "combined must be (B, T, 3*H)");
        TORCH_CHECK(state.dim() == 3, "state must be (B, 1, H)");
        TORCH_CHECK(state.size(0) == combined.size(0), "B must match");
        TORCH_CHECK(state.size(1) == 1, "state T dim must be 1");
        TORCH_CHECK(combined.size(2) == 3 * state.size(2), "combined must be 3*H");
        TORCH_CHECK(combined.is_contiguous() && state.is_contiguous(),
                    "All tensors must be contiguous");

        auto device = combined.device();
        auto B = combined.size(0);
        auto T = combined.size(1);
        auto H = state.size(2);
        auto T_buf = T + 1;

        auto out = torch::empty({B, T, H}, state.options());
        auto next_state = torch::empty({B, 1, H}, state.options());

        // Sparse checkpoint buffers (still needed for backward recomputation)
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto a_star = torch::empty({B, T_buf, H}, options_float);
        auto s_vals = torch::empty({B, T_buf, H}, options_float);
        auto log_values_buf = torch::empty({B, T_buf, H}, options_float);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        launch_fused_scan_forward_checkpointed(
            (precision_t*)out.data_ptr(),
            (precision_t*)next_state.data_ptr(),
            a_star.data_ptr<float>(),
            s_vals.data_ptr<float>(),
            log_values_buf.data_ptr<float>(),
            (const precision_t*)combined.data_ptr(),
            (const precision_t*)state.data_ptr(),
            static_cast<int>(T),
            static_cast<int>(H),
            static_cast<int>(B),
            stream
        );

        // Save all tensors for backward (ensures proper cleanup after backward)
        ctx->save_for_backward({combined, state, a_star, s_vals, log_values_buf});

        return {out, next_state};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto combined = saved[0];
        auto state = saved[1];
        auto a_star_buf = saved[2];
        auto s_vals = saved[3];
        auto log_values_buf = saved[4];

        auto grad_out = grad_outputs[0];
        TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");

        auto grad_next_state = grad_outputs[1];
        TORCH_CHECK(grad_next_state.is_contiguous(), "grad_next_state must be contiguous");

        auto B = combined.size(0);
        auto T = combined.size(1);
        auto H = state.size(2);

        auto grad_combined = torch::empty_like(combined);
        auto grad_state = torch::empty_like(state);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        launch_fused_scan_backward_checkpointed(
            (precision_t*)grad_combined.data_ptr(),
            (precision_t*)grad_state.data_ptr(),
            (const precision_t*)grad_out.data_ptr(),
            (const precision_t*)grad_next_state.data_ptr(),
            (const precision_t*)combined.data_ptr(),
            (const precision_t*)state.data_ptr(),
            a_star_buf.data_ptr<float>(),
            s_vals.data_ptr<float>(),
            log_values_buf.data_ptr<float>(),
            static_cast<int>(T),
            static_cast<int>(H),
            static_cast<int>(B),
            stream
        );

        return {grad_combined, grad_state};
    }
};

// Named entrypoint: fused_scan_checkpointed(combined, state) -> {out, next_state}
// Same interface as fused_scan but uses checkpointed kernels for reduced memory
torch::autograd::tensor_list fused_scan_checkpointed(
    torch::Tensor combined,
    torch::Tensor state
) {
    return FusedScanCheckpointedFunction::apply(combined, state);
}

class LogCumsumExpFunction : public torch::autograd::Function<LogCumsumExpFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x  // (B, T, H)
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
        auto device = x.device();
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto out = torch::empty({B, T, H}, x.options());
        //auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        auto s_buf = torch::empty({B, T, H}, options_double);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        launch_logcumsumexp_forward(
            (precision_t*)out.data_ptr(),
            s_buf.data_ptr<double>(),
            (const precision_t*)x.data_ptr(),
            (int)T, (int)H, (int)B,
            stream
        );

        ctx->save_for_backward({x, out, s_buf});
        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0].contiguous();  // input x might not be contiguous
        auto s_buf = saved[2];  // s_buf was from torch::empty, already contiguous

        auto grad_out = grad_outputs[0].contiguous();  // incoming grad might not be contiguous
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto grad_x = torch::empty_like(x);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        launch_logcumsumexp_backward(
            (precision_t*)grad_x.data_ptr(),
            (const precision_t*)grad_out.data_ptr(),
            (const precision_t*)x.data_ptr(),
            s_buf.data_ptr<double>(),
            (int)T, (int)H, (int)B,
            stream
        );

        return {grad_x};
    }
};

// Entry point
torch::Tensor logcumsumexp_cuda(torch::Tensor x) {
    return LogCumsumExpFunction::apply(x)[0];
}

class PPOFusedLossOptimizedFunction : public torch::autograd::Function<PPOFusedLossOptimizedFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor logits,           // (N, T, A_total) where A_total = sum(act_sizes); for continuous: mean
        torch::Tensor logstd,           // (N, T, num_atns) for continuous; empty tensor for discrete
        torch::Tensor values_pred,      // (N, T, 1) or (N, T)
        torch::Tensor actions,          // (N, T, num_atns) - float64 for both continuous and discrete
        torch::Tensor old_logprobs,     // (N, T)
        torch::Tensor advantages,       // (N, T)
        torch::Tensor prio,             // (N, 1) — importance weights
        torch::Tensor values,           // (N, T)
        torch::Tensor returns,          // (N, T)
        torch::Tensor adv_mean,         // (1)
        torch::Tensor adv_var,          // (1) - variance, kernel computes sqrt
        torch::Tensor ratio_out,        // (N, T) - output for ratio
        torch::Tensor newvalue_out,     // (N, T) - output for newvalue
        torch::Tensor act_sizes,        // (num_atns,) int32 CUDA tensor - size of each action head
        double clip_coef,
        double vf_clip_coef,
        double vf_coef,
        double ent_coef
    ) {
        TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
        TORCH_CHECK(act_sizes.is_cuda(), "act_sizes must be on CUDA");
        TORCH_CHECK(act_sizes.dtype() == torch::kInt32, "act_sizes must be int32");
        auto dtype = logits.dtype();
        TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kBFloat16,
                    "Only float32 and bfloat16 supported");

        bool is_continuous = logstd.defined() && logstd.numel() > 0;

        // Create empty CUDA tensor for logstd if discrete (undefined tensors can't be saved)
        auto logstd_to_save = is_continuous ? logstd : torch::empty({0}, logits.options());

        auto device = logits.device();
        auto N = logits.size(0);
        auto T = logits.size(1);
        auto A_total = logits.size(2);
        auto num_atns = act_sizes.size(0);

        // Reshape actions from (N, T, num_atns) to (N*T, num_atns) for kernel
        auto actions_flat = actions.reshape({N * T, num_atns}).contiguous();

        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto options_double = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        auto loss_output = torch::empty({1}, options_float);

        // saved_for_backward not used by optimized backward, but kernel still writes to it
        auto saved_for_backward = torch::zeros({N * T, 5}, options_double);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // strides for non-contiguous tensor support
        auto logits_strides = logits.strides();
        auto values_strides = values_pred.strides();
        int logits_stride_n = logits_strides[0];
        int logits_stride_t = logits_strides[1];
        int logits_stride_a = logits_strides[2];
        int values_stride_n = values_strides[0];
        int values_stride_t = values_strides[1];

        launch_ppo_loss_forward_optimized(
            loss_output.data_ptr<float>(),
            saved_for_backward.data_ptr<double>(),
            (precision_t*)ratio_out.data_ptr(),
            (precision_t*)newvalue_out.data_ptr(),
            (const precision_t*)logits.data_ptr(),
            is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
            (const precision_t*)values_pred.data_ptr(),
            actions_flat.data_ptr<double>(),
            (const precision_t*)old_logprobs.data_ptr(),
            advantages.data_ptr<float>(),
            (const precision_t*)prio.data_ptr(),
            (const precision_t*)values.data_ptr(),
            (const precision_t*)returns.data_ptr(),
            adv_mean.data_ptr<float>(),
            adv_var.data_ptr<float>(),
            act_sizes.data_ptr<int>(),
            static_cast<int>(num_atns),
            static_cast<float>(clip_coef),
            static_cast<float>(vf_clip_coef),
            static_cast<float>(vf_coef),
            static_cast<float>(ent_coef),
            T, A_total, N,
            logits_stride_n, logits_stride_t, logits_stride_a,
            values_stride_n, values_stride_t,
            is_continuous,
            stream
        );

        ctx->saved_data["clip_coef"] = clip_coef;
        ctx->saved_data["vf_clip_coef"] = vf_clip_coef;
        ctx->saved_data["vf_coef"] = vf_coef;
        ctx->saved_data["ent_coef"] = ent_coef;
        ctx->saved_data["is_continuous"] = is_continuous;

        // Save tensors for backward - use logstd_to_save (empty CUDA tensor for discrete)
        ctx->save_for_backward({logits, logstd_to_save, values_pred, actions_flat, old_logprobs, advantages,
                                prio, values, returns, adv_mean, adv_var, act_sizes});

        return {loss_output};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto logits = saved[0].contiguous();
        auto logstd = saved[1];  // may be empty for discrete
        auto values_pred = saved[2].contiguous();
        auto actions_flat = saved[3].contiguous();  // already (N*T, num_atns) from forward
        auto old_logprobs = saved[4].contiguous();
        auto advantages = saved[5].contiguous();
        auto prio = saved[6].contiguous();
        auto values = saved[7].contiguous();
        auto returns = saved[8].contiguous();
        auto adv_mean = saved[9].contiguous();
        auto adv_var = saved[10].contiguous();
        auto act_sizes = saved[11];  // already on CUDA and contiguous

        auto N = logits.size(0);
        auto T = logits.size(1);
        auto A_total = logits.size(2);
        auto num_atns = act_sizes.size(0);

        float clip_coef = ctx->saved_data["clip_coef"].to<double>();
        float vf_clip_coef = ctx->saved_data["vf_clip_coef"].to<double>();
        float vf_coef = ctx->saved_data["vf_coef"].to<double>();
        float ent_coef = ctx->saved_data["ent_coef"].to<double>();
        bool is_continuous = ctx->saved_data["is_continuous"].to<bool>();

        auto grad_loss = grad_outputs[0].to(torch::kFloat32).contiguous();

        // keep gradients in fp32 for precision / training stability issues
        auto grad_logits = torch::empty(logits.sizes(), logits.options().dtype(torch::kFloat32));
        auto grad_values_pred = torch::empty(values_pred.sizes(), values_pred.options().dtype(torch::kFloat32));
        // For continuous: grad_logstd has same shape as logstd
        torch::Tensor grad_logstd;
        if (is_continuous) {
            logstd = logstd.contiguous();
            grad_logstd = torch::empty(logstd.sizes(), logstd.options().dtype(torch::kFloat32));
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // need strides
        auto logits_strides = logits.strides();
        auto values_strides = values_pred.strides();
        int logits_stride_n = logits_strides[0];
        int logits_stride_t = logits_strides[1];
        int logits_stride_a = logits_strides[2];
        int values_stride_n = values_strides[0];
        int values_stride_t = values_strides[1];

        launch_ppo_loss_backward_optimized(
            grad_logits.data_ptr<float>(),
            is_continuous ? grad_logstd.data_ptr<float>() : nullptr,
            grad_values_pred.data_ptr<float>(),
            grad_loss.data_ptr<float>(),
            (const precision_t*)logits.data_ptr(),
            is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
            (const precision_t*)values_pred.data_ptr(),
            actions_flat.data_ptr<double>(),
            (const precision_t*)old_logprobs.data_ptr(),
            advantages.data_ptr<float>(),
            (const precision_t*)prio.data_ptr(),
            (const precision_t*)values.data_ptr(),
            (const precision_t*)returns.data_ptr(),
            adv_mean.data_ptr<float>(),
            adv_var.data_ptr<float>(),
            act_sizes.data_ptr<int>(),
            static_cast<int>(num_atns),
            clip_coef, vf_clip_coef,
            vf_coef, ent_coef,
            T, A_total, N,
            logits_stride_n, logits_stride_t, logits_stride_a,
            values_stride_n, values_stride_t,
            is_continuous,
            stream
        );

        return {
            grad_logits,
            is_continuous ? grad_logstd : torch::Tensor(),  // grad_logstd
            grad_values_pred,
            {}, {}, {}, {}, {}, {},  // actions, old_logprobs, advantages, prio, values, returns
            {}, {},                   // adv_mean, adv_std
            {}, {},                   // ratio_out, newvalue_out (no grad needed)
            {},                       // act_sizes (no grad needed)
            {}, {}, {}, {}           // clip_coef, vf_clip_coef, vf_coef, ent_coef
        };
    }
};

torch::autograd::tensor_list fused_ppo_loss_optimized(
    torch::Tensor logits,           // For continuous: mean
    torch::Tensor logstd,           // For continuous: log std; empty tensor for discrete
    torch::Tensor values_pred,
    torch::Tensor actions,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor prio,
    torch::Tensor values,
    torch::Tensor returns,
    torch::Tensor adv_mean,
    torch::Tensor adv_var,  // variance, kernel does sqrt
    torch::Tensor ratio_out,
    torch::Tensor newvalue_out,
    torch::Tensor act_sizes,  // (num_atns,) int32 CUDA tensor - size of each action head
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef
) {
    return PPOFusedLossOptimizedFunction::apply(logits, logstd, values_pred, actions,
        old_logprobs, advantages, prio, values, returns, adv_mean,
        adv_var, ratio_out, newvalue_out, act_sizes, clip_coef, vf_clip_coef, vf_coef, ent_coef);
}

// Reference implementation for mingru_gate (inference path)
// Takes combined (B, 1, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Returns {out, next_state} where:
//   out = sigmoid(proj) * mingru_out
//   next_state = mingru_out (for recurrence)
std::vector<torch::Tensor> mingru_gate_cpp(torch::Tensor state, torch::Tensor combined) {
    auto chunks = combined.chunk(3, 2);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    auto h = torch::where(hidden >= 0, hidden + 0.5, hidden.sigmoid());
    auto g = gate.sigmoid();
    auto mingru_out = torch::lerp(state, h, g);
    auto out = torch::sigmoid(proj) * mingru_out;
    return {out, mingru_out};
}

torch::autograd::tensor_list log_coeffs_and_values_cpp(torch::Tensor gate, torch::Tensor hidden) {
    auto log_coeffs = -torch::nn::functional::softplus(gate);
    auto log_z = -torch::nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (torch::nn::functional::relu(hidden) + 0.5).log(),
        -torch::nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;
    return {log_coeffs, log_values};
}

torch::Tensor logcumsumexp_cpp(torch::Tensor x) {
    return x.exp().cumsum(1).log();
}

// Reference implementation for fused_scan (training path)
// Takes combined (B, T, 3*H) = [hidden, gate, proj] and state (B, 1, H)
// Returns {out, next_state} where:
//   out (B, T, H) = sigmoid(proj) * scan_result
//   next_state (B, 1, H) = raw scan_result at T (for recurrence)
std::vector<torch::Tensor> fused_scan_cpp(torch::Tensor combined, torch::Tensor state) {
    auto seq_len = combined.size(1);

    // Split combined into hidden, gate, proj
    auto chunks = combined.chunk(3, 2);
    auto hidden = chunks[0];
    auto gate = chunks[1];
    auto proj = chunks[2];

    // Compute log_coeffs and log_values
    auto log_coeffs = -torch::nn::functional::softplus(gate);
    auto log_z = -torch::nn::functional::softplus(-gate);
    auto log_tilde_h = torch::where(hidden >= 0,
        (torch::nn::functional::relu(hidden) + 0.5).log(),
        -torch::nn::functional::softplus(-hidden));
    auto log_values = log_z + log_tilde_h;

    // Cat state and pad for scan
    log_values = torch::cat({state.log(), log_values}, 1);
    log_coeffs = torch::pad(log_coeffs, {0, 0, 1, 0});

    // Heinsen associative scan
    auto a_star = log_coeffs.cumsum(1);
    auto log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1);
    auto log_h = a_star + log_h0_plus_b_star;
    auto scan_result = log_h.exp();

    // Extract output and next_state
    scan_result = scan_result.narrow(1, scan_result.size(1) - seq_len, seq_len);
    auto next_state = scan_result.narrow(1, scan_result.size(1) - 1, 1);

    // Apply sigmoid(proj) * scan_result for output
    auto out = torch::sigmoid(proj) * scan_result;

    return {out, next_state};
}

torch::Tensor fused_ppo_loss_cpp(
    torch::Tensor logits,
    torch::Tensor newvalue,
    torch::Tensor actions,
    torch::Tensor old_logprobs,
    torch::Tensor advantages,
    torch::Tensor prio,
    torch::Tensor values,
    torch::Tensor returns,
    torch::Tensor adv_mean,
    torch::Tensor adv_std,
    float clip_coef,
    float vf_clip_coef,
    float vf_coef,
    float ent_coef
) {
    auto segments = logits.size(0);
    auto horizon = logits.size(1);

    auto flat_logits = logits.reshape({-1, logits.size(-1)});
    auto flat_actions = actions.reshape({-1});
    auto logprobs_new = torch::log_softmax(flat_logits, 1);

    auto probs_new = logprobs_new.exp();
    auto entropy = -(probs_new * logprobs_new).sum(1).mean();

    auto newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);
    auto newlogprob = newlogprob_flat.reshape({segments, horizon});
    auto logratio = newlogprob - old_logprobs;
    auto ratio_new = logratio.exp();

    auto adv_normalized = prio.unsqueeze(1) * (advantages - adv_mean) / (adv_std + 1e-8);
    auto pg_loss1 = -adv_normalized * ratio_new;
    auto pg_loss2 = -adv_normalized * torch::clamp(ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
    auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

    auto nv = newvalue.view(returns.sizes());
    auto v_clipped = values + torch::clamp(nv - values, -vf_clip_coef, vf_clip_coef);
    auto v_loss_unclipped = (nv - returns).pow(2);
    auto v_loss_clipped = (v_clipped - returns).pow(2);
    auto v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

    return pg_loss + vf_coef * v_loss - ent_coef * entropy;
}

// Fused sample_logits: handles both discrete and continuous action sampling
// For discrete: nan_to_num + log_softmax + multinomial + gather + value copy
// For continuous: sample from Normal(mean, exp(logstd)) + compute log_prob
// Writes directly to pre-allocated output tensors to avoid copy overhead
//
// logits: (B, A) - For discrete: raw logits. For continuous: mean values.
// logstd: Tensor or empty - For continuous: log standard deviation. Empty for discrete.
// value: (B, 1) or (B,) - value from fused output (may be non-contiguous)
// actions_out: (B, num_atns) float64 - output actions
// logprobs_out: (B,) same dtype as logits - output log probabilities (sum over action dims)
// value_out: (B,) same dtype as logits - output value (flattened copy)
// act_sizes: (num_atns,) int32 - size of each action head (all 1s for continuous)
// seed: RNG seed
// offset: RNG offset tensor (int64, read at kernel execution time for CUDA graph support)
//
// NOTE: Unlike other kernels, this supports non-contiguous logits/value input via stride.
// NOTE: offset is a tensor (not scalar) so that CUDA graphs read the current value at
// replay time. Increment offset with a CUDA tensor op after calling this function.
void sample_logits(
    torch::Tensor logits,
    torch::Tensor logstd,  // Empty tensor for discrete, defined for continuous
    torch::Tensor value,
    torch::Tensor actions_out,
    torch::Tensor logprobs_out,
    torch::Tensor value_out,
    torch::Tensor act_sizes,
    uint64_t seed,
    torch::Tensor offset
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D (B, num_atns or sum(act_sizes))");
    TORCH_CHECK(logits.stride(1) == 1, "logits must be contiguous in last dim");
    TORCH_CHECK(actions_out.is_contiguous(), "actions_out must be contiguous");
    TORCH_CHECK(logprobs_out.is_contiguous(), "logprobs_out must be contiguous");
    TORCH_CHECK(value_out.is_contiguous(), "value_out must be contiguous");
    TORCH_CHECK(actions_out.dtype() == torch::kFloat64, "actions_out must be float64");
    TORCH_CHECK(offset.dtype() == torch::kInt64, "offset must be int64");
    TORCH_CHECK(offset.is_cuda(), "offset must be on CUDA");
    TORCH_CHECK(act_sizes.dtype() == torch::kInt32, "act_sizes must be int32");
    TORCH_CHECK(act_sizes.is_cuda(), "act_sizes must be on CUDA");

    bool is_continuous = logstd.defined() && logstd.numel() > 0;

    auto B = logits.size(0);
    auto num_atns = act_sizes.size(0);
    auto logits_stride = logits.stride(0);  // row stride (may be > sum(act_sizes) for fused output)
    // logstd may have different stride (e.g., 0 for broadcast from [1, num_atns] expanded to [B, num_atns])
    auto logstd_stride = is_continuous ? logstd.stride(0) : 0;
    // value may be (B, 1) or (B,) - stride(0) works for both (gives stride between elements)
    auto value_stride = value.stride(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_sample_logits(
        actions_out.data_ptr<double>(),
        (precision_t*)logprobs_out.data_ptr(),
        (precision_t*)value_out.data_ptr(),
        (const precision_t*)logits.data_ptr(),
        is_continuous ? (const precision_t*)logstd.data_ptr() : nullptr,
        (const precision_t*)value.data_ptr(),
        act_sizes.data_ptr<int>(),
        seed,
        offset.data_ptr<int64_t>(),
        static_cast<int>(num_atns),
        static_cast<int>(B),
        static_cast<int>(logits_stride),
        static_cast<int>(logstd_stride),
        static_cast<int>(value_stride),
        is_continuous,
        stream
    );
}

// Reference implementation for sample_logits (for correctness testing)
std::vector<torch::Tensor> sample_logits_cpp(
    torch::Tensor logits
) {
    // nan_to_num
    auto clean_logits = torch::nan_to_num(logits);

    // log_softmax
    auto log_probs = torch::log_softmax(clean_logits, 1);

    // multinomial sampling
    auto probs = log_probs.exp();
    auto actions = torch::multinomial(probs, 1, /*replacement=*/false).squeeze(1);

    // gather logprobs
    auto sampled_logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1);

    return {actions, sampled_logprobs};
}

// Reference implementation for testing
torch::Tensor fc_relu_fc_max_cpp(
    torch::Tensor x,      // (B, N, D_in)
    torch::Tensor W1,     // (D_mid, D_in)
    torch::Tensor b1,     // (D_mid)
    torch::Tensor W2,     // (D_out, D_mid)
    torch::Tensor b2      // (D_out)
) {
    // FC1: x @ W1.T + b1 -> (B, N, D_mid)
    auto fc1 = torch::addmm(b1, x.flatten(0, 1), W1.t()).view({x.size(0), x.size(1), -1});
    // ReLU
    auto relu_out = torch::relu(fc1);
    // FC2: relu_out @ W2.T + b2 -> (B, N, D_out)
    auto fc2 = torch::addmm(b2, relu_out.flatten(0, 1), W2.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc2.max(1));
}

// =============================================================================
// FCMax: Simple FC -> Max (no intermediate ReLU layer)
// =============================================================================

class FCMaxFunction : public torch::autograd::Function<FCMaxFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,      // (B, N, D_in)
        torch::Tensor W,      // (D_out, D_in)
        torch::Tensor b       // (D_out)
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
        TORCH_CHECK(W.is_cuda(), "W must be on CUDA");
        TORCH_CHECK(b.is_cuda(), "b must be on CUDA");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
        TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

        int B = x.size(0);
        int N = x.size(1);
        int D_in = x.size(2);
        int D_out = W.size(0);

        auto out = torch::empty({B, D_out}, x.options());
        auto argmax = torch::empty({B, D_out}, torch::dtype(torch::kInt32).device(x.device()));

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        auto W_f32 = W.dtype() == torch::kFloat32 ? W : W.to(torch::kFloat32);
        auto b_f32 = b.dtype() == torch::kFloat32 ? b : b.to(torch::kFloat32);
        launch_fc_max_forward(
            (precision_t*)out.data_ptr(),
            argmax.data_ptr<int>(),
            (const precision_t*)x.data_ptr(),
            W_f32.data_ptr<float>(),
            b_f32.data_ptr<float>(),
            B, N, D_in, D_out, stream);

        ctx->save_for_backward({x, W, argmax});
        ctx->saved_data["B"] = B;
        ctx->saved_data["N"] = N;
        ctx->saved_data["D_in"] = D_in;
        ctx->saved_data["D_out"] = D_out;

        return {out, argmax};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto W = saved[1];
        auto argmax = saved[2];
        auto grad_out = grad_outputs[0].contiguous();

        auto dtype = x.dtype();
        int B = ctx->saved_data["B"].toInt();
        int N = ctx->saved_data["N"].toInt();
        int D_in = ctx->saved_data["D_in"].toInt();
        int D_out = ctx->saved_data["D_out"].toInt();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // Accumulate in fp32 (atomicAdd requires fp32), W is always fp32
        auto opts_f32 = x.options().dtype(torch::kFloat32);
        auto grad_x_f32 = torch::zeros({B, N, D_in}, opts_f32);
        auto grad_W_f32 = torch::zeros({D_out, D_in}, opts_f32);
        auto grad_b_f32 = torch::zeros({D_out}, opts_f32);
        auto W_f32 = W.dtype() == torch::kFloat32 ? W : W.to(torch::kFloat32);

        launch_fc_max_backward(
            grad_x_f32.data_ptr<float>(),
            grad_W_f32.data_ptr<float>(),
            grad_b_f32.data_ptr<float>(),
            (const precision_t*)grad_out.data_ptr(),
            (const precision_t*)x.data_ptr(),
            W_f32.data_ptr<float>(),
            argmax.data_ptr<int>(),
            B, N, D_in, D_out, stream);

        auto grad_x = (dtype == torch::kBFloat16) ? grad_x_f32.to(torch::kBFloat16) : grad_x_f32;
        return {grad_x, grad_W_f32, grad_b_f32};
    }
};

// Named entrypoint: fc_max(x, W, b) -> out
torch::Tensor fc_max(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    return FCMaxFunction::apply(x, W, b)[0];
}

// Reference implementation for testing
torch::Tensor fc_max_cpp(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    // FC: x @ W.T + b -> (B, N, D_out)
    auto fc = torch::addmm(b, x.flatten(0, 1), W.t()).view({x.size(0), x.size(1), -1});
    // Max over N dimension
    return std::get<0>(fc.max(1));
}

#endif // PUFFERLIB_MODULES_CPP
