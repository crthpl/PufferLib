#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>

torch::Tensor mingru_gate(
    torch::Tensor state,
    torch::Tensor gate,
    torch::Tensor hidden
) {
    // Validate
    TORCH_CHECK(state.is_cuda(), "state must be on CUDA");
    TORCH_CHECK(gate.is_cuda(), "gate must be on CUDA");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be on CUDA");
    TORCH_CHECK(state.dtype() == gate.dtype() && gate.dtype() == hidden.dtype(),
                "All tensors must have the same dtype");
    TORCH_CHECK(state.sizes() == gate.sizes() && gate.sizes() == hidden.sizes(),
                "All tensors must have the same shape");
    TORCH_CHECK(state.is_contiguous() && gate.is_contiguous() && hidden.is_contiguous(),
                "All tensors must be contiguous");

    auto dtype = state.dtype();
    auto device = state.device();
    auto sizes = state.sizes();
    const int N = state.numel();

    auto out = torch::empty(sizes, state.options());

    if (dtype == torch::kFloat32) {
        launch_mingru_gate_inference<float>(
            out.data_ptr<float>(),
            gate.data_ptr<float>(),
            hidden.data_ptr<float>(),
            state.data_ptr<float>(),
            N
        );
    } else if (dtype == torch::kBFloat16) {
        launch_mingru_gate_inference<at::BFloat16>(
            out.data_ptr<at::BFloat16>(),
            gate.data_ptr<at::BFloat16>(),
            hidden.data_ptr<at::BFloat16>(),
            state.data_ptr<at::BFloat16>(),
            N
        );
    } else {
        TORCH_CHECK(false,
            "Unsupported dtype. Supported dtypes are float32 and bfloat16");
    }
    return out;
}

class LogCoeffsAndValuesFunction : public torch::autograd::Function<LogCoeffsAndValuesFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor gate,
        torch::Tensor hidden
    ) {

        auto dtype = gate.dtype();
        auto log_coeffs = torch::empty_like(gate, gate.options().dtype(dtype));
        auto log_values = torch::empty_like(gate, gate.options().dtype(dtype));

        const int N = gate.numel();

        if (dtype == torch::kFloat32) {
            launch_log_coeffs_and_values<float>(
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                gate.data_ptr<float>(),
                hidden.data_ptr<float>(),
                N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_log_coeffs_and_values<at::BFloat16>(
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                gate.data_ptr<at::BFloat16>(),
                hidden.data_ptr<at::BFloat16>(),
                N
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype. Supported dtypes are float32 and bfloat16");
        }

        ctx->save_for_backward({gate, hidden});
        return {log_coeffs, log_values};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto gate = saved[0];
        auto hidden = saved[1];
        auto grad_log_values = grad_outputs[1].contiguous();
        auto grad_log_coeffs = grad_outputs[0].contiguous();

        auto grad_gate = torch::empty_like(gate);
        auto grad_hidden = torch::empty_like(hidden);

        const int N = gate.numel();
        auto dtype = gate.dtype();

        if (dtype == torch::kFloat32) {
            launch_log_coeffs_and_values_backward<float>(
                grad_gate.data_ptr<float>(),
                grad_hidden.data_ptr<float>(),
                grad_log_coeffs.data_ptr<float>(),
                grad_log_values.data_ptr<float>(),
                gate.data_ptr<float>(),
                hidden.data_ptr<float>(),
                N
            );
        } else if (dtype == torch::kBFloat16) {
            launch_log_coeffs_and_values_backward<at::BFloat16>(
                grad_gate.data_ptr<at::BFloat16>(),
                grad_hidden.data_ptr<at::BFloat16>(),
                grad_log_coeffs.data_ptr<at::BFloat16>(),
                grad_log_values.data_ptr<at::BFloat16>(),
                gate.data_ptr<at::BFloat16>(),
                hidden.data_ptr<at::BFloat16>(),
                N
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype in backward");
        }

        return {grad_gate, grad_hidden};
    }
};

torch::autograd::tensor_list log_coeffs_and_values(
    torch::Tensor gate,
    torch::Tensor hidden
) {
    return LogCoeffsAndValuesFunction::apply(gate, hidden);
}


class FusedScanFunction : public torch::autograd::Function<FusedScanFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor log_coeffs,
        torch::Tensor log_values
    ) {
        TORCH_CHECK(log_coeffs.is_cuda(), "log_coeffs must be on CUDA");
        TORCH_CHECK(log_values.is_cuda(), "log_values must be on CUDA");
        TORCH_CHECK(log_coeffs.dtype() == log_values.dtype(), "dtypes must match");
        TORCH_CHECK(log_coeffs.dim() == 3 && log_values.dim() == 3, "must be (B, T, H)");
        TORCH_CHECK(log_values.size(1) == log_coeffs.size(1), "T must match");

        auto dtype = log_coeffs.dtype();        // e.g., kBFloat16 or kFloat32
        auto device = log_coeffs.device();      // e.g., cuda:0
        auto B = log_coeffs.size(0);
        auto T = log_coeffs.size(1);            // T = T_total (e.g. T+1)
        auto H = log_coeffs.size(2);

        // Output: same dtype and device as input
        auto out = torch::empty({B, T, H}, log_coeffs.options());

        // Intermediates: must be float32, but on same device!
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto a_star = torch::empty({B, T, H}, options_float);
        auto s_vals = torch::empty({B, T, H}, options_float);

        // Launch kernel
        if (dtype == torch::kFloat32) {
            launch_fused_scan_forward<float>(
                out.data_ptr<float>(),
                a_star.data_ptr<float>(),  // float*
                s_vals.data_ptr<float>(),  // float*
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );

        } else if (dtype == torch::kBFloat16) {
            launch_fused_scan_forward<at::BFloat16>(
                out.data_ptr<at::BFloat16>(),
                a_star.data_ptr<float>(),  // float* ← critical!
                s_vals.data_ptr<float>(),  // float* ← critical!
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                static_cast<int>(T),
                static_cast<int>(H),
                static_cast<int>(B)
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype. Only float32 and bfloat16 supported.");
        }

        // Save for backward (a_star and s_vals are float32, but that's fine)
        ctx->save_for_backward({log_coeffs, log_values, out, a_star, s_vals});

        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto log_coeffs = saved[0].contiguous();
        auto log_values = saved[1].contiguous();
        auto out = saved[2].contiguous();
        auto a_star_buf = saved[3].contiguous();  // float tensor
        auto s_vals = saved[4].contiguous();      // float tensor

        auto grad_out = grad_outputs[0].contiguous();
        auto dtype = log_coeffs.dtype();

        auto B = log_coeffs.size(0);
        auto T = log_coeffs.size(1);
        auto H = log_coeffs.size(2);

        auto grad_log_coeffs = torch::empty_like(log_coeffs);
        auto grad_log_values = torch::empty_like(log_values);

        if (dtype == torch::kFloat32) {
            launch_fused_scan_backward<float>(
                grad_log_coeffs.data_ptr<float>(),
                grad_log_values.data_ptr<float>(),
                grad_out.data_ptr<float>(),
                log_coeffs.data_ptr<float>(),
                log_values.data_ptr<float>(),
                out.data_ptr<float>(),
                a_star_buf.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                T, H, B
            );
        } else if (dtype == torch::kBFloat16) {
            launch_fused_scan_backward<at::BFloat16>(
                grad_log_coeffs.data_ptr<at::BFloat16>(),
                grad_log_values.data_ptr<at::BFloat16>(),
                grad_out.data_ptr<at::BFloat16>(),
                log_coeffs.data_ptr<at::BFloat16>(),
                log_values.data_ptr<at::BFloat16>(),
                out.data_ptr<at::BFloat16>(),
                a_star_buf.data_ptr<float>(),
                s_vals.data_ptr<float>(),
                T, H, B
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype");
        }

        return {grad_log_coeffs, grad_log_values};
    }
};

// Named entrypoint: fused_scan(log_coeffs, log_values) -> out
torch::autograd::tensor_list fused_scan(
    torch::Tensor log_coeffs,
    torch::Tensor log_values
) {
    return FusedScanFunction::apply(log_coeffs, log_values);
}

class LogCumsumExpFunction : public torch::autograd::Function<LogCumsumExpFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x  // (B, T, H)
    ) {
        TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
        auto dtype = x.dtype();
        auto device = x.device();
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto out = torch::empty({B, T, H}, x.options());
        auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto s_buf = torch::empty({B, T, H}, options_float);

        if (dtype == torch::kFloat32) {
            launch_logcumsumexp_forward<float>(
                out.data_ptr<float>(),
                s_buf.data_ptr<float>(),
                x.data_ptr<float>(),
                (int)T, (int)H, (int)B
            );
        } else if (dtype == torch::kBFloat16) {
            launch_logcumsumexp_forward<at::BFloat16>(
                out.data_ptr<at::BFloat16>(),
                s_buf.data_ptr<float>(),
                x.data_ptr<at::BFloat16>(),
                (int)T, (int)H, (int)B
            );
        } else {
            TORCH_CHECK(false, "Only float32 and bfloat16 supported");
        }

        ctx->save_for_backward({x, out, s_buf});
        return {out};
    }
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto x = saved[0].contiguous();
        auto s_buf = saved[2].contiguous();  // s_buf is saved, out is not needed

        auto grad_out = grad_outputs[0].contiguous();
        auto dtype = x.dtype();
        auto B = x.size(0), T = x.size(1), H = x.size(2);

        auto grad_x = torch::empty_like(x);

        if (dtype == torch::kFloat32) {
            launch_logcumsumexp_backward<float>(
                grad_x.data_ptr<float>(),
                grad_out.data_ptr<float>(),
                x.data_ptr<float>(),
                s_buf.data_ptr<float>(),
                (int)T, (int)H, (int)B
            );
        } else if (dtype == torch::kBFloat16) {
            launch_logcumsumexp_backward<at::BFloat16>(
                grad_x.data_ptr<at::BFloat16>(),
                grad_out.data_ptr<at::BFloat16>(),
                x.data_ptr<at::BFloat16>(),
                s_buf.data_ptr<float>(),
                (int)T, (int)H, (int)B
            );
        } else {
            TORCH_CHECK(false, "Unsupported dtype in backward");
        }

        return {grad_x};
    }
};

// Entry point
torch::Tensor logcumsumexp_cuda(torch::Tensor x) {
    return LogCumsumExpFunction::apply(x)[0];
}
