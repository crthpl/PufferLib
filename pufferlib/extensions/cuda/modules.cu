#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include "ops.cuh"
#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>


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
