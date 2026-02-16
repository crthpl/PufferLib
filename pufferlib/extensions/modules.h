#ifndef PUFFERLIB_MODULES_H
#define PUFFERLIB_MODULES_H

#include <torch/extension.h>
#include <torch/torch.h>

// Loss component indices for the shared accumulator tensor
enum LossIdx {
    LOSS_PG = 0,
    LOSS_VF = 1,
    LOSS_ENT = 2,
    LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4,
    LOSS_APPROX_KL = 5,
    LOSS_CLIPFRAC = 6,
    LOSS_N = 7,           // number of accumulations (also == number of loss components)
    NUM_LOSSES = 8,
};

// Autograd modules (implementations in cuda/modules.cu)
using tensor_list = torch::autograd::tensor_list;
using AutogradCtx = torch::autograd::AutogradContext;

class PrefixScan : public torch::autograd::Function<PrefixScan> {
public:
    static tensor_list forward(AutogradCtx* ctx,
        torch::Tensor combined, torch::Tensor state);
    static tensor_list backward(AutogradCtx* ctx, tensor_list grad_outputs);
};

class LogCumsumExp : public torch::autograd::Function<LogCumsumExp> {
public:
    static tensor_list forward(AutogradCtx* ctx, torch::Tensor x);
    static tensor_list backward(AutogradCtx* ctx, tensor_list grad_outputs);
};

class PPOLoss : public torch::autograd::Function<PPOLoss> {
public:
    static tensor_list forward(AutogradCtx* ctx,
        torch::Tensor logits, torch::Tensor logstd,
        torch::Tensor values_pred, torch::Tensor actions,
        torch::Tensor old_logprobs, torch::Tensor advantages,
        torch::Tensor prio, torch::Tensor values, torch::Tensor returns,
        torch::Tensor ratio_out, torch::Tensor newvalue_out,
        torch::Tensor act_sizes, torch::Tensor losses_acc,
        double clip_coef, double vf_clip_coef, double vf_coef, double ent_coef);
    static tensor_list backward(AutogradCtx* ctx, tensor_list grad_outputs);
};

class FCMax : public torch::autograd::Function<FCMax> {
public:
    static tensor_list forward(AutogradCtx* ctx,
        torch::Tensor x, torch::Tensor W, torch::Tensor b);
    static tensor_list backward(AutogradCtx* ctx, tensor_list grad_outputs);
};

// Fused mingru gate inference — writes into pre-allocated out and next_state
void mingru_gate(torch::Tensor state, torch::Tensor combined,
    torch::Tensor out, torch::Tensor next_state);

// Fused sample_logits: handles both discrete and continuous action sampling
void sample_logits(
    torch::Tensor logits, torch::Tensor logstd, torch::Tensor value,
    torch::Tensor actions_out, torch::Tensor logprobs_out, torch::Tensor value_out,
    torch::Tensor act_sizes, uint64_t seed, torch::Tensor offset);

// Priority replay: fused priority sampling for minibatch selection
std::tuple<torch::Tensor, torch::Tensor> prio_replay_cuda(
    torch::Tensor advantages, float prio_alpha,
    int minibatch_segments, int total_agents, float anneal_beta);

// Select + Copy: fused index_select and copy for minibatch preparation
void train_select_and_copy_cuda(
    torch::Tensor observations, torch::Tensor actions,
    torch::Tensor logprobs, torch::Tensor values, torch::Tensor advantages,
    torch::Tensor idx, torch::Tensor mb_prio,
    torch::Tensor dst_obs, torch::Tensor dst_state,
    torch::Tensor dst_actions, torch::Tensor dst_logprobs,
    torch::Tensor dst_advantages, torch::Tensor dst_prio,
    torch::Tensor dst_values, torch::Tensor dst_returns);

// Direct (non-autograd) prefix scan buffers — pre-allocated via Allocator
struct PrefixScanBuffers {
    // Forward buffers (saved for backward)
    torch::Tensor combined;       // (B, T, 3*H) precision_t — reference to matmul result
    torch::Tensor state;          // (B, 1, H) precision_t — reference to state input
    torch::Tensor a_star;         // (B, T+1, H) float32 — pre-allocated
    torch::Tensor s_vals;         // (B, T+1, H) float32 — pre-allocated
    torch::Tensor log_values_buf; // (B, T+1, H) float32 — pre-allocated
    torch::Tensor out;            // (B, T, H) precision_t — pre-allocated
    torch::Tensor next_state;     // (B, 1, H) precision_t — pre-allocated scratch (output discarded)
    // Backward buffers
    torch::Tensor grad_combined;  // (B, T, 3*H) precision_t — pre-allocated
    torch::Tensor grad_state;     // (B, 1, H) precision_t — pre-allocated
};

// PPO gradient outputs from fused forward+backward
struct PPOGrads {
    torch::Tensor grad_logits;    // (N, T, A_total) float32
    torch::Tensor grad_logstd;    // (N, T, A_total) float32, or empty for discrete
    torch::Tensor grad_values;    // (N, T, 1) float32
};

// Pre-allocated buffers for ppo_loss_fwd_bwd (avoids per-call allocations)
struct PPOBuffers {
    torch::Tensor loss_output;       // (1,) float32
    torch::Tensor saved_for_bwd;     // (N*T, 5) float64
    torch::Tensor grad_loss;         // (1,) float32, constant 1.0
    torch::Tensor grad_logits;       // (N, T, A_total) float32
    torch::Tensor grad_values;       // (N, T, 1) float32
    torch::Tensor grad_logstd;       // (N, T, A_total) float32, or undefined for discrete

    void create(int N, int T, int A_total, bool is_continuous, torch::Device device) {
        auto f32 = torch::dtype(torch::kFloat32).device(device);
        auto f64 = torch::dtype(torch::kFloat64).device(device);
        loss_output = torch::empty({1}, f32);
        saved_for_bwd = torch::zeros({N * T, 5}, f64);
        grad_loss = torch::ones({1}, f32);
        grad_logits = torch::empty({N, T, A_total}, f32);
        grad_values = torch::empty({N, T, 1}, f32);
        if (is_continuous) {
            grad_logstd = torch::empty({N, T, A_total}, f32);
        }
    }
};

// Direct (non-autograd) prefix scan forward/backward with pre-allocated buffers
void prefix_scan_forward_direct(torch::Tensor combined, torch::Tensor state,
    PrefixScanBuffers& bufs);
void prefix_scan_backward_direct(
    torch::Tensor grad_out, torch::Tensor grad_next_state,
    PrefixScanBuffers& bufs);

// Fused PPO loss forward + backward (no autograd) — uses pre-allocated buffers
void ppo_loss_fwd_bwd(
    torch::Tensor logits, torch::Tensor logstd, torch::Tensor values_pred,
    torch::Tensor actions, torch::Tensor old_logprobs, torch::Tensor advantages,
    torch::Tensor prio, torch::Tensor values, torch::Tensor returns,
    torch::Tensor ratio_out, torch::Tensor newvalue_out,
    torch::Tensor act_sizes, torch::Tensor losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
    PPOBuffers& bufs);

// Contiguous memory allocator for params/grads
struct Allocator {
    struct Registration {
        torch::Tensor* ptr;
        int64_t size;
        std::vector<int64_t> shape;
    };
    struct ActivationReg {
        torch::Tensor* ptr;
        std::vector<int64_t> shape;
        torch::ScalarType dtype;  // explicit dtype (may differ from allocator default)
    };
    std::vector<Registration> params, grads, zero_activations;
    std::vector<ActivationReg> activations;
    torch::Tensor param_buffer, grad_buffer, zero_buffer;

    void register_param(torch::Tensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        params.push_back({ptr, size, shape});
    }

    void register_grad(torch::Tensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        grads.push_back({ptr, size, shape});
    }

    void register_activation(torch::Tensor* ptr, std::vector<int64_t> shape,
                             torch::ScalarType dtype) {
        activations.push_back({ptr, shape, dtype});
    }

    // Activations that must be zeroed before use — contiguous buffer, single zero_() call
    void register_zero_activation(torch::Tensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        zero_activations.push_back({ptr, size, shape});
    }

    void create(torch::Device device, torch::ScalarType dtype) {
        // Allocate contiguous param buffer
        int64_t total_params = 0;
        for (auto& r : params) total_params += r.size;
        if (total_params > 0) {
            param_buffer = torch::zeros({total_params},
                torch::dtype(dtype).device(device));
            int64_t offset = 0;
            for (auto& r : params) {
                *r.ptr = param_buffer.narrow(0, offset, r.size).view(r.shape);
                offset += r.size;
            }
        }

        // Allocate contiguous grad buffer
        int64_t total_grads = 0;
        for (auto& r : grads) total_grads += r.size;
        if (total_grads > 0) {
            grad_buffer = torch::zeros({total_grads},
                torch::dtype(dtype).device(device));
            int64_t offset = 0;
            for (auto& r : grads) {
                *r.ptr = grad_buffer.narrow(0, offset, r.size).view(r.shape);
                offset += r.size;
            }
        }

        // Allocate contiguous zero-before-use activation buffer (same dtype as allocator)
        int64_t total_zero = 0;
        for (auto& r : zero_activations) total_zero += r.size;
        if (total_zero > 0) {
            zero_buffer = torch::zeros({total_zero}, torch::dtype(dtype).device(device));
            int64_t offset = 0;
            for (auto& r : zero_activations) {
                *r.ptr = zero_buffer.narrow(0, offset, r.size).view(r.shape);
                offset += r.size;
            }
        }

        // Allocate activation tensors individually (mixed dtypes)
        for (auto& r : activations) {
            *r.ptr = torch::empty(r.shape, torch::dtype(r.dtype).device(device));
        }
    }
};

// Puff Advantage CUDA dispatch
namespace pufferlib {
void puff_advantage_cuda(
    torch::Tensor values, torch::Tensor rewards,
    torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
    double gamma, double lambda, double rho_clip, double c_clip);
}

#endif // PUFFERLIB_MODULES_H
