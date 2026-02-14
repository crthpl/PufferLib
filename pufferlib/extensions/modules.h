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

// Fused mingru gate inference
std::vector<torch::Tensor> mingru_gate(torch::Tensor state, torch::Tensor combined);

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

// Direct (non-autograd) prefix scan buffers
struct PrefixScanBuffers {
    torch::Tensor combined;       // (B, T, 3*H) precision_t
    torch::Tensor state;          // (B, 1, H) precision_t
    torch::Tensor a_star;         // (B, T+1, H) float32
    torch::Tensor s_vals;         // (B, T+1, H) float32
    torch::Tensor log_values_buf; // (B, T+1, H) float32
    torch::Tensor out;            // (B, T, H) precision_t
};

// PPO gradient outputs from fused forward+backward
struct PPOGrads {
    torch::Tensor grad_logits;    // (N, T, A_total) float32
    torch::Tensor grad_logstd;    // (N, T, A_total) float32, or empty for discrete
    torch::Tensor grad_values;    // (N, T, 1) float32
};

// Direct (non-autograd) prefix scan forward/backward
PrefixScanBuffers prefix_scan_forward_direct(torch::Tensor combined, torch::Tensor state);
std::vector<torch::Tensor> prefix_scan_backward_direct(
    torch::Tensor grad_out, torch::Tensor grad_next_state,
    const PrefixScanBuffers& bufs);

// Fused PPO loss forward + backward (no autograd)
PPOGrads ppo_loss_fwd_bwd(
    torch::Tensor logits, torch::Tensor logstd, torch::Tensor values_pred,
    torch::Tensor actions, torch::Tensor old_logprobs, torch::Tensor advantages,
    torch::Tensor prio, torch::Tensor values, torch::Tensor returns,
    torch::Tensor ratio_out, torch::Tensor newvalue_out,
    torch::Tensor act_sizes, torch::Tensor losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef);

// Contiguous memory allocator for params/grads
struct Allocator {
    struct Registration {
        torch::Tensor* ptr;
        int64_t size;
        std::vector<int64_t> shape;
    };
    std::vector<Registration> params, grads;
    torch::Tensor param_buffer, grad_buffer;

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
