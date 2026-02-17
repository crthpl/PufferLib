#ifndef PUFFERLIB_MODULES_H
#define PUFFERLIB_MODULES_H

#include <torch/extension.h>
#include <torch/torch.h>

// Minimal tensor: raw pointer + shape, no torch dependency in the struct itself.
// Memory is owned by an Allocator buffer — PufTensor is just a view.
struct PufTensor {
    void* data;
    int64_t shape[4];   // up to 4D, unused dims = 1
    int ndim;
    int64_t numel;
    int dtype_size;      // bytes per element (2 for bf16/f16, 4 for f32, 8 for f64)

    PufTensor() : data(nullptr), ndim(0), numel(0), dtype_size(0) {
        for (int i = 0; i < 4; i++) shape[i] = 1;
    }

    int64_t size(int dim) const { return shape[dim]; }
    int64_t nbytes() const { return numel * dtype_size; }

    // Cast to torch::Tensor for interop (no copy — shares memory)
    torch::Tensor to_torch(torch::ScalarType dtype) const {
        return torch::from_blob(data, {shape, shape + ndim},
            torch::dtype(dtype).device(torch::kCUDA));
    }

    // Create a PufTensor view of a torch::Tensor (no copy)
    static PufTensor from_torch(const torch::Tensor& t) {
        PufTensor p;
        p.data = t.data_ptr();
        p.ndim = t.dim();
        p.numel = t.numel();
        p.dtype_size = t.element_size();
        for (int i = 0; i < 4; i++)
            p.shape[i] = (i < t.dim()) ? t.size(i) : 1;
        return p;
    }
};

// Helper: bytes per element for a torch ScalarType
inline int dtype_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return 4;
        case torch::kFloat64: return 8;
        case torch::kBFloat16: return 2;
        case torch::kFloat16: return 2;
        case torch::kInt32: return 4;
        case torch::kInt64: return 8;
        default: TORCH_CHECK(false, "Unsupported dtype for PufTensor"); return 0;
    }
}

// cuBLAS matmuls: all row-major PufTensors, bf16 with f32 compute
void puf_mm(PufTensor& a, PufTensor& b, PufTensor& out);     // out(M,N) = a(M,K) @ b(N,K)^T
void puf_mm_tn(PufTensor& a, PufTensor& b, PufTensor& out);  // out(M,N) = a(K,M)^T @ b(K,N)
void puf_mm_nn(PufTensor& a, PufTensor& b, PufTensor& out);  // out(M,N) = a(M,K) @ b(K,N)

// out = alpha * a @ b^T + beta * out (bf16, f32 compute)
void puf_addmm(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta);
// out = alpha * a @ b + beta * out (bf16, f32 compute, no transpose)
void puf_addmm_nn(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta);

// Copy from torch::Tensor into PufTensor with dtype conversion (e.g. f32→bf16)
void puf_copy_from_torch(PufTensor& dst, torch::Tensor src);

// Cast bf16→f32
void puf_cast_bf16_to_f32(PufTensor& dst, const PufTensor& src);

// Cast f32→bf16
void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src);

// Cast f32(R,C)→bf16(C,R) with transpose
void puf_cast_f32_to_bf16_transpose(PufTensor& dst, const PufTensor& src);

// Transpose f32(R,C) → f32(C,R)
void puf_transpose_f32(PufTensor& dst, const PufTensor& src);

// Frobenius norm → device scalar. Writes sqrt(sum(x^2)) to *out_ptr. src must be bf16.
void puf_norm(const PufTensor& src, float* out_ptr);

// dst *= 1.0 / max(*norm_ptr, eps) — normalize by device-resident norm
void puf_normalize(PufTensor& dst, const float* norm_ptr, float eps);

// PufTensor→PufTensor memcpy (same dtype, same size)
void puf_copy(PufTensor& dst, const PufTensor& src);

// Zero a PufTensor
void puf_zero(PufTensor& dst);

// dst *= alpha (f32)
void puf_scale(PufTensor& dst, float alpha);

// dst *= *alpha_ptr (f32, reads scalar from device memory)
void puf_scale_dev(PufTensor& dst, const float* alpha_ptr);

// dst += alpha * src (f32)
void puf_axpy(PufTensor& dst, const PufTensor& src, float alpha);

// dst += (*alpha_ptr) * src (f32, reads scalar from device memory)
void puf_axpy_dev(PufTensor& dst, const PufTensor& src, const float* alpha_ptr);

// Compute derived lr scalars on device: neg_lr = -lr, wd_scale = 1 - lr * wd
void compute_lr_scalars(const float* lr_ptr, float weight_decay, float* neg_lr_ptr, float* wd_scale_ptr);

// dst += src with mixed precision (fp32 += bf16)
void puf_add(PufTensor& dst, const PufTensor& src);

// dst(1, C) += src(R, C).sum(dim=0) — column-wise sum reduction, both f32
void puf_sum_rows_add(PufTensor& dst, PufTensor& src);

// Assemble fused decoder grad: dst(B_TT, output+1) = [grad_logits(B_TT, od) | grad_value(B_TT, 1)]
// Handles f32→bf16 cast from fp32 PufTensor grad outputs into bf16 PufTensor
void puf_assemble_decoder_grad(PufTensor& dst, PufTensor& grad_logits, PufTensor& grad_value);

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

// PufTensor overload — all PufTensors
void mingru_gate(PufTensor& state, PufTensor& combined,
    PufTensor& out, PufTensor& next_state);

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
struct PrefixScan {
    // Forward inputs saved for backward (raw pointers — data lives in pre-allocated buffers)
    void* combined_ptr;           // (B, T, 3*H) precision_t
    void* state_ptr;              // (B, 1, H) precision_t
    int B, T, H;                  // dimensions saved from forward
    // Pre-allocated activation buffers
    PufTensor a_star;             // (B, T+1, H) float32
    PufTensor s_vals;             // (B, T+1, H) float32
    PufTensor log_values_buf;     // (B, T+1, H) float32
    PufTensor out;                // (B, T, H) precision_t
    PufTensor next_state;         // (B, 1, H) precision_t — scratch (output discarded)
    // Backward buffers
    PufTensor grad_combined;      // (B, T, 3*H) precision_t
    PufTensor grad_state;         // (B, 1, H) precision_t

    PrefixScan() : combined_ptr(nullptr), state_ptr(nullptr), B(0), T(0), H(0) {}
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

// Prefix scan forward/backward with pre-allocated buffers
void prefix_scan_forward(PufTensor& combined, PufTensor& state,
    PrefixScan& bufs);
void prefix_scan_backward(
    PufTensor& grad_out, PufTensor& grad_next_state,
    PrefixScan& bufs);

// Fused PPO loss forward + backward (no autograd) — uses pre-allocated buffers
void ppo_loss_fwd_bwd(
    torch::Tensor logits, torch::Tensor logstd, torch::Tensor values_pred,
    torch::Tensor actions, torch::Tensor old_logprobs, torch::Tensor advantages,
    torch::Tensor prio, torch::Tensor values, torch::Tensor returns,
    torch::Tensor ratio_out, torch::Tensor newvalue_out,
    torch::Tensor act_sizes, torch::Tensor losses_acc,
    float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef,
    PPOBuffers& bufs);

// Contiguous memory allocator for params/grads/activations
struct Allocator {
    struct PufRegistration {
        PufTensor* ptr;
        int64_t size;           // total elements
        std::vector<int64_t> shape;
        int elem_size;          // bytes per element
    };
    std::vector<PufRegistration> params, grads, puf_activations;
    torch::Tensor param_buffer, grad_buffer, puf_buffer;

    void register_param(PufTensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        params.push_back({ptr, size, shape, 0});  // elem_size set in create()
    }

    void register_grad(PufTensor* ptr, std::vector<int64_t> shape) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        grads.push_back({ptr, size, shape, 0});
    }

    // Register a PufTensor activation — allocated from a contiguous byte buffer
    void register_puf(PufTensor* ptr, std::vector<int64_t> shape, int elem_size) {
        int64_t size = 1;
        for (auto s : shape) size *= s;
        puf_activations.push_back({ptr, size, shape, elem_size});
    }

    // Helper: assign PufTensor views into a contiguous torch buffer
    void assign_puf_views(std::vector<PufRegistration>& regs, torch::Tensor& buffer) {
        char* base = (char*)buffer.data_ptr();
        int esz = buffer.element_size();
        int64_t offset = 0;
        for (auto& r : regs) {
            r.ptr->data = base + offset * esz;
            r.ptr->ndim = r.shape.size();
            r.ptr->numel = r.size;
            r.ptr->dtype_size = esz;
            for (int i = 0; i < 4; i++)
                r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 1;
            offset += r.size;
        }
    }

    void create(torch::Device device, torch::ScalarType dtype) {
        // Allocate contiguous param buffer
        int64_t total_params = 0;
        for (auto& r : params) total_params += r.size;
        if (total_params > 0) {
            param_buffer = torch::zeros({total_params},
                torch::dtype(dtype).device(device));
            assign_puf_views(params, param_buffer);
        }

        // Allocate contiguous grad buffer
        int64_t total_grads = 0;
        for (auto& r : grads) total_grads += r.size;
        if (total_grads > 0) {
            grad_buffer = torch::zeros({total_grads},
                torch::dtype(dtype).device(device));
            assign_puf_views(grads, grad_buffer);
        }

        // Allocate PufTensor activations from a contiguous byte buffer
        int64_t total_puf_bytes = 0;
        for (auto& r : puf_activations) total_puf_bytes += r.size * r.elem_size;
        if (total_puf_bytes > 0) {
            puf_buffer = torch::zeros({total_puf_bytes},
                torch::dtype(torch::kUInt8).device(device));
            char* base = (char*)puf_buffer.data_ptr();
            int64_t offset = 0;
            for (auto& r : puf_activations) {
                r.ptr->data = base + offset;
                r.ptr->ndim = r.shape.size();
                r.ptr->numel = r.size;
                r.ptr->dtype_size = r.elem_size;
                for (int i = 0; i < 4; i++)
                    r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 1;
                offset += r.size * r.elem_size;
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
