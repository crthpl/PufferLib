#ifndef PUFFERLIB_MODULES_H
#define PUFFERLIB_MODULES_H

#ifdef PUFFERLIB_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include <vector>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>

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

    const char* dtype_name() const {
        switch (dtype_size) {
            case 1: return "i8";
            case 2: return "bf16";
            case 4: return "f32";
            case 8: return "f64";
            default: return "?";
        }
    }

    std::string repr() const {
        std::string s = "PufTensor(";
        if (!data) return s + "empty)";
        s += dtype_name();
        s += ", [";
        for (int i = 0; i < ndim; i++) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "], ";
        s += std::to_string(numel) + " elems)";
        return s;
    }

#ifdef PUFFERLIB_TORCH
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
#endif
};

#ifdef PUFFERLIB_TORCH
// Helper: bytes per element for a torch ScalarType
inline int dtype_size(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return sizeof(float);
        case torch::kFloat64: return sizeof(double);
        case torch::kBFloat16: return 2;  // no native C++ type
        case torch::kFloat16: return 2;   // no native C++ type
        case torch::kInt32: return sizeof(int32_t);
        case torch::kInt64: return sizeof(int64_t);
        default: TORCH_CHECK(false, "Unsupported dtype for PufTensor"); return 0;
    }
}
#endif

// cuBLAS matmuls: all row-major PufTensors, bf16 with f32 compute
void puf_mm(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream);     // out(M,N) = a(M,K) @ b(N,K)^T
void puf_mm_tn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream);  // out(M,N) = a(K,M)^T @ b(K,N)
void puf_mm_nn(PufTensor& a, PufTensor& b, PufTensor& out, cudaStream_t stream);  // out(M,N) = a(M,K) @ b(K,N)

// out = alpha * a @ b^T + beta * out (bf16, f32 compute)
void puf_addmm(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta, cudaStream_t stream);
// out = alpha * a @ b + beta * out (bf16, f32 compute, no transpose)
void puf_addmm_nn(PufTensor& a, PufTensor& b, PufTensor& out, float alpha, float beta, cudaStream_t stream);

// Cast bf16->f32
void puf_cast_bf16_to_f32(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Cast f32->bf16
void puf_cast_f32_to_bf16(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Cast f32(R,C)->bf16(C,R) with transpose
void puf_cast_f32_to_bf16_transpose(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Transpose f32(R,C) -> f32(C,R)
void puf_transpose_f32(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Swap dims 0 and 1: src(A, B, ...) -> dst(B, A, ...). Any dtype.
// For 2D: standard matrix transpose. For 3D: (A,B,C)->(B,A,C).
void puf_transpose_01(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Orthogonal initialization (QR-based, matches torch.nn.init.orthogonal_)
// dst must be 2D. gain scales the result. Uses cuSOLVER + cuRAND internally.
void puf_orthogonal_init(PufTensor& dst, float gain, uint64_t seed, cudaStream_t stream);

// Frobenius norm -> device scalar. Writes sqrt(sum(x^2)) to *out_ptr. src must be bf16.
void puf_norm(const PufTensor& src, float* out_ptr, cudaStream_t stream);

// dst *= 1.0 / max(*norm_ptr, eps) — normalize by device-resident norm
void puf_normalize(PufTensor& dst, const float* norm_ptr, float eps, cudaStream_t stream);

// Clip gradient norm: dst(f32) *= min(max_norm / (norm(dst) + 1e-6), 1.0). scratch is 1 device float.
void puf_clip_grad_norm(PufTensor& grad, float max_norm, float* scratch, cudaStream_t stream);

// PufTensor->PufTensor memcpy (same dtype, same size)
void puf_copy(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Zero a PufTensor
void puf_zero(PufTensor& dst, cudaStream_t stream);

// Fill a PufTensor with a scalar value (bf16 or f32)
void puf_fill(PufTensor& dst, float val, cudaStream_t stream);

// Clamp PufTensor in-place (bf16)
void puf_clamp(PufTensor& dst, float lo, float hi, cudaStream_t stream);

// dst *= alpha (f32)
void puf_scale(PufTensor& dst, float alpha, cudaStream_t stream);

// dst *= *alpha_ptr (f32, reads scalar from device memory)
void puf_scale_dev(PufTensor& dst, const float* alpha_ptr, cudaStream_t stream);

// dst += alpha * src (f32)
void puf_axpy(PufTensor& dst, const PufTensor& src, float alpha, cudaStream_t stream);

// dst += (*alpha_ptr) * src (f32, reads scalar from device memory)
void puf_axpy_dev(PufTensor& dst, const PufTensor& src, const float* alpha_ptr, cudaStream_t stream);

// Compute derived lr scalars on device: neg_lr = -lr, wd_scale = 1 - lr * wd
void compute_lr_scalars(const float* lr_ptr, float weight_decay, float* neg_lr_ptr, float* wd_scale_ptr, cudaStream_t stream);

// dst += src with mixed precision (fp32 += bf16)
void puf_add(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// dst(1, C) += src(R, C).sum(dim=0) — column-wise sum reduction, both f32
void puf_sum_rows_add(PufTensor& dst, PufTensor& src, cudaStream_t stream);

// Assemble fused decoder grad: dst(B_TT, output+1) = [grad_logits(B_TT, od) | grad_value(B_TT, 1)]
// Handles f32->bf16 cast from fp32 PufTensor grad outputs into bf16 PufTensor
void puf_assemble_decoder_grad(PufTensor& dst, PufTensor& grad_logits, PufTensor& grad_value, cudaStream_t stream);

// Compute variance and mean of a float PufTensor into two device floats
void puf_var_mean(const PufTensor& src, float* var_out, float* mean_out, cudaStream_t stream);

// Scatter rows: dst[idx[i], :] = src[i, :] for i in [0, num_idx)
// idx is int64, src/dst have same row width and dtype
void puf_index_copy(PufTensor& dst, const PufTensor& idx, const PufTensor& src, cudaStream_t stream);

// Cast uint8 src to precision_t dst (bf16 or f32 depending on compile flag)
void puf_cast_u8_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Cast f32 src to precision_t dst (copy if f32 mode, cast if bf16 mode)
void puf_cast_f32_to_precision(PufTensor& dst, const PufTensor& src, cudaStream_t stream);

// Increment a single float at device pointer: *ptr += val
void puf_add_scalar(float* ptr, float val, cudaStream_t stream);

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

// PufTensor overload — all PufTensors
void mingru_gate(PufTensor& state, PufTensor& combined,
    PufTensor& out, PufTensor& next_state, cudaStream_t stream);

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


// Prefix scan forward/backward with pre-allocated buffers
void prefix_scan_forward(PufTensor& combined, PufTensor& state,
    PrefixScan& bufs, cudaStream_t stream);
void prefix_scan_backward(
    PufTensor& grad_out, PufTensor& grad_next_state,
    PrefixScan& bufs, cudaStream_t stream);


// Contiguous memory allocator for params/grads/activations
struct Allocator {
    struct PufRegistration {
        PufTensor* ptr;
        int64_t size;           // total elements
        std::vector<int64_t> shape;
        int elem_size;          // bytes per element
    };
    std::vector<PufRegistration> params, grads, puf_activations;
#ifdef PUFFERLIB_TORCH
    torch::Tensor param_buffer, grad_buffer, puf_buffer;  // Tensor views (non-owning)
#endif
    void* param_mem = nullptr;   // backing cudaMalloc memory
    void* grad_mem = nullptr;
    void* puf_mem = nullptr;

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

    // Helper: assign PufTensor views into contiguous memory
    void assign_puf_views(std::vector<PufRegistration>& regs, void* base, int esz) {
        int64_t offset = 0;
        for (auto& r : regs) {
            r.ptr->data = (char*)base + offset * esz;
            r.ptr->ndim = r.shape.size();
            r.ptr->numel = r.size;
            r.ptr->dtype_size = esz;
            for (int i = 0; i < 4; i++)
                r.ptr->shape[i] = (i < (int)r.shape.size()) ? r.shape[i] : 1;
            offset += r.size;
        }
    }

    void destroy() {
#ifdef PUFFERLIB_TORCH
        param_buffer = torch::Tensor();
        grad_buffer = torch::Tensor();
        puf_buffer = torch::Tensor();
#endif
        if (param_mem) { cudaFree(param_mem); param_mem = nullptr; }
        if (grad_mem) { cudaFree(grad_mem); grad_mem = nullptr; }
        if (puf_mem) { cudaFree(puf_mem); puf_mem = nullptr; }
    }

    void create(int esz) {
        // Allocate contiguous param buffer
        int64_t total_params = 0;
        for (auto& r : params) total_params += r.size;
        if (total_params > 0) {
            cudaMalloc(&param_mem, total_params * esz);
            cudaMemset(param_mem, 0, total_params * esz);
            assign_puf_views(params, param_mem, esz);
        }
        total_param_elems = total_params;

        // Allocate contiguous grad buffer
        int64_t total_grads = 0;
        for (auto& r : grads) total_grads += r.size;
        if (total_grads > 0) {
            cudaMalloc(&grad_mem, total_grads * esz);
            cudaMemset(grad_mem, 0, total_grads * esz);
            assign_puf_views(grads, grad_mem, esz);
        }
        total_grad_elems = total_grads;

        // Allocate PufTensor activations from a contiguous byte buffer
        int64_t total_puf_bytes = 0;
        for (auto& r : puf_activations) total_puf_bytes += r.size * r.elem_size;
        if (total_puf_bytes > 0) {
            cudaMalloc(&puf_mem, total_puf_bytes);
            cudaMemset(puf_mem, 0, total_puf_bytes);
            char* base = (char*)puf_mem;
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
        elem_size = esz;
    }

#ifdef PUFFERLIB_TORCH
    // Legacy: create with torch types (builds Tensor views for bindings/cpp paths)
    void create(torch::Device device, torch::ScalarType dtype) {
        int esz = (dtype == torch::kFloat32) ? sizeof(float) : (dtype == torch::kBFloat16) ? 2 : sizeof(float);
        create(esz);
        // Create Tensor views over the cudaMalloc'd memory
        if (param_mem) {
            param_buffer = torch::from_blob(param_mem, {total_param_elems},
                torch::dtype(dtype).device(device));
        }
        if (grad_mem) {
            grad_buffer = torch::from_blob(grad_mem, {total_grad_elems},
                torch::dtype(dtype).device(device));
        }
    }
#endif

    int elem_size = 0;
    int64_t total_param_elems = 0;
    int64_t total_grad_elems = 0;
};


#endif // PUFFERLIB_MODULES_H
