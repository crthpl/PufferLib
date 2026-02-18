#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <cassert>
#include "modules.h"  // PufTensor, puf_copy, puf_zero, etc.

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

// Newton-Schulz scratch buffers (all PufTensor, bf16)
struct NSScratch {
    PufTensor x, A, gram, tmp;
    PufTensor result_f32;   // f32 output buffer (same max size as x)
    float* norm_ptr;        // device scalar for norm
    int64_t max_M, max_N;

    // Slice scratch to actual size needed
    PufTensor slice(PufTensor& buf, int64_t rows, int64_t cols) {
        PufTensor s = buf;
        s.shape[0] = rows; s.shape[1] = cols; s.ndim = 2;
        s.numel = rows * cols;
        return s;
    }
};

struct Muon {
    // Param shape info for Newton-Schulz (no torch dependency)
    struct ParamShape {
        int64_t numel;
        std::vector<int64_t> shape;
        int ndim;
    };

    // Hyperparameters
    double momentum;
    double weight_decay;
    double eps;
    float lr_val_init;                // initial lr value for post_create

    // State — all raw CUDA, no torch
    float* lr_ptr = nullptr;          // alias into lr_puf.data after post_create
    float* lr_derived_ptr = nullptr;  // alias into lr_derived_puf.data after post_create
    PufTensor lr_puf;                 // (1,) f32 — registered in external Allocator
    PufTensor lr_derived_puf;         // (2,) f32 — registered in external Allocator
    PufTensor ns_norm_puf;            // (1,) f32 — norm scalar for Newton-Schulz
    PufTensor wb_puf, gb_puf, mb_puf, gc_puf, up_puf;
    bool bufs_initialized = false;
    NSScratch ns;
    std::vector<ParamShape> param_shapes;

    // Multi-GPU
    ncclComm_t nccl_comm = nullptr;
    int world_size = 1;

    Muon(std::vector<ParamShape> param_shapes, PufTensor weight_buffer,
         PufTensor grad_buffer, double lr_val, double momentum,
         double eps, double weight_decay)
        : momentum(momentum), weight_decay(weight_decay), eps(eps),
          lr_val_init((float)lr_val),
          wb_puf(weight_buffer), gb_puf(grad_buffer),
          param_shapes(std::move(param_shapes))
    {
        assert(lr_val >= 0 && "Invalid learning rate");
        assert(eps >= 0 && "Invalid epsilon value");
        assert(weight_decay >= 0 && "Invalid weight_decay value");
    }

    ~Muon() {} // Memory owned by external Allocator

    // Non-copyable
    Muon(const Muon&) = delete;
    Muon& operator=(const Muon&) = delete;

    // Register all Muon buffers into an external Allocator (call before alloc.create())
    void register_buffers(Allocator& alloc) {
        int64_t n = wb_puf.numel;
        alloc.register_puf(&lr_puf, {1}, sizeof(float));
        alloc.register_puf(&lr_derived_puf, {2}, sizeof(float));
        alloc.register_puf(&mb_puf, {n}, sizeof(float));
        alloc.register_puf(&gc_puf, {n}, sizeof(float));
        alloc.register_puf(&up_puf, {n}, sizeof(float));

        // Pre-compute Newton-Schulz scratch sizes for largest 2D param
        int64_t max_M = 0, max_N = 0;
        for (auto& ps : param_shapes) {
            if (ps.ndim >= 2) {
                int64_t R = ps.shape[0];
                int64_t C = ps.numel / R;
                int64_t M = std::min(R, C), N = std::max(R, C);
                max_M = std::max(max_M, M);
                max_N = std::max(max_N, N);
            }
        }
        if (max_M > 0) {
            ns.max_M = max_M; ns.max_N = max_N;
            alloc.register_puf(&ns.x, {max_M, max_N}, 2);           // bf16
            alloc.register_puf(&ns.A, {max_M, max_M}, 2);           // bf16
            alloc.register_puf(&ns.gram, {max_M, max_M}, 2);        // bf16
            alloc.register_puf(&ns.tmp, {max_M, max_N}, 2);         // bf16
            alloc.register_puf(&ns.result_f32, {max_M, max_N}, sizeof(float));
            alloc.register_puf(&ns_norm_puf, {1}, sizeof(float));
        }
        bufs_initialized = true;
    }

    // Set up pointer aliases and initial values (call after alloc.create())
    void post_create() {
        lr_ptr = (float*)lr_puf.data;
        lr_derived_ptr = (float*)lr_derived_puf.data;
        if (ns_norm_puf.data) ns.norm_ptr = (float*)ns_norm_puf.data;
        cudaMemcpy(lr_ptr, &lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(lr_derived_ptr, 0, 2 * sizeof(float));
        // Zero momentum buffer
        cudaMemset(mb_puf.data, 0, mb_puf.numel * sizeof(float));
    }

    void step(cudaStream_t stream = 0) {
        if (wb_puf.data == nullptr) return;

        // Copy grads into persistent clone buffer
        puf_copy(gc_puf, gb_puf, stream);

        // Multi-GPU gradient sync
        if (nccl_comm != nullptr && world_size > 1) {
            ncclAllReduce(gc_puf.data, gc_puf.data, gc_puf.numel,
                          ncclFloat, ncclAvg, nccl_comm, stream);
        }

        // Nesterov momentum
        puf_scale(mb_puf, (float)momentum, stream);
        puf_axpy(mb_puf, gc_puf, 1.0f, stream);
        puf_axpy(gc_puf, mb_puf, (float)momentum, stream);

        // Newton-Schulz per param
        puf_zero(up_puf, stream);
        int64_t offset = 0;
        for (auto& ps : param_shapes) {
            float* gc_ptr = (float*)gc_puf.data + offset;
            float* up_ptr = (float*)up_puf.data + offset;

            if (ps.ndim >= 2) {
                int64_t R = ps.shape[0];
                int64_t C = ps.numel / R;
                bool transposed = R > C;
                int64_t M = transposed ? C : R;
                int64_t N = transposed ? R : C;

                // PufTensor view of f32 input
                PufTensor G_f32;
                G_f32.data = gc_ptr; G_f32.shape[0] = R; G_f32.shape[1] = C;
                G_f32.ndim = 2; G_f32.numel = R*C; G_f32.dtype_size = 4;

                // Slice scratch to actual size
                PufTensor x = ns.slice(ns.x, M, N);
                PufTensor A = ns.slice(ns.A, M, M);
                PufTensor gram = ns.slice(ns.gram, M, M);
                PufTensor tmp = ns.slice(ns.tmp, M, N);

                // Cast f32 → bf16 (with transpose if needed)
                if (transposed) {
                    PufTensor x_view = x; // (C, R) = (M, N)
                    puf_cast_f32_to_bf16_transpose(x_view, G_f32, stream);
                } else {
                    puf_cast_f32_to_bf16(x, G_f32, stream);
                }

                // Normalize: x /= max(norm(x), 1e-7)
                puf_norm(x, ns.norm_ptr, stream);
                puf_normalize(x, ns.norm_ptr, 1e-7f, stream);

                // 5 Newton-Schulz iterations, ping-pong between x and tmp
                for (int i = 0; i < 5; ++i) {
                    float a = (float)ns_coeffs[i][0];
                    float b = (float)ns_coeffs[i][1];
                    float c = (float)ns_coeffs[i][2];
                    PufTensor& src = (i % 2 == 0) ? x : tmp;
                    PufTensor& dst = (i % 2 == 0) ? tmp : x;
                    // A = src @ src^T
                    puf_mm(src, src, A, stream);
                    // gram = c * A @ A + b * A  (reuse A as input, write to gram)
                    puf_copy(gram, A, stream);
                    puf_addmm_nn(A, A, gram, c, b, stream);
                    // dst = 1.0 * gram @ src + a * src  (ping-pong)
                    puf_copy(dst, src, stream);
                    puf_addmm_nn(gram, src, dst, 1.0f, a, stream);
                }
                // After 5 (odd) iterations, result is in tmp
                PufTensor& result_bf16 = tmp;

                // Cast bf16 result → f32, scale, and optional transpose back
                double ratio = (double)M / (double)N;
                float scale = (float)std::sqrt(std::max(1.0, ratio));
                PufTensor out_f32;
                out_f32.data = up_ptr; out_f32.shape[0] = R; out_f32.shape[1] = C;
                out_f32.ndim = 2; out_f32.numel = R*C; out_f32.dtype_size = 4;

                PufTensor res_f32 = ns.slice(ns.result_f32, M, N);
                res_f32.dtype_size = 4; res_f32.numel = M * N;
                puf_cast_bf16_to_f32(res_f32, result_bf16, stream);
                if (scale != 1.0f) puf_scale(res_f32, scale, stream);

                if (transposed) {
                    // res_f32 is (M=C, N=R), need (R, C) in out_f32
                    puf_transpose_f32(out_f32, res_f32, stream);
                } else {
                    puf_copy(out_f32, res_f32, stream);
                }
            } else {
                // 1D param: just copy as-is
                PufTensor src_puf, dst_puf;
                src_puf.data = gc_ptr; src_puf.numel = ps.numel; src_puf.dtype_size = 4;
                dst_puf.data = up_ptr; dst_puf.numel = ps.numel; dst_puf.dtype_size = 4;
                puf_copy(dst_puf, src_puf, stream);
            }
            offset += ps.numel;
        }

        // Compute derived lr scalars on device (graph-safe)
        compute_lr_scalars(lr_ptr, (float)weight_decay, lr_derived_ptr, lr_derived_ptr + 1, stream);

        // Apply update
        if (weight_decay != 0) {
            puf_scale_dev(wb_puf, lr_derived_ptr + 1, stream);
        }
        puf_axpy_dev(wb_puf, up_puf, lr_derived_ptr, stream);
    }

    void zero_grad(cudaStream_t stream) {
        if (gb_puf.data != nullptr) {
            puf_zero(gb_puf, stream);
        }
    }
};
