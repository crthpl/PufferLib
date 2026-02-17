#pragma once

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>
#include "modules.h"  // PufTensor, puf_copy, puf_zero

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

    // State
    torch::Tensor lr;              // (1,) f32 CUDA — written by lr annealing, read by graph
    torch::Tensor lr_derived;      // (2,) f32 CUDA — [neg_lr, wd_scale], computed from lr
    torch::Tensor momentum_buffer; // contiguous momentum buffer (torch for state_dict)
    torch::Tensor grad_clone;      // pre-allocated clone buffer (torch for state_dict)
    torch::Tensor updates;         // pre-allocated buffer (torch for state_dict)
    PufTensor wb_puf, gb_puf, mb_puf, gc_puf, up_puf;  // cached PufTensor views
    bool bufs_initialized = false;
    NSScratch ns;
    torch::Tensor ns_buffer;  // keeps NS scratch memory alive
    std::vector<ParamShape> param_shapes;  // shapes for per-param Newton-Schulz

    // Multi-GPU
    ncclComm_t nccl_comm = nullptr;
    int world_size = 1;

    Muon(std::vector<ParamShape> param_shapes, PufTensor weight_buffer,
         PufTensor grad_buffer, double lr_val, double momentum,
         double eps, double weight_decay)
        : momentum(momentum), weight_decay(weight_decay), eps(eps),
          wb_puf(weight_buffer), gb_puf(grad_buffer),
          param_shapes(std::move(param_shapes))
    {
        TORCH_CHECK(lr_val >= 0, "Invalid learning rate: ", lr_val);
        TORCH_CHECK(eps >= 0, "Invalid epsilon value: ", eps);
        TORCH_CHECK(weight_decay >= 0, "Invalid weight_decay value: ", weight_decay);
        lr = torch::tensor(lr_val, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
        lr_derived = torch::zeros({2}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    }

    void step() {
        if (wb_puf.data == nullptr) return;

        // Initialize persistent buffers and cache PufTensor views (once)
        if (!bufs_initialized) {
            auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
            momentum_buffer = torch::zeros({wb_puf.numel}, opts);
            grad_clone = torch::empty({wb_puf.numel}, opts);
            updates = torch::empty({wb_puf.numel}, opts);
            mb_puf = PufTensor::from_torch(momentum_buffer);
            gc_puf = PufTensor::from_torch(grad_clone);
            up_puf = PufTensor::from_torch(updates);

            // Pre-allocate Newton-Schulz scratch for largest 2D param
            // M = min(R,C), N = max(R,C) — matching transposed convention in NS
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
                int bf16sz = 2, f32sz = 4;
                // Allocate one contiguous buffer for all NS scratch
                int64_t total = (max_M*max_N + max_M*max_M + max_M*max_M + max_M*max_N) * bf16sz
                              + max_M*max_N * f32sz + sizeof(float);
                torch::Tensor ns_buf = torch::empty({total},
                    torch::dtype(torch::kUInt8).device(torch::kCUDA));
                char* p = (char*)ns_buf.data_ptr();
                auto mk_bf16 = [&](int64_t r, int64_t c) -> PufTensor {
                    PufTensor t; t.data = p; t.shape[0] = r; t.shape[1] = c;
                    t.ndim = 2; t.numel = r*c; t.dtype_size = bf16sz;
                    p += r * c * bf16sz; return t;
                };
                ns.x = mk_bf16(max_M, max_N);
                ns.A = mk_bf16(max_M, max_M);
                ns.gram = mk_bf16(max_M, max_M);
                ns.tmp = mk_bf16(max_M, max_N);
                // f32 result buffer
                ns.result_f32.data = p; ns.result_f32.shape[0] = max_M; ns.result_f32.shape[1] = max_N;
                ns.result_f32.ndim = 2; ns.result_f32.numel = max_M*max_N; ns.result_f32.dtype_size = f32sz;
                p += max_M * max_N * f32sz;
                ns.norm_ptr = (float*)p;
                // Keep ns_buf alive by storing it (hack: reuse updates tensor? No, just add a field)
                // Actually the torch tensor going out of scope would free the memory.
                // Store it as a member.
                ns_buffer = ns_buf;
            }
            bufs_initialized = true;
        }

        // Copy grads into persistent clone buffer
        puf_copy(gc_puf, gb_puf);

        // Multi-GPU gradient sync
        if (nccl_comm != nullptr && world_size > 1) {
            ncclAllReduce(gc_puf.data, gc_puf.data, gc_puf.numel,
                          ncclFloat, ncclAvg, nccl_comm,
                          at::cuda::getCurrentCUDAStream());
        }

        // Nesterov momentum
        puf_scale(mb_puf, (float)momentum);
        puf_axpy(mb_puf, gc_puf, 1.0f);
        puf_axpy(gc_puf, mb_puf, (float)momentum);

        // Newton-Schulz per param
        puf_zero(up_puf);
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
                    puf_cast_f32_to_bf16_transpose(x_view, G_f32);
                } else {
                    puf_cast_f32_to_bf16(x, G_f32);
                }

                // Normalize: x /= max(norm(x), 1e-7)
                puf_norm(x, ns.norm_ptr);
                puf_normalize(x, ns.norm_ptr, 1e-7f);

                // 5 Newton-Schulz iterations, ping-pong between x and tmp
                for (int i = 0; i < 5; ++i) {
                    float a = (float)ns_coeffs[i][0];
                    float b = (float)ns_coeffs[i][1];
                    float c = (float)ns_coeffs[i][2];
                    PufTensor& src = (i % 2 == 0) ? x : tmp;
                    PufTensor& dst = (i % 2 == 0) ? tmp : x;
                    // A = src @ src^T
                    puf_mm(src, src, A);
                    // gram = c * A @ A + b * A  (reuse A as input, write to gram)
                    puf_copy(gram, A);
                    puf_addmm_nn(A, A, gram, c, b);
                    // dst = 1.0 * gram @ src + a * src  (ping-pong)
                    puf_copy(dst, src);
                    puf_addmm_nn(gram, src, dst, 1.0f, a);
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
                puf_cast_bf16_to_f32(res_f32, result_bf16);
                if (scale != 1.0f) puf_scale(res_f32, scale);

                if (transposed) {
                    // res_f32 is (M=C, N=R), need (R, C) in out_f32
                    puf_transpose_f32(out_f32, res_f32);
                } else {
                    puf_copy(out_f32, res_f32);
                }
            } else {
                // 1D param: just copy as-is
                PufTensor src_puf, dst_puf;
                src_puf.data = gc_ptr; src_puf.numel = ps.numel; src_puf.dtype_size = 4;
                dst_puf.data = up_ptr; dst_puf.numel = ps.numel; dst_puf.dtype_size = 4;
                puf_copy(dst_puf, src_puf);
            }
            offset += ps.numel;
        }

        // Compute derived lr scalars on device (graph-safe — reads lr from device memory)
        float* lr_ptr = (float*)lr.data_ptr();
        float* neg_lr_ptr = (float*)lr_derived.data_ptr();
        float* wd_scale_ptr = neg_lr_ptr + 1;
        compute_lr_scalars(lr_ptr, (float)weight_decay, neg_lr_ptr, wd_scale_ptr);

        // Apply update
        if (weight_decay != 0) {
            puf_scale_dev(wb_puf, wd_scale_ptr);
        }
        puf_axpy_dev(wb_puf, up_puf, neg_lr_ptr);
    }

    void zero_grad() {
        if (gb_puf.data != nullptr) {
            puf_zero(gb_puf);
        }
    }

    std::unordered_map<std::string, torch::Tensor> state_dict() const {
        std::unordered_map<std::string, torch::Tensor> state;
        state["lr"] = lr;
        if (wb_puf.data != nullptr) state["weight_buffer"] = wb_puf.to_torch(torch::kFloat32);
        if (momentum_buffer.defined()) state["momentum_buffer"] = momentum_buffer;
        return state;
    }

    void load_state_dict(const std::unordered_map<std::string, torch::Tensor>& state) {
        auto it = state.find("lr");
        if (it != state.end()) lr.copy_(it->second);
        it = state.find("weight_buffer");
        if (it != state.end() && wb_puf.data != nullptr) {
            // Copy into the PufTensor-backed weight buffer
            torch::Tensor wb_view = wb_puf.to_torch(torch::kFloat32);
            wb_view.copy_(it->second);
        }
        it = state.find("momentum_buffer");
        if (it != state.end()) {
            momentum_buffer.copy_(it->second);
            if (!grad_clone.defined()) {
                auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
                grad_clone = torch::empty({wb_puf.numel}, opts);
                updates = torch::empty({wb_puf.numel}, opts);
            }
        }
    }
};
