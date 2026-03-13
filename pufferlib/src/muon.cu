#ifndef PUFFERLIB_MUON_CU
#define PUFFERLIB_MUON_CU

#include <nccl.h>

__global__ void norm_reduce_kernel(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void norm_partials_kernel(float* __restrict__ partials, const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void norm_apply_kernel(precision_t* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// Nesterov with f32 momentum accumulator and precision_t gradients
__global__ void nesterov_momentum_kernel(float* __restrict__ mb, precision_t* __restrict__ gc, float mu, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + to_float(gc[idx]);
        mb[idx] = m;
        gc[idx] = from_float(to_float(gc[idx]) + mu * m);
    }
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * scale * update
__global__ void muon_weight_update_kernel(float* __restrict__ wb, const precision_t* __restrict__ update,
                                           const float* __restrict__ lr_ptr, float wd, float scale, int n) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * scale * to_float(update[idx]);
    }
}

__global__ void clip_by_norm_partials_kernel(precision_t* __restrict__ dst, const float* __restrict__ sum_sq_ptr,
                                               float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * clip_coef);
    }
}

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct Muon {
    double momentum, weight_decay, eps;
    float lr_val_init;
    float* lr_ptr;
    float* lr_derived_ptr;
    float* norm_ptr;
    float* grad_norm_ptr;
    PufTensor lr_puf, lr_derived_puf, ns_norm_puf, grad_norm_puf;
    PufTensor mb_puf;
    PufTensor gram, gram_buf, x_buf, norm_partials;
    long max_M, max_N;
    Allocator* param_alloc;  // fp32 params allocator — shapes used by muon_step
    ncclComm_t nccl_comm;
    int world_size;
};

void muon_init(Muon* m, Allocator* param_alloc,
               double lr_val, double momentum, double eps, double weight_decay,
               Allocator* alloc) {
    m->momentum = momentum;
    m->weight_decay = weight_decay;
    m->eps = eps;
    m->lr_val_init = (float)lr_val;
    m->lr_ptr = nullptr;
    m->lr_derived_ptr = nullptr;
    m->param_alloc = param_alloc;
    m->nccl_comm = nullptr;
    m->world_size = 1;
    m->max_M = 0; m->max_N = 0;
    long n = param_alloc->total_elems;
    int f = sizeof(float);
    m->lr_puf = {.shape = {1}, .dtype_size = f};
    m->lr_derived_puf = {.shape = {2}, .dtype_size = f};
    m->mb_puf = {.shape = {n}, .dtype_size = f};
    m->norm_partials = {.shape = {256}, .dtype_size = f};
    m->grad_norm_puf = {.shape = {1}, .dtype_size = f};
    alloc_register(alloc, &m->lr_puf);
    alloc_register(alloc, &m->lr_derived_puf);
    alloc_register(alloc, &m->mb_puf);
    alloc_register(alloc, &m->norm_partials);
    alloc_register(alloc, &m->grad_norm_puf);
    long max_M = 0, max_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        PufTensor* t = param_alloc->regs[_i];
        if (t->ndim() >= 2) {
            long R = t->shape[0], C = t->numel() / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
        }
    }
    if (max_M > 0) {
        m->max_M = max_M; m->max_N = max_N;
        int ns_esz = PRECISION_SIZE;
        m->gram = {.shape = {max_M, max_M}, .dtype_size = ns_esz};
        m->gram_buf = {.shape = {max_M, max_M}, .dtype_size = ns_esz};
        m->x_buf = {.shape = {max_M, max_N}, .dtype_size = ns_esz};
        m->ns_norm_puf = {.shape = {1}, .dtype_size = f};
        alloc_register(alloc, &m->gram);
        alloc_register(alloc, &m->gram_buf);
        alloc_register(alloc, &m->x_buf);
        alloc_register(alloc, &m->ns_norm_puf);
    }
}

void muon_post_create(Muon* m) {
    m->lr_ptr = (float*)m->lr_puf.bytes;
    m->lr_derived_ptr = (float*)m->lr_derived_puf.bytes;
    m->grad_norm_ptr = (float*)m->grad_norm_puf.bytes;
    if (m->ns_norm_puf.bytes) m->norm_ptr = (float*)m->ns_norm_puf.bytes;
    cudaMemcpy(m->lr_ptr, &m->lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(m->lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(m->mb_puf.bytes, 0, m->mb_puf.numel() * sizeof(float));
}

void muon_step(Muon* m, PufTensor weights, PufTensor grads, float max_grad_norm, cudaStream_t stream = 0) {
    // Multi-GPU support: simple all-reduce over a contiguous grad buffer
    if (m->nccl_comm != nullptr && m->world_size > 1) {
        ncclAllReduce(grads.bytes, grads.bytes, grads.numel(),
            NCCL_PRECISION, ncclAvg, m->nccl_comm, stream);
    }

    // Clip gradients by norm
    int clip_blocks = min((int)grid_size(grads.numel()), 256);
    norm_partials_kernel<<<clip_blocks, 256, 0, stream>>>(
        (float*)m->norm_partials.bytes, (const precision_t*)grads.bytes, grads.numel());
    norm_reduce_kernel<<<1, 256, 0, stream>>>(m->grad_norm_ptr, (float*)m->norm_partials.bytes, clip_blocks);
    clip_by_norm_partials_kernel<<<grid_size(grads.numel()), BLOCK_SIZE, 0, stream>>>(
        (precision_t*)grads.bytes, m->grad_norm_ptr, max_grad_norm, 1e-6f, grads.numel());

    // Nesterov momentum works better than Nesterov EMA in our experiments
    nesterov_momentum_kernel<<<grid_size(m->mb_puf.numel()), BLOCK_SIZE, 0, stream>>>(
        (float*)m->mb_puf.bytes, (precision_t*)grads.bytes, (float)m->momentum, m->mb_puf.numel());

    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        PufTensor* t = m->param_alloc->regs[_i];
        precision_t* gc_ptr = (precision_t*)grads.bytes + offset;
        float* wb_ptr = (float*)weights.bytes + offset;
        long numel = t->numel();
        const precision_t* update_ptr = gc_ptr;
        float scale = 1.0f;

        // Orthogonalize the update
        if (t->ndim() >= 2) {
            long R = t->shape[0], C = numel / R;
            long M = min(R, C), N = max(R, C);
            bool tall = R > C;
            PufTensor x = {.bytes = (char*)gc_ptr, .shape = {R, C}, .dtype_size = PRECISION_SIZE};
            PufTensor x_buf = {.bytes = m->x_buf.bytes, .shape = {R, C}, .dtype_size = PRECISION_SIZE};
            PufTensor gram = {.bytes = m->gram.bytes, .shape = {M, M}, .dtype_size = PRECISION_SIZE};
            PufTensor gram_buf = {.bytes = m->gram_buf.bytes, .shape = {M, M}, .dtype_size = PRECISION_SIZE};

            //x = x / clamp(x.norm(dim=(-2, -1)), min=eps)
            int nblk = min((int)grid_size(x.numel()), 256);
            norm_partials_kernel<<<nblk, 256, 0, stream>>>(
                (float*)m->norm_partials.bytes, (const precision_t*)x.bytes, x.numel());
            norm_reduce_kernel<<<1, 256, 0, stream>>>(m->norm_ptr, (float*)m->norm_partials.bytes, nblk);
            norm_apply_kernel<<<grid_size(x.numel()), BLOCK_SIZE, 0, stream>>>(
                (precision_t*)x.bytes, m->norm_ptr, 1e-7f, x.numel());

            // Gram matrix is symmetric -> we can skip tranposing x
            cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
            for (int i = 0; i < 5; ++i) {
                PufTensor& src = (i % 2 == 0) ? x : x_buf;
                PufTensor& dst = (i % 2 == 0) ? x_buf : x;
                cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
                    src.bytes, src.bytes, gram.bytes, stream);
                puf_copy(gram_buf, gram, stream);
                puf_addmm_nn(gram, gram, gram_buf, ns_coeffs[i][2], ns_coeffs[i][1], stream);
                puf_copy(dst, src, stream);
                cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
                    tall ? src.bytes : gram_buf.bytes, tall ? gram_buf.bytes : src.bytes, dst.bytes,
                    stream, 1.0f, ns_coeffs[i][0]);
            }

            // Scaling matches Keller
            update_ptr = (const precision_t*)x_buf.bytes;
            scale = sqrtf(fmaxf(1.0f, (float)M / (float)N));
        }

        // Orthogonalized for 2D+, simple SGD + Nesterov momentum for 1D
        muon_weight_update_kernel<<<grid_size(numel), BLOCK_SIZE, 0, stream>>>(
            wb_ptr, update_ptr, m->lr_ptr, (float)m->weight_decay, scale, (int)numel);
        offset += numel;
    }
}

#endif // PUFFERLIB_MUON_CU
