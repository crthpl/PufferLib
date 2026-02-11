#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define PRIO_WARP_SIZE 32
#define PRIO_FULL_MASK 0xffffffff
#define PRIO_BLOCK_SIZE 256
#define PRIO_NUM_WARPS (PRIO_BLOCK_SIZE / PRIO_WARP_SIZE)

namespace pufferlib {

__global__ void compute_prio_part1(
    const float* __restrict__ advantages,
    float* prio_weights,
    float prio_alpha,
    int stride
) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int offset = row * stride;

    float local_sum = 0.0f;
    for (int t = tx; t < stride; t += blockDim.x) {
        local_sum += fabsf(advantages[offset + t]);
    }

    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (tx == 0) {
        float pw = __powf(local_sum, prio_alpha);
        if (isnan(pw) || isinf(pw)) pw = 0.0f;
        prio_weights[row] = pw;
    }
}

__global__ void compute_prio_part2(
    float* prio_weights,
    int length
) {
    __shared__ float shmem[PRIO_NUM_WARPS];
    __shared__ float block_sum;

    int tx = threadIdx.x;
    int lane = tx % PRIO_WARP_SIZE;
    int warp_id = tx / PRIO_WARP_SIZE;
    const float eps = 1e-6f;

    float local_sum = 0.0f;
    for (int t = tx; t < length; t += blockDim.x) {
        local_sum += prio_weights[t];
    }
    for (int s = PRIO_WARP_SIZE / 2; s >= 1; s /= 2) {
        local_sum += __shfl_down_sync(PRIO_FULL_MASK, local_sum, s);
    }
    if (lane == 0) shmem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < PRIO_NUM_WARPS) ? shmem[lane] : 0.0f;
        for (int s = PRIO_NUM_WARPS / 2; s >= 1; s /= 2) {
            val += __shfl_down_sync(PRIO_FULL_MASK, val, s);
        }
        if (tx == 0) block_sum = val + eps;
    }
    __syncthreads();

    for (int t = tx; t < length; t += blockDim.x) {
        prio_weights[t] = (prio_weights[t] + eps) / block_sum;
    }
}

// Part 3: compute importance weights for sampled indices
// mb_prio[i] = pow(total_agents * prio_probs[idx[i]], -anneal_beta)
__global__ void compute_prio_part3(
    const int64_t* __restrict__ indices,
    const float* __restrict__ prio_probs,
    float* mb_prio,
    int total_agents,
    float anneal_beta,
    int minibatch_segments
) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx < minibatch_segments) {
        float value = prio_probs[indices[tx]] * (float)total_agents;
        mb_prio[tx] = __powf(value, -anneal_beta);
    }
}

// Host dispatch: replaces ~9 PyTorch kernel launches with 3 custom + multinomial
std::tuple<torch::Tensor, torch::Tensor> compute_prio_cuda(
    torch::Tensor advantages,       // (S, T) float32
    float prio_alpha,
    int minibatch_segments,
    int total_agents,
    float anneal_beta
) {
    int S = advantages.size(0);
    int T = advantages.size(1);

    auto prio_probs = torch::empty({S}, advantages.options());

    compute_prio_part1<<<S, PRIO_WARP_SIZE>>>(
        advantages.data_ptr<float>(), prio_probs.data_ptr<float>(),
        prio_alpha, T);

    compute_prio_part2<<<1, PRIO_BLOCK_SIZE>>>(
        prio_probs.data_ptr<float>(), S);

    auto idx = at::multinomial(prio_probs, minibatch_segments, true);

    auto mb_prio = torch::empty({minibatch_segments, 1}, advantages.options());
    int p3_blocks = (minibatch_segments + PRIO_BLOCK_SIZE - 1) / PRIO_BLOCK_SIZE;
    compute_prio_part3<<<p3_blocks, PRIO_BLOCK_SIZE>>>(
        idx.data_ptr<int64_t>(), prio_probs.data_ptr<float>(),
        mb_prio.data_ptr<float>(),
        total_agents, anneal_beta, minibatch_segments);

    return {idx, mb_prio};
}

} // namespace pufferlib
