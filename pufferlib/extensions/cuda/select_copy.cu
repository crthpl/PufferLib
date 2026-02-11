#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define SELECT_COPY_THREADS 256

namespace pufferlib {

__device__ __forceinline__ void copy_bytes(
    const char* __restrict__ src, char* __restrict__ dst,
    int src_row, int dst_row, int row_bytes
) {
    const int* soffset = (const int*)(src + (int64_t)src_row * row_bytes);
    int* doffset = (int*)(dst + (int64_t)dst_row * row_bytes);
    for (int i = threadIdx.x; i < row_bytes / 4; i += blockDim.x)
        doffset[i] = soffset[i];
}

template<typename T>
__device__ __forceinline__ void copy_values_adv_returns(
    const T* __restrict__ src_values, T* __restrict__ dst_values,
    const float* __restrict__ src_advantages, float* __restrict__ dst_advantages,
    T* __restrict__ dst_returns,
    int src_row, int dst_row, int horizon
) {
    int srh = (int64_t)src_row * horizon;
    int drh = (int64_t)dst_row * horizon;
    const T* s_values = src_values + srh;
    const float* s_adv = src_advantages + srh;
    T* d_values = dst_values + drh;
    float* d_adv = dst_advantages + drh;
    T* d_returns = dst_returns + drh;
    for (int i = threadIdx.x; i < horizon; i += blockDim.x) {
        T val = s_values[i];
        float adv = s_adv[i];
        d_values[i] = val;
        d_adv[i] = adv;
        d_returns[i] = (T)((float)val + adv);
    }
}

template<typename T>
__global__ void select_copy_kernel(
    const int64_t* __restrict__ idx,
    const char* __restrict__ src_obs, char* __restrict__ dst_obs, int obs_row_bytes,
    const char* __restrict__ src_actions, char* __restrict__ dst_actions, int actions_row_bytes,
    const char* __restrict__ src_logprobs, char* __restrict__ dst_logprobs, int logprobs_row_bytes,
    const T* __restrict__ src_values, T* __restrict__ dst_values,
    const float* __restrict__ src_advantages, float* __restrict__ dst_advantages,
    T* __restrict__ dst_returns, int horizon,
    const T* __restrict__ src_prio, T* __restrict__ dst_prio
) {
    int mb = blockIdx.x;
    int ch = blockIdx.y;
    int src_row = (int)idx[mb];

    switch (ch) {
    case 0:
        copy_bytes(src_obs, dst_obs, src_row, mb, obs_row_bytes);
        break;
    case 1:
        copy_bytes(src_actions, dst_actions, src_row, mb, actions_row_bytes);
        break;
    case 2:
        copy_bytes(src_logprobs, dst_logprobs, src_row, mb, logprobs_row_bytes);
        break;
    case 3:
        copy_values_adv_returns(src_values, dst_values, src_advantages,
                dst_advantages, dst_returns, src_row, mb, horizon);
        break;
    case 4:
        if (threadIdx.x == 0) {
            dst_prio[mb] = src_prio[mb];
            break;
        }
    }
}


template<typename T>
void launch_select_copy(
    torch::Tensor& idx, int mb_segs, int horizon,
    torch::Tensor& observations, torch::Tensor& dst_obs, int obs_row_bytes,
    torch::Tensor& actions, torch::Tensor& dst_actions, int actions_row_bytes,
    torch::Tensor& logprobs, torch::Tensor& dst_logprobs, int logprobs_row_bytes,
    torch::Tensor& values, torch::Tensor& dst_values,
    torch::Tensor& advantages, torch::Tensor& dst_advantages,
    torch::Tensor& dst_returns,
    torch::Tensor& mb_prio, torch::Tensor& dst_prio
) {
    select_copy_kernel<T><<<dim3(mb_segs, 5), SELECT_COPY_THREADS>>>(
        idx.data_ptr<int64_t>(),
        (const char*)observations.data_ptr(), (char*)dst_obs.data_ptr(), obs_row_bytes,
        (const char*)actions.data_ptr(), (char*)dst_actions.data_ptr(), actions_row_bytes,
        (const char*)logprobs.data_ptr(), (char*)dst_logprobs.data_ptr(), logprobs_row_bytes,
        (const T*)values.data_ptr(), (T*)dst_values.data_ptr(),
        advantages.data_ptr<float>(), dst_advantages.data_ptr<float>(),
        (T*)dst_returns.data_ptr(), horizon,
        (const T*)mb_prio.data_ptr(), (T*)dst_prio.data_ptr());
}


void train_select_and_copy_cuda(
    torch::Tensor observations, torch::Tensor actions,
    torch::Tensor logprobs, torch::Tensor values, torch::Tensor advantages,
    torch::Tensor idx, torch::Tensor mb_prio,
    torch::Tensor dst_obs, torch::Tensor dst_state,
    torch::Tensor dst_actions, torch::Tensor dst_logprobs,
    torch::Tensor dst_advantages, torch::Tensor dst_prio,
    torch::Tensor dst_values, torch::Tensor dst_returns
) {
    int mb_segs = idx.size(0);
    int horizon = values.size(1);
    int obs_rb = observations.stride(0) * observations.element_size();
    int act_rb = actions.stride(0) * actions.element_size();
    int lp_rb = logprobs.stride(0) * logprobs.element_size();

    dst_state.zero_();

    if (values.scalar_type() == at::kBFloat16)
        launch_select_copy<__nv_bfloat16>(idx, mb_segs, horizon,
            observations, dst_obs, obs_rb, actions, dst_actions, act_rb,
            logprobs, dst_logprobs, lp_rb, values, dst_values,
            advantages, dst_advantages, dst_returns, mb_prio, dst_prio);
    else
        launch_select_copy<float>(idx, mb_segs, horizon,
            observations, dst_obs, obs_rb, actions, dst_actions, act_rb,
            logprobs, dst_logprobs, lp_rb, values, dst_values,
            advantages, dst_advantages, dst_returns, mb_prio, dst_prio);
}

}
