// kernels.h - Launch function declarations for CUDA kernels
// All dtype-dependent args use void* with a dtype parameter
// This header has no torch dependencies

#ifndef PUFFERLIB_KERNELS_H
#define PUFFERLIB_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

#define DTYPE_FLOAT 0
#define DTYPE_BF16 1

void launch_mingru_gate_inference(void* out, void* next_state, const void* combined, const void* state_in, int H, int B, cudaStream_t stream, int dtype);

void launch_fused_scan_forward_checkpointed(void* out, void* next_state, float* a_star, float* s_vals, float* log_values_buf, const void* combined, const void* state, int T_seq, int H, int B, cudaStream_t stream, int dtype);

void launch_fused_scan_backward_checkpointed(void* grad_combined, void* grad_state, const void* grad_out, const void* grad_next_state, const void* combined, const void* state, const float* a_star_buf, const float* s_buf, const float* log_values_buf, int T_seq, int H, int B, cudaStream_t stream, int dtype);

void launch_logcumsumexp_forward(void* out, double* s_buf, const void* x, int T_total, int H, int B, cudaStream_t stream, int dtype);

void launch_logcumsumexp_backward(void* grad_x, const void* grad_out, const void* x, const double* s_buf, int T_total, int H, int B, cudaStream_t stream, int dtype);

void launch_ppo_loss_forward_optimized(float* loss_output, double* saved_for_backward, void* ratio_out, void* newvalue_out, const void* logits, const void* logstd, const void* values_pred, const double* actions, const void* old_logprobs, const float* advantages, const void* prio, const void* values, const void* returns, const float* adv_mean, const float* adv_var, const int* act_sizes, int num_atns, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A_total, int N, int logits_stride_n, int logits_stride_t, int logits_stride_a, int values_stride_n, int values_stride_t, bool is_continuous, cudaStream_t stream, int dtype);

void launch_ppo_loss_backward_optimized(float* grad_logits, float* grad_logstd, float* grad_values_pred, const float* grad_loss, const void* logits, const void* logstd, const void* values_pred, const double* actions, const void* old_logprobs, const float* advantages, const void* prio, const void* values, const void* returns, const float* adv_mean, const float* adv_var, const int* act_sizes, int num_atns, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A_total, int N, int logits_stride_n, int logits_stride_t, int logits_stride_a, int values_stride_n, int values_stride_t, bool is_continuous, cudaStream_t stream, int dtype);

void launch_sample_logits(double* actions, void* logprobs, void* value_out, const void* logits, const void* logstd, const void* value, const int* act_sizes, uint64_t seed, const int64_t* offset_ptr, int num_atns, int B, int logits_stride, int logstd_stride, int value_stride, bool is_continuous, cudaStream_t stream, int dtype);

void launch_fc_max_forward(void* out, int* argmax_indices, const void* x, const float* W, const float* b, int B, int N, int D_in, int D_out, cudaStream_t stream, int dtype);

void launch_fc_max_backward(float* grad_x, float* grad_W, float* grad_b, const void* grad_out, const void* x, const float* W, const int* argmax_indices, int B, int N, int D_in, int D_out, cudaStream_t stream, int dtype);

#endif // PUFFERLIB_KERNELS_H
