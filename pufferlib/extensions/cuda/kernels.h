// kernels.h - Launch function declarations for CUDA kernels
// Precision is selected at compile time via -DPRECISION_FLOAT
// This header has no torch dependencies

#ifndef PUFFERLIB_KERNELS_H
#define PUFFERLIB_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Compile-time precision: default bf16, pass -DPRECISION_FLOAT for float32
#ifdef PRECISION_FLOAT
typedef float precision_t;
#else
typedef __nv_bfloat16 precision_t;
#endif

void launch_mingru_gate_inference(precision_t* out, precision_t* next_state, const precision_t* combined, const precision_t* state_in, int H, int B, cudaStream_t stream);

void launch_fused_scan_forward_checkpointed(precision_t* out, precision_t* next_state, float* a_star, float* s_vals, float* log_values_buf, const precision_t* combined, const precision_t* state, int T_seq, int H, int B, cudaStream_t stream);

void launch_fused_scan_backward_checkpointed(precision_t* grad_combined, precision_t* grad_state, const precision_t* grad_out, const precision_t* grad_next_state, const precision_t* combined, const precision_t* state, const float* a_star_buf, const float* s_buf, const float* log_values_buf, int T_seq, int H, int B, cudaStream_t stream);

void launch_logcumsumexp_forward(precision_t* out, double* s_buf, const precision_t* x, int T_total, int H, int B, cudaStream_t stream);

void launch_logcumsumexp_backward(precision_t* grad_x, const precision_t* grad_out, const precision_t* x, const double* s_buf, int T_total, int H, int B, cudaStream_t stream);

void launch_ppo_loss_forward_optimized(float* loss_output, double* saved_for_backward, precision_t* ratio_out, precision_t* newvalue_out, const precision_t* logits, const precision_t* logstd, const precision_t* values_pred, const double* actions, const precision_t* old_logprobs, const float* advantages, const precision_t* prio, const precision_t* values, const precision_t* returns, const float* adv_mean, const float* adv_var, const int* act_sizes, int num_atns, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A_total, int N, int logits_stride_n, int logits_stride_t, int logits_stride_a, int values_stride_n, int values_stride_t, bool is_continuous, cudaStream_t stream);

void launch_ppo_loss_backward_optimized(float* grad_logits, float* grad_logstd, float* grad_values_pred, const float* grad_loss, const precision_t* logits, const precision_t* logstd, const precision_t* values_pred, const double* actions, const precision_t* old_logprobs, const float* advantages, const precision_t* prio, const precision_t* values, const precision_t* returns, const float* adv_mean, const float* adv_var, const int* act_sizes, int num_atns, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A_total, int N, int logits_stride_n, int logits_stride_t, int logits_stride_a, int values_stride_n, int values_stride_t, bool is_continuous, cudaStream_t stream);

void launch_sample_logits(double* actions, precision_t* logprobs, precision_t* value_out, const precision_t* logits, const precision_t* logstd, const precision_t* value, const int* act_sizes, uint64_t seed, const int64_t* offset_ptr, int num_atns, int B, int logits_stride, int logstd_stride, int value_stride, bool is_continuous, cudaStream_t stream);

void launch_fc_max_forward(precision_t* out, int* argmax_indices, const precision_t* x, const float* W, const float* b, int B, int N, int D_in, int D_out, cudaStream_t stream);

void launch_fc_max_backward(float* grad_x, float* grad_W, float* grad_b, const precision_t* grad_out, const precision_t* x, const float* W, const int* argmax_indices, int B, int N, int D_in, int D_out, cudaStream_t stream);

#endif // PUFFERLIB_KERNELS_H
