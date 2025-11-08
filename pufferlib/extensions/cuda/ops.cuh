#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Thes functions are ported from pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/
// PyTorch defines these kernels anonymously and with extra saftey and features
// This is a lower-overhead port of the math for use in fused kernels
// The caller handles templating, so these can just be simple C-style code

#define SOFTPLUS_BETA 1.0f
#define SOFTPLUS_THRESHOLD 20.0f
__device__ __forceinline__ float softplus_fwd(float x) {
    float x_scaled = x * SOFTPLUS_BETA;
    if (x_scaled > SOFTPLUS_THRESHOLD) {
        return x;
    } else {
        return log1pf(expf(x_scaled)) / SOFTPLUS_BETA;
    }
}

__device__ __forceinline__ float softplus_bwd(float grad_output, float x) {
    float beta_x = SOFTPLUS_BETA * x;
    if (beta_x > SOFTPLUS_THRESHOLD) {
        return grad_output;
    } else {
        float exp_beta_x = expf(beta_x);
        return grad_output * (exp_beta_x / (1.0f + exp_beta_x));
    }
}

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float relu_backward(float x, float grad_output) {
    return (x > 0.0f) ? grad_output : 0.0f;
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float sigmoid_backward(float x, float grad_output) {
    float sig = sigmoid(x);
    return grad_output * sig * (1.0f - sig);
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    if (fabsf(w) < 0.5f) {
        return a + w * diff;
    } else {
        return b - diff * (1.0f - w);
    }
}

__device__ __forceinline__ float lerp_backward_a(float a, float b, float w, float grad_output) {
    return grad_output * (1.0f - w);
}

__device__ __forceinline__ float lerp_backward_b(float a, float b, float w, float grad_output) {
    return grad_output * w;
}

__device__ __forceinline__ float lerp_backward_w(float a, float b, float w, float grad_output) {
    float diff = b - a;
    if (fabsf(w) < 0.5f) {
        return grad_output * diff;
    } else {
        return grad_output * (-diff) * -1.0f; // derivative of (1 - w)
    }
}

__device__ __forceinline__ float mingru_gate(float h) {
    return h >= 0.0f ? h + 0.5f : sigmoid(h);
}

__device__ __forceinline__ float mingru_gate_backward(float h, float grad_output) {
    if (h > 0.0f) {
        return grad_output * 1.0f;
    } else {
        float sig = sigmoid(h);
        return grad_output * sig * (1.0f - sig);
    }
}
