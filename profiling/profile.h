// profiling/profile.h
// Shared infrastructure for CUDA kernel profiling
//
// Build without torch: nvcc -O3 -arch=sm_80 -DPRECISION_FLOAT profiling/main.cu -o profile_v2 -I.
// Build with torch:    python setup.py build_profiler_v2 --env=<env_name>
// Run:                 ./profile_v2 <profile>

#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <chrono>


#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/cuda/kernels.cu"
#include "pufferlib/extensions/modules.cu"

// models.cpp provides Policy, MinGRU, DefaultEncoder, DefaultDecoder,
// reference impls, PRECISION_DTYPE, cuda_f32/f64/i32/i64, Tensor typedef
namespace pufferlib {
#include "pufferlib/extensions/models.cpp"
}
using namespace pufferlib;

// Muon optimizer (production uses Muon, not Adam)
#include "pufferlib/extensions/muon.h"

// Advantage computation (puff_advantage CUDA kernel)
// Redefine TORCH_LIBRARY_IMPL to skip dispatch registration —
// profiler calls compute_puff_advantage_cuda directly, no dispatch needed.
#undef TORCH_LIBRARY_IMPL
#define TORCH_LIBRARY_IMPL(ns, k, m) \
    static void _profiler_noop_advantage([[maybe_unused]] torch::Library& m)
#include "pufferlib/extensions/cuda/advantage.cu"

#endif

#ifndef USE_TORCH
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

// ============================================================================
// Constants
// ============================================================================

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;
const float TIMEOUT_SEC = 3.0f;
const int BATCH_CHECK = 100;  // Check timeout every BATCH_CHECK iterations

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;
const int INPUT_SIZE = 96;

typedef void (*kernel_fn)(void*);

// ============================================================================
// Helpers
// ============================================================================

inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

inline void print_timing_pct(const char* name, float ms, int N, float total_ms) {
    float pct = (total_ms > 0) ? (ms / total_ms * 100.0f) : 0.0f;
    printf("  %-28s %8.1f us  %5.1f%%\n", name, ms * 1000, pct);
}

inline float get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9f;
}

inline void warmup_gpu() {
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 100; i++) {
        cudaMemset(dummy, 0, 64 * 1024 * 1024);
    }
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

inline float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Safe host-to-device copy: converts float[] → precision_t[] before cudaMemcpy.
// Prevents buffer overflow when precision_t is bf16 (2 bytes) but source is float (4 bytes).
inline void float_to_device(precision_t* dst, const float* src, int count) {
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; ++i) tmp[i] = (precision_t)src[i];
    cudaMemcpy(dst, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

// ============================================================================
// Profiling harness
// ============================================================================

float profile_kernel(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup with timeout
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_CHECK == 0) {
            cudaDeviceSynchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    cudaDeviceSynchronize();

    // Timed runs with timeout
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_CHECK; ++i) {
            fn(args);
        }
        iters += BATCH_CHECK;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
#ifdef USE_TORCH
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return ms / iters;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    // Warmup with timeout
    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    float warmup_start = get_time_sec();
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fn(args);
        if (i % BATCH_CHECK == 0) {
            warmup_stream.synchronize();
            if (get_time_sec() - warmup_start > TIMEOUT_SEC) break;
        }
    }
    warmup_stream.synchronize();

    // Capture graph
    at::cuda::CUDAStream cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    cuda_graph.capture_begin();
    fn(args);
    cuda_graph.capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();
    at::cuda::setCurrentCUDAStream(current_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timed runs with timeout
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    float timing_start = get_time_sec();
    long iters = 0;
    float elapsed = 0;
    while (elapsed < TIMEOUT_SEC) {
        for (int i = 0; i < BATCH_CHECK; ++i) {
            cuda_graph.replay();
        }
        iters += BATCH_CHECK;
        cudaDeviceSynchronize();
        elapsed = get_time_sec() - timing_start;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return ms / iters;
}
#endif
