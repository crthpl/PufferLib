#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

static constexpr unsigned char NOOP   = 0;
static constexpr unsigned char DOWN   = 1;
static constexpr unsigned char UP     = 2;
static constexpr unsigned char LEFT   = 3;
static constexpr unsigned char RIGHT  = 4;

static constexpr unsigned char EMPTY  = 0;
static constexpr unsigned char AGENT  = 1;
static constexpr unsigned char TARGET = 2;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

struct Squared {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    int r, c;
    curandState rng;
};

__global__ void init_environments(
    Squared* envs,
    unsigned char* obs_mem,
    int* actions_mem,
    float* rewards_mem,
    unsigned char* terminals_mem,
    int num_envs,
    int grid_size
);

__global__ void step_environments(Squared* envs, int num_envs);
__global__ void reset_environments(Squared* envs, int* indices, int num_reset);
