//#include "squared.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "squared.hpp"


// Device: Reset environment
__device__ void cuda_reset(Squared* env, curandState* rng) {
    int tiles = env->size * env->size;
    int center = env->size / 2 * env->size + env->size / 2;

    // Clear grid
    for (int i = 0; i < tiles; i++) {
        env->observations[i] = EMPTY;
    }

    // Place agent at center
    env->observations[center] = AGENT;
    env->r = env->size / 2;
    env->c = env->size / 2;
    env->tick = 0;

    // Place target randomly (not on agent)
    int target_idx;
    do {
        target_idx = curand(rng) % tiles;
    } while (target_idx == center);

    env->observations[target_idx] = TARGET;
}

// Device: Step environment
__device__ void cuda_step(Squared* env) {
    env->tick += 1;
    int action = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    int pos = env->r * env->size + env->c;
    env->observations[pos] = EMPTY;  // Clear old agent pos

    // Move agent
    if (action == DOWN) {
        env->r += 1;
    } else if (action == UP) {
        env->r -= 1;
    } else if (action == RIGHT) {
        env->c += 1;
    } else if (action == LEFT) {
        env->c -= 1;
    }

    pos = env->r * env->size + env->c;

    // Check bounds and timeout
    if (env->r < 0 || env->c < 0 || env->r >= env->size || env->c >= env->size ||
        env->tick > 3 * env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        env->log.perf += 0;
        env->log.score += -1.0f;
        env->log.episode_return += -1.0f;
        env->log.episode_length += env->tick;
        env->log.n += 1;
        cuda_reset(env, &env->rng);
        return;
    }

    // Check if reached target
    if (env->observations[pos] == TARGET) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        env->log.perf += 1;
        env->log.score += 1.0f;
        env->log.episode_return += 1.0f;
        env->log.episode_length += env->tick;
        env->log.n += 1;
        cuda_reset(env, &env->rng);
        return;
    }

    // Place agent
    env->observations[pos] = AGENT;
}

// Kernel: Step all environments
__global__ void step_environments(Squared* envs, int num_envs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    cuda_step(&envs[idx]);
}

// Kernel: Reset specific environments
__global__ void reset_environments(Squared* envs, int* indices, int num_reset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_reset) return;
    int env_idx = indices[idx];
    cuda_reset(&envs[env_idx], &envs[env_idx].rng);
}

// Kernel: Initialize all environments
__global__ void init_environments(Squared* envs,
                                 unsigned char* obs_mem,
                                 int* actions_mem,
                                 float* rewards_mem,
                                 unsigned char* terminals_mem,
                                 int num_envs,
                                 int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    Squared* env = &envs[idx];

    // Initialize log
    env->log.perf = 0;
    env->log.score = 0;
    env->log.episode_return = 0;
    env->log.episode_length = 0;
    env->log.n = 0;

    // Set pointers into memory pools
    env->observations = obs_mem + idx * grid_size * grid_size;
    env->actions = actions_mem + idx;
    env->rewards = rewards_mem + idx;
    env->terminals = terminals_mem + idx;

    env->size = grid_size;
    env->tick = 0;
    env->r = grid_size / 2;
    env->c = grid_size / 2;

    // Initialize RNG
    curand_init(clock64(), idx, 0, &env->rng);

    // Initial reset
    cuda_reset(env, &env->rng);
}


