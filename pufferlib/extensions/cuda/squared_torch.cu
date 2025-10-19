#include <torch/extension.h>
#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

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

struct __align__(16) Squared {
    curandState rng;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int size;
    int tick;
    int r, c;
    Log log;
    int padding[3];
};


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
    /*
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
    */
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


inline dim3 make_grid(int n) {
    return dim3((n + 255) / 256);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_squared_environments(int64_t num_envs, int64_t grid_size, torch::Tensor dummy) {
    auto device = dummy.device();
    TORCH_CHECK(device.type() == at::kCUDA, "Dummy tensor must be on CUDA device");

    auto envs_tensor = torch::empty({static_cast<int64_t>(num_envs * sizeof(Squared))}, torch::kUInt8).to(device);
    auto obs = torch::zeros({num_envs, grid_size, grid_size}, torch::kUInt8).to(device);
    auto actions = torch::zeros({num_envs}, torch::kInt32).to(device);
    auto rewards = torch::zeros({num_envs}, torch::kFloat32).to(device);
    auto terminals = torch::zeros({num_envs}, torch::kUInt8).to(device);

    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());

    init_environments<<<make_grid(num_envs), 256>>>(
        envs,
        obs.data_ptr<unsigned char>(),
        actions.data_ptr<int>(),
        rewards.data_ptr<float>(),
        terminals.data_ptr<unsigned char>(),
        num_envs,
        grid_size
    );
    cudaDeviceSynchronize();

    return std::make_tuple(envs_tensor, obs, actions, rewards, terminals);
}

void step_environments_cuda(torch::Tensor envs_tensor) {
    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());
    // YOU HARDCODED THIS HERE
    int num_envs = 2048;

    step_environments<<<make_grid(num_envs), 256>>>(envs, num_envs);
    cudaDeviceSynchronize();
}

void reset_environments_cuda(torch::Tensor envs_tensor, torch::Tensor indices_tensor) {
    Squared* envs = reinterpret_cast<Squared*>(envs_tensor.data_ptr<unsigned char>());
    auto indices = indices_tensor.data_ptr<int>();
    int num_reset = indices_tensor.size(0);

    reset_environments<<<make_grid(num_reset), 256>>>(envs, indices, num_reset);
    cudaDeviceSynchronize();
}

TORCH_LIBRARY(squared, m) {
    m.def("create_squared_environments(int num_envs, int grid_size, Tensor dummy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("step_environments(Tensor envs) -> ()");
    m.def("reset_environments(Tensor envs, Tensor indices) -> ()");
}

TORCH_LIBRARY_IMPL(squared, CUDA, m) {
    m.impl("create_squared_environments", &create_squared_environments);
    m.impl("step_environments", &step_environments_cuda);
    m.impl("reset_environments", &reset_environments_cuda);
}
