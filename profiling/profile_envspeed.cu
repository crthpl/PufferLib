// profiling/profile_envspeed.cu
// Environment step throughput profiling (static linked env)
#pragma once
#include "profile.h"

#ifdef USE_STATIC_ENV

#include "pufferlib/extensions/env_binding.h"
#include "pufferlib/extensions/ini.h"

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}

static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs;
    int num_buffers;
    int num_threads;
    int horizon;
    int obs_size;
    int num_atns;
} EnvSpeedArgs;

static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) {
        dict_set(env_kwargs, strdup(name), atof(value));
    }
    return 1;
}

typedef struct { int total_agents; int num_buffers; } VecDefaults;
static int ini_handler_vec(void* user, const char* section,
                           const char* name, const char* value) {
    VecDefaults* defaults = (VecDefaults*)user;
    if (strcmp(section, "vec") == 0) {
        if (strcmp(name, "total_agents") == 0) defaults->total_agents = atoi(value);
        else if (strcmp(name, "num_buffers") == 0) defaults->num_buffers = atoi(value);
    }
    return 1;
}

EnvSpeedArgs* create_envspeedargs(int total_agents, int num_buffers, int num_threads, int horizon) {
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "pufferlib/config/ocean/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    if (ini_parse(ini_path, ini_handler_vec, &defaults) < 0) {
        fprintf(stderr, "Warning: Could not load config %s\n", ini_path);
    }

    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    Dict* env_kwargs = create_dict(64);
    if (ini_parse(ini_path, ini_handler_env, env_kwargs) < 0) {
        fprintf(stderr, "Warning: Could not load [env] config from %s\n", ini_path);
    }

    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    if (!vec) {
        fprintf(stderr, "Failed to create environments\n");
        return nullptr;
    }

    int num_envs = vec->size;
    printf("Created %d envs (%s) for %d total_agents\n", num_envs, TOSTRING(ENV_NAME), total_agents);

    create_static_threads(vec, num_threads, horizon, nullptr, empty_net_callback, empty_thread_init);

    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();

    return args;
}

void free_envspeedargs(EnvSpeedArgs* args) {
    free(args);
}

void run_env_rollout(EnvSpeedArgs* args) {
    static_vec_omp_step(args->vec);
}

float profile_env_rollout(EnvSpeedArgs* args, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        run_env_rollout(args);
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > TIMEOUT_SEC) break;
    }

    start_time = std::chrono::steady_clock::now();
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        run_env_rollout(args);
        completed += 1;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > TIMEOUT_SEC) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / completed;
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);

    EnvSpeedArgs* args = create_envspeedargs(total_agents, num_buffers, num_threads, horizon);
    if (!args) {
        printf("  Failed to create env - skipping\n\n");
        return;
    }

    printf("  num_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    float rollout_ms = profile_env_rollout(args, "env_rollout");
    int total_steps = total_agents * horizon;
    printf("  rollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);

    float sps = total_steps / rollout_ms * 1000.0f;
    printf("  throughput: %.2f M steps/s\n", sps / 1e6);

    free_envspeedargs(args);
    printf("\n");
}

#endif  // USE_STATIC_ENV
