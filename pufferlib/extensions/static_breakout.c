// static_breakout.c - Compiled with clang into libstatic_breakout.a
// Then linked into the torch extension

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <omp.h>
#include <stdatomic.h>
#include <pthread.h>
#include <cuda_runtime.h>

// Dict from vecenv.h
typedef struct {
    const char* key;
    double value;
    void* ptr;
} DictItem;

typedef struct {
    DictItem* items;
    int size;
    int capacity;
} Dict;

static inline DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item == NULL) printf("dict_get failed to find key: %s\n", key);
    assert(item != NULL);
    return item;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    assert(dict->size < dict->capacity);
    DictItem* item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->value = value;
        return;
    }
    dict->items[dict->size].key = key;
    dict->items[dict->size].value = value;
    dict->size++;
}

// Breakout config
#define OBS_SIZE 118
#define NUM_ATNS 1

// Include breakout env
#include "../ocean/breakout/breakout.h"

#define Env Breakout

// Init for breakout
void breakout_env_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->frameskip = (int)dict_get(kwargs, "frameskip")->value;
    env->width = (int)dict_get(kwargs, "width")->value;
    env->height = (int)dict_get(kwargs, "height")->value;
    env->initial_paddle_width = (int)dict_get(kwargs, "paddle_width")->value;
    env->paddle_height = (int)dict_get(kwargs, "paddle_height")->value;
    env->ball_width = (int)dict_get(kwargs, "ball_width")->value;
    env->ball_height = (int)dict_get(kwargs, "ball_height")->value;
    env->brick_width = (int)dict_get(kwargs, "brick_width")->value;
    env->brick_height = (int)dict_get(kwargs, "brick_height")->value;
    env->brick_rows = (int)dict_get(kwargs, "brick_rows")->value;
    env->brick_cols = (int)dict_get(kwargs, "brick_cols")->value;
    env->initial_ball_speed = (int)dict_get(kwargs, "initial_ball_speed")->value;
    env->max_ball_speed = (int)dict_get(kwargs, "max_ball_speed")->value;
    env->paddle_speed = (int)dict_get(kwargs, "paddle_speed")->value;
    env->continuous = (int)dict_get(kwargs, "continuous")->value;
    init(env);
}

// Log for breakout
void breakout_env_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

// Threading state
#define OMP_WAITING 5
#define OMP_RUNNING 6

typedef void (*net_callback_fn)(void* ctx, int buf, int t);
typedef void (*thread_init_fn)(void* ctx, int buf);

typedef struct StaticThreading {
    atomic_int* buffer_states;
    int num_threads;
    int num_buffers;
    pthread_t* threads;
} StaticThreading;

typedef struct StaticOMPArg {
    void* vec;
    int buf;
    int horizon;
    void* ctx;
    net_callback_fn net_callback;
    thread_init_fn thread_init;
} StaticOMPArg;

// Minimal VecEnv for static breakout
typedef struct StaticVec {
    Env* envs;
    int size;
    int total_agents;
    int buffers;
    int agents_per_buffer;
    int* buffer_env_starts;
    int* buffer_env_counts;
    float* observations;
    double* actions;
    float* rewards;
    float* terminals;
    float* gpu_observations;
    double* gpu_actions;
    float* gpu_rewards;
    float* gpu_terminals;
    cudaStream_t* streams;
    StaticThreading* threading;
} StaticVec;

// OMP thread manager
static void* static_omp_threadmanager(void* arg) {
    StaticOMPArg* worker_arg = (StaticOMPArg*)arg;
    StaticVec* vec = (StaticVec*)worker_arg->vec;
    StaticThreading* threading = vec->threading;
    int buf = worker_arg->buf;
    int horizon = worker_arg->horizon;
    void* ctx = worker_arg->ctx;
    net_callback_fn net_callback = worker_arg->net_callback;
    thread_init_fn thread_init = worker_arg->thread_init;

    if (thread_init != NULL) {
        thread_init(ctx, buf);
    }

    int agents_per_buffer = vec->agents_per_buffer;
    int agent_start = buf * agents_per_buffer;
    int env_start = vec->buffer_env_starts[buf];
    int env_count = vec->buffer_env_counts[buf];
    atomic_int* buffer_states = threading->buffer_states;
    int num_workers = threading->num_threads / vec->buffers;
    if (num_workers < 1) num_workers = 1;

    while (1) {
        while (atomic_load(&buffer_states[buf]) != OMP_RUNNING) {}
        cudaStream_t stream = vec->streams[buf];

        for (int t = 0; t < horizon; t++) {
            net_callback(ctx, buf, t);

            cudaMemcpyAsync(
                &vec->actions[agent_start * NUM_ATNS],
                &vec->gpu_actions[agent_start * NUM_ATNS],
                agents_per_buffer * NUM_ATNS * sizeof(double),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            #pragma omp parallel for schedule(static) num_threads(num_workers)
            for (int i = env_start; i < env_start + env_count; i++) {
                c_step(&vec->envs[i]);
            }

            cudaMemcpyAsync(
                &vec->gpu_observations[agent_start * OBS_SIZE],
                &vec->observations[agent_start * OBS_SIZE],
                agents_per_buffer * OBS_SIZE * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_rewards[agent_start],
                &vec->rewards[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(
                &vec->gpu_terminals[agent_start],
                &vec->terminals[agent_start],
                agents_per_buffer * sizeof(float),
                cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);
        atomic_store(&buffer_states[buf], OMP_WAITING);
    }
}

void static_vec_omp_step(StaticVec* vec) {
    StaticThreading* threading = vec->threading;
    for (int buf = 0; buf < vec->buffers; buf++) {
        atomic_store(&threading->buffer_states[buf], OMP_RUNNING);
    }
    for (int buf = 0; buf < vec->buffers; buf++) {
        while (atomic_load(&threading->buffer_states[buf]) != OMP_WAITING) {}
    }
}

StaticVec* create_static_vec(int total_agents, int num_buffers, Dict* env_kwargs) {
    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    vec->total_agents = total_agents;
    vec->buffers = num_buffers;
    vec->agents_per_buffer = total_agents / num_buffers;
    vec->size = total_agents;

    vec->envs = (Env*)calloc(total_agents, sizeof(Env));
    vec->buffer_env_starts = (int*)calloc(num_buffers, sizeof(int));
    vec->buffer_env_counts = (int*)calloc(num_buffers, sizeof(int));

    int envs_per_buffer = total_agents / num_buffers;
    for (int b = 0; b < num_buffers; b++) {
        vec->buffer_env_starts[b] = b * envs_per_buffer;
        vec->buffer_env_counts[b] = envs_per_buffer;
    }

    cudaHostAlloc((void**)&vec->observations, total_agents * OBS_SIZE * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, total_agents * NUM_ATNS * sizeof(double), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, total_agents * sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, total_agents * sizeof(float), cudaHostAllocPortable);

    cudaMalloc((void**)&vec->gpu_observations, total_agents * OBS_SIZE * sizeof(float));
    cudaMalloc((void**)&vec->gpu_actions, total_agents * NUM_ATNS * sizeof(double));
    cudaMalloc((void**)&vec->gpu_rewards, total_agents * sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, total_agents * sizeof(float));

    cudaMemset(vec->gpu_observations, 0, total_agents * OBS_SIZE * sizeof(float));
    cudaMemset(vec->gpu_actions, 0, total_agents * NUM_ATNS * sizeof(double));
    cudaMemset(vec->gpu_rewards, 0, total_agents * sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, total_agents * sizeof(float));

    vec->streams = (cudaStream_t*)calloc(num_buffers, sizeof(cudaStream_t));
    for (int i = 0; i < num_buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }

    for (int i = 0; i < total_agents; i++) {
        Env* env = &vec->envs[i];
        env->observations = vec->observations + i * OBS_SIZE;
        env->actions = vec->actions + i * NUM_ATNS;
        env->rewards = vec->rewards + i;
        env->terminals = vec->terminals + i;
        srand(i);
        breakout_env_init(env, env_kwargs);
    }

    return vec;
}

void static_vec_reset(StaticVec* vec) {
    for (int i = 0; i < vec->size; i++) {
        c_reset(&vec->envs[i]);
    }
    cudaMemcpy(vec->gpu_observations, vec->observations,
        vec->total_agents * OBS_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_rewards, vec->rewards,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec->gpu_terminals, vec->terminals,
        vec->total_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void create_static_threads(StaticVec* vec, int num_threads, int horizon,
        void* ctx, net_callback_fn net_callback, thread_init_fn thread_init) {
    vec->threading = (StaticThreading*)calloc(1, sizeof(StaticThreading));
    vec->threading->num_threads = num_threads;
    vec->threading->num_buffers = vec->buffers;
    vec->threading->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    vec->threading->threads = (pthread_t*)calloc(vec->buffers, sizeof(pthread_t));

    StaticOMPArg* args = (StaticOMPArg*)calloc(vec->buffers, sizeof(StaticOMPArg));
    for (int i = 0; i < vec->buffers; i++) {
        args[i].vec = vec;
        args[i].buf = i;
        args[i].horizon = horizon;
        args[i].ctx = ctx;
        args[i].net_callback = net_callback;
        args[i].thread_init = thread_init;
        pthread_create(&vec->threading->threads[i], NULL, static_omp_threadmanager, &args[i]);
    }
}

void static_vec_log(StaticVec* vec, Dict* out) {
    Log aggregate = {0};
    for (int i = 0; i < vec->size; i++) {
        aggregate.perf += vec->envs[i].log.perf;
        aggregate.score += vec->envs[i].log.score;
        aggregate.episode_return += vec->envs[i].log.episode_return;
        aggregate.episode_length += vec->envs[i].log.episode_length;
        aggregate.n += vec->envs[i].log.n;
        memset(&vec->envs[i].log, 0, sizeof(Log));
    }
    if (aggregate.n > 0) {
        float n = aggregate.n;
        aggregate.perf /= n;
        aggregate.score /= n;
        aggregate.episode_return /= n;
        aggregate.episode_length /= n;
        dict_set(out, "n", n);
        breakout_env_log(&aggregate, out);
    }
}
