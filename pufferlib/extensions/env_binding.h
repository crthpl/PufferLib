#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "vecenv.h"
#include <stdatomic.h>

#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3

__attribute__((visibility("default"))) const int OBS_N = OBS_SIZE;
__attribute__((visibility("default"))) const int ACT_N = ACT_SIZE;
__attribute__((visibility("default"))) const int OBS_T = OBS_TYPE;
__attribute__((visibility("default"))) const int ACT_T = ACT_TYPE;

typedef struct Threading {
    atomic_long work_index;
    long start_index;
    atomic_long end_index;
    int num_threads;
    pthread_cond_t wake_cond;
    pthread_mutex_t wake_mutex;
    pthread_cond_t all_done_cond;
    pthread_mutex_t all_done_mutex;
    pthread_t* threads;
    bool* actions_ready_on_gpu;
    bool* obs_ready_on_cpu;
} Threading;

// Forward declarations for env-specific functions supplied by user
void my_log(Log* log, Dict* out);
void my_init(Env* env, Dict* args);

void* my_shared(Env* env, Dict* kwargs);
#ifndef MY_SHARED
void* my_shared(Env* env, Dict* kwargs) {
    return NULL;
}
#endif

void my_shared_close(Env* env);
#ifndef MY_SHARED_CLOSE
void my_shared_close(Env* env) {}
#endif

void* my_get(Env* env, Dict* out);
#ifndef MY_GET
void* my_get(Env* env, Dict* out) {
    return NULL;
}
#endif

int my_put(Env* env, Dict* kwargs);
#ifndef MY_PUT
int my_put(Env* env, Dict* kwargs) {
    return 0;
}
#endif

static void* c_threadstep(void* arg)
{
    VecEnv* vec = (VecEnv*)arg;
    Threading* threading = vec->threading;

    int num_envs = vec->size;
    atomic_long* work_index = &threading->work_index;
    atomic_long* end_index = &threading->end_index;
    int index;
    while (1) {
        // Wait for work
        pthread_mutex_lock(&threading->wake_mutex);
        pthread_cond_wait(&threading->wake_cond, &threading->wake_mutex);
        pthread_mutex_unlock(&threading->wake_mutex);
        int end = atomic_load_explicit(end_index, memory_order_relaxed);
        do
        {
            // This is important: Go do a bunch of work in our thread, without context switches or locks
            // or any new allocs. This is the main speedup and core to ensuring the threads do as little work
            // as part of their main loop as possible. We can afford to this as the load balancing naturally happens
            // with mutually exclusive index values spread across threads.

            index = atomic_fetch_add_explicit(work_index, 1, memory_order_relaxed);
            if (index < end) {
                c_step(&vec->envs[index % num_envs]);
            }
        }
        while (index < end);
    }
    return NULL;
}

static void* c_threadmanager(void* arg) {
    VecEnv* vec = (VecEnv*)arg;
    Threading* threading = vec->threading;

    atomic_long* work_index = &threading->work_index;
    atomic_long* end_index = &threading->end_index;
 
    int block_size = vec->size / vec->buffers;
    bool* actions_ready_on_gpu = threading->actions_ready_on_gpu;
    bool* obs_ready_on_cpu = threading->obs_ready_on_cpu;

    printf("Thread manager initialized\n");
    // TODO: Init?
    while (1) {
        for (int buf=0; buf < vec->buffers; buf++) {
            if (threading->actions_ready_on_gpu[buf] && cudaStreamQuery(vec->streams[buf]) == cudaSuccess) {
                printf("Actions ready on CPU\n");
                // Actions are ready on CPU
                atomic_fetch_add_explicit(end_index, block_size, memory_order_relaxed);
                pthread_cond_broadcast(&threading->wake_cond);
                actions_ready_on_gpu[buf] = false;
            }

            // TODO: race?
            int work = atomic_load_explicit(work_index, memory_order_relaxed);
            if ( work >= block_size + threading->start_index) {
                threading->start_index += block_size;
                printf("Observations ready on CPU\n");

                // Observations are ready on CPU
                int block_size = vec->size / vec->buffers;
                int start = buf * block_size;

                cudaMemcpyAsync(
                    &vec->gpu_observations[start],
                    &vec->observations[start],
                    block_size*OBS_SIZE*sizeof(float),
                    cudaMemcpyHostToDevice,
                    vec->streams[buf]
                );
                cudaMemcpyAsync(
                    &vec->gpu_rewards[start],
                    &vec->rewards[start],
                    block_size*sizeof(float),
                    cudaMemcpyHostToDevice,
                    vec->streams[buf]
                );
                cudaMemcpyAsync(
                    &vec->gpu_terminals[start],
                    &vec->terminals[start],
                    block_size*sizeof(unsigned char),
                    cudaMemcpyHostToDevice,
                    vec->streams[buf]
                );
                obs_ready_on_cpu[buf] = true;
            }
        }
    }
}
 
__attribute__((visibility("default")))
VecEnv* create_environments(int num_envs, int threads, int buffers, Dict* kwargs) {
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->envs = envs;
    vec->size = num_envs;
    vec->buffers = buffers;
    vec->threading = calloc(1, sizeof(Threading));

    Threading* threading = vec->threading;
    threading->num_threads = threads;
    threading->actions_ready_on_gpu = (bool*)calloc(threads, sizeof(bool));
    threading->obs_ready_on_cpu = (bool*)calloc(threads, sizeof(bool));

    if (threads > 0) {
        vec->streams = (cudaStream_t*)calloc(buffers, sizeof(cudaStream_t));
        for (int i = 0; i < buffers; i++) {
            cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
        }

        threading->threads = (pthread_t*)calloc(threads + 1, sizeof(pthread_t));
        assert(threading->threads != NULL && "create_vecenv failed to allocate memory for threads\n");
        assert(pthread_cond_init(&threading->wake_cond, NULL) == 0 && "create_vecenv failed to initialize wake_cond\n");
        assert(pthread_mutex_init(&threading->wake_mutex, NULL) == 0 && "create_vecenv failed to initialize wake_mutex\n");
        atomic_store(&threading->end_index, 0);
        atomic_store(&threading->work_index, 0);

        for (int i = 0; i < threads; i++) {
            int err = pthread_create(&threading->threads[i], NULL, c_threadstep, (void*)(vec));
            assert(err == 0 && "create_vecenv failed to create thread\n");
        }

        // Last thread manages host device syncs
        int err = pthread_create(&threading->threads[threads], NULL, c_threadmanager, (void*)(vec));
        assert(err == 0 && "create_vecenv failed to create manager thread\n");
    }

    int num_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        srand(i);
        my_init(&envs[i], kwargs);
        //num_agents += envs[i].num_agents;
        num_agents += 1;
    }

    /*
    vec->observations = (float*)calloc(num_agents*OBS_SIZE, sizeof(float));
    vec->actions = (float*)calloc(num_agents*ACT_SIZE, sizeof(float));
    vec->rewards = (float*)calloc(num_agents, sizeof(float));
    vec->terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    */
    cudaHostAlloc((void**)&vec->observations, num_agents*OBS_SIZE*sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->actions, num_agents*ACT_SIZE*sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->rewards, num_agents*sizeof(float), cudaHostAllocPortable);
    cudaHostAlloc((void**)&vec->terminals, num_agents*sizeof(unsigned char), cudaHostAllocPortable);
    memset(vec->observations, 0, num_agents*OBS_SIZE*sizeof(float));
    memset(vec->actions, 0, num_agents*ACT_SIZE*sizeof(float));
    memset(vec->rewards, 0, num_agents*sizeof(float));
    memset(vec->terminals, 0, num_agents*sizeof(unsigned char));

    cudaMalloc((void**)&vec->gpu_observations, num_agents*OBS_SIZE*sizeof(float));
    cudaMalloc((void**)&vec->gpu_actions, num_agents*ACT_SIZE*sizeof(float));
    cudaMalloc((void**)&vec->gpu_rewards, num_agents*sizeof(float));
    cudaMalloc((void**)&vec->gpu_terminals, num_agents*sizeof(unsigned char));

    int agent = 0;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->observations = vec->observations + agent*OBS_SIZE;
        env->actions = vec->actions + agent*ACT_SIZE;
        env->rewards = vec->rewards + agent;
        env->terminals = vec->terminals + agent;
        //agent += env->num_agents;
        agent += 1;
    }

    return vec;
}

Env* env_init(float* observations, float* actions, float* rewards,
        unsigned char* terminals, int seed, Dict* kwargs) {
    Env* env = (Env*)calloc(1, sizeof(Env));
    assert(env != NULL && "env_init failed to allocated memory\n");

    // TODO: Types can vary
    env->observations = observations;
    env->actions = actions;
    env->rewards = rewards;
    env->terminals = terminals;

    srand(seed);
    my_init(env, kwargs);
    return env;
}

void vec_reset(VecEnv* vec) {
    for (int i = 0; i < vec->size; i++) {
        Env* env = &vec->envs[i];
        c_reset(env);
    }
    bool* obs_ready_on_cpu = vec->threading->obs_ready_on_cpu;
    for (int buf=0; buf < vec->buffers; buf++) {
        int block_size = vec->size / vec->buffers;
        int start = buf * block_size;

        cudaMemcpy(
            &vec->gpu_observations[start],
            &vec->observations[start],
            block_size*OBS_SIZE*sizeof(float),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            &vec->gpu_rewards[start],
            &vec->rewards[start],
            block_size*sizeof(float),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            &vec->gpu_terminals[start],
            &vec->terminals[start],
            block_size*sizeof(unsigned char),
            cudaMemcpyHostToDevice
        );
        obs_ready_on_cpu[buf] = true;
    }
}

void vec_send(VecEnv* vec, int buffer) {
    int block_size = vec->size / vec->buffers;
    int start = buffer * block_size;

    cudaMemcpyAsync(
        &vec->actions[start],
        &vec->gpu_actions[start],
        block_size*ACT_SIZE*sizeof(float),
        cudaMemcpyDeviceToHost,
        vec->streams[buffer]
    );

    Threading* threading = vec->threading;
    bool* actions_ready_on_gpu = threading->actions_ready_on_gpu;
    actions_ready_on_gpu[buffer] = true;

    // Single threaded
    if (threading->num_threads == 0) {
        for (int i = 0; i < vec->size; i++) {
            Env* env = &vec->envs[i];
            c_step(env);
        }
        return;
    }
}

void vec_recv(VecEnv* vec, int buffer) {
    Threading* threading = vec->threading;
    // TODO: Single stream architecture requires busy waiting here
    while (!threading->obs_ready_on_cpu[buffer]) {}
    cudaStreamSynchronize(vec->streams[buffer]);
    threading->obs_ready_on_cpu[buffer] = false;
}

void vec_step(VecEnv* vec, int buffer) {
    vec_send(vec, buffer);
    vec_recv(vec, buffer);
}

void env_close(Env* env) {
    c_close(env);
    free(env);
}

void vec_close(VecEnv* vec) {
    for (int i = 0; i < vec->size; i++) {
        Env* env = &vec->envs[i];
        c_close(env);
    }
    free(vec->envs);
}

void vec_render(VecEnv* vec, int env_idx) {
    Env* env = &vec->envs[env_idx];
    c_render(env);
}

void vec_log(VecEnv* vec, Dict* out) {
    Log aggregate = {0};
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec->size; i++) {
        Env* env = &vec->envs[i];
        for (int j = 0; j < num_keys; j++) {
            ((float*)&aggregate)[j] += ((float*)&env->log)[j];
            ((float*)&env->log)[j] = 0.0f;
        }
    }

    if (aggregate.n == 0.0f) {
        return;
    }

    // Average
    float n = aggregate.n;
    for (int i = 0; i < num_keys; i++) {
        ((float*)&aggregate)[i] /= n;
    }

    // User populates dict
    dict_set_float(out, "n", n);
    my_log(&aggregate, out);
}
