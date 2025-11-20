#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "vecenv.h"

#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3

__attribute__((visibility("default"))) const int OBS_N = OBS_SIZE;
__attribute__((visibility("default"))) const int ACT_N = ACT_SIZE;
__attribute__((visibility("default"))) const int OBS_T = OBS_TYPE;
__attribute__((visibility("default"))) const int ACT_T = ACT_TYPE;

typedef struct Threading {
    atomic_int work_index;
    atomic_int num_running_threads;
    int num_threads;
    pthread_cond_t wake_cond;
    pthread_mutex_t wake_mutex;
    pthread_cond_t all_done_cond;
    pthread_mutex_t all_done_mutex;
    pthread_t* threads;
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

    atomic_int* work_index = &threading->work_index;
    atomic_int* num_running_threads = &threading->num_running_threads;
    int index;
    atomic_fetch_add_explicit(num_running_threads, 1, memory_order_relaxed);
    while (1) {
        // Wait for work
        pthread_mutex_lock(&threading->wake_mutex);
        pthread_cond_wait(&threading->wake_cond, &threading->wake_mutex);
        pthread_mutex_unlock(&threading->wake_mutex);
        do
        {
            // This is important: Go do a bunch of work in our thread, without context switches or locks
            // or any new allocs. This is the main speedup and core to ensuring the threads do as little work
            // as part of their main loop as possible. We can afford to this as the load balancing naturally happens
            // with mutually exclusive index values spread across threads.
            index = atomic_fetch_sub_explicit(work_index, 1, memory_order_relaxed);
            if (index >= 0) {
                c_step(&vec->envs[index]);
            }
        }
        while (index > 0);
        if (atomic_fetch_sub_explicit(num_running_threads, 1, memory_order_relaxed) == 1) {
            pthread_mutex_lock(&threading->all_done_mutex);
            pthread_cond_signal(&threading->all_done_cond);
            pthread_mutex_unlock(&threading->all_done_mutex);
        }
    }
    return NULL;
}

__attribute__((visibility("default")))
VecEnv* create_environments(int num_envs, int threads, Dict* kwargs) {
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->envs = envs;
    vec->size = num_envs;
    vec->threading = calloc(1, sizeof(Threading));

    Threading* threading = vec->threading;
    threading->num_threads = threads;

    if (threads > 0) {
        threading->threads = (pthread_t*)calloc(threads, sizeof(pthread_t));
        assert(threading->threads != NULL && "create_vecenv failed to allocate memory for threads\n");
        assert(pthread_cond_init(&threading->wake_cond, NULL) == 0 && "create_vecenv failed to initialize wake_cond\n");
        assert(pthread_mutex_init(&threading->wake_mutex, NULL) == 0 && "create_vecenv failed to initialize wake_mutex\n");
        atomic_store(&threading->num_running_threads, 0);
        atomic_store(&threading->work_index, -1);

        for (int i = 0; i < threads; i++) {
            int err = pthread_create(&threading->threads[i], NULL, c_threadstep, (void*)(vec));
            assert(err == 0 && "create_vecenv failed to create thread\n");
        }

        // Wait for all threads to initialize
        while (atomic_load_explicit(&threading->num_running_threads, memory_order_relaxed) < threading->num_threads) {}
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
}

void vec_step(VecEnv* vec) {
    // Single threaded
    Threading* threading = vec->threading;
    if (threading->num_threads == 0) {
        for (int i = 0; i < vec->size; i++) {
            Env* env = &vec->envs[i];
            c_step(env);
        }
        return;
    }

    atomic_store_explicit(&threading->num_running_threads, threading->num_threads, memory_order_relaxed);
    atomic_store_explicit(&threading->work_index, vec->size-1, memory_order_relaxed);
    pthread_cond_broadcast(&threading->wake_cond);
    pthread_mutex_lock(&threading->all_done_mutex);
    while (atomic_load_explicit(&threading->num_running_threads, memory_order_relaxed) > 0) {
        pthread_cond_wait(&threading->all_done_cond, &threading->all_done_mutex);
    }
    pthread_mutex_unlock(&threading->all_done_mutex);
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
