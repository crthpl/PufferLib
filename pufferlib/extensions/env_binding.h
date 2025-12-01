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

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s: %s (error %d)\n",    \
                    #call, cudaGetErrorString(err), (int)err);      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0)

__attribute__((visibility("default"))) const int OBS_N = OBS_SIZE;
__attribute__((visibility("default"))) const int ACT_N = ACT_SIZE;
__attribute__((visibility("default"))) const int OBS_T = OBS_TYPE;
__attribute__((visibility("default"))) const int ACT_T = ACT_TYPE;

typedef struct Threading {
    atomic_long work_index;
    atomic_long completed_index;
    long start_index;
    atomic_long end_index;
    int num_threads;
    pthread_cond_t wake_cond;
    pthread_mutex_t wake_mutex;
    pthread_cond_t all_done_cond;
    pthread_mutex_t all_done_mutex;
    pthread_t* threads;
    atomic_bool* actions_ready_on_gpu;
    atomic_bool* obs_ready_on_cpu;
    int block_size;
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

    int block_size = threading->block_size;
    int num_envs = vec->size;
    atomic_long* work_index = &threading->work_index;
    atomic_long* completed_index = &threading->completed_index;
    atomic_long* end_index = &threading->end_index;
    int index, end;
    while (1) {
        // Wait for work
        pthread_mutex_lock(&threading->wake_mutex);
	//printf("worker waiting on wake\n");
	if (atomic_load(work_index) >= atomic_load(end_index)){
            pthread_cond_wait(&threading->wake_cond, &threading->wake_mutex);
	}
	//printf("worker woke up\n");
        pthread_mutex_unlock(&threading->wake_mutex);
        end = atomic_load_explicit(end_index, memory_order_relaxed);
        do
        {
            // This is important: Go do a bunch of work in our thread, without context switches or locks
            // or any new allocs. This is the main speedup and core to ensuring the threads do as little work
            // as part of their main loop as possible. We can afford to this as the load balancing naturally happens
            // with mutually exclusive index values spread across threads.

            index = atomic_fetch_add_explicit(work_index, block_size, memory_order_relaxed);
	        //printf("Index %d, End %d\n", index, end);
            if (index < end) {
                for (int i = index; i < index + block_size; i++) {
                    c_step(&vec->envs[i % num_envs]);
                }
		        atomic_fetch_add_explicit(completed_index, block_size, memory_order_relaxed);
            } else {
		        atomic_fetch_sub_explicit(work_index, block_size, memory_order_relaxed);
	        } 
        }
        while (index < end - 1);
	//printf("Completed Index %d, End %d\n", index, end);
    }
    return NULL;
}

static void* c_threadmanager(void* arg) {
    VecEnv* vec = (VecEnv*)arg;
    Threading* threading = vec->threading;

    atomic_long* completed_index = &threading->completed_index;
    atomic_long* end_index = &threading->end_index;
 
    int block_size = vec->size / vec->buffers;
    atomic_bool* actions_ready_on_gpu = threading->actions_ready_on_gpu;
    atomic_bool* obs_ready_on_cpu = threading->obs_ready_on_cpu;

    //printf("Thread manager initialized\n");
    // TODO: Init?
    int done_buf = 0;
    while (1) {
        for (int buf=0; buf < vec->buffers; buf++) {
            if (atomic_load(&actions_ready_on_gpu[buf]) && cudaStreamQuery(vec->streams[buf]) == cudaSuccess) {
                // Actions are ready on CPU
                atomic_fetch_add_explicit(end_index, block_size, memory_order_relaxed);
                //printf("Buffer %d Actions ready on CPU. end idx: %d \n", buf, atomic_load(end_index));
                pthread_cond_broadcast(&threading->wake_cond);
		atomic_store(&actions_ready_on_gpu[buf], false);
            }

	    // Note: you can skip blocks (I think?) if you have waaay too many threads or only a few envs
            int completed = atomic_load_explicit(completed_index, memory_order_relaxed);
            if ( completed >= block_size + threading->start_index) {
                //printf("Buffer %d Observations ready on CPU. start_index = %d, completed = %d\n", done_buf, threading->start_index, completed);
                threading->start_index += block_size;

                // Observations are ready on CPU
                int block_size = vec->size / vec->buffers;
                int start = done_buf * block_size;

                cudaMemcpyAsync(
                    &vec->gpu_observations[start],
                    &vec->observations[start],
                    block_size*OBS_SIZE*sizeof(float),
                    cudaMemcpyHostToDevice,
                    vec->streams[done_buf]
                );
                cudaMemcpyAsync(
                    &vec->gpu_rewards[start],
                    &vec->rewards[start],
                    block_size*sizeof(float),
                    cudaMemcpyHostToDevice,
                    vec->streams[done_buf]
                );
                cudaMemcpyAsync(
                    &vec->gpu_terminals[start],
                    &vec->terminals[start],
                    block_size*sizeof(unsigned char),
                    cudaMemcpyHostToDevice,
                    vec->streams[done_buf]
                );
		atomic_store(&obs_ready_on_cpu[done_buf], true);
		done_buf = (done_buf + 1) % vec->buffers;
            }
        }
    }
}
 
__attribute__((visibility("default")))
VecEnv* create_environments(int num_envs, int threads, int buffers, int block_size, Dict* kwargs) {
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->envs = envs;
    vec->size = num_envs;
    vec->buffers = buffers;
    vec->threading = calloc(1, sizeof(Threading));

    Threading* threading = vec->threading;
    threading->num_threads = threads;
    threading->block_size = block_size;
    threading->actions_ready_on_gpu = (atomic_bool*)calloc(buffers, sizeof(bool));
    threading->obs_ready_on_cpu = (atomic_bool*)calloc(buffers, sizeof(bool));

    vec->streams = (cudaStream_t*)calloc(buffers, sizeof(cudaStream_t));
    for (int i = 0; i < buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }

    if (threads > 0) {
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
    //printf("Size of alloc: %d\n", num_agents*OBS_SIZE*sizeof(float));
    cudaSetDevice(0);
    CHECK_CUDA(cudaHostAlloc((void**)&vec->observations, num_agents*OBS_SIZE*sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&vec->actions, num_agents*ACT_SIZE*sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&vec->rewards, num_agents*sizeof(float), cudaHostAllocPortable));
    CHECK_CUDA(cudaHostAlloc((void**)&vec->terminals, num_agents*sizeof(unsigned char), cudaHostAllocPortable));
    memset(vec->observations, 0, num_agents*OBS_SIZE*sizeof(float));
    memset(vec->actions, 0, num_agents*ACT_SIZE*sizeof(float));
    memset(vec->rewards, 0, num_agents*sizeof(float));
    memset(vec->terminals, 0, num_agents*sizeof(unsigned char));

    CHECK_CUDA(cudaMalloc((void**)&vec->gpu_observations, num_agents*OBS_SIZE*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&vec->gpu_actions, num_agents*ACT_SIZE*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&vec->gpu_rewards, num_agents*sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&vec->gpu_terminals, num_agents*sizeof(unsigned char)));
    cudaMemset(vec->gpu_observations, 0, num_agents*OBS_SIZE*sizeof(float));
    cudaMemset(vec->gpu_actions, 0, num_agents*ACT_SIZE*sizeof(float));
    cudaMemset(vec->gpu_rewards, 0, num_agents*sizeof(float));
    cudaMemset(vec->gpu_terminals, 0, num_agents*sizeof(unsigned char));

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

    printf("Finished creating %d envs\n", num_envs);

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
    atomic_bool* obs_ready_on_cpu = vec->threading->obs_ready_on_cpu;
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
	    //atomic_store(&obs_ready_on_cpu[buf], true);
    }
}

void vec_send(VecEnv* vec, int buffer) {
    int block_size = vec->size / vec->buffers;
    int start = buffer * block_size;

    // For testing
    //int val = rand() % 8192;
    //cudaMemset(&vec->gpu_actions[start], val, block_size*ACT_SIZE*sizeof(float));

    Threading* threading = vec->threading;

    // Single threaded
    if (threading->num_threads == 0) {
        //float val = rand()%3;
        //cudaMemcpy(&vec->gpu_actions[start], &val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(
            &vec->actions[start*ACT_SIZE],
            &vec->gpu_actions[start*ACT_SIZE],
            block_size*ACT_SIZE*sizeof(float),
            cudaMemcpyDeviceToHost
        );
        for (int i = start; i < start + block_size; i++) {
            Env* env = &vec->envs[i];
            c_step(env);
        }
        cudaMemcpy(
            &vec->gpu_observations[start*OBS_SIZE],
            &vec->observations[start*OBS_SIZE],
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
    } else {
        cudaMemcpyAsync(
            &vec->actions[start*ACT_SIZE],
            &vec->gpu_actions[start*ACT_SIZE],
            block_size*ACT_SIZE*sizeof(float),
            cudaMemcpyDeviceToHost,
            vec->streams[buffer]
        );

        atomic_bool* actions_ready_on_gpu = threading->actions_ready_on_gpu;
        atomic_store(&actions_ready_on_gpu[buffer], true);
        //printf("vec_send initiated actions->CPU\n");
    }


}

void vec_recv(VecEnv* vec, int buffer) {
    Threading* threading = vec->threading;
    // TODO: Single stream architecture requires busy waiting here
    if (threading->num_threads > 0) {
        atomic_bool* obs_ready_on_cpu = &threading->obs_ready_on_cpu[buffer];
        //printf("vec_recv waiting on CPU obs\n");
        while (!atomic_load(obs_ready_on_cpu)) {}
        //printf("vec_recv waiting on obs->GPU\n");
        cudaStreamSynchronize(vec->streams[buffer]);
        //printf("vec_recv got obs on GPU\n");
        atomic_store(obs_ready_on_cpu, false);
    }
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
