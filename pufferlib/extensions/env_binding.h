#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <cuda_runtime.h>

#include "vecenv.h"
#include <stdatomic.h>

#define FLOAT 1
#define INT 2
#define UNSIGNED_CHAR 3
#define DOUBLE 4

#if OBS_TYPE == FLOAT
    #define OBS_DTYPE float
#elif OBS_TYPE == INT
    #define OBS_DTYPE int
#elif OBS_TYPE == UNSIGNED_CHAR
    #define OBS_DTYPE unsigned char
#elif OBS_TYPE == DOUBLE
    #define OBS_DTYPE double
#elif OBS_TYPE == CHAR
    #define OBS_DTYPE char
#endif

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
__attribute__((visibility("default"))) const int NUM_ATNS_EXPORT = NUM_ATNS;
__attribute__((visibility("default"))) const int ACT_SIZES_EXPORT[NUM_ATNS] = ACT_SIZES;
__attribute__((visibility("default"))) const int OBS_T = OBS_TYPE;
__attribute__((visibility("default"))) const int ACT_T = ACT_TYPE;

#define INIT 0
#define OBS_READY_ON_CPU 1
#define OBS_READY_ON_GPU 2
#define ATN_READY_ON_GPU 3
#define ATN_READY_ON_CPU 4

typedef struct Threading {
    atomic_long* completed;
    long start_index;
    long end_index;
    int num_threads;
    pthread_cond_t wake_cond;
    pthread_mutex_t wake_mutex;
    pthread_cond_t all_done_cond;
    pthread_mutex_t all_done_mutex;
    pthread_t* threads;
    atomic_int* buffer_states;
    int block_size;
    int num_envs;
    int num_buffers;
    bool use_gpu;
    int test_idx;
    long min_expected;
    int iters;
} Threading;

typedef struct WorkerArg {
    VecEnv* vec;
    int idx;
} WorkerArg;

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

void update_buffer_state(Threading* threading, int buf, int val) {
    atomic_int* states = threading->buffer_states;
    int old_val = atomic_load(&states[buf]);
    atomic_store(&states[buf], val);
    //printf("Updated vecenv %d buf %d from %d to %d \n", threading->test_idx, buf, old_val, val);
}

static void* c_threadstep(void* arg)
{
    WorkerArg* worker_arg = (WorkerArg*)arg;
    VecEnv* vec = worker_arg->vec;
    Threading* threading = vec->threading;

    int block_size = threading->block_size;
    int num_envs = vec->size;
    atomic_long* completed = &threading->completed[worker_arg->idx];
    long end = 0;
    long block_start = 0;
    while (1) {
        pthread_mutex_lock(&threading->wake_mutex);
        while (threading->start_index >= threading->end_index) {
            atomic_store(completed, threading->start_index);
            //printf("Min completed %d on thread %d. end %d test idx %d\n", threading->start_index, worker_arg->idx, threading->end_index, threading->test_idx);
            pthread_cond_wait(&threading->wake_cond, &threading->wake_mutex);
        }
 
        long start = threading->start_index;
        long end = threading->start_index + block_size;
        if (end > threading->end_index) {
            end = threading->end_index;
        }
        threading->start_index = end;
        pthread_mutex_unlock(&threading->wake_mutex);

        for (int i=start; i<end; i++) {
            c_step(&vec->envs[i % num_envs]);
        }
        atomic_store(completed, end);
    }
    return NULL;
}

static void* c_threadmanager(void* arg) {
    VecEnv* vec = (VecEnv*)arg;
    Threading* threading = vec->threading;

    int envs_per_buffer = vec->size / vec->buffers;
    int agents_per_buffer = vec->num_agents / vec->buffers;
    atomic_int* buffer_states = threading->buffer_states;
    long iters = 0;
    int curr_buf = 0;

    while (1) {
        for (int buf=0; buf < vec->buffers; buf++) {
            int state = atomic_load(&buffer_states[buf]);
            bool cuda_ready = !threading->use_gpu || cudaStreamQuery(vec->streams[buf]) == cudaSuccess;
            if (state == ATN_READY_ON_GPU && cuda_ready) {
                update_buffer_state(threading, buf, ATN_READY_ON_CPU);
                pthread_mutex_lock(&threading->wake_mutex);
                threading->end_index += envs_per_buffer;
                pthread_cond_broadcast(&threading->wake_cond);
                pthread_mutex_unlock(&threading->wake_mutex);
            }

            if (buf != curr_buf) {
                continue;
            }

            long min_expected = (iters + 1) * envs_per_buffer;
            threading->min_expected = min_expected;
            long min_completed = LONG_MAX;
            for (int i=0; i<threading->num_threads; i++) {
                long completed = atomic_load(threading->completed + i);
                if (completed < min_completed) {
                    min_completed = completed;
                }
            }
            if (min_completed < min_expected) {
                continue;
            }

            if (state == ATN_READY_ON_CPU) {
                curr_buf = (curr_buf + 1) % vec->buffers;
                iters++;
                threading->iters = iters;

                int start = buf * agents_per_buffer;

                if (threading->use_gpu) {
                    cudaMemcpyAsync(
                        &((OBS_DTYPE*)vec->gpu_observations)[start*OBS_SIZE],
                        &((OBS_DTYPE*)vec->observations)[start*OBS_SIZE],
                        agents_per_buffer*OBS_SIZE*sizeof(OBS_DTYPE),
                        cudaMemcpyHostToDevice,
                        vec->streams[buf]
                    );
                    cudaMemcpyAsync(
                        &vec->gpu_rewards[start],
                        &vec->rewards[start],
                        agents_per_buffer*sizeof(float),
                        cudaMemcpyHostToDevice,
                        vec->streams[buf]
                    );
                    cudaMemcpyAsync(
                        &vec->gpu_terminals[start],
                        &vec->terminals[start],
                        agents_per_buffer*sizeof(float),
                        cudaMemcpyHostToDevice,
                        vec->streams[buf]
                    );
                }
                update_buffer_state(threading, buf, OBS_READY_ON_CPU);
            }
        }
    }
}
 
__attribute__((visibility("default")))
VecEnv* create_environments(int num_envs, int buffers, bool use_gpu, int test_idx, Dict* kwargs) {
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    VecEnv* vec = (VecEnv*)calloc(1, sizeof(VecEnv));
    vec->envs = envs;
    vec->size = num_envs;
    vec->buffers = buffers;
    vec->threading = calloc(1, sizeof(Threading));
    vec->threading->use_gpu = use_gpu;
    vec->threading->test_idx = test_idx;

    int num_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        srand(i);
        my_init(&envs[i], kwargs);
        num_agents += envs[i].num_agents;
    }

    /*
    vec->observations = calloc(num_agents*OBS_SIZE, sizeof(OBS_DTYPE));
    vec->actions = (float*)calloc(num_agents*NUM_ATNS, sizeof(float));
    vec->rewards = (float*)calloc(num_agents, sizeof(float));
    vec->terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));
    */
    //printf("Size of alloc: %d\n", num_agents*OBS_SIZE*sizeof(OBS_DTYPE));
    //printf("Before allocated mem host\n");
    if (use_gpu) {
        cudaSetDevice(0);
        CHECK_CUDA(cudaHostAlloc((void**)&vec->observations, num_agents*OBS_SIZE*sizeof(OBS_DTYPE), cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc((void**)&vec->actions, num_agents*NUM_ATNS*sizeof(double), cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc((void**)&vec->rewards, num_agents*sizeof(float), cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc((void**)&vec->terminals, num_agents*sizeof(float), cudaHostAllocPortable));
    } else {
        vec->observations = calloc(num_agents*OBS_SIZE, sizeof(OBS_DTYPE));
        vec->actions = calloc(num_agents*NUM_ATNS, sizeof(double));
        vec->rewards = calloc(num_agents, sizeof(float));
        vec->terminals = calloc(num_agents, sizeof(float));
    }

    memset(vec->observations, 0, num_agents*OBS_SIZE*sizeof(OBS_DTYPE));
    memset(vec->actions, 0, num_agents*NUM_ATNS*sizeof(double));
    memset(vec->rewards, 0, num_agents*sizeof(float));
    memset(vec->terminals, 0, num_agents*sizeof(float));
    //printf("allocated mem host\n");

    if (use_gpu) {
        CHECK_CUDA(cudaMalloc((void**)&vec->gpu_observations, num_agents*OBS_SIZE*sizeof(OBS_DTYPE)));
        CHECK_CUDA(cudaMalloc((void**)&vec->gpu_actions, num_agents*NUM_ATNS*sizeof(double)));
        CHECK_CUDA(cudaMalloc((void**)&vec->gpu_rewards, num_agents*sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&vec->gpu_terminals, num_agents*sizeof(float)));
        cudaMemset(vec->gpu_observations, 0, num_agents*OBS_SIZE*sizeof(OBS_DTYPE));
        cudaMemset(vec->gpu_actions, 0, num_agents*NUM_ATNS*sizeof(double));
        cudaMemset(vec->gpu_rewards, 0, num_agents*sizeof(float));
        cudaMemset(vec->gpu_terminals, 0, num_agents*sizeof(float));
    } else {
        vec->gpu_observations = vec->observations;
        vec->gpu_actions = vec->actions;
        vec->gpu_rewards = vec->rewards;
        vec->gpu_terminals = vec->terminals;
    }

    vec->num_agents = num_agents;

    int agent = 0;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->observations = (OBS_DTYPE*)vec->observations + agent*OBS_SIZE;
        env->actions = vec->actions + agent*NUM_ATNS;
        env->rewards = vec->rewards + agent;
        env->terminals = vec->terminals + agent;
        agent += env->num_agents;
    }

    return vec;
}

void create_threads(VecEnv* vec, int threads, int block_size) {
    //printf("Finished creating %d envs\n", num_envs);
    Threading* threading = vec->threading;
    threading->num_threads = threads;
    threading->block_size = block_size;
    threading->completed = (atomic_long*)calloc(threads, sizeof(atomic_long));
    threading->buffer_states = (atomic_int*)calloc(vec->buffers, sizeof(atomic_int));
    threading->num_envs = vec->size;
    threading->num_buffers = vec->buffers;

    vec->streams = (cudaStream_t*)calloc(vec->buffers, sizeof(cudaStream_t));
    for (int i = 0; i < vec->buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }


    if (threads > 0) {
        WorkerArg* worker_args = (WorkerArg*)calloc(threads, sizeof(WorkerArg));

        threading->threads = (pthread_t*)calloc(threads + 1, sizeof(pthread_t));
        assert(threading->threads != NULL && "create_vecenv failed to allocate memory for threads\n");
        assert(pthread_cond_init(&threading->wake_cond, NULL) == 0 && "create_vecenv failed to initialize wake_cond\n");
        assert(pthread_mutex_init(&threading->wake_mutex, NULL) == 0 && "create_vecenv failed to initialize wake_mutex\n");
        //atomic_store(&threading->end_index, 0);
        //atomic_store(&threading->work_index, 0);

        for (int i = 0; i < threads; i++) {
            WorkerArg* arg = &worker_args[i];
            arg->vec = vec;
            arg->idx = i;

            int err = pthread_create(&threading->threads[i], NULL, c_threadstep, (void*)(arg));
            assert(err == 0 && "create_vecenv failed to create thread\n");
        }

        // Last thread manages host device syncs
        int err = pthread_create(&threading->threads[threads], NULL, c_threadmanager, (void*)(vec));
        assert(err == 0 && "create_vecenv failed to create manager thread\n");
    }
}

Env* env_init(OBS_DTYPE* observations, double* actions, float* rewards,
        float* terminals, int seed, Dict* kwargs) {
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
    cudaMemcpy(
        vec->gpu_observations,
        vec->observations,
        vec->num_agents*OBS_SIZE*sizeof(OBS_DTYPE),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        vec->gpu_rewards,
        vec->rewards,
        vec->num_agents*sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        vec->gpu_terminals,
        vec->terminals,
        vec->num_agents*sizeof(float),
        cudaMemcpyHostToDevice
    );
    cudaDeviceSynchronize();
 
    Threading* threading = vec->threading;
    if (threading->num_threads > 0) {
        for (int buf=0; buf < vec->buffers; buf++) {
            update_buffer_state(threading, buf, OBS_READY_ON_CPU);
        }
    }
}

void vec_send(VecEnv* vec, int buffer) {
    int envs_per_buffer = vec->size / vec->buffers;
    int env_start = buffer * envs_per_buffer;
    int agents_per_buffer = vec->num_agents / vec->buffers;
    int start = buffer * agents_per_buffer;

    Threading* threading = vec->threading;

    // Single threaded
    if (threading->num_threads == 0) {
        cudaDeviceSynchronize();
        cudaMemcpy(
            &vec->actions[start*NUM_ATNS],
            &vec->gpu_actions[start*NUM_ATNS],
            agents_per_buffer*NUM_ATNS*sizeof(double),
            cudaMemcpyDeviceToHost
        );
        for (int i = env_start; i < env_start + envs_per_buffer; i++) {
            Env* env = &vec->envs[i];
            c_step(env);
        }
        cudaMemcpy(
            &((OBS_DTYPE*)vec->gpu_observations)[start*OBS_SIZE],
            &((OBS_DTYPE*)vec->observations)[start*OBS_SIZE],
            agents_per_buffer*OBS_SIZE*sizeof(OBS_DTYPE),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            &vec->gpu_rewards[start],
            &vec->rewards[start],
            agents_per_buffer*sizeof(float),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            &vec->gpu_terminals[start],
            &vec->terminals[start],
            agents_per_buffer*sizeof(float),
            cudaMemcpyHostToDevice
        );
        cudaDeviceSynchronize();
    } else {
        if (threading->use_gpu) {
            cudaMemcpy(
                &vec->actions[start*NUM_ATNS],
                &vec->gpu_actions[start*NUM_ATNS],
                agents_per_buffer*NUM_ATNS*sizeof(double),
                cudaMemcpyDeviceToHost
            );
        }

        atomic_int* buffer_states = threading->buffer_states;
        update_buffer_state(threading, buffer, ATN_READY_ON_GPU);
    }
}

void vec_recv(VecEnv* vec, int buffer) {
    cudaDeviceSynchronize();
    Threading* threading = vec->threading;
    // TODO: Single stream architecture requires busy waiting here
    //printf("Recv buf %d\n", buffer);

    if (threading->num_threads > 0) {
        atomic_int* buffer_states = threading->buffer_states;
        //printf("vec_recv waiting on CPU obs\n");
        while (atomic_load(&buffer_states[buffer]) != OBS_READY_ON_CPU) {}
        //printf("vec_recv waiting on obs->GPU\n");
        if (threading->use_gpu) {
            cudaStreamSynchronize(vec->streams[buffer]);
        }
        update_buffer_state(vec->threading, buffer, OBS_READY_ON_GPU);
        //buffer_states[buffer] = OBS_READY_ON_GPU;
        //printf("vec_recv got obs on GPU for buffer %d\n", buffer);
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
    dict_set(out, "n", n);
    my_log(&aggregate, out);
}
