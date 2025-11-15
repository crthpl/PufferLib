#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "vecenv.h"

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

__attribute__((visibility("default")))
VecEnv create_environments(int num_envs, Dict* kwargs) {
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    VecEnv vec = {
        .envs = envs,
        .size = num_envs
    };

    int num_agents = 0;
    for (int i = 0; i < num_envs; i++) {
        srand(i);
        my_init(&envs[i], kwargs);
        //num_agents += envs[i].num_agents;
        num_agents += 1;
    }

    vec.observations = (float*)calloc(num_agents*OBS_SIZE, sizeof(float));
    vec.actions = (float*)calloc(num_agents*ACT_SIZE, sizeof(float));
    vec.rewards = (float*)calloc(num_agents, sizeof(float));
    vec.terminals = (unsigned char*)calloc(num_agents, sizeof(unsigned char));

    int agent = 0;
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->observations = vec.observations + agent*OBS_SIZE;
        env->actions = vec.actions + agent*ACT_SIZE;
        env->rewards = vec.rewards + agent;
        env->terminals = vec.terminals + agent;
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

void vec_reset(VecEnv vec) {
    for (int i = 0; i < vec.size; i++) {
        Env* env = &vec.envs[i];
        c_reset(env);
    }
}

void vec_step(VecEnv vec) {
    for (int i = 0; i < vec.size; i++) {
        Env* env = &vec.envs[i];
        c_step(env);
    }
}

void env_close(Env* env) {
    c_close(env);
    free(env);
}

void vec_close(VecEnv vec) {
    for (int i = 0; i < vec.size; i++) {
        Env* env = &vec.envs[i];
        c_close(env);
    }
    free(vec.envs);
}

void vec_render(VecEnv vec, int env_idx) {
    Env* env = &vec.envs[env_idx];
    c_render(env);
}

void vec_log(VecEnv vec, Dict* out) {
    Log aggregate = {0};
    int num_keys = sizeof(Log) / sizeof(float);
    for (int i = 0; i < vec.size; i++) {
        Env* env = &vec.envs[i];
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
    my_log(&aggregate, out);
}
