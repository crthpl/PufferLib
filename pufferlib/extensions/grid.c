#include "../ocean/grid/grid.h"
#define OBS_SIZE 121
#define NUM_ATNS 1
#define ACT_SIZES {5}
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE DOUBLE

#define MY_VEC_INIT
#define Env Grid
#include "env_binding.h"

Env* my_vec_init(int* num_envs_out, Dict* vec_kwargs, Dict* env_kwargs) {
    int num_envs = (int)dict_get(vec_kwargs, "total_agents")->value;

    int max_size = (int)dict_get(env_kwargs, "max_size")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;
    int map_size = (int)dict_get(env_kwargs, "map_size")->value;

    if (max_size <= 5) {
        *num_envs_out = 0;
        return NULL;
    }

    // Generate maze levels (shared across all envs)
    State* levels = calloc(num_maps, sizeof(State));

    // Temporary env used to generate maps
    Grid temp_env;
    temp_env.max_size = max_size;
    init_grid(&temp_env);

    srand(time(NULL));
    int start_seed = rand();
    for (int i = 0; i < num_maps; i++) {
        int sz = map_size;
        if (map_size == -1) {
            sz = 5 + (rand() % (max_size - 5));
        }

        if (sz % 2 == 0) {
            sz -= 1;
        }

        float difficulty = (float)rand() / (float)(RAND_MAX);
        create_maze_level(&temp_env, sz, sz, difficulty, start_seed + i);
        init_state(&levels[i], max_size, 1);
        get_state(&temp_env, &levels[i]);
    }

    // Free temp env internal allocations
    free(temp_env.grid);
    free(temp_env.counts);
    free(temp_env.agents);

    // Allocate all environments
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));

    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        env->max_size = max_size;
        env->num_maps = num_maps;
        env->num_agents = 1;
        env->levels = levels;
        init_grid(env);
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->max_size = (int)dict_get(kwargs, "max_size")->value;
    env->num_maps = (int)dict_get(kwargs, "num_maps")->value;
    env->num_agents = 1;
    init_grid(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
