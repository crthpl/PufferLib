#include "mcenv.h"
#define OBS_SIZE MC_OBS_TOTAL
#define NUM_ATNS 7
#define ACT_SIZES {3, 3, 2, 2, 5, 5, 2}
#define OBS_TENSOR_T FloatTensor

#define Env MCEnv
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->mc = NULL;
    env->renderer = NULL;
    env->start_x = MC_START_X;
    env->start_y = MC_START_Y;
    env->start_z = MC_START_Z;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_x", log->max_x);
    dict_set(out, "fell", log->fell);
    dict_set(out, "blocks_placed", log->blocks_placed);
}
