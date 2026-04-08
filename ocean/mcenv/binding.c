#include "mcenv.h"
#define OBS_SIZE MC_OBS_TOTAL
#define NUM_ATNS 8
#define ACT_SIZES {3, 3, 2, 2, 2, 7, 7, 2}
#define OBS_TENSOR_T FloatTensor

#define Env MCEnv
#include "vecenv.h"

static double dict_get_or(Dict* d, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(d, key);
    return item ? item->value : fallback;
}

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->mc = NULL;
    env->renderer = NULL;
    env->start_x = MC_START_X;
    env->start_y = MC_START_Y;
    env->start_z = MC_START_Z;
    env->rw_survival     = (float)dict_get_or(kwargs, "rw_survival",     MC_SURVIVAL_DEFAULT);
    env->rw_block        = (float)dict_get_or(kwargs, "rw_block",        MC_BLOCK_DEFAULT);
    env->rw_fall         = (float)dict_get_or(kwargs, "rw_fall",         MC_FALL_DEFAULT);
    env->rw_speed        = (float)dict_get_or(kwargs, "rw_speed",        MC_SPEED_DEFAULT);
    env->rw_target_reach = (float)dict_get_or(kwargs, "rw_target_reach", MC_TARGET_DEFAULT);
    env->curriculum_phase = (int)dict_get_or(kwargs, "curriculum_phase", 0);
    env->phase_transition = (int)dict_get_or(kwargs, "phase_transition", 0);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_dist", log->max_dist);
    dict_set(out, "fell", log->fell);
    dict_set(out, "blocks_placed", log->blocks_placed);
    dict_set(out, "targets_reached", log->targets_reached);
    dict_set(out, "sneak_frac", log->sneak_frac);
    dict_set(out, "place_frac", log->place_frac);
    dict_set(out, "avg_pitch", log->avg_pitch);
    dict_set(out, "on_ground_frac", log->on_ground_frac);
}
