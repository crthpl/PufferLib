#include "mcenv.h"
#define OBS_SIZE MC_OBS_TOTAL
#define NUM_ATNS 9
#define ACT_SIZES {3, 3, 2, 2, 2, 7, 7, 3, 5}
#define OBS_TENSOR_T FloatTensor

#define Env MCEnv
#define MY_GPU_NATIVE
#include "vecenv.h"

static double dict_get_or(Dict* d, const char* key, double fallback) {
    DictItem* item = dict_get_unsafe(d, key);
    return item ? item->value : fallback;
}

// ---------------------------------------------------------------------------
// GPU-native env stepping — extern declarations for mcenv_cuda.cu functions
// ---------------------------------------------------------------------------

// Forward declare McEnvState/McEnvConfig/McEnvLog as opaque for C
typedef struct McEnvState McEnvState;
typedef struct McEnvConfig McEnvConfig;
typedef struct McEnvLog McEnvLog;

extern float* mcenv_cuda_create_sine_table(void);
extern void   mcenv_cuda_free_sine_table(float* dev_table);
extern void   mcenv_cuda_alloc(int N, McEnvState** d_states, unsigned char** d_blocks,
                               McEnvConfig** d_config, McEnvLog** d_logs);
extern void   mcenv_cuda_free(McEnvState* d_states, unsigned char* d_blocks,
                              McEnvConfig* d_config, McEnvLog* d_logs);
extern void   mcenv_cuda_upload_config(McEnvConfig* d_config, const McEnvConfig* h_config);
extern void   mcenv_cuda_pufferlib_step(
    McEnvState* d_states, const float* d_actions,
    float* d_observations, float* d_rewards, float* d_terminals,
    unsigned char* d_blocks, McEnvLog* d_logs, const float* d_sine_table,
    const McEnvConfig* d_config, int N,
    cudaStream_t stream);
extern void   mcenv_cuda_pufferlib_reset(
    McEnvState* d_states, float* d_observations,
    unsigned char* d_blocks, const float* d_sine_table,
    const McEnvConfig* d_config, int N);
extern void   mcenv_cuda_download_logs(McEnvLog* d_logs, McEnvLog* h_logs, int N);
extern int    mcenv_cuda_grid_size(void);

void my_gpu_native_init(StaticVec* vec, Dict* env_kwargs) {
    int N = vec->total_agents;

    McEnvState* d_states;
    unsigned char* d_blocks;
    McEnvConfig* d_config;
    McEnvLog* d_logs;
    mcenv_cuda_alloc(N, &d_states, &d_blocks, &d_config, &d_logs);

    float* d_sine = mcenv_cuda_create_sine_table();

    // Build config and upload — layout must match McEnvConfig in mcenv_cuda.cuh
    struct {
        int   max_ticks;
        float rw_survival;
        float rw_block;
        float rw_fall;
        float rw_speed;
        float rw_target_reach;
        int   curriculum_phase;
        int   phase_transition;
        int   force_phase2;
    } h_config;
    h_config.max_ticks       = (int)dict_get(env_kwargs, "max_ticks")->value;
    h_config.rw_survival     = (float)dict_get_or(env_kwargs, "rw_survival",     MC_SURVIVAL_DEFAULT);
    h_config.rw_block        = (float)dict_get_or(env_kwargs, "rw_block",        MC_BLOCK_DEFAULT);
    h_config.rw_fall         = (float)dict_get_or(env_kwargs, "rw_fall",         MC_FALL_DEFAULT);
    h_config.rw_speed        = (float)dict_get_or(env_kwargs, "rw_speed",        MC_SPEED_DEFAULT);
    h_config.rw_target_reach = (float)dict_get_or(env_kwargs, "rw_target_reach", MC_TARGET_DEFAULT);
    h_config.curriculum_phase = (int)dict_get_or(env_kwargs, "curriculum_phase", 0);
    h_config.phase_transition = (int)dict_get_or(env_kwargs, "phase_transition", 0);
    h_config.force_phase2    = (int)dict_get_or(env_kwargs, "force_phase2",     0);
    mcenv_cuda_upload_config(d_config, (const McEnvConfig*)&h_config);

    vec->gpu_env_states = d_states;
    vec->gpu_blocks     = d_blocks;
    vec->gpu_sine_table = d_sine;
    vec->gpu_env_config = d_config;
    vec->gpu_env_logs   = d_logs;
}

extern int mcenv_cuda_state_size(void);
extern int mcenv_cuda_log_size(void);

void my_gpu_native_step(StaticVec* vec) {
    int start = vec->gpu_native_agent_start;
    int count = vec->gpu_native_agent_count;
    int obs_size = vec->obs_size;
    int num_atns = vec->num_atns;
    mcenv_cuda_pufferlib_step(
        (McEnvState*)((char*)vec->gpu_env_states + start * mcenv_cuda_state_size()),
        (const float*)vec->gpu_actions + start * num_atns,
        (float*)vec->gpu_observations + start * obs_size,
        vec->gpu_rewards + start,
        vec->gpu_terminals + start,
        (unsigned char*)vec->gpu_blocks + start * mcenv_cuda_grid_size(),
        (McEnvLog*)((char*)vec->gpu_env_logs + start * mcenv_cuda_log_size()),
        (const float*)vec->gpu_sine_table,
        (const McEnvConfig*)vec->gpu_env_config,
        count,
        vec->gpu_native_stream);
}

void my_gpu_native_reset(StaticVec* vec) {
    mcenv_cuda_pufferlib_reset(
        (McEnvState*)vec->gpu_env_states,
        (float*)vec->gpu_observations,
        (unsigned char*)vec->gpu_blocks,
        (const float*)vec->gpu_sine_table,
        (const McEnvConfig*)vec->gpu_env_config,
        vec->total_agents);
}

void my_gpu_native_close(StaticVec* vec) {
    if (vec->gpu_sine_table)
        mcenv_cuda_free_sine_table((float*)vec->gpu_sine_table);
    if (vec->gpu_env_states)
        mcenv_cuda_free((McEnvState*)vec->gpu_env_states,
                        (unsigned char*)vec->gpu_blocks,
                        (McEnvConfig*)vec->gpu_env_config,
                        (McEnvLog*)vec->gpu_env_logs);
}

void my_gpu_native_log(StaticVec* vec) {
    // Download GPU logs into CPU env Log structs, then zero GPU logs.
    // McEnvLog layout matches Log exactly (same 14 floats in same order).
    int N = vec->total_agents;
    Log* h_logs = (Log*)calloc(N, sizeof(Log));
    mcenv_cuda_download_logs((McEnvLog*)vec->gpu_env_logs, (McEnvLog*)h_logs, N);

    // Merge into CPU env logs so static_vec_aggregate_logs picks them up
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        int num_keys = sizeof(Log) / sizeof(float);
        for (int j = 0; j < num_keys; j++) {
            ((float*)&envs[i].log)[j] += ((float*)&h_logs[i])[j];
        }
    }
    free(h_logs);
}

// ---------------------------------------------------------------------------

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
    env->lifetime_blocks = 0;
    env->avg_targets_ema = 0.0f;
    env->phase2_unlocked = 0;
    env->force_phase2 = (int)dict_get_or(kwargs, "force_phase2", 0);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_dist", log->max_dist);
    dict_set(out, "max_height", log->max_height);
    dict_set(out, "fell", log->fell);
    dict_set(out, "blocks_placed", log->blocks_placed);
    dict_set(out, "targets_reached", log->targets_reached);
    dict_set(out, "sneak_frac", log->sneak_frac);
    dict_set(out, "avg_pitch", log->avg_pitch);
    dict_set(out, "on_ground_frac", log->on_ground_frac);
    dict_set(out, "phase", log->phase);
    dict_set(out, "rw_air", log->rw_air);
    dict_set(out, "rw_sprint_jump", log->rw_sprint_jump);
    dict_set(out, "rw_speed", log->rw_speed);
    dict_set(out, "rw_block", log->rw_block);
    dict_set(out, "rw_target", log->rw_target);
}
