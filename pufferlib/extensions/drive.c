#include "../ocean/drive/drive.h"
#define OBS_SIZE 1848
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TYPE FLOAT
#define ACT_TYPE DOUBLE

#define MY_VEC_INIT
#define Env Drive
#include "env_binding.h"

// Test version: find first map with 8 agents and fill buffer with copies
Env* my_vec_init(int* num_envs_out, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;

    int agents_per_buffer = total_agents / num_buffers;

    // Get config from env_kwargs
    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision")->value;
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision")->value;
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn")->value;
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn")->value;
    int spawn_immunity_timer = (int)dict_get(env_kwargs, "spawn_immunity_timer")->value;
    int human_agent_idx = (int)dict_get(env_kwargs, "human_agent_idx")->value;

    // Find first map with exactly 8 agents
    int target_map_id = -1;
    for (int map_id = 0; map_id < num_maps; map_id++) {
        char map_file[100];
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);

        Env temp_env = {0};
        temp_env.map_name = map_file;
        temp_env.num_agents = 64;
        init(&temp_env);

        int agent_count = temp_env.active_agent_count;
        int total_agent_count = temp_env.num_agents;
        c_close(&temp_env);

        if (agent_count == 8) {
            target_map_id = map_id;
            printf("Found map %d with 8 active agents and %d total agents\n", map_id, agent_count);
            break;
        }
    }

    if (target_map_id < 0) {
        printf("ERROR: No map found with exactly 8 agents\n");
        *num_envs_out = 0;
        return NULL;
    }

    // Calculate how many envs we need (8 agents per env)
    int envs_per_buffer = agents_per_buffer / 8;
    int total_envs = envs_per_buffer * num_buffers;

    Env* envs = (Env*)calloc(total_envs, sizeof(Env));

    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", target_map_id);

    for (int i = 0; i < total_envs; i++) {
        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));

        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->spawn_immunity_timer = spawn_immunity_timer;
        env->num_agents = 8;

        init(env);
    }

    printf("Created %d envs with %d agents each (%d total agents)\n",
           total_envs, 8, total_envs * 8);

    *num_envs_out = total_envs;
    return envs;
}

/*
// Original my_vec_init with random maps and buffer boundary handling
Env* my_vec_init(int* num_envs_out, Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int num_maps = (int)dict_get(env_kwargs, "num_maps")->value;

    int agents_per_buffer = total_agents / num_buffers;

    // Get config from env_kwargs
    float reward_vehicle_collision = dict_get(env_kwargs, "reward_vehicle_collision")->value;
    float reward_offroad_collision = dict_get(env_kwargs, "reward_offroad_collision")->value;
    float reward_goal_post_respawn = dict_get(env_kwargs, "reward_goal_post_respawn")->value;
    float reward_vehicle_collision_post_respawn = dict_get(env_kwargs, "reward_vehicle_collision_post_respawn")->value;
    int spawn_immunity_timer = (int)dict_get(env_kwargs, "spawn_immunity_timer")->value;
    int human_agent_idx = (int)dict_get(env_kwargs, "human_agent_idx")->value;

    // Allocate max possible envs (1 agent per env worst case)
    Env* envs = (Env*)calloc(total_agents, sizeof(Env));

    int num_envs = 0;
    int current_buffer = 0;
    int current_buffer_agents = 0;

    while (current_buffer < num_buffers) {
        // Seed srand with current loop index over envs
        srand(num_envs);
        int map_id = rand() % num_maps;

        char map_file[100];
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);

        // Initialize env struct
        Env* env = &envs[num_envs];
        memset(env, 0, sizeof(Env));

        // Set config
        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->spawn_immunity_timer = spawn_immunity_timer;
        env->num_agents = 0;  // Let init determine via set_active_agents

        // Call init (loads map, sets active agents, etc.)
        init(env);

        int map_agent_count = env->active_agent_count;

        // Check if map fits in current buffer
        if (current_buffer_agents + map_agent_count > agents_per_buffer) {
            // Doesn't fit - close env and move to next buffer (padding)
            c_close(env);
            free(env->map_name);
            memset(env, 0, sizeof(Env));

            current_buffer++;
            current_buffer_agents = 0;
            continue;
        }

        // Map fits
        env->num_agents = map_agent_count;
        current_buffer_agents += map_agent_count;
        num_envs++;
    }

    // Shrink to actual size needed
    envs = (Env*)realloc(envs, num_envs * sizeof(Env));
    *num_envs_out = num_envs;
    return envs;
}
*/

void my_init(Env* env, Dict* kwargs) {
    env->human_agent_idx = dict_get(kwargs, "human_agent_idx")->value;
    env->reward_vehicle_collision = dict_get(kwargs, "reward_vehicle_collision")->value;
    env->reward_offroad_collision = dict_get(kwargs, "reward_offroad_collision")->value;
    env->reward_goal_post_respawn = dict_get(kwargs, "reward_goal_post_respawn")->value;
    env->reward_vehicle_collision_post_respawn = dict_get(kwargs, "reward_vehicle_collision_post_respawn")->value;
    env->spawn_immunity_timer = dict_get(kwargs, "spawn_immunity_timer")->value;
    int map_id = dict_get(kwargs, "map_id")->value;
    int max_agents = dict_get(kwargs, "max_agents")->value;

    char map_file[100];
    sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "offroad_rate", log->offroad_rate);
    dict_set(out, "collision_rate", log->collision_rate);
    dict_set(out, "dnf_rate", log->dnf_rate);
    dict_set(out, "n", log->n);
    dict_set(out, "completion_rate", log->completion_rate);
    dict_set(out, "clean_collision_rate", log->clean_collision_rate);
}
