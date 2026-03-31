#include "drive.h"
#define OBS_SIZE 1848
#define NUM_ATNS 2
#define ACT_SIZES {7, 13}
#define OBS_TENSOR_T FloatTensor

#define MY_VEC_INIT
#define Env Drive
#include "vecenv.h"

// Test version: find first map with 8 agents and fill buffer with copies
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
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

    // Verify first map exists
    char first_map[100];
    sprintf(first_map, "resources/drive/binaries/map_%03d.bin", 0);
    FILE* test_fp = fopen(first_map, "rb");
    if (!test_fp) {
        printf("ERROR: Cannot find map files at resources/drive/binaries/. "
               "Make sure you are running from the root directory.\n");
        *num_envs_out = 0;
        return NULL;
    }
    fclose(test_fp);

    // Probe each map to get agent counts
    int agents_per_map[num_maps];
    for (int m = 0; m < num_maps; m++) {
        char map_file[100];
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", m);

        Env temp_env = {0};
        temp_env.map_name = map_file;
        temp_env.num_agents = 0;
        init(&temp_env);
        int map_agents = temp_env.active_agent_count;
        c_close(&temp_env);

        agents_per_map[m] = map_agents < MAX_CARS ? map_agents : MAX_CARS;
        printf("Map %d: %d active agents, using %d (MAX_CARS=%d)\n",
               m, map_agents, agents_per_map[m], MAX_CARS);
    }

    // Calculate envs per buffer per map
    // Each buffer gets copies of every map, repeated to fill agents_per_buffer
    int agents_per_map_set = 0;
    for (int m = 0; m < num_maps; m++)
        agents_per_map_set += agents_per_map[m];

    int sets_per_buffer = agents_per_buffer / agents_per_map_set;
    int envs_per_buffer = sets_per_buffer * num_maps;
    int total_envs = envs_per_buffer * num_buffers;

    // Fill buffer info
    for (int b = 0; b < num_buffers; b++) {
        buffer_env_starts[b] = b * envs_per_buffer;
        buffer_env_counts[b] = envs_per_buffer;
    }

    Env* envs = (Env*)calloc(total_envs, sizeof(Env));

    for (int i = 0; i < total_envs; i++) {
        int map_id = i % num_maps;
        char map_file[100];
        sprintf(map_file, "resources/drive/binaries/map_%03d.bin", map_id);

        Env* env = &envs[i];
        memset(env, 0, sizeof(Env));

        env->map_name = strdup(map_file);
        env->human_agent_idx = human_agent_idx;
        env->reward_vehicle_collision = reward_vehicle_collision;
        env->reward_offroad_collision = reward_offroad_collision;
        env->reward_goal_post_respawn = reward_goal_post_respawn;
        env->reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn;
        env->spawn_immunity_timer = spawn_immunity_timer;
        env->num_agents = agents_per_map[map_id];

        init(env);
    }

    int total_agent_count = total_envs / num_maps * agents_per_map_set;
    printf("Created %d envs across %d maps (%d total agents)\n",
           total_envs, num_maps, total_agent_count);

    *num_envs_out = total_envs;
    return envs;
}

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
