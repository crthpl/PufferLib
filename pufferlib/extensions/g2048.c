#include "../ocean/g2048/g2048.h"

#define OBS_SIZE 289
#define ACT_SIZE 1
#define OBS_TYPE UNSIGNED_CHAR
#define ACT_TYPE INT


#define Env Game
#include "env_binding.h"

void my_init(Env* env, Dict* kwargs) {
    env->can_go_over_65536 = dict_get(kwargs, "can_go_over_65536")->value;
    env->reward_scaler = dict_get(kwargs, "reward_scaler")->value;
    env->endgame_env_prob = dict_get(kwargs, "endgame_env_prob")->value;
    env->scaffolding_ratio = dict_get(kwargs, "scaffolding_ratio")->value;
    env->use_heuristic_rewards = dict_get(kwargs, "use_heuristic_rewards")->value;
    env->snake_reward_weight = dict_get(kwargs, "snake_reward_weight")->value;
    env->use_sparse_reward = dict_get(kwargs, "use_sparse_reward")->value;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "merge_score", log->merge_score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "lifetime_max_tile", log->lifetime_max_tile);
    dict_set(out, "reached_32768", log->reached_32768);
    dict_set(out, "reached_65536", log->reached_65536);
    dict_set(out, "monotonicity_reward", log->monotonicity_reward);
    dict_set(out, "snake_state", log->snake_state);
    dict_set(out, "snake_reward", log->snake_reward);
}
