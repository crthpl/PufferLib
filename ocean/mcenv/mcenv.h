// MCEnv: Minecraft bridging + navigation environment using mcenv-codex.
// Agent bridges to random waypoints in a 100x100 area.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mcenv_codex.h"

#define MC_OBS_PLAYER 11
#define MC_OBS_TARGET 2
#define MC_GRID_X 7
#define MC_GRID_Y 3
#define MC_GRID_Z 3
#define MC_OBS_GRID (MC_GRID_X * MC_GRID_Y * MC_GRID_Z)
#define MC_OBS_TOTAL (MC_OBS_PLAYER + MC_OBS_TARGET + MC_OBS_GRID)

#define MC_START_X 25.5      // Center of 50x50 area
#define MC_START_Y 3.0
#define MC_START_Z 25.5
#define MC_START_YAW -90.0f
#define MC_START_PITCH -45.0f

#define MC_VOID_Y 0.0
#define MC_PLATFORM_MAX_X 26 // Single block at x=25, ends at x=26
#define MC_AREA_SIZE 50
#define MC_TARGET_REACH 1.0f // Within 1 block = reached

// Default reward constants
#define MC_SURVIVAL_DEFAULT  0.001f
#define MC_BLOCK_DEFAULT     1.0f
#define MC_FALL_DEFAULT     -1.0f
#define MC_SPEED_DEFAULT     1.0f
#define MC_TARGET_DEFAULT    10.0f

// Yaw/pitch delta lookup tables (degrees)
static const float YAW_DELTAS[7]   = {-15.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 15.0f};
static const float PITCH_DELTAS[7] = {-15.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 15.0f};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float max_dist;
    float fell;
    float blocks_placed;
    float targets_reached;
    float sneak_frac;
    float place_frac;
    float avg_pitch;
    float on_ground_frac;
    float n;
} Log;

typedef struct MCEnv MCEnv;
struct MCEnv {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;

    Environment* mc;
    float start_x;
    float start_y;
    float start_z;
    float prev_dist;       // Previous distance to target
    float yaw;
    float pitch;
    int tick;
    int max_ticks;
    float episode_return;
    float max_dist_from_start;
    int blocks_placed;
    int targets_reached;
    float target_x;
    float target_z;
    float rw_survival;
    float rw_block;
    float rw_fall;
    float rw_speed;
    float rw_target_reach;
    Demo3dRenderer* renderer;
    struct timespec render_last_time;

    // Diagnostic accumulators
    int sneak_ticks;
    int place_ticks;
    float pitch_sum;
    int on_ground_ticks;
};

static float mc_dist_to_target(MCEnv* env, double px, double pz) {
    float dx = (float)px - env->target_x;
    float dz = (float)pz - env->target_z;
    return sqrtf(dx*dx + dz*dz);
}

static void mc_new_target(MCEnv* env) {
    // Random target in [0.5, 99.5] x [0.5, 99.5], at least 10 blocks from player
    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);
    for (int attempt = 0; attempt < 100; attempt++) {
        float tx = 0.5f + (float)(rand_r(&env->rng) % MC_AREA_SIZE);
        float tz = 0.5f + (float)(rand_r(&env->rng) % MC_AREA_SIZE);
        float dx = tx - (float)state.pos.x;
        float dz = tz - (float)state.pos.z;
        if (dx*dx + dz*dz >= 25.0f) { // >= 5 blocks away
            env->target_x = tx;
            env->target_z = tz;
            env->prev_dist = mc_dist_to_target(env, state.pos.x, state.pos.z);
            return;
        }
    }
    // Fallback: just pick something
    env->target_x = 0.5f + (float)(rand_r(&env->rng) % MC_AREA_SIZE);
    env->target_z = 0.5f + (float)(rand_r(&env->rng) % MC_AREA_SIZE);
    env->prev_dist = mc_dist_to_target(env, state.pos.x, state.pos.z);
}

static void setup_platform(MCEnv* env) {
    // Single block at y=2 under start position — must bridge to move
    CBlockPos pos = {(int)env->start_x, 2, (int)env->start_z};
    mcenv_environment_set_block(env->mc, pos, CBLOCK_FULL_CUBE);
}

static void compute_observations(MCEnv* env) {
    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    int idx = 0;

    // Player state (11 values)
    env->observations[idx++] = (float)(state.pos.x - env->start_x) / 50.0f;
    env->observations[idx++] = (float)(state.pos.y - env->start_y) / 10.0f;
    env->observations[idx++] = (float)(state.pos.z - env->start_z) / 10.0f;
    env->observations[idx++] = (float)state.vel.x;
    env->observations[idx++] = (float)state.vel.y;
    env->observations[idx++] = (float)state.vel.z;
    float yaw_rad = env->yaw * (float)M_PI / 180.0f;
    float pitch_rad = env->pitch * (float)M_PI / 180.0f;
    env->observations[idx++] = sinf(yaw_rad);
    env->observations[idx++] = cosf(yaw_rad);
    env->observations[idx++] = sinf(pitch_rad);
    env->observations[idx++] = cosf(pitch_rad);
    env->observations[idx++] = state.on_ground ? 1.0f : 0.0f;

    // Target relative position (2 values)
    env->observations[idx++] = (env->target_x - (float)state.pos.x) / 50.0f;
    env->observations[idx++] = (env->target_z - (float)state.pos.z) / 50.0f;

    // Local block grid: 7x3x3
    int bx = (int)floorf((float)state.pos.x);
    int by = (int)floorf((float)state.pos.y);
    int bz = (int)floorf((float)state.pos.z);

    for (int dx = -1; dx < MC_GRID_X - 1; dx++) {
        for (int dy = -MC_GRID_Y; dy < 0; dy++) {
            for (int dz = -(MC_GRID_Z/2); dz <= MC_GRID_Z/2; dz++) {
                CBlockPos pos = {bx + dx, by + dy, bz + dz};
                CBlock block;
                mcenv_environment_get_block(env->mc, pos, &block);
                env->observations[idx++] = (block == CBLOCK_FULL_CUBE) ? 1.0f : 0.0f;
            }
        }
    }
}

void add_log(MCEnv* env) {
    int t = env->tick > 0 ? env->tick : 1;
    env->log.score += env->targets_reached;
    env->log.perf += env->targets_reached / 10.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.max_dist += env->max_dist_from_start;
    env->log.fell += (env->tick < env->max_ticks) ? 1.0f : 0.0f;
    env->log.blocks_placed += env->blocks_placed;
    env->log.targets_reached += env->targets_reached;
    env->log.sneak_frac += (float)env->sneak_ticks / t;
    env->log.place_frac += (float)env->place_ticks / t;
    env->log.avg_pitch += env->pitch_sum / t;
    env->log.on_ground_frac += (float)env->on_ground_ticks / t;
    env->log.n++;
}

void c_reset(MCEnv* env) {
    if (env->mc != NULL) {
        mcenv_environment_free(env->mc);
    }
    env->mc = mcenv_environment_new();
    setup_platform(env);

    CPlayerState pstate;
    mcenv_environment_get_player(env->mc, &pstate);
    pstate.pos = (CVec3){env->start_x, MC_START_Y, env->start_z};
    pstate.prev_pos = pstate.pos;
    pstate.vel = (CVec3){0, 0, 0};
    mcenv_environment_set_player(env->mc, &pstate);

    env->yaw = MC_START_YAW;
    env->pitch = MC_START_PITCH;
    env->tick = 0;
    env->episode_return = 0.0f;
    env->max_dist_from_start = 0.0f;
    env->blocks_placed = 0;
    env->targets_reached = 0;
    env->sneak_ticks = 0;
    env->place_ticks = 0;
    env->pitch_sum = 0.0f;
    env->on_ground_ticks = 0;

    mc_new_target(env);

    CPlayerInput input = mcenv_player_input_default();
    input.has_look = true;
    input.look_yaw = env->yaw;
    input.look_pitch = env->pitch;
    mcenv_environment_input(env->mc, input);
    mcenv_environment_tick(env->mc);

    if (env->renderer != NULL) {
        mcenv_demo3d_renderer_reset(env->renderer, env->mc);
    }

    memset(env->observations, 0, MC_OBS_TOTAL * sizeof(float));
    compute_observations(env);
}

void c_step(MCEnv* env) {
    env->tick++;

    int a_idx = 0;
    int forward_action  = (int)env->actions[a_idx++];
    int strafe_action   = (int)env->actions[a_idx++];
    int jump_action     = (int)env->actions[a_idx++];
    int sneak_action    = (int)env->actions[a_idx++];
    int yaw_action      = (int)env->actions[a_idx++];
    int pitch_action    = (int)env->actions[a_idx++];
    int place_action    = (int)env->actions[a_idx++];

    env->yaw += YAW_DELTAS[yaw_action];
    env->pitch += PITCH_DELTAS[pitch_action];
    if (env->pitch > 90.0f) env->pitch = 90.0f;
    if (env->pitch < -90.0f) env->pitch = -90.0f;

    CPlayerInput input = mcenv_player_input_default();
    input.forward = (float)(forward_action - 1);
    input.strafe = (float)(strafe_action - 1);
    input.jump = (jump_action == 1);
    input.sneak = (sneak_action == 1);
    input.sprint = false;
    input.has_look = true;
    input.look_yaw = env->yaw;
    input.look_pitch = env->pitch;
    input.place = (place_action == 1);
    input.break_block = false;

    // Count nearby non-platform blocks BEFORE tick to detect new placements
    int blocks_before = 0;
    {
        CPlayerState pre;
        mcenv_environment_get_player(env->mc, &pre);
        int bx = (int)floorf((float)pre.pos.x);
        int bz = (int)floorf((float)pre.pos.z);
        int px = (int)env->start_x, pz = (int)env->start_z;
        for (int cx = bx - 1; cx <= bx + 2; cx++) {
            for (int cz = bz - 1; cz <= bz + 2; cz++) {
                CBlockPos bp = {cx, 2, cz};
                CBlock blk;
                mcenv_environment_get_block(env->mc, bp, &blk);
                if (blk == CBLOCK_FULL_CUBE &&
                    !(cx == px && cz == pz)) {
                    blocks_before++;
                }
            }
        }
    }

    mcenv_environment_input(env->mc, input);
    mcenv_environment_tick(env->mc);

    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    // Count blocks AFTER tick (same area — player barely moved in 1 tick)
    int blocks_after = 0;
    {
        int bx = (int)floorf((float)state.pos.x);
        int bz = (int)floorf((float)state.pos.z);
        int px = (int)env->start_x, pz = (int)env->start_z;
        for (int cx = bx - 1; cx <= bx + 2; cx++) {
            for (int cz = bz - 1; cz <= bz + 2; cz++) {
                CBlockPos bp = {cx, 2, cz};
                CBlock blk;
                mcenv_environment_get_block(env->mc, bp, &blk);
                if (blk == CBLOCK_FULL_CUBE &&
                    !(cx == px && cz == pz)) {
                    blocks_after++;
                }
            }
        }
    }
    int placed_this_tick = blocks_after - blocks_before;
    if (placed_this_tick > 0) env->blocks_placed += placed_this_tick;

    // Diagnostics
    if (sneak_action == 1) env->sneak_ticks++;
    if (place_action == 1) env->place_ticks++;
    env->pitch_sum += env->pitch;
    if (state.on_ground) env->on_ground_ticks++;

    // ---- Reward computation ----
    float reward = 0.0f;

    // 1. Survival
    if (state.on_ground) {
        reward += env->rw_survival;
    }

    // 2. Getting closer to target (only while on solid ground)
    float dist = mc_dist_to_target(env, state.pos.x, state.pos.z);
    if (state.on_ground) {
        float delta_dist = env->prev_dist - dist;
        reward += env->rw_speed * delta_dist;
    }
    env->prev_dist = dist;

    // 3. Block placement reward (only for genuinely new blocks)
    if (placed_this_tick > 0) {
        reward += env->rw_block * placed_this_tick;
    }

    // 4. Target reached
    if (dist < MC_TARGET_REACH) {
        reward += env->rw_target_reach;
        env->targets_reached++;
        mc_new_target(env);
        dist = env->prev_dist; // prev_dist updated by mc_new_target
    }

    // Track progress
    float dist_from_start = sqrtf(
        (float)(state.pos.x - env->start_x) * (float)(state.pos.x - env->start_x) +
        (float)(state.pos.z - env->start_z) * (float)(state.pos.z - env->start_z));
    if (dist_from_start > env->max_dist_from_start) {
        env->max_dist_from_start = dist_from_start;
    }

    // Termination
    int done = 0;
    if (state.pos.y < MC_VOID_Y) {
        reward = env->rw_fall;
        done = 1;
    } else if (env->tick >= env->max_ticks) {
        done = 1;
    }

    env->rewards[0] = reward;
    env->episode_return += reward;
    env->terminals[0] = done ? 1.0f : 0.0f;

    if (done) {
        add_log(env);
        c_reset(env);
    } else {
        compute_observations(env);
    }
}

static const char* FWD_NAMES[3]   = {"back", "none", "fwd"};
static const char* STR_NAMES[3]   = {"right", "none", "left"};
static const char* BOOL_NAMES[2]  = {"no", "yes"};

void c_render(MCEnv* env) {
    if (env->renderer == NULL) {
        env->renderer = mcenv_demo3d_renderer_new(env->mc);
        clock_gettime(CLOCK_MONOTONIC, &env->render_last_time);
    }

    // Sleep until next 50ms tick boundary to pace eval at 20 TPS
    double tick_sec = mcenv_demo3d_tick_seconds();
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - env->render_last_time.tv_sec)
                   + (now.tv_nsec - env->render_last_time.tv_nsec) * 1e-9;
    double remaining = tick_sec - elapsed;
    if (remaining > 0.001) {
        struct timespec sleep_ts = {0, (long)(remaining * 1e9)};
        nanosleep(&sleep_ts, NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &env->render_last_time);

    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);
    mcenv_demo3d_renderer_push_player_state(env->renderer, &state);

    char hud[256];
    int fwd   = (int)env->actions[0]; if ((unsigned)fwd   >= 3) fwd   = 1;
    int str   = (int)env->actions[1]; if ((unsigned)str   >= 3) str   = 1;
    int jump  = (int)env->actions[2]; if ((unsigned)jump  >= 2) jump  = 0;
    int sneak = (int)env->actions[3]; if ((unsigned)sneak >= 2) sneak = 0;
    int yaw_i = (int)env->actions[4]; if ((unsigned)yaw_i >= 7) yaw_i = 3;
    int pit_i = (int)env->actions[5]; if ((unsigned)pit_i >= 7) pit_i = 3;
    int place = (int)env->actions[6]; if ((unsigned)place >= 2) place = 0;
    snprintf(hud, sizeof(hud),
        "fwd=%s str=%s snk=%s place=%s yaw=%+.0f pit=%+.0f | t=%d tgt=%.0f,%.0f dist=%.1f reached=%d blk=%d",
        FWD_NAMES[fwd], STR_NAMES[str],
        BOOL_NAMES[sneak], BOOL_NAMES[place],
        YAW_DELTAS[yaw_i], PITCH_DELTAS[pit_i],
        env->tick, env->target_x, env->target_z,
        mc_dist_to_target(env, 0, 0), // will recompute properly below
        env->targets_reached, env->blocks_placed);
    // Fix: compute actual distance for HUD
    {
        CPlayerState s;
        mcenv_environment_get_player(env->mc, &s);
        float d = mc_dist_to_target(env, s.pos.x, s.pos.z);
        snprintf(hud, sizeof(hud),
            "fwd=%s str=%s snk=%s place=%s yaw=%+.0f pit=%+.0f | t=%d tgt=%.0f,%.0f dist=%.1f reached=%d blk=%d",
            FWD_NAMES[fwd], STR_NAMES[str],
            BOOL_NAMES[sneak], BOOL_NAMES[place],
            YAW_DELTAS[yaw_i], PITCH_DELTAS[pit_i],
            env->tick, env->target_x, env->target_z,
            d, env->targets_reached, env->blocks_placed);
    }
    mcenv_demo3d_renderer_set_hud_text(env->renderer, hud);

    mcenv_demo3d_renderer_render(env->renderer, env->mc);
}

void c_close(MCEnv* env) {
    if (env->renderer != NULL) {
        mcenv_demo3d_renderer_free(env->renderer);
        env->renderer = NULL;
    }
    if (env->mc != NULL) {
        mcenv_environment_free(env->mc);
        env->mc = NULL;
    }
}
