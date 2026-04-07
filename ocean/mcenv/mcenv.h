// MCEnv: Minecraft bridging environment using mcenv-codex physics engine.
// Goal: maximize x position by learning to place blocks and bridge forward.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "mcenv_codex.h"

#define MC_OBS_PLAYER 11
#define MC_GRID_X 7
#define MC_GRID_Y 3
#define MC_GRID_Z 3
#define MC_OBS_GRID (MC_GRID_X * MC_GRID_Y * MC_GRID_Z)
#define MC_OBS_TOTAL (MC_OBS_PLAYER + MC_OBS_GRID)

#define MC_START_X 1.5       // Right at edge of platform
#define MC_START_Y 3.0
#define MC_START_Z 1.5
#define MC_START_YAW -90.0f  // Facing +x
#define MC_START_PITCH -45.0f // Looking down — close to bridging angle

#define MC_VOID_Y 0.0
#define MC_MAX_Z_DRIFT 10.0
#define MC_PLATFORM_MAX_X 2  // Platform blocks span x: -1..1, block x=1 ends at x=2

// Reward constants
#define MC_SURVIVAL_REWARD  0.001f // Tiny per-tick survival (full ep = ~1.0)
#define MC_BLOCK_REWARD     20.0f  // Huge reward for placing a block beyond platform
#define MC_FALL_PENALTY    -1.0f   // Penalty for falling

// Yaw/pitch delta lookup tables (degrees)
static const float YAW_DELTAS[7]   = {-15.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 15.0f};
static const float PITCH_DELTAS[7] = {-15.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 15.0f};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float max_x;
    float fell;
    float blocks_placed;
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
    float prev_x;
    float yaw;
    float pitch;
    int tick;
    int max_ticks;
    float episode_return;
    float max_x;
    float max_block_x;
    int blocks_placed;
    Demo3dRenderer* renderer;

    // Diagnostic accumulators
    int sneak_ticks;
    int place_ticks;
    float pitch_sum;
    int on_ground_ticks;
};

static void setup_platform(MCEnv* env) {
    for (int x = -1; x <= 1; x++) {
        for (int z = 0; z <= 2; z++) {
            CBlockPos pos = {x, 2, z};
            mcenv_environment_set_block(env->mc, pos, CBLOCK_FULL_CUBE);
        }
    }
}

static void compute_observations(MCEnv* env) {
    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    int idx = 0;

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
    env->log.score += env->max_x;
    env->log.perf += env->max_x / 50.0f;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.max_x += env->max_x;
    env->log.blocks_placed += env->blocks_placed;
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
    pstate.pos = (CVec3){MC_START_X, MC_START_Y, MC_START_Z};
    pstate.prev_pos = pstate.pos;
    pstate.vel = (CVec3){0, 0, 0};
    mcenv_environment_set_player(env->mc, &pstate);

    env->yaw = MC_START_YAW;
    env->pitch = MC_START_PITCH;
    env->prev_x = MC_START_X;
    env->tick = 0;
    env->episode_return = 0.0f;
    env->max_x = 0.0f;
    env->max_block_x = (float)MC_PLATFORM_MAX_X;
    env->blocks_placed = 0;
    env->sneak_ticks = 0;
    env->place_ticks = 0;
    env->pitch_sum = 0.0f;
    env->on_ground_ticks = 0;

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

    // Decode actions
    int a_idx = 0;
    int forward_action  = (int)env->actions[a_idx++];
    int strafe_action   = (int)env->actions[a_idx++];
    int jump_action     = (int)env->actions[a_idx++];
    int sneak_action    = (int)env->actions[a_idx++];
    int yaw_action      = (int)env->actions[a_idx++];
    int pitch_action    = (int)env->actions[a_idx++];
    int place_action    = (int)env->actions[a_idx++];

    // Update look direction
    env->yaw += YAW_DELTAS[yaw_action];
    env->pitch += PITCH_DELTAS[pitch_action];
    if (env->pitch > 90.0f) env->pitch = 90.0f;
    if (env->pitch < -90.0f) env->pitch = -90.0f;

    // Build input
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

    mcenv_environment_input(env->mc, input);
    mcenv_environment_tick(env->mc);

    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    // Accumulate diagnostics
    if (sneak_action == 1) env->sneak_ticks++;
    if (place_action == 1) env->place_ticks++;
    env->pitch_sum += env->pitch;
    if (state.on_ground) env->on_ground_ticks++;

    // ---- Reward computation ----
    float reward = 0.0f;

    // 1. Survival: reward for being on ground (teaches not-falling)
    if (state.on_ground) {
        reward += MC_SURVIVAL_REWARD;
    }

    // 2. Block placement: big reward for extending the bridge
    if (place_action == 1) {
        int bx = (int)floorf((float)state.pos.x);
        for (int checkx = bx; checkx <= bx + 3; checkx++) {
            if (checkx < MC_PLATFORM_MAX_X) continue;
            for (int checkz = (int)floorf((float)state.pos.z) - 1;
                 checkz <= (int)floorf((float)state.pos.z) + 1; checkz++) {
                CBlockPos bp = {checkx, 2, checkz};
                CBlock blk;
                mcenv_environment_get_block(env->mc, bp, &blk);
                if (blk == CBLOCK_FULL_CUBE && (float)checkx >= env->max_block_x) {
                    env->max_block_x = (float)(checkx + 1);
                    env->blocks_placed++;
                    reward += MC_BLOCK_REWARD;
                }
            }
        }
    }

    // Track progress
    env->prev_x = (float)state.pos.x;
    float x_progress = (float)(state.pos.x - env->start_x);
    if (x_progress > env->max_x) {
        env->max_x = x_progress;
    }

    // Check termination
    int done = 0;
    if (state.pos.y < MC_VOID_Y) {
        reward = MC_FALL_PENALTY;
        done = 1;
        env->log.fell++;
    } else if (fabs(state.pos.z - env->start_z) > MC_MAX_Z_DRIFT) {
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
    }

    char hud[256];
    int fwd   = (int)env->actions[0]; if ((unsigned)fwd   >= 3) fwd   = 1;
    int str   = (int)env->actions[1]; if ((unsigned)str   >= 3) str   = 1;
    int jump  = (int)env->actions[2]; if ((unsigned)jump  >= 2) jump  = 0;
    int sneak = (int)env->actions[3]; if ((unsigned)sneak >= 2) sneak = 0;
    int yaw_i = (int)env->actions[4]; if ((unsigned)yaw_i >= 7) yaw_i = 3;
    int pit_i = (int)env->actions[5]; if ((unsigned)pit_i >= 7) pit_i = 3;
    int place = (int)env->actions[6]; if ((unsigned)place >= 2) place = 0;
    snprintf(hud, sizeof(hud),
        "Actions: fwd=%s  strafe=%s  jump=%s  sneak=%s  yaw=%+.0f  pitch=%+.0f  place=%s | tick=%d  max_x=%.1f  blocks=%d",
        FWD_NAMES[fwd], STR_NAMES[str],
        BOOL_NAMES[jump], BOOL_NAMES[sneak],
        YAW_DELTAS[yaw_i], PITCH_DELTAS[pit_i],
        BOOL_NAMES[place],
        env->tick, env->max_x, env->blocks_placed);
    mcenv_demo3d_renderer_set_hud_text(env->renderer, hud);

    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);
    mcenv_demo3d_renderer_push_player_state(env->renderer, &state);
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
