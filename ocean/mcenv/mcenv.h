// MCEnv: Minecraft bridging + navigation environment using mcenv-codex.
// Agent bridges to random waypoints in a 100x100 area.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mcenv_codex.h"

#define MC_OBS_PLAYER 14
#define MC_OBS_TARGET 2
#define MC_GRID_X 7  // -3 to +3
#define MC_GRID_Y 3
#define MC_GRID_Z 7  // -3 to +3
#define MC_OBS_GRID (MC_GRID_X * MC_GRID_Y * MC_GRID_Z)
#define MC_OBS_TOTAL (MC_OBS_PLAYER + MC_OBS_TARGET + MC_OBS_GRID)

#define MC_START_X 25.5      // Center of 50x50 area
#define MC_START_Y 3.0
#define MC_START_Z 25.5
#define MC_START_YAW -90.0f
#define MC_START_PITCH -45.0f

#define MC_VOID_Y 2.5   // Just below platform (y=2..3) — falling = instant death
#define MC_PLATFORM_MAX_X 26 // Single block at x=25, ends at x=26
#define MC_AREA_SIZE 50
#define MC_TARGET_REACH 1.0f // Within 1 block = reached

// Default reward constants
#define MC_SURVIVAL_DEFAULT  0.001f
#define MC_BLOCK_DEFAULT     1.0f
#define MC_FALL_DEFAULT     -1.0f
#define MC_SPEED_DEFAULT     1.0f
#define MC_TARGET_DEFAULT    10.0f
#define MC_LOOK_REVERSAL_DEFAULT 0.05f
#define MC_SPRINT_JUMP_DEFAULT   0.0f
#define MC_REQUIRE_GROUND_DEFAULT 1
#define MC_FALL_SCALE_PHASE1_DEFAULT 1.0f
#define MC_SPEED_POWER_DEFAULT   1.0f
#define MC_ROT_PCT_ENABLED_DEFAULT 0
#define MC_PLACE_REPEAT_ENABLED_DEFAULT 0

// Yaw/pitch delta lookup tables (degrees)
#if defined(MC_FULL_ANGLE)
static const float YAW_DELTAS[153] = {-180.0f, -178.0f, -174.0f, -170.0f, -166.0f, -162.0f, -158.0f, -154.0f, -150.0f, -146.0f, -142.0f, -138.0f, -134.0f, -130.0f, -126.0f, -122.0f, -118.0f, -114.0f, -110.0f, -106.0f, -102.0f, -98.0f, -94.0f, -90.0f, -88.0f, -86.0f, -84.0f, -82.0f, -80.0f, -78.0f, -76.0f, -74.0f, -72.0f, -70.0f, -68.0f, -66.0f, -64.0f, -62.0f, -60.0f, -58.0f, -56.0f, -54.0f, -52.0f, -50.0f, -48.0f, -46.0f, -44.0f, -42.0f, -40.0f, -38.0f, -36.0f, -34.0f, -32.0f, -30.0f, -28.0f, -26.0f, -24.0f, -22.0f, -20.0f, -18.0f, -16.0f, -15.0f, -14.0f, -13.0f, -12.0f, -11.0f, -10.0f, -9.0f, -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f, 38.0f, 40.0f, 42.0f, 44.0f, 46.0f, 48.0f, 50.0f, 52.0f, 54.0f, 56.0f, 58.0f, 60.0f, 62.0f, 64.0f, 66.0f, 68.0f, 70.0f, 72.0f, 74.0f, 76.0f, 78.0f, 80.0f, 82.0f, 84.0f, 86.0f, 88.0f, 90.0f, 94.0f, 98.0f, 102.0f, 106.0f, 110.0f, 114.0f, 118.0f, 122.0f, 126.0f, 130.0f, 134.0f, 138.0f, 142.0f, 146.0f, 150.0f, 154.0f, 158.0f, 162.0f, 166.0f, 170.0f, 174.0f, 178.0f, 180.0f};
static const float PITCH_DELTAS[107] = {-90.0f, -88.0f, -86.0f, -84.0f, -82.0f, -80.0f, -78.0f, -76.0f, -74.0f, -72.0f, -70.0f, -68.0f, -66.0f, -64.0f, -62.0f, -60.0f, -58.0f, -56.0f, -54.0f, -52.0f, -50.0f, -48.0f, -46.0f, -44.0f, -42.0f, -40.0f, -38.0f, -36.0f, -34.0f, -32.0f, -30.0f, -28.0f, -26.0f, -24.0f, -22.0f, -20.0f, -18.0f, -16.0f, -15.0f, -14.0f, -13.0f, -12.0f, -11.0f, -10.0f, -9.0f, -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f, 38.0f, 40.0f, 42.0f, 44.0f, 46.0f, 48.0f, 50.0f, 52.0f, 54.0f, 56.0f, 58.0f, 60.0f, 62.0f, 64.0f, 66.0f, 68.0f, 70.0f, 72.0f, 74.0f, 76.0f, 78.0f, 80.0f, 82.0f, 84.0f, 86.0f, 88.0f, 90.0f};
#elif defined(MC_BC_ACTIONS)
static const float YAW_DELTAS[7]   = {-180.0f, -15.0f, -1.0f, 0.0f, 1.0f, 15.0f, 180.0f};
static const float PITCH_DELTAS[7] = {-180.0f, -15.0f, -1.0f, 0.0f, 1.0f, 15.0f, 180.0f};
#else
static const float YAW_DELTAS[25]  = {-180.0f,-135.0f,-90.0f,-45.0f,-15.0f,-11.25f,-7.5f,-3.75f,-1.0f,-0.75f,-0.5f,-0.25f,0.0f,0.25f,0.5f,0.75f,1.0f,3.75f,7.5f,11.25f,15.0f,45.0f,90.0f,135.0f,180.0f};
static const float PITCH_DELTAS[25]= {-180.0f,-135.0f,-90.0f,-45.0f,-15.0f,-11.25f,-7.5f,-3.75f,-1.0f,-0.75f,-0.5f,-0.25f,0.0f,0.25f,0.5f,0.75f,1.0f,3.75f,7.5f,11.25f,15.0f,45.0f,90.0f,135.0f,180.0f};
#endif

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float max_dist;
    float max_height;
    float fell;
    float blocks_placed;
    float targets_reached;
    float sneak_frac;
    float avg_pitch;
    float on_ground_frac;
    float phase;
    float rw_air;
    float rw_sprint_jump;
    float rw_speed;
    float rw_block;
    float rw_target;
    float rw_look_reversal;
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
    float max_height;
    int blocks_placed;
    int targets_reached;
    int lifetime_blocks;   // Persists across resets — drives global curriculum
    float avg_targets_ema; // EMA of targets/episode (~20 ep window)
    int phase2_unlocked;   // Ratchet: once phase 2 activates, stays permanent
    int phase3_unlocked;
    float target_x;
    float target_z;
    float rw_survival;
    float rw_block;
    float rw_fall;
    float rw_speed;
    float rw_target_reach;
    float rw_look_reversal;
    int curriculum_phase;      // 0=block only, 1=block+target, 2=target only
    int phase_transition;      // blocks_placed threshold to advance phase
    Demo3dRenderer* renderer;
    struct timespec render_last_time;
    int force_phase2;  // Toggle via P key during eval render
    float rw_sprint_jump;
    int require_ground;
    float fall_scale_phase1;
    float speed_power;
    int rot_pct_enabled;
    int place_repeat_enabled;
    int camera_locked; // Toggle via L key: lock camera to agent's look

    // Sprint-jump progress tracking
    int sj_timer;
    float sj_dist;
    float sj_dot;

    // Diagnostic accumulators
    int sneak_ticks;
    float pitch_sum;
    int on_ground_ticks;

    // Reward source accumulators (per episode)
    float rw_air_sum;
    float rw_sprint_jump_sum;
    float rw_speed_sum;
    float rw_block_sum;
    float rw_target_sum;
    float rw_look_reversal_sum;

    // Yaw reversal tracking
    float prev_yaw_delta;
};

static float mc_dist_to_target(MCEnv* env, double px, double pz) {
    float dx = (float)px - env->target_x;
    float dz = (float)pz - env->target_z;
    return sqrtf(dx*dx + dz*dz);
}

// Match glibc rand_r: LCG returning 15 bits (0-32767)
static unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static void mc_new_target(MCEnv* env) {
    // Random target in [0.5, 99.5] x [0.5, 99.5], at least 3 blocks from player
    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);
    for (int attempt = 0; attempt < 100; attempt++) {
        float tx = 0.5f + (float)(xorshift32(&env->rng) % MC_AREA_SIZE);
        float tz = 0.5f + (float)(xorshift32(&env->rng) % MC_AREA_SIZE);
        float dx = tx - (float)state.pos.x;
        float dz = tz - (float)state.pos.z;
        if (dx*dx + dz*dz >= 9.0f) {
            env->target_x = tx;
            env->target_z = tz;
            env->prev_dist = mc_dist_to_target(env, state.pos.x, state.pos.z);
            return;
        }
    }
    env->target_x = 0.5f + (float)(xorshift32(&env->rng) % MC_AREA_SIZE);
    env->target_z = 0.5f + (float)(xorshift32(&env->rng) % MC_AREA_SIZE);
    env->prev_dist = mc_dist_to_target(env, state.pos.x, state.pos.z);
}

static void setup_platform(MCEnv* env) {
    // Single block at y=2 under start
    CBlockPos platform_pos = {(int)env->start_x, 2, (int)env->start_z};
    mcenv_environment_set_block(env->mc, platform_pos, CBLOCK_FULL_CUBE);
}

static int mc_get_phase(MCEnv* env) {
    if (env->force_phase2) return 2;
    if (env->phase_transition <= 0) return env->curriculum_phase;
    int pt = env->phase_transition;
    if (env->phase3_unlocked) return 3;
    if (env->lifetime_blocks >= pt * 5) {
        if (!env->phase2_unlocked && env->avg_targets_ema > 2.0f)
            env->phase2_unlocked = 1;
        if (env->phase2_unlocked) {
            if (!env->phase3_unlocked && env->avg_targets_ema > 4.0f)
                env->phase3_unlocked = 1;
            if (env->phase3_unlocked) return 3;
            return 2;
        }
    }
    if (env->lifetime_blocks >= pt) return 1;
    return 0;
}

static void compute_observations(MCEnv* env) {
    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    int idx = 0;

    // Player state (14 values)
    env->observations[idx++] = (float)(state.pos.x - env->start_x) / 50.0f;
    env->observations[idx++] = (float)(state.pos.y - env->start_y) / 10.0f;
    env->observations[idx++] = (float)(state.pos.z - env->start_z) / 10.0f;
    env->observations[idx++] = (float)(state.pos.x - floorf((float)state.pos.x));
    env->observations[idx++] = (float)(state.pos.y - floorf((float)state.pos.y));
    env->observations[idx++] = (float)(state.pos.z - floorf((float)state.pos.z));
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

    // Local block grid: 7x3x7 (±3 in x and z, symmetric)
    int bx = (int)floorf((float)state.pos.x);
    int by = (int)floorf((float)state.pos.y);
    int bz = (int)floorf((float)state.pos.z);

    for (int dx = -(MC_GRID_X/2); dx <= MC_GRID_X/2; dx++) {
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
    env->log.max_height += env->max_height;
    env->log.fell += (env->tick < env->max_ticks) ? 1.0f : 0.0f;
    env->log.blocks_placed += env->blocks_placed;
    env->log.targets_reached += env->targets_reached;
    env->log.sneak_frac += (float)env->sneak_ticks / t;
    env->log.avg_pitch += env->pitch_sum / t;
    env->log.on_ground_frac += (float)env->on_ground_ticks / t;
    env->log.phase += (float)mc_get_phase(env);
    float rw_total = fabsf(env->rw_air_sum) + fabsf(env->rw_sprint_jump_sum)
                   + fabsf(env->rw_speed_sum) + fabsf(env->rw_block_sum)
                   + fabsf(env->rw_target_sum) + fabsf(env->rw_look_reversal_sum);
    if (rw_total > 0.0f) {
        float inv = 1.0f / rw_total;
        env->log.rw_air += fabsf(env->rw_air_sum) * inv;
        env->log.rw_sprint_jump += fabsf(env->rw_sprint_jump_sum) * inv;
        env->log.rw_speed += fabsf(env->rw_speed_sum) * inv;
        env->log.rw_block += fabsf(env->rw_block_sum) * inv;
        env->log.rw_target += fabsf(env->rw_target_sum) * inv;
        env->log.rw_look_reversal += fabsf(env->rw_look_reversal_sum) * inv;
    }
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
    env->max_height = 0.0f;
    // Update EMA of targets/episode (~20 episode window)
    env->avg_targets_ema = 0.95f * env->avg_targets_ema + 0.05f * (float)env->targets_reached;
    env->blocks_placed = 0;
    env->targets_reached = 0;
    // NOTE: lifetime_* fields are NOT reset — they drive global curriculum
    env->sj_timer = 0;
    env->sj_dist = 0.0f;
    env->sj_dot = 0.0f;

    env->sneak_ticks = 0;
    env->pitch_sum = 0.0f;
    env->on_ground_ticks = 0;

    env->rw_air_sum = 0.0f;
    env->rw_sprint_jump_sum = 0.0f;
    env->rw_speed_sum = 0.0f;
    env->rw_block_sum = 0.0f;
    env->rw_target_sum = 0.0f;
    env->rw_look_reversal_sum = 0.0f;
    env->prev_yaw_delta = 0.0f;

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
    int sprint_action   = (int)env->actions[a_idx++];
    int yaw_action      = (int)env->actions[a_idx++];
    int pitch_action    = (int)env->actions[a_idx++];
    int place_action    = (int)env->actions[a_idx++];
#ifdef MC_BC_ACTIONS
    int rot_pct_action  = (int)env->actions[a_idx++];
    float rot_pct = rot_pct_action * 0.25f;
    float yaw_delta = YAW_DELTAS[yaw_action] * rot_pct;
    env->yaw += yaw_delta;
    env->pitch += PITCH_DELTAS[pitch_action] * rot_pct;
#else
    float yaw_delta = YAW_DELTAS[yaw_action];
    env->yaw += yaw_delta;
    env->pitch += PITCH_DELTAS[pitch_action];
#endif
    if (env->pitch > 90.0f) env->pitch = 90.0f;
    if (env->pitch < -90.0f) env->pitch = -90.0f;

    int phase = mc_get_phase(env);

    CPlayerInput input = mcenv_player_input_default();
    input.forward = (float)(forward_action - 1);
    input.strafe = (float)(strafe_action - 1);
    input.jump = (jump_action == 1);
    input.sneak = (sneak_action == 1);
    input.sprint = (sprint_action == 1);
    input.has_look = true;
    input.look_yaw = env->yaw;
    input.look_pitch = env->pitch;
    // place_action: 0=no, 1=single (edge-triggered), 2=repeat (every tick)
    input.place = (place_action >= 1);
    input.place_repeat = (place_action == 2);
    input.break_block = false;



    // --- Tick + block placement detection ---
    mcenv_environment_input(env->mc, input);
    mcenv_environment_tick(env->mc);

    float place_scale = 0.0f;
    CBlockPos placed_pos;
    if (mcenv_environment_last_placed_block(env->mc, &placed_pos)) {
        int px = (int)env->start_x, pz = (int)env->start_z;
        if (!(placed_pos.x == px && placed_pos.z == pz) && placed_pos.y == 2) {
            place_scale = 1.0f;
            env->blocks_placed++;
            env->lifetime_blocks++;
        }
    }

    CPlayerState state;
    mcenv_environment_get_player(env->mc, &state);

    // Diagnostics
    if (sneak_action == 1) env->sneak_ticks++;
    env->pitch_sum += env->pitch;
    if (state.on_ground) env->on_ground_ticks++;

    // ---- Global curriculum (phase computed above for input overrides) ----
    // Phase 0: block reward only — learn bridging mechanics
    // Phase 1: block + target + speed, on_ground gated — learn navigation
    // Phase 2: same as phase 1 but on_ground gate removed + sprint-jump reward

    float block_weight = 1.0f;
    float target_weight = 0.0f;
    int require_ground = env->require_ground;
    if (phase >= 1) {
        target_weight = 1.0f;
        block_weight = 1.0f;
        if (phase >= 2) {
            block_weight = 0.0f;
        }
    }

    // ---- Reward computation ----
    float reward = 0.0f;
    float rw_air = 0.0f, rw_sprint_jump = 0.0f, rw_speed = 0.0f;
    float rw_block = 0.0f, rw_target = 0.0f;

    // Facing-toward-target dot product (yaw only)
    float yaw_rad = env->yaw * (3.14159265f / 180.0f);
    float face_x = -sinf(yaw_rad), face_z = cosf(yaw_rad);
    float tgt_dx = env->target_x - (float)state.pos.x;
    float tgt_dz = env->target_z - (float)state.pos.z;
    float tgt_len = sqrtf(tgt_dx * tgt_dx + tgt_dz * tgt_dz);
    float facing_dot = 0.0f;
    if (tgt_len > 0.01f) facing_dot = (face_x * tgt_dx + face_z * tgt_dz) / tgt_len;
    // facing_dot can be negative (facing away from target)

    // 2. Speed: getting closer to target (phases 1+2)
    float dist = mc_dist_to_target(env, state.pos.x, state.pos.z);

    // Sprint-jump: progress reward + airborne block bonus timer
    if (phase >= 1) {
        if (env->sj_timer > 0) {
            env->sj_timer--;
            if (env->sj_dist - dist > 0.05f) {
                float sj_rw = env->rw_sprint_jump * env->sj_dot;
                reward += sj_rw; rw_sprint_jump += sj_rw;
                env->sj_timer = 0;
            }
        }
        if (forward_action == 2 && sprint_action == 1 && jump_action == 1 && state.on_ground && env->sj_timer == 0) {
            env->sj_timer = 20;
            env->sj_dist = dist;
            env->sj_dot = facing_dot;
        }
    }
    float delta_dist = env->prev_dist - dist;
    if (target_weight > 0.0f && (!require_ground || state.on_ground)) {
        float abs_delta = fabsf(delta_dist);
        float shaped_delta = (delta_dist >= 0.0f ? 1.0f : -1.0f) * powf(abs_delta, env->speed_power);
        float s_rw = env->rw_speed * shaped_delta * target_weight;
        reward += s_rw; rw_speed += s_rw;
    }
    env->prev_dist = dist;

    // Facing bonus: disabled (old code had none)
    float facing_rw = 0.0f;

    // 3. Block placement reward
    if (place_scale > 0.0f && block_weight > 0.0f) {
        float b_rw = env->rw_block * place_scale * block_weight;
        reward += b_rw; rw_block += b_rw;
    }

    // 4. Target reached — always detect, reward with time bonus in phases 1+2
    if (dist < MC_TARGET_REACH && (!require_ground || state.on_ground)) {
        if (target_weight > 0.0f) {
            float escalate = 1.0f + 0.5f * (float)env->targets_reached;
            float time_bonus = 1.0f + (float)(env->max_ticks - env->tick) / (float)env->max_ticks;
            float t_rw = env->rw_target_reach * escalate * time_bonus * target_weight;
            reward += t_rw; rw_target += t_rw;
        }
        env->targets_reached++;
        mc_new_target(env);
        dist = env->prev_dist;
    }

    // Yaw reversal penalty: penalize sign changes in consecutive yaw turns
    float rw_look_reversal = 0.0f;
    if (phase >= 1 && env->prev_yaw_delta != 0.0f && yaw_delta != 0.0f &&
        ((env->prev_yaw_delta > 0.0f) != (yaw_delta > 0.0f))) {
        rw_look_reversal = -env->rw_look_reversal;
        reward += rw_look_reversal;
    }
    env->prev_yaw_delta = yaw_delta;

    env->rw_air_sum += rw_air;
    env->rw_sprint_jump_sum += rw_sprint_jump;
    env->rw_speed_sum += rw_speed;
    env->rw_block_sum += rw_block;
    env->rw_target_sum += rw_target;
    env->rw_look_reversal_sum += rw_look_reversal;

    // Track progress
    float dist_from_start = sqrtf(
        (float)(state.pos.x - env->start_x) * (float)(state.pos.x - env->start_x) +
        (float)(state.pos.z - env->start_z) * (float)(state.pos.z - env->start_z));
    if (dist_from_start > env->max_dist_from_start) {
        env->max_dist_from_start = dist_from_start;
    }
    float height = (float)state.pos.y - env->start_y;
    if (height > env->max_height) {
        env->max_height = height;
    }


    // Termination
    int done = 0;
    if (state.pos.y < MC_VOID_Y) {
        reward = (phase >= 1) ? env->rw_fall * env->fall_scale_phase1 : env->rw_fall;
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

static const char* FWD_NAMES[3]   = {"back", "none", " fwd"};
static const char* STR_NAMES[3]   = {" rgt", "none", "left"};
static const char* BOOL_NAMES[2]  = {" no", "yes"};
static const char* PLACE_NAMES[3] = {"  no", " sgl", " rpt"};

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
    int sprint= (int)env->actions[4]; if ((unsigned)sprint>= 2) sprint= 0;
#if defined(MC_FULL_ANGLE)
    int yaw_i = (int)env->actions[5]; if ((unsigned)yaw_i >= 153) yaw_i = 76;
    int pit_i = (int)env->actions[6]; if ((unsigned)pit_i >= 107) pit_i = 53;
#elif defined(MC_BC_ACTIONS)
    int yaw_i = (int)env->actions[5]; if ((unsigned)yaw_i >= 7) yaw_i = 3;
    int pit_i = (int)env->actions[6]; if ((unsigned)pit_i >= 7) pit_i = 3;
#else
    int yaw_i = (int)env->actions[5]; if ((unsigned)yaw_i >= 25) yaw_i = 12;
    int pit_i = (int)env->actions[6]; if ((unsigned)pit_i >= 25) pit_i = 12;
#endif
    int place = (int)env->actions[7]; if ((unsigned)place >= 3) place = 0;
    {
        CPlayerState s;
        mcenv_environment_get_player(env->mc, &s);
        float d = mc_dist_to_target(env, s.pos.x, s.pos.z);
        int phase = mc_get_phase(env);
        // Sprint-jump flash: show dot when timer active
        char sj_buf[16] = "";
        if (env->sj_timer > 0 && env->sj_dot > 0.0f)
            snprintf(sj_buf, sizeof(sj_buf), " SPRJ=%.2f", env->sj_dot);
        snprintf(hud, sizeof(hud),
            "fwd=%s str=%s jmp=%s snk=%s spr=%s plc=%d yaw=%+7.1f pit=%+7.1f | t=%4d tgt=%5.0f,%5.0f dist=%5.1f rch=%2d blk=%3d ph=%d%s%s%s",
            FWD_NAMES[fwd], STR_NAMES[str], BOOL_NAMES[jump],
            BOOL_NAMES[sneak], BOOL_NAMES[sprint], place,
            YAW_DELTAS[yaw_i], PITCH_DELTAS[pit_i],
            env->tick, env->target_x, env->target_z,
            d, env->targets_reached, env->blocks_placed,
            phase, env->force_phase2 ? "[P]" : "",
            env->camera_locked ? "[L]" : "",
            sj_buf);
    }
    mcenv_demo3d_renderer_set_hud_text(env->renderer, hud);

    if (mcenv_demo3d_is_key_pressed(MCENV_DEMO3D_KEY_P))
        env->force_phase2 = !env->force_phase2;

    if (mcenv_demo3d_is_key_pressed(MCENV_DEMO3D_KEY_L)) {
        env->camera_locked = !env->camera_locked;
        mcenv_demo3d_renderer_set_camera_locked(env->renderer, env->camera_locked);
    }

    mcenv_demo3d_renderer_set_nav_target(env->renderer, env->target_x, env->target_z);
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
