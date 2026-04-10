#ifndef MCENV_CUDA_CUH
#define MCENV_CUDA_CUH

#include <cstdint>
#include "math.cuh"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define GRAVITY         0.08
#define DRAG            0.98
#define FRICTION        0.91
#define TINY_VEL_THRESH 0.003
#define STEP_HEIGHT     0.6
#define COLLISION_EPS   1.0e-7
#define SURVIVAL_REACH  4.5
#define MAX_COLLIDERS   128

// ---------------------------------------------------------------------------
// AABB — mirrors Rust aabb::Aabb
// ---------------------------------------------------------------------------

struct Aabb {
    Vec3 min, max;
};

__host__ __device__ inline Aabb aabb_new(Vec3 mn, Vec3 mx) {
    Aabb a; a.min = mn; a.max = mx; return a;
}

// Mirrors Aabb::for_entity — f32 rounding for width/height
__host__ __device__ inline Aabb aabb_for_entity(Vec3 pos, double width, double height) {
    float width_f = (float)width;
    float height_f = (float)height;
    double half = (double)(width_f * 0.5f);
    return aabb_new(
        vec3_new(pos.x - half, pos.y, pos.z - half),
        vec3_new(pos.x + half, pos.y + (double)height_f, pos.z + half)
    );
}

__host__ __device__ inline double aabb_width(Aabb a) { return a.max.x - a.min.x; }
__host__ __device__ inline double aabb_height(Aabb a) { return a.max.y - a.min.y; }
__host__ __device__ inline double aabb_depth(Aabb a) { return a.max.z - a.min.z; }

__host__ __device__ inline Aabb aabb_offset(Aabb a, Vec3 d) {
    return aabb_new(vec3_add(a.min, d), vec3_add(a.max, d));
}

__host__ __device__ inline Aabb aabb_stretch(Aabb a, Vec3 d) {
    Vec3 mn = a.min, mx = a.max;
    if (d.x < 0.0) mn.x += d.x; else mx.x += d.x;
    if (d.y < 0.0) mn.y += d.y; else mx.y += d.y;
    if (d.z < 0.0) mn.z += d.z; else mx.z += d.z;
    return aabb_new(mn, mx);
}

__host__ __device__ inline Aabb aabb_expand(Aabb a, double amt) {
    return aabb_new(
        vec3_new(a.min.x - amt, a.min.y - amt, a.min.z - amt),
        vec3_new(a.max.x + amt, a.max.y + amt, a.max.z + amt)
    );
}

__host__ __device__ inline bool aabb_intersects(Aabb a, Aabb b) {
    return a.max.x > b.min.x && a.min.x < b.max.x
        && a.max.y > b.min.y && a.min.y < b.max.y
        && a.max.z > b.min.z && a.min.z < b.max.z;
}

__host__ __device__ inline Vec3 aabb_center(Aabb a) {
    return vec3_new(
        (a.min.x + a.max.x) * 0.5,
        (a.min.y + a.max.y) * 0.5,
        (a.min.z + a.max.z) * 0.5
    );
}

// ---------------------------------------------------------------------------
// BlockPos
// ---------------------------------------------------------------------------

struct BlockPos {
    int x, y, z;
};

__host__ __device__ inline BlockPos blockpos_new(int x, int y, int z) {
    BlockPos p; p.x = x; p.y = y; p.z = z; return p;
}

__host__ __device__ inline BlockPos blockpos_from_vec3(Vec3 v) {
    return blockpos_new((int)floor(v.x), (int)floor(v.y), (int)floor(v.z));
}

// ---------------------------------------------------------------------------
// McEnvInput — matches Rust PlayerInput via recording format
// ---------------------------------------------------------------------------

struct McEnvInput {
    float forward;
    float strafe;
    int jump;
    int sprint;
    int sneak;
    int has_look;
    float look_yaw;
    float look_pitch;
    int place;
    int place_repeat;
    int break_block;
};

// ---------------------------------------------------------------------------
// McEnvState — full per-environment mutable state
// ---------------------------------------------------------------------------

struct McEnvState {
    // Player physics (f64)
    double pos_x, pos_y, pos_z;
    double prev_pos_x, prev_pos_y, prev_pos_z;
    double vel_x, vel_y, vel_z;
    double bb_min_x, bb_min_y, bb_min_z;
    double bb_max_x, bb_max_y, bb_max_z;

    // Player state
    float yaw;
    float pitch;
    int on_ground;
    int horizontal_collision;
    int vertical_collision;
    int sneaking;
    int sprinting;
    uint8_t sprint_toggle_timer;
    int prev_forward_positive;

    // Constants (never change after init)
    double width;   // 0.6
    double height;  // 1.8
    double step_height; // 0.6
    float movement_speed_attr; // 0.1

    // Edge detection for place/break
    int prev_place_pressed;
    int prev_break_pressed;
    int latched_place;
    int latched_break;

    // Last placed block (set during tick, -1 y = none)
    int last_placed_x, last_placed_y, last_placed_z;
    int last_placed_valid;

    // ---- MCEnv episode/curriculum state (from mcenv.h) ----

    float start_x, start_y, start_z;
    float target_x, target_z;
    float prev_dist;           // distance to target last tick
    float episode_return;
    float max_dist_from_start;
    float max_height;
    int   tick;
    int   max_ticks;
    int   blocks_placed;
    int   targets_reached;

    // Curriculum (persists across resets)
    int   lifetime_blocks;
    float avg_targets_ema;
    int   phase2_unlocked;

    // Sprint-jump progress tracking
    int   sj_timer;
    float sj_dist;

    // Diagnostics
    int   sneak_ticks;
    float pitch_sum;
    int   on_ground_ticks;

    // Reward source accumulators (per episode)
    float rw_air_sum;
    float rw_sprint_jump_sum;
    float rw_speed_sum;
    float rw_block_sum;
    float rw_target_sum;

    // RNG
    unsigned int rng;
};

// ---------------------------------------------------------------------------
// McEnvConfig — shared read-only config, one per launch
// ---------------------------------------------------------------------------

struct McEnvConfig {
    int   max_ticks;
    float rw_survival;
    float rw_block;
    float rw_fall;
    float rw_speed;
    float rw_target_reach;
    int   curriculum_phase;
    int   phase_transition;
    int   force_phase2;
};

// ---------------------------------------------------------------------------
// McEnvLog — per-env episode log accumulator (mirrors mcenv.h Log)
// Layout must match exactly so we can memcpy into CPU Log structs.
// ---------------------------------------------------------------------------

struct McEnvLog {
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
    float n;
};

// ---------------------------------------------------------------------------
// MCEnv constants (from mcenv.h)
// ---------------------------------------------------------------------------

#define MC_OBS_PLAYER 14
#define MC_OBS_TARGET 2
#define MC_GRID_X 7  // -3 to +3
#define MC_GRID_Y 3
#define MC_GRID_Z 7  // -3 to +3
#define MC_OBS_GRID (MC_GRID_X * MC_GRID_Y * MC_GRID_Z)
#define MC_OBS_TOTAL (MC_OBS_PLAYER + MC_OBS_TARGET + MC_OBS_GRID)
#define MC_NUM_ATNS 9

#define MC_START_X 25.5f
#define MC_START_Y 3.0f
#define MC_START_Z 25.5f
#define MC_START_YAW  (-90.0f)
#define MC_START_PITCH (-45.0f)

#define MC_VOID_Y     2.5f
#define MC_AREA_SIZE  50
#define MC_TARGET_REACH 1.0f

// ---------------------------------------------------------------------------
// Lookup tables (action decoding)
// ---------------------------------------------------------------------------

__device__ __constant__ float d_YAW_DELTAS[7]   = {-180.0f, -15.0f, -1.0f, 0.0f, 1.0f, 15.0f, 180.0f};
__device__ __constant__ float d_PITCH_DELTAS[7]  = {-180.0f, -15.0f, -1.0f, 0.0f, 1.0f, 15.0f, 180.0f};

// ---------------------------------------------------------------------------
// RNG — xorshift32 (replaces rand_r for GPU)
// ---------------------------------------------------------------------------

__host__ __device__ inline unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ---------------------------------------------------------------------------
// Helper: Vec3 <-> state fields
// ---------------------------------------------------------------------------

__host__ __device__ inline Vec3 state_pos(const McEnvState* s) {
    return vec3_new(s->pos_x, s->pos_y, s->pos_z);
}
__host__ __device__ inline Vec3 state_vel(const McEnvState* s) {
    return vec3_new(s->vel_x, s->vel_y, s->vel_z);
}
__host__ __device__ inline Aabb state_bb(const McEnvState* s) {
    return aabb_new(
        vec3_new(s->bb_min_x, s->bb_min_y, s->bb_min_z),
        vec3_new(s->bb_max_x, s->bb_max_y, s->bb_max_z)
    );
}
__host__ __device__ inline void state_set_pos(McEnvState* s, Vec3 p) {
    s->pos_x = p.x; s->pos_y = p.y; s->pos_z = p.z;
}
__host__ __device__ inline void state_set_vel(McEnvState* s, Vec3 v) {
    s->vel_x = v.x; s->vel_y = v.y; s->vel_z = v.z;
}
__host__ __device__ inline void state_set_bb(McEnvState* s, Aabb bb) {
    s->bb_min_x = bb.min.x; s->bb_min_y = bb.min.y; s->bb_min_z = bb.min.z;
    s->bb_max_x = bb.max.x; s->bb_max_y = bb.max.y; s->bb_max_z = bb.max.z;
}

// ---------------------------------------------------------------------------
// sync_to_bounding_box — recompute pos from bb center (mirrors Rust)
// ---------------------------------------------------------------------------

__host__ __device__ inline void sync_to_bounding_box(McEnvState* s) {
    double cx = (s->bb_min_x + s->bb_max_x) * 0.5;
    double cz = (s->bb_min_z + s->bb_max_z) * 0.5;
    s->pos_x = cx;
    s->pos_y = s->bb_min_y;
    s->pos_z = cz;
}

// ---------------------------------------------------------------------------
// Init state to defaults matching Player::new_at
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Init player physics to defaults matching Player::new_at
// ---------------------------------------------------------------------------

__host__ __device__ inline void mcenv_player_init(McEnvState* s, Vec3 pos) {
    s->width = 0.6;
    s->height = 1.8;
    s->step_height = 0.6;
    s->movement_speed_attr = 0.1f;

    state_set_pos(s, pos);
    s->prev_pos_x = pos.x; s->prev_pos_y = pos.y; s->prev_pos_z = pos.z;
    state_set_vel(s, vec3_zero());

    Aabb bb = aabb_for_entity(pos, s->width, s->height);
    state_set_bb(s, bb);

    s->yaw = 0.0f;
    s->pitch = 0.0f;
    s->on_ground = 0;
    s->horizontal_collision = 0;
    s->vertical_collision = 0;
    s->sneaking = 0;
    s->sprinting = 0;
    s->sprint_toggle_timer = 0;
    s->prev_forward_positive = 0;

    s->prev_place_pressed = 0;
    s->prev_break_pressed = 0;
    s->latched_place = 0;
    s->latched_break = 0;
}

// ---------------------------------------------------------------------------
// MCEnv helpers (from mcenv.h)
// ---------------------------------------------------------------------------

// MCEnv helpers that don't depend on world/physics are here.
// Functions that depend on WorldDims/physics are in mcenv_env.cuh.

__host__ __device__ inline float mc_dist_to_target(const McEnvState* s, double px, double pz) {
    float dx = (float)px - s->target_x;
    float dz = (float)pz - s->target_z;
    return sqrtf(dx*dx + dz*dz);
}

__host__ __device__ inline void mc_new_target(McEnvState* s) {
    for (int attempt = 0; attempt < 100; attempt++) {
        float tx = 0.5f + (float)(xorshift32(&s->rng) % MC_AREA_SIZE);
        float tz = 0.5f + (float)(xorshift32(&s->rng) % MC_AREA_SIZE);
        float dx = tx - (float)s->pos_x;
        float dz = tz - (float)s->pos_z;
        if (dx*dx + dz*dz >= 9.0f) {
            s->target_x = tx;
            s->target_z = tz;
            s->prev_dist = mc_dist_to_target(s, s->pos_x, s->pos_z);
            return;
        }
    }
    s->target_x = 0.5f + (float)(xorshift32(&s->rng) % MC_AREA_SIZE);
    s->target_z = 0.5f + (float)(xorshift32(&s->rng) % MC_AREA_SIZE);
    s->prev_dist = mc_dist_to_target(s, s->pos_x, s->pos_z);
}

__host__ __device__ inline int mc_get_phase(const McEnvState* s, const McEnvConfig* cfg) {
    // TEMP: phase 2 disabled — cap at 1
    if (cfg->force_phase2) return 2;
    int pt = cfg->phase_transition;
    if (pt <= 0) return cfg->curriculum_phase < 2 ? cfg->curriculum_phase : 1;
    if (s->lifetime_blocks >= pt) return 1;
    return 0;
}

__host__ __device__ inline int mc_get_phase_mut(McEnvState* s, const McEnvConfig* cfg) {
    // TEMP: phase 2 disabled — cap at 1
    if (cfg->force_phase2) return 2;
    int pt = cfg->phase_transition;
    if (pt <= 0) return cfg->curriculum_phase < 2 ? cfg->curriculum_phase : 1;
    if (s->lifetime_blocks >= pt) return 1;
    return 0;
}

__host__ __device__ inline void mcenv_add_log(McEnvLog* log, const McEnvState* s, const McEnvConfig* cfg) {
    int t = s->tick > 0 ? s->tick : 1;
    log->score += s->targets_reached;
    log->perf += s->targets_reached / 10.0f;
    log->episode_return += s->episode_return;
    log->episode_length += s->tick;
    log->max_dist += s->max_dist_from_start;
    log->max_height += s->max_height;
    log->fell += (s->tick < s->max_ticks) ? 1.0f : 0.0f;
    log->blocks_placed += s->blocks_placed;
    log->targets_reached += s->targets_reached;
    log->sneak_frac += (float)s->sneak_ticks / t;
    log->avg_pitch += s->pitch_sum / t;
    log->on_ground_frac += (float)s->on_ground_ticks / t;
    log->phase += (float)mc_get_phase(s, cfg);
    float rw_total = fabsf(s->rw_air_sum) + fabsf(s->rw_sprint_jump_sum)
                    + fabsf(s->rw_speed_sum) + fabsf(s->rw_block_sum)
                    + fabsf(s->rw_target_sum);
    if (rw_total > 0.0f) {
        float inv = 1.0f / rw_total;
        log->rw_air += fabsf(s->rw_air_sum) * inv;
        log->rw_sprint_jump += fabsf(s->rw_sprint_jump_sum) * inv;
        log->rw_speed += fabsf(s->rw_speed_sum) * inv;
        log->rw_block += fabsf(s->rw_block_sum) * inv;
        log->rw_target += fabsf(s->rw_target_sum) * inv;
    }
    log->n++;
}

__host__ __device__ inline void mcenv_full_init(
    McEnvState* s, const McEnvConfig* cfg, int env_idx)
{
    s->start_x = MC_START_X;
    s->start_y = MC_START_Y;
    s->start_z = MC_START_Z;
    s->max_ticks = cfg->max_ticks;
    s->lifetime_blocks = 0;
    s->avg_targets_ema = 0.0f;
    s->phase2_unlocked = 0;
    s->targets_reached = 0;
    s->blocks_placed = 0;
    s->rng = (unsigned int)(env_idx + 1);
}

#endif // MCENV_CUDA_CUH
