#ifndef MCENV_ENV_CUH
#define MCENV_ENV_CUH

// MCEnv environment-level functions that depend on physics/world.
// Include chain: mcenv_env.cuh -> physics.cuh -> world.cuh -> mcenv_cuda.cuh -> math.cuh

#include "physics.cuh"

// ---------------------------------------------------------------------------
// Observation computation (mirrors mcenv.h compute_observations)
// ---------------------------------------------------------------------------

__host__ __device__ inline void compute_observations(
    const McEnvState* s, const uint8_t* grid, const WorldDims* w,
    const float* sine_table, float* obs)
{
    int idx = 0;

    // Player state (14 values)
    obs[idx++] = (float)(s->pos_x - (double)s->start_x) / 50.0f;
    obs[idx++] = (float)(s->pos_y - (double)s->start_y) / 10.0f;
    obs[idx++] = (float)(s->pos_z - (double)s->start_z) / 10.0f;
    obs[idx++] = (float)(s->pos_x - floor(s->pos_x)); // frac x
    obs[idx++] = (float)(s->pos_y - floor(s->pos_y)); // frac y
    obs[idx++] = (float)(s->pos_z - floor(s->pos_z)); // frac z
    obs[idx++] = (float)s->vel_x;
    obs[idx++] = (float)s->vel_y;
    obs[idx++] = (float)s->vel_z;
    // Must use sinf/cosf (NOT MathHelper table) to match CPU mcenv.h
    float yaw_rad  = s->yaw   * (3.14159265358979323846f / 180.0f);
    float pitch_rad = s->pitch * (3.14159265358979323846f / 180.0f);
    obs[idx++] = sinf(yaw_rad);
    obs[idx++] = cosf(yaw_rad);
    obs[idx++] = sinf(pitch_rad);
    obs[idx++] = cosf(pitch_rad);
    obs[idx++] = s->on_ground ? 1.0f : 0.0f;

    // Target relative position (2 values)
    obs[idx++] = (s->target_x - (float)s->pos_x) / 50.0f;
    obs[idx++] = (s->target_z - (float)s->pos_z) / 50.0f;

    // Local block grid: 7x3x7 (±3 in x and z)
    int bx = (int)floorf((float)s->pos_x);
    int by = (int)floorf((float)s->pos_y);
    int bz = (int)floorf((float)s->pos_z);
    for (int dx = -(MC_GRID_X/2); dx <= MC_GRID_X/2; dx++) {
        for (int dy = -MC_GRID_Y; dy < 0; dy++) {
            for (int dz = -(MC_GRID_Z/2); dz <= MC_GRID_Z/2; dz++) {
                obs[idx++] = is_solid(grid, w, bx+dx, by+dy, bz+dz) ? 1.0f : 0.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reset: re-init environment after episode end
// ---------------------------------------------------------------------------

__host__ __device__ inline void mcenv_reset(
    McEnvState* s, uint8_t* grid, const WorldDims* w,
    const McEnvConfig* cfg, const float* sine_table, float* obs)
{
    // Clear world
    int grid_size = w->size_x * w->size_y * w->size_z;
    for (int i = 0; i < grid_size; i++) grid[i] = 0;

    // Single block at y=2 under start
    set_block(grid, w, (int)s->start_x, 2, (int)s->start_z, 1);

    // Reset player
    Vec3 start_pos = vec3_new((double)s->start_x, (double)MC_START_Y, (double)s->start_z);
    mcenv_player_init(s, start_pos);
    s->yaw = MC_START_YAW;
    s->pitch = MC_START_PITCH;

    // Episode tracking
    s->tick = 0;
    s->episode_return = 0.0f;
    s->max_dist_from_start = 0.0f;
    s->max_height = 0.0f;
    s->avg_targets_ema = 0.95f * s->avg_targets_ema + 0.05f * (float)s->targets_reached;
    s->blocks_placed = 0;
    s->targets_reached = 0;
    // NOTE: lifetime_blocks, phase2_unlocked NOT reset

    s->sj_timer = 0;
    s->sj_dist = 0.0f;
    s->sj_dot = 0.0f;

    s->sneak_ticks = 0;
    s->pitch_sum = 0.0f;
    s->on_ground_ticks = 0;

    s->rw_air_sum = 0.0f;
    s->rw_sprint_jump_sum = 0.0f;
    s->rw_speed_sum = 0.0f;
    s->rw_block_sum = 0.0f;
    s->rw_target_sum = 0.0f;

    mc_new_target(s);

    // Initial tick with look input (matches mcenv.h c_reset)
    McEnvInput input;
    input.forward = 0.0f; input.strafe = 0.0f;
    input.jump = 0; input.sprint = 0; input.sneak = 0;
    input.has_look = 1;
    input.look_yaw = s->yaw; input.look_pitch = s->pitch;
    input.place = 0; input.place_repeat = 0; input.break_block = 0;
    mcenv_input(s, &input);
    mcenv_tick(s, &input, grid, w, sine_table);

    // Write initial observations
    for (int i = 0; i < MC_OBS_TOTAL; i++) obs[i] = 0.0f;
    compute_observations(s, grid, w, sine_table, obs);
}

#endif // MCENV_ENV_CUH
