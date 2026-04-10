#include "mcenv_env.cuh"
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// World dimensions for mcenv (matching CUDA.md spec)
// ---------------------------------------------------------------------------

#define MCENV_WORLD_X 64
#define MCENV_WORLD_Y 8
#define MCENV_WORLD_Z 64
#define MCENV_WORLD_OFF_X 0
#define MCENV_WORLD_OFF_Y 0
#define MCENV_WORLD_OFF_Z 0
#define MCENV_GRID_SIZE (MCENV_WORLD_X * MCENV_WORLD_Y * MCENV_WORLD_Z)

static __host__ __device__ inline WorldDims mcenv_world_dims() {
    WorldDims wd;
    wd.size_x = MCENV_WORLD_X; wd.size_y = MCENV_WORLD_Y; wd.size_z = MCENV_WORLD_Z;
    wd.off_x = MCENV_WORLD_OFF_X; wd.off_y = MCENV_WORLD_OFF_Y; wd.off_z = MCENV_WORLD_OFF_Z;
    return wd;
}

// ---------------------------------------------------------------------------
// GPU kernel: PufferLib-integrated step
//
// One thread per environment. Reads flat float actions, writes flat float
// obs/rewards/terminals. Handles physics, reward, terminal, auto-reset.
// Zero CPU<->GPU copies.
// ---------------------------------------------------------------------------

__global__ void mcenv_pufferlib_step_kernel(
    McEnvState*       states,       // [N]
    const float*      actions,      // [N * 8]  discrete actions as floats
    float*            observations, // [N * 76] output
    float*            rewards,      // [N]      output
    float*            terminals,    // [N]      output
    uint8_t*          blocks,       // [N * MCENV_GRID_SIZE]
    McEnvLog*         logs,         // [N] per-env log accumulators
    const float*      sine_table,   // [65536]
    const McEnvConfig* config,      // [1]
    int               N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    McEnvState* s = &states[tid];
    const float* act = actions + tid * MC_NUM_ATNS;
    float* obs = observations + tid * MC_OBS_TOTAL;
    uint8_t* env_grid = blocks + tid * MCENV_GRID_SIZE;
    WorldDims wd = mcenv_world_dims();

    s->tick++;

    // --- Decode actions ---
    int forward_action  = (int)act[0];
    int strafe_action   = (int)act[1];
    int jump_action     = (int)act[2];
    int sneak_action    = (int)act[3];
    int sprint_action   = (int)act[4];
    int yaw_action      = (int)act[5];
    int pitch_action    = (int)act[6];
    int place_action    = (int)act[7];
    int rot_pct_action  = (int)act[8];

    float rot_pct = rot_pct_action * 0.25f;  // 0, 0.25, 0.5, 0.75, 1.0
    s->yaw   += d_YAW_DELTAS[yaw_action] * rot_pct;
    s->pitch += d_PITCH_DELTAS[pitch_action] * rot_pct;
    if (s->pitch > 90.0f) s->pitch = 90.0f;
    if (s->pitch < -90.0f) s->pitch = -90.0f;

    int phase = mc_get_phase_mut(s, config);

    // --- Build McEnvInput for physics ---
    McEnvInput input;
    input.forward = (float)(forward_action - 1);
    input.strafe  = (float)(strafe_action - 1);
    input.jump    = (jump_action == 1);
    input.sneak   = (sneak_action == 1);
    input.sprint  = (sprint_action == 1);
    input.has_look = 1;
    input.look_yaw = s->yaw;
    input.look_pitch = s->pitch;
    input.place = (place_action >= 1);
    input.place_repeat = (place_action == 2);
    input.break_block = 0;

    // --- Tick + block placement detection ---
    mcenv_input(s, &input);
    mcenv_tick(s, &input, env_grid, &wd, sine_table);

    float place_scale = 0.0f;
    if (s->last_placed_valid) {
        int px = (int)s->start_x, pz = (int)s->start_z;
        if (!(s->last_placed_x == px && s->last_placed_z == pz) && s->last_placed_y <= 5) {
            place_scale = 1.0f;
            s->blocks_placed++;
            s->lifetime_blocks++;
        }
    }

    // --- Diagnostics ---
    if (sneak_action == 1) s->sneak_ticks++;
    s->pitch_sum += s->pitch;
    if (s->on_ground) s->on_ground_ticks++;

    // --- Reward computation (from mcenv.h) ---
    float block_weight = 1.0f;
    float target_weight = 0.0f;
    int require_ground = 0;
    if (phase >= 1) {
        target_weight = 1.0f;
        block_weight = 0.25f;
        if (phase >= 2) {
            block_weight = 0.0f;
        }
    }

    float reward = 0.0f;
    float rw_air = 0.0f, rw_sprint_jump = 0.0f, rw_speed = 0.0f;
    float rw_block = 0.0f, rw_target = 0.0f;

    // 1. Survival / movement shaping
    if (phase < 1) {
        if (s->on_ground) reward += config->rw_survival;
    } else {
        if (!s->on_ground) { reward += 0.1f; rw_air += 0.1f; }
    }

    // 2. Speed: getting closer to target
    float dist = mc_dist_to_target(s, s->pos_x, s->pos_z);

    // Sprint-jump progress reward, scaled by facing-toward-target dot product
    if (phase >= 1) {
        float yaw_rad = s->yaw * (3.14159265f / 180.0f);
        float face_x = -sinf(yaw_rad), face_z = cosf(yaw_rad);
        float tgt_dx = s->target_x - (float)s->pos_x;
        float tgt_dz = s->target_z - (float)s->pos_z;
        float tgt_len = sqrtf(tgt_dx * tgt_dx + tgt_dz * tgt_dz);
        float dot = 0.0f;
        if (tgt_len > 0.01f) dot = (face_x * tgt_dx + face_z * tgt_dz) / tgt_len;
        if (dot < 0.0f) dot = 0.0f;

        if (s->sj_timer > 0) {
            s->sj_timer--;
            if (s->sj_dist - dist > 0.3f) {
                float sj_rw = 100.0f * dot;
                reward += sj_rw; rw_sprint_jump += sj_rw;
                s->sj_timer = 0;
            }
        }
        if (forward_action == 2 && sprint_action == 1 && jump_action == 1) {
            s->sj_timer = 20;
            s->sj_dist = dist;
        }
    }
    if (target_weight > 0.0f && (!require_ground || s->on_ground)) {
        float delta_dist = s->prev_dist - dist;
        float s_rw = config->rw_speed * delta_dist * target_weight;
        reward += s_rw; rw_speed += s_rw;
    }
    s->prev_dist = dist;

    // 3. Block placement reward
    if (place_scale > 0.0f && block_weight > 0.0f) {
        float air_mult = s->on_ground ? 1.0f : 4.0f;
        float b_rw = config->rw_block * place_scale * block_weight * air_mult;
        reward += b_rw; rw_block += b_rw;
    }

    // 4. Target reached
    if (dist < MC_TARGET_REACH && (!require_ground || s->on_ground)) {
        if (target_weight > 0.0f) {
            float escalate = 1.0f + 0.5f * (float)s->targets_reached;
            float time_bonus = 1.0f + (float)(s->max_ticks - s->tick) / (float)s->max_ticks;
            float t_rw = config->rw_target_reach * escalate * time_bonus * target_weight;
            reward += t_rw; rw_target += t_rw;
        }
        s->targets_reached++;
        mc_new_target(s);
        dist = s->prev_dist;
    }

    s->rw_air_sum += rw_air;
    s->rw_sprint_jump_sum += rw_sprint_jump;
    s->rw_speed_sum += rw_speed;
    s->rw_block_sum += rw_block;
    s->rw_target_sum += rw_target;

    // Track progress
    float dist_from_start = sqrtf(
        (float)(s->pos_x - (double)s->start_x) * (float)(s->pos_x - (double)s->start_x) +
        (float)(s->pos_z - (double)s->start_z) * (float)(s->pos_z - (double)s->start_z));
    if (dist_from_start > s->max_dist_from_start)
        s->max_dist_from_start = dist_from_start;
    float height = (float)s->pos_y - MC_START_Y;
    if (height > s->max_height)
        s->max_height = height;

    // --- Termination ---
    int done = 0;
    if (s->pos_y < (double)MC_VOID_Y) {
        reward = config->rw_fall;
        done = 1;
    } else if (s->tick >= s->max_ticks) {
        done = 1;
    }

    rewards[tid] = reward;
    s->episode_return += reward;
    terminals[tid] = done ? 1.0f : 0.0f;

    if (done) {
        mcenv_add_log(&logs[tid], s, config);
        mcenv_reset(s, env_grid, &wd, config, sine_table, obs);
    } else {
        compute_observations(s, env_grid, &wd, sine_table, obs);
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: reset all environments
// ---------------------------------------------------------------------------

__global__ void mcenv_pufferlib_reset_kernel(
    McEnvState*        states,
    float*             observations,
    uint8_t*           blocks,
    const float*       sine_table,
    const McEnvConfig* config,
    int                N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    McEnvState* s = &states[tid];
    float* obs = observations + tid * MC_OBS_TOTAL;
    uint8_t* env_grid = blocks + tid * MCENV_GRID_SIZE;
    WorldDims wd = mcenv_world_dims();

    mcenv_full_init(s, config, tid);
    mcenv_reset(s, env_grid, &wd, config, sine_table, obs);
}

// ---------------------------------------------------------------------------
// Host API — called from PufferLib vecenv.h
// ---------------------------------------------------------------------------

extern "C" {

float* mcenv_cuda_create_sine_table() {
    float host_table[65536];
    generate_sine_table(host_table);
    float* dev_table = nullptr;
    cudaMalloc(&dev_table, 65536 * sizeof(float));
    cudaMemcpy(dev_table, host_table, 65536 * sizeof(float), cudaMemcpyHostToDevice);
    return dev_table;
}

void mcenv_cuda_free_sine_table(float* dev_table) {
    cudaFree(dev_table);
}

void mcenv_cuda_alloc(int N, McEnvState** d_states, uint8_t** d_blocks,
                      McEnvConfig** d_config, McEnvLog** d_logs) {
    cudaMalloc(d_states, N * sizeof(McEnvState));
    cudaMalloc(d_blocks, N * MCENV_GRID_SIZE * sizeof(uint8_t));
    cudaMemset(*d_blocks, 0, N * MCENV_GRID_SIZE * sizeof(uint8_t));
    cudaMalloc(d_config, sizeof(McEnvConfig));
    cudaMalloc(d_logs, N * sizeof(McEnvLog));
    cudaMemset(*d_logs, 0, N * sizeof(McEnvLog));
}

void mcenv_cuda_free(McEnvState* d_states, uint8_t* d_blocks, McEnvConfig* d_config,
                     McEnvLog* d_logs) {
    cudaFree(d_states);
    cudaFree(d_blocks);
    cudaFree(d_config);
    cudaFree(d_logs);
}

void mcenv_cuda_upload_config(McEnvConfig* d_config, const McEnvConfig* h_config) {
    cudaMemcpy(d_config, h_config, sizeof(McEnvConfig), cudaMemcpyHostToDevice);
}

void mcenv_cuda_pufferlib_step(
    McEnvState* d_states, const float* d_actions,
    float* d_observations, float* d_rewards, float* d_terminals,
    uint8_t* d_blocks, McEnvLog* d_logs, const float* d_sine_table,
    const McEnvConfig* d_config, int N,
    cudaStream_t stream)
{
    int threads = 256;
    int nblocks = (N + threads - 1) / threads;
    mcenv_pufferlib_step_kernel<<<nblocks, threads, 0, stream>>>(
        d_states, d_actions, d_observations, d_rewards, d_terminals,
        d_blocks, d_logs, d_sine_table, d_config, N);
}

void mcenv_cuda_pufferlib_reset(
    McEnvState* d_states, float* d_observations,
    uint8_t* d_blocks, const float* d_sine_table,
    const McEnvConfig* d_config, int N)
{
    int threads = 256;
    int nblocks = (N + threads - 1) / threads;
    mcenv_pufferlib_reset_kernel<<<nblocks, threads>>>(
        d_states, d_observations, d_blocks, d_sine_table, d_config, N);
    cudaDeviceSynchronize();
}

// Download per-env logs to host, then zero them on device.
// Called once per log interval (not per tick).
void mcenv_cuda_download_logs(McEnvLog* d_logs, McEnvLog* h_logs, int N) {
    cudaMemcpy(h_logs, d_logs, N * sizeof(McEnvLog), cudaMemcpyDeviceToHost);
    cudaMemset(d_logs, 0, N * sizeof(McEnvLog));
}

int mcenv_cuda_grid_size() {
    return MCENV_GRID_SIZE;
}

int mcenv_cuda_state_size() {
    return (int)sizeof(McEnvState);
}

int mcenv_cuda_log_size() {
    return (int)sizeof(McEnvLog);
}

} // extern "C"
