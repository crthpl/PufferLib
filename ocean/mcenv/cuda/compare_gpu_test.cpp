// Same test as compare_test.c but using the CUDA physics (CPU fallback mode)
#define __host__
#define __device__
#define __global__
#define __constant__

#include "mcenv_env.cuh"
#include <cstdio>
#include <cstring>

#define WORLD_X 64
#define WORLD_Y 8
#define WORLD_Z 64
#define GRID_SIZE (WORLD_X * WORLD_Y * WORLD_Z)

int main() {
    float sine_table[65536];
    generate_sine_table(sine_table);

    WorldDims wd = {WORLD_X, WORLD_Y, WORLD_Z, 0, 0, 0};
    uint8_t grid[GRID_SIZE];
    float obs[MC_OBS_TOTAL];
    McEnvState s;
    McEnvConfig cfg;

    cfg.max_ticks = 1000;
    cfg.rw_survival = 0.001f;
    cfg.rw_block = 20.0f;
    cfg.rw_fall = -100.0f;
    cfg.rw_speed = 5.0f;
    cfg.rw_target_reach = 100.0f;
    cfg.curriculum_phase = 0;
    cfg.phase_transition = 30;
    cfg.force_phase2 = 0;

    mcenv_full_init(&s, &cfg, 0);
    mcenv_reset(&s, grid, &wd, &cfg, sine_table, obs);

    // Same actions: forward=2, strafe=1, jump=0, sneak=1, sprint=0, yaw=5(0), pitch=3(0), place=1
    for (int t = 0; t < 500; t++) {
        s.tick++;

        int forward_action = 2, strafe_action = 1, jump_action = 0;
        int sneak_action = 1, sprint_action = 0;
        int yaw_action = 5, pitch_action = 3, place_action = 1;

        // Same as GPU kernel: add yaw/pitch deltas
        float YAW_DELTAS[11] = {-180,  -90, -15, -5, -1, 0, 1, 5, 15, 90, 180};
        float PITCH_DELTAS[7] = {-15, -5, -1, 0, 1, 5, 15};
        s.yaw   += YAW_DELTAS[yaw_action];
        s.pitch += PITCH_DELTAS[pitch_action];
        if (s.pitch > 90.0f) s.pitch = 90.0f;
        if (s.pitch < -90.0f) s.pitch = -90.0f;

        int phase = mc_get_phase_mut(&s, &cfg);

        McEnvInput input;
        input.forward = (float)(forward_action - 1);
        input.strafe  = (float)(strafe_action - 1);
        input.jump    = (jump_action == 1);
        input.sneak   = (sneak_action == 1);
        input.sprint  = (sprint_action == 1);
        input.has_look = 1;
        input.look_yaw = s.yaw;
        input.look_pitch = s.pitch;
        input.place = (place_action >= 1);
        input.place_repeat = (place_action == 2);
        input.break_block = 0;

        // Block placement detection (same as GPU kernel)
        int placed_this_tick = 0;
        if (place_action >= 1) {
            int scan_bx = (int)floor(s.pos_x);
            int scan_bz = (int)floor(s.pos_z);
            int px = (int)s.start_x, pz = (int)s.start_z;
            int blocks_before = 0;
            for (int cx = scan_bx-5; cx <= scan_bx+5; cx++)
                for (int cz = scan_bz-5; cz <= scan_bz+5; cz++) {
                    if (cx == px && cz == pz) continue;
                    for (int cy = 2; cy <= 3; cy++)
                        if (is_solid(grid, &wd, cx, cy, cz)) blocks_before++;
                }

            mcenv_input(&s, &input);
            mcenv_tick(&s, &input, grid, &wd, sine_table);

            int blocks_after = 0;
            for (int cx = scan_bx-5; cx <= scan_bx+5; cx++)
                for (int cz = scan_bz-5; cz <= scan_bz+5; cz++) {
                    if (cx == px && cz == pz) continue;
                    for (int cy = 2; cy <= 3; cy++)
                        if (is_solid(grid, &wd, cx, cy, cz)) blocks_after++;
                }
            placed_this_tick = blocks_after - blocks_before;
            if (placed_this_tick < 0) placed_this_tick = 0;
        } else {
            mcenv_input(&s, &input);
            mcenv_tick(&s, &input, grid, &wd, sine_table);
        }

        // Reward (simplified — just check state)
        float reward = 0.0f;
        if (phase < 2 && s.on_ground) reward += cfg.rw_survival;

        if (t % 50 == 0) {
            printf("tick %3d: pos=(%.6f, %.6f, %.6f) vel=(%.6f, %.6f, %.6f) on_ground=%d reward=%.4f\n",
                t, s.pos_x, s.pos_y, s.pos_z, s.vel_x, s.vel_y, s.vel_z,
                s.on_ground, reward);
        }

        if (s.pos_y < 2.5 || s.tick >= s.max_ticks) {
            printf("Episode ended at tick %d\n", t);
            mcenv_reset(&s, grid, &wd, &cfg, sine_table, obs);
        }
    }

    printf("\nFinal CUDA state after 500 ticks:\n");
    printf("  pos=(%.15f, %.15f, %.15f)\n", s.pos_x, s.pos_y, s.pos_z);
    printf("  vel=(%.15f, %.15f, %.15f)\n", s.vel_x, s.vel_y, s.vel_z);
    printf("  on_ground=%d yaw=%.2f pitch=%.2f\n", s.on_ground, s.yaw, s.pitch);

    return 0;
}
