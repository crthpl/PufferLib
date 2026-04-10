// Quick comparison: run CPU (Rust via FFI) and CUDA physics side by side
// with the same action sequence. Print first divergence.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../mcenv.h"  // CPU mcenv (uses Rust FFI)

// Manually define the CUDA structs to match mcenv_cuda.cuh layout
// (Can't include CUDA headers from C, so we replicate the struct)
typedef struct {
    double pos_x, pos_y, pos_z;
    double prev_pos_x, prev_pos_y, prev_pos_z;
    double vel_x, vel_y, vel_z;
    double bb_min_x, bb_min_y, bb_min_z;
    double bb_max_x, bb_max_y, bb_max_z;
    float yaw, pitch;
    int on_ground, horizontal_collision, vertical_collision;
    int sneaking, sprinting;
    unsigned char sprint_toggle_timer;
    int prev_forward_positive;
    double width, height, step_height;
    float movement_speed_attr;
    int prev_place_pressed, prev_break_pressed;
    int latched_place, latched_break;
    // Episode fields
    float start_x, start_y, start_z;
    float target_x, target_z;
    float prev_dist, episode_return, max_dist_from_start;
    int tick, max_ticks, blocks_placed, targets_reached;
    int lifetime_blocks;
    float avg_targets_ema;
    int phase2_unlocked;
    int sneak_ticks;
    float pitch_sum;
    int on_ground_ticks;
    unsigned int rng;
} GpuState;

int main(void) {
    // Setup CPU env
    MCEnv cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.num_agents = 1;
    cpu.max_ticks = 1000;
    cpu.start_x = MC_START_X;
    cpu.start_y = MC_START_Y;
    cpu.start_z = MC_START_Z;
    cpu.rw_survival = 0.001f;
    cpu.rw_block = 20.0f;
    cpu.rw_fall = -100.0f;
    cpu.rw_speed = 5.0f;
    cpu.rw_target_reach = 100.0f;
    cpu.curriculum_phase = 0;
    cpu.phase_transition = 30;
    cpu.mc = NULL;

    float obs_buf[MC_OBS_TOTAL];
    float act_buf[8];
    float rew_buf[1];
    float term_buf[1];
    cpu.observations = obs_buf;
    cpu.actions = act_buf;
    cpu.rewards = rew_buf;
    cpu.terminals = term_buf;

    c_reset(&cpu);

    // Run 500 ticks with a fixed action pattern
    // forward=2(fwd), strafe=1(none), jump=0, sneak=1, sprint=0, yaw=5(0deg), pitch=3(0deg), place=1(single)
    float actions[8] = {2, 1, 0, 1, 0, 5, 3, 1};

    for (int t = 0; t < 500; t++) {
        memcpy(act_buf, actions, sizeof(actions));
        rew_buf[0] = 0; term_buf[0] = 0;
        c_step(&cpu);

        CPlayerState s;
        mcenv_environment_get_player(cpu.mc, &s);

        // Print state every 50 ticks
        if (t % 50 == 0) {
            printf("tick %3d: pos=(%.6f, %.6f, %.6f) vel=(%.6f, %.6f, %.6f) on_ground=%d reward=%.4f\n",
                t, s.pos.x, s.pos.y, s.pos.z, s.vel.x, s.vel.y, s.vel.z,
                s.on_ground, rew_buf[0]);
        }

        if (term_buf[0] > 0.5f) {
            printf("Episode ended at tick %d (reward=%.4f)\n", t, rew_buf[0]);
            // c_step already called c_reset
        }
    }

    printf("\nFinal CPU state after 500 ticks:\n");
    CPlayerState final_s;
    mcenv_environment_get_player(cpu.mc, &final_s);
    printf("  pos=(%.15f, %.15f, %.15f)\n", final_s.pos.x, final_s.pos.y, final_s.pos.z);
    printf("  vel=(%.15f, %.15f, %.15f)\n", final_s.vel.x, final_s.vel.y, final_s.vel.z);
    printf("  on_ground=%d yaw=%.2f pitch=%.2f\n", final_s.on_ground, cpu.yaw, cpu.pitch);

    c_close(&cpu);
    return 0;
}
