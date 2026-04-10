// Compare observations from CPU (Rust) and GPU (CUDA) paths after identical steps.
// Both use xorshift RNG with same seeds.

// Include CUDA physics first (defines xorshift32, constants, etc.)
#define __host__
#define __device__
#define __global__
#define __constant__
#include "mcenv_env.cuh"

// Now include CPU mcenv — undef conflicting macros first
#undef MC_START_X
#undef MC_START_Y
#undef MC_START_Z
#undef MC_START_YAW
#undef MC_START_PITCH
#undef MC_VOID_Y
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../mcenv.h"

#define WORLD_X 64
#define WORLD_Y 8
#define WORLD_Z 64
#define GRID_SIZE (WORLD_X * WORLD_Y * WORLD_Z)

int main(void) {
    // ---- CPU (Rust) env ----
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
    cpu.rng = 1; // same as GPU: env_idx(0) + 1

    float cpu_obs[MC_OBS_TOTAL], cpu_act[8], cpu_rew[1], cpu_term[1];
    cpu.observations = cpu_obs;
    cpu.actions = cpu_act;
    cpu.rewards = cpu_rew;
    cpu.terminals = cpu_term;
    c_reset(&cpu);

    // ---- GPU (CUDA fallback) env ----
    float sine_table[65536];
    generate_sine_table(sine_table);
    WorldDims wd = {WORLD_X, WORLD_Y, WORLD_Z, 0, 0, 0};
    uint8_t grid[GRID_SIZE];
    float gpu_obs[MC_OBS_TOTAL];
    McEnvState gs;
    McEnvConfig cfg = {1000, 0.001f, 20.0f, -100.0f, 5.0f, 100.0f, 0, 30, 0};
    mcenv_full_init(&gs, &cfg, 0);
    mcenv_reset(&gs, grid, &wd, &cfg, sine_table, gpu_obs);

    // Compare initial observations
    printf("=== Initial observations after reset ===\n");
    int diffs = 0;
    for (int i = 0; i < MC_OBS_TOTAL; i++) {
        if (memcmp(&cpu_obs[i], &gpu_obs[i], sizeof(float)) != 0) {
            printf("  obs[%2d]: CPU=%.9g (0x%08x)  GPU=%.9g (0x%08x)  DIFF\n",
                i, cpu_obs[i], *(unsigned*)&cpu_obs[i],
                gpu_obs[i], *(unsigned*)&gpu_obs[i]);
            diffs++;
        }
    }
    if (diffs == 0) printf("  ALL MATCH\n");
    printf("  CPU target: (%.1f, %.1f)  GPU target: (%.1f, %.1f)\n",
        cpu.target_x, cpu.target_z, gs.target_x, gs.target_z);

    // Step with identical actions for 20 ticks
    float actions[][8] = {
        {2,1,0,1,0,5,3,1},  // fwd, sneak, place
        {2,1,1,0,1,5,3,0},  // fwd, jump, sprint
        {0,2,0,1,0,7,4,1},  // back, left, sneak, yaw+5, pitch+1, place
        {2,1,0,0,1,5,3,0},  // fwd, sprint
    };
    int num_patterns = 4;

    for (int t = 0; t < 20; t++) {
        float* act = actions[t % num_patterns];

        // CPU step
        memcpy(cpu_act, act, sizeof(float)*8);
        cpu_rew[0] = 0; cpu_term[0] = 0;
        c_step(&cpu);

        // GPU step (replicate kernel logic)
        gs.tick++;
        int fa=(int)act[0], sa=(int)act[1], ja=(int)act[2], sna=(int)act[3];
        int spa=(int)act[4], ya=(int)act[5], pa=(int)act[6], pla=(int)act[7];
        float YD[11]={-180,-90,-15,-5,-1,0,1,5,15,90,180};
        float PD[7]={-15,-5,-1,0,1,5,15};
        gs.yaw+=YD[ya]; gs.pitch+=PD[pa];
        if(gs.pitch>90)gs.pitch=90; if(gs.pitch<-90)gs.pitch=-90;
        int phase = mc_get_phase_mut(&gs, &cfg);

        McEnvInput inp;
        inp.forward=(float)(fa-1); inp.strafe=(float)(sa-1);
        inp.jump=(ja==1); inp.sneak=(sna==1); inp.sprint=(spa==1);
        inp.has_look=1; inp.look_yaw=gs.yaw; inp.look_pitch=gs.pitch;
        inp.place=(pla>=1); inp.place_repeat=(pla==2); inp.break_block=0;

        // Block placement detection (same as kernel)
        int placed=0;
        if(pla>=1){
            int sbx=(int)floor(gs.pos_x),sbz=(int)floor(gs.pos_z);
            int px=(int)gs.start_x,pz=(int)gs.start_z;
            int before=0;
            for(int cx=sbx-5;cx<=sbx+5;cx++)for(int cz=sbz-5;cz<=sbz+5;cz++){
                if(cx==px&&cz==pz)continue;
                for(int cy=2;cy<=3;cy++)if(is_solid(grid,&wd,cx,cy,cz))before++;}
            mcenv_input(&gs,&inp); mcenv_tick(&gs,&inp,grid,&wd,sine_table);
            int after=0;
            for(int cx=sbx-5;cx<=sbx+5;cx++)for(int cz=sbz-5;cz<=sbz+5;cz++){
                if(cx==px&&cz==pz)continue;
                for(int cy=2;cy<=3;cy++)if(is_solid(grid,&wd,cx,cy,cz))after++;}
            placed=after-before; if(placed<0)placed=0;
        } else {
            mcenv_input(&gs,&inp); mcenv_tick(&gs,&inp,grid,&wd,sine_table);
        }

        // Reward (simplified for comparison)
        float gpu_rew = 0;
        if(phase<2 && gs.on_ground) gpu_rew+=cfg.rw_survival;
        float dist=mc_dist_to_target(&gs,gs.pos_x,gs.pos_z);
        gs.prev_dist=dist;

        // Terminal
        int done = (gs.pos_y < 2.5) || (gs.tick >= gs.max_ticks);
        float gpu_term = done ? 1.0f : 0.0f;

        if (done) {
            mcenv_reset(&gs, grid, &wd, &cfg, sine_table, gpu_obs);
        } else {
            compute_observations(&gs, grid, &wd, sine_table, gpu_obs);
        }

        // Compare obs
        int tick_diffs = 0;
        for (int i = 0; i < MC_OBS_TOTAL; i++) {
            if (memcmp(&cpu_obs[i], &gpu_obs[i], sizeof(float)) != 0) {
                if (tick_diffs < 3) {
                    printf("  tick %2d obs[%2d]: CPU=%.9g (0x%08x)  GPU=%.9g (0x%08x)\n",
                        t, i, cpu_obs[i], *(unsigned*)&cpu_obs[i],
                        gpu_obs[i], *(unsigned*)&gpu_obs[i]);
                }
                tick_diffs++;
            }
        }
        if (tick_diffs > 0) {
            printf("  tick %2d: %d obs diffs, reward CPU=%.6f GPU=%.6f, term CPU=%.1f GPU=%.1f\n",
                t, tick_diffs, cpu_rew[0], gpu_rew, cpu_term[0], gpu_term);
        } else {
            printf("  tick %2d: ALL obs match, reward CPU=%.6f GPU=%.6f\n", t, cpu_rew[0], gpu_rew);
        }
    }

    c_close(&cpu);
    return 0;
}
