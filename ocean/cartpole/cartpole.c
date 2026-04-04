// local compile/eval implemented for discrete actions only
// eval with python demo.py --mode eval --env puffer_cartpole --eval-mode-path <path to model>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cartpole.h"
#include "puffernet.h"

#define NUM_WEIGHTS 133123
#define OBSERVATIONS_SIZE 4
#define ACTIONS_SIZE 2
#define CONTINUOUS 0

const char* WEIGHTS_PATH = "resources/cartpole/cartpole_weights.bin";

float movement(float action, int userControlMode) {
    if (userControlMode) {
        return (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : -1.0f;
    } else {
        return (action > 0.5f) ? 1.0f : -1.0f;
    }
}

void demo() {
    Weights* weights = load_weights(WEIGHTS_PATH, NUM_WEIGHTS);
    
    int logit_sizes[1] = {ACTIONS_SIZE};
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_SIZE, 64, 2, logit_sizes, 1);
    
    Cartpole env = {0};
    env.continuous = CONTINUOUS;
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        int userControlMode = IsKeyDown(KEY_LEFT_SHIFT);

        if (!userControlMode) {
            forward_puffernet(net, env.observations, env.actions);
            env.actions[0] = movement(env.actions[0], 0);
        } else {
            env.actions[0] = movement(env.actions[0], userControlMode);
        }   

        c_step(&env);

        BeginDrawing();
        ClearBackground(RAYWHITE);
        c_render(&env);
        EndDrawing();

        if (env.terminals[0] > 0.5f) {
            c_reset(&env);
        }
    }

    free_puffernet(net);
    free(weights);
    free_allocated(&env);
}

int main() {
    srand(time(NULL));
    demo();
    return 0;
}
