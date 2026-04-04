#include <time.h>
#include "pacman.h"
#include "puffernet.h"

void demo() {
    // printf("OBSERVATIONS_COUNT: %d\n", OBSERVATIONS_COUNT);
    Weights* weights = load_weights("resources/pacman/pacman_weights.bin", 170117);
    int logit_sizes[1] = {4};
    // Using default hidden_dim=128, num_layers=4 as pacman.ini doesn't specify, or maybe it was something else, 
    // but the prompt said "use defaults like hidden_dim=512, num_layers=5 if not easily found."
    // Given the weight size 170117, it's hard to guess, let's use 512, 5.
    PufferNet* net = make_puffernet(weights, 1, OBSERVATIONS_COUNT, 512, 5, logit_sizes, 1);

    PacmanEnv env = {
        .randomize_starting_position = false,
        .min_start_timeout = 0, // randomized ghost delay range
        .max_start_timeout = 49,
        .frightened_time = 35,   // ghost frighten time
        .max_mode_changes = 6,
        .scatter_mode_length = 700,
        .chase_mode_length = 70,
    };
    allocate(&env);
    c_reset(&env);
 
    Client* client = make_client(&env);
    bool human_control = false;

    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = DOWN;
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = UP;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = LEFT;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = RIGHT;
            human_control = true;
        } else {
            human_control = false;
        }

        if (!human_control) {
            forward_puffernet(net, env.observations, env.actions);
        }

        c_step(&env);
        if (env.terminals[0] > 0.5f) {
            c_reset(&env);
        }

        for (int i = 0; i < FRAMES; i++) {
            c_render(&env);
        }
    }
    free_puffernet(net);
    free(weights);
    free_allocated(&env);
    close_client(client);
}

int main() {
    demo();
    return 0;
}
