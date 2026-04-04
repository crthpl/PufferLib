#include "maze.h"
#include "puffernet.h"

void demo() {
    Weights* weights = load_weights("resources/maze/maze_weights.bin", 0); // 0 is dummy if no weights
    int logit_sizes[1] = {5};
    PufferNet* net = make_puffernet(weights, 1, 121, 64, 2, logit_sizes, 1);

    int max_size = 32;
    int num_agents = 1;
    int horizon = 128;
    float speed = 1;
    int vision = 5;
    bool discretize = true;

    int seed = 0;

    Grid* env = allocate_maze(max_size, num_agents, horizon,
        vision, speed, discretize);

    State* levels = calloc(1, sizeof(State));

    create_maze_level(env, 31, 31, 0.85, seed);
    init_state(levels, max_size, num_agents);
    get_state(env, levels);
    env->num_maps = 1;
    env->levels = levels;
 
    int tick = 0;
    c_render(env);
    while (!WindowShouldClose()) {
        env->actions[0] = ATN_PASS;
        Agent* agent = &env->agents[0];

        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)){
                agent->direction = 3.0*PI/2.0;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) {
                agent->direction = PI/2.0;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) {
                agent->direction = PI;
                env->actions[0] = ATN_FORWARD;
            } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
                agent->direction = 0;
                env->actions[0] = ATN_FORWARD;
            } else {
                env->actions[0] = ATN_PASS;
            }
        } else {
            forward_puffernet(net, env->observations, env->actions);
        }

        tick = (tick + 1)%12;
        if (tick % 1 == 0) {
            c_step(env);
        }
        c_render(env);
    }
    
    free_puffernet(net);
    free(weights);
    free_allocated_maze(env);
    free(levels);
}

int main() {
    demo();
    return 0;
}
