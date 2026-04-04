#include "drone.h"
#include "puffernet.h"
#include "render.h"
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef __EMSCRIPTEN__
typedef struct {
    DroneEnv* env;
    PufferNet* net;
    Weights* weights;
} WebRenderArgs;

void emscriptenStep(void* e) {
    WebRenderArgs* args = (WebRenderArgs*)e;
    DroneEnv* env = args->env;
    PufferNet* net = args->net;
    size_t obs_size = 23;

    for (int i = 0; i < env->num_agents; i++) {
        int base = i * obs_size;
        env->observations[base + 19] = 0.0f;
        env->observations[base + 20] = 0.0f;
        env->observations[base + 21] = 0.0f;
        env->observations[base + 22] = 0.0f;
    }

    forward_puffernet(net, env->observations, env->actions);
    c_step(env);
    c_render(env);
}

WebRenderArgs* web_args = NULL;
#endif

void demo() {
    srand(time(NULL));

    DroneEnv* env = calloc(1, sizeof(DroneEnv));
    size_t obs_size = 23;

    env->num_agents = 64;
    env->max_rings = 10;
    env->task = HOVER;
    env->hover_target_dist = 0.5f;
    env->hover_dist = 0.05f;
    env->hover_omega = 0.05;
    env->hover_vel = 0.01;
    init(env);

    allocate(env);

    Weights* weights = load_weights("resources/drone/puffer_drone_weights.bin", 4841);
    int logit_sizes[4] = {1, 1, 1, 1};
    // make_puffernet(weights, num_agents, obs_size, hidden_size, num_layers, logit_sizes, num_actions)
    PufferNet* net = make_puffernet(weights, env->num_agents, obs_size, 64, 2, logit_sizes, 4);

    c_reset(env);

#ifdef __EMSCRIPTEN__
    WebRenderArgs* args = calloc(1, sizeof(WebRenderArgs));
    args->env = env;
    args->net = net;
    args->weights = weights;
    web_args = args;

    emscripten_set_main_loop_arg(emscriptenStep, args, 0, true);
#else
    c_render(env);
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        for (int i = 0; i < env->num_agents; i++) {
            int base = i * obs_size;
            env->observations[base + 19] = 0.0f;
            env->observations[base + 20] = 0.0f;
            env->observations[base + 21] = 0.0f;
            env->observations[base + 22] = 0.0f;
        }
        forward_puffernet(net, env->observations, env->actions);
        c_step(env);
        c_render(env);
    }

    c_close(env);
    free_puffernet(net);
    free(weights);
    free_allocated(env);
    free(env);
#endif
}

int main() {
    demo();
    return 0;
}
