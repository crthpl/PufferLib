#include <stdio.h>
#include <dlfcn.h>
#include "vecenv.h"


int main() {
    void* handle = dlopen("./test_binding.so", RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen error: %s\n", dlerror());
        return 1;
    }

    // Clear any existing errors
    dlerror();

    // Load the function pointer
    create_environments_fn create_environments = (create_environments_fn)dlsym(handle, "create_environments");
    env_init_fn env_init = (env_init_fn)dlsym(handle, "env_init");
    vec_reset_fn vec_reset = (vec_reset_fn)dlsym(handle, "vec_reset");
    vec_step_fn vec_step = (vec_step_fn)dlsym(handle, "vec_step");
    env_close_fn env_close = (env_close_fn)dlsym(handle, "env_close");
    vec_close_fn vec_close = (vec_close_fn)dlsym(handle, "vec_close");
    vec_log_fn vec_log = (vec_log_fn)dlsym(handle, "vec_log");
    vec_render_fn vec_render = (vec_render_fn)dlsym(handle, "vec_render");
    
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        fprintf(stderr, "dlsym error: %s\n", dlsym_error);
        dlclose(handle);
        return 1;
    }

    // Now call it!
    Dict* kwargs = create_dict(32);
    dict_set_int(kwargs, "frameskip", 4);
    dict_set_int(kwargs, "width", 576);
    dict_set_int(kwargs, "height", 330);
    dict_set_int(kwargs, "paddle_width", 62);
    dict_set_int(kwargs, "paddle_height", 8);
    dict_set_int(kwargs, "ball_width", 32);
    dict_set_int(kwargs, "ball_height", 32);
    dict_set_int(kwargs, "brick_width", 32);
    dict_set_int(kwargs, "brick_height", 12);
    dict_set_int(kwargs, "brick_rows", 6);
    dict_set_int(kwargs, "brick_cols", 18);
    dict_set_int(kwargs, "initial_ball_speed", 256);
    dict_set_int(kwargs, "max_ball_speed", 448);
    dict_set_int(kwargs, "paddle_speed", 620);
    dict_set_int(kwargs, "continuous", 0);


    VecEnv vec = create_environments(4, kwargs);
    vec_reset(vec);
    for (int i = 0; i < 300; i++) {
        vec_step(vec);
        vec_render(vec, 0);
    }

    printf("Created VecEnv with %d environments\n", vec.size);

    // TODO: Add a `close_vecenv` function to clean up
    // vec.envs, etc.

    // Close the library
    dlclose(handle);
    return 0;
}
