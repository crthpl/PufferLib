#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "cJSON.h"
#include "raylib.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "rcamera.h"

#if defined(PLATFORM_DESKTOP) || defined(PLATFORM_DESKTOP_SDL)
    #if defined(GRAPHICS_API_OPENGL_ES2)
        #include "glad_gles2.h"       // Required for: OpenGL functionality
        #define glGenVertexArrays glGenVertexArraysOES
        #define glBindVertexArray glBindVertexArrayOES
        #define glDeleteVertexArrays glDeleteVertexArraysOES
        #define GLSL_VERSION            100
    #else
        #if defined(__APPLE__)
            #define GL_SILENCE_DEPRECATION // Silence Opengl API deprecation warnings
            #include <OpenGL/gl3.h>     // OpenGL 3 library for OSX
            #include <OpenGL/gl3ext.h>  // OpenGL 3 extensions library for OSX
        #else
            #include "glad.h"       // Required for: OpenGL functionality
        #endif
        #define GLSL_VERSION            330
    #endif
#else   // PLATFORM_ANDROID, PLATFORM_WEB
    #define GLSL_VERSION            100
#endif

#include "rlgl.h"
#include "raymath.h"

#define CAMERA_ORBITAL_SPEED 0.1f
#define CAMERA_MOUSE_MOVE_SENSITIVITY 0.005f
#define CAMERA_MOVE_SPEED 5.4f
#define CAMERA_ROTATION_SPEED 0.03f
#define CAMERA_PAN_SPEED 0.2f

// Camera mouse movement sensitivity
#define CAMERA_MOUSE_MOVE_SENSITIVITY                   0.003f
void CustomUpdateCamera(Camera *camera, int mode)
{
    Vector2 mousePositionDelta = GetMouseDelta();

    bool moveInWorldPlane = ((mode == CAMERA_FIRST_PERSON) || (mode == CAMERA_THIRD_PERSON));
    bool rotateAroundTarget = ((mode == CAMERA_THIRD_PERSON) || (mode == CAMERA_ORBITAL));
    bool lockView = ((mode == CAMERA_FREE) || (mode == CAMERA_FIRST_PERSON) || (mode == CAMERA_THIRD_PERSON) || (mode == CAMERA_ORBITAL));
    bool rotateUp = false;

    // Camera speeds based on frame time
    float cameraMoveSpeed = CAMERA_MOVE_SPEED*GetFrameTime();
    float cameraRotationSpeed = CAMERA_ROTATION_SPEED*GetFrameTime();
    float cameraPanSpeed = CAMERA_PAN_SPEED*GetFrameTime();
    float cameraOrbitalSpeed = CAMERA_ORBITAL_SPEED*GetFrameTime();

    // Orbital can just orbit
    Matrix rotation = MatrixRotate(GetCameraUp(camera), cameraOrbitalSpeed);
    Vector3 view = Vector3Subtract(camera->position, camera->target);
    view = Vector3Transform(view, rotation);
    camera->position = Vector3Add(camera->target, view);
    // Zoom target distance
    CameraMoveToTarget(camera, -GetMouseWheelMove());
    if (IsKeyPressed(KEY_KP_SUBTRACT)) CameraMoveToTarget(camera, 2.0f);
    if (IsKeyPressed(KEY_KP_ADD)) CameraMoveToTarget(camera, -2.0f);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

Color COLORS[] = {
    BLUE, MAROON, ORANGE, DARKGREEN, DARKBLUE, DARKPURPLE, DARKBROWN,
    GRAY, RED, GOLD, LIME, VIOLET, LIGHTGRAY, PINK, YELLOW,
    GREEN, SKYBLUE, PURPLE, BEIGE
};

const float EMPTY = -4242.0f;

#define MAX_PARTICLES       1000

typedef struct Particle {
    float x;
    float y;
    float i;
    float r;
    float g;
    float b;
    float a;
} Particle;

typedef struct VertexBuffer {
    float* vertices;
    int n;
} VertexBuffer;

#define SEP 4
#define SETTINGS_HEIGHT 20
#define TOGGLE_WIDTH 60
#define DROPDOWN_WIDTH 136
#define BUCKETS 8

typedef struct {
    char *key;
    float *ary;
    int n;
} Hyper;

typedef struct {
    char *key;
    Hyper *hypers;
    int n;
} Env;

typedef struct {
    Env *envs;
    int n;
} Dataset;

Hyper* get_hyper(Dataset *data, char *env, char* hyper) {
    for (int i = 0; i < data->n; i++) {
        if (strcmp(data->envs[i].key, env) != 0) {
            continue;
        }
        for (int j = 0; j < data->envs[i].n; j++) {
            if (strcmp(data->envs[i].hypers[j].key, hyper) == 0) {
                return &data->envs[i].hypers[j];
            }
        }
    }
    printf("Error: hyper %s not found in env %s\n", hyper, env);
    exit(1);
    return NULL;
}

// TODO: Slow as fuck
/*
Color rgb(float h) {
    float r = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f, 6.f) - 3.f) - 1.f));
    float g = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f + 4.f, 6.f) - 3.f) - 1.f));
    float b = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f + 2.f, 6.f) - 3.f) - 1.f));
    //return (Color){255.f, 255.f, 255.f, 255};
    return (Color){r * 255.f + .5f, g * 255.f + .5f, b * 255.f + .5f, 255};
}
*/

Color rgb(float h) {
    //return ColorFromHSV(180, h, 1.0);
    h = 120.0f * (1.0 + h);
    //return ColorFromHSV(h, 1.0, 1.0);
    return ColorFromHSV(h, 0.8f, 0.15f);
}

typedef struct PlotArgs {
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;
    int width;
    int height;
    int title_font_size;
    int axis_font_size;
    int axis_tick_font_size;
    int legend_font_size;
    int line_width;
    int tick_length;
    int top_margin;
    int bottom_margin;
    int left_margin;
    int right_margin;
    int tick_margin;
    Color font_color;
    Color background_color;
    Color axis_color;
    char* x_label;
    char* y_label;
    char* z_label;
    Font font;
    Font font_small;
} PlotArgs;

PlotArgs DEFAULT_PLOT_ARGS = {
    .x_min = EMPTY,
    .x_max = EMPTY,
    .y_min = EMPTY,
    .y_max = EMPTY,
    .z_min = EMPTY,
    .z_max = EMPTY,
    .width = 960,
    .height = 540 - SETTINGS_HEIGHT,
    .title_font_size = 32,
    .axis_font_size = 32,
    .axis_tick_font_size = 16,
    .legend_font_size = 12,
    .line_width = 2,
    .tick_length = 8,
    .tick_margin = 8,
    .top_margin = 70,
    .bottom_margin = 70,
    .left_margin = 100,
    .right_margin = 100,
    .font_color = PUFF_WHITE,
    .background_color = PUFF_BACKGROUND,
    .axis_color = PUFF_WHITE,
    .x_label = "Cost",
    .y_label = "Score",
    .z_label = "Train/Learning Rate",
};

float signed_log10(float x) {
    if (fabs(x) < 1e-8) {
        return -8.0f;
    }
    if (x > 0) {
        return log10(x);
    }
    return -log10(-x);
}

const char* format_tick_label(double value) {
    static char buffer[32];
    int precision = 2;

    if (fabs(value) < 1e-10) {
        strcpy(buffer, "0");
        return buffer;
    }

    if (fabs(value) < 0.01 || fabs(value) > 10000) {
        snprintf(buffer, sizeof(buffer), "%.2e", value);
    } else {
        snprintf(buffer, sizeof(buffer), "%.*f", precision, value);

        char *end = buffer + strlen(buffer) - 1;
        while (end > buffer && *end == '0') *end-- = '\0';
        if (end > buffer && *end == '.') *end = '\0';
    }

    return buffer;
}

void draw_axes(PlotArgs args) {
    DrawLine(args.left_margin, args.top_margin,
        args.left_margin, args.height - args.bottom_margin, PUFF_WHITE);
    DrawLine(args.left_margin, args.height - args.bottom_margin,
        args.width - args.right_margin, args.height - args.bottom_margin, PUFF_WHITE);
}

void draw_labels(PlotArgs args) {
    // X label
    Vector2 x_font_size = MeasureTextEx(args.font, args.x_label, args.axis_font_size, 0);
    DrawTextEx(
        args.font,
        args.x_label,
        (Vector2){
            args.width/2 - x_font_size.x/2,
            args.height - x_font_size.y,
        },
        args.axis_font_size,
        0,
        PUFF_WHITE
    );

    // Y label
    Vector2 y_font_size = MeasureTextEx(args.font, args.y_label, args.axis_font_size, 0);
    DrawTextPro(
        args.font,
        args.y_label,
        (Vector2){
            0,
            args.height/2 + y_font_size.x/2
        },
        (Vector2){ 0, 0 },
        -90,
        args.axis_font_size,
        0,
        PUFF_WHITE
    );
}


void draw_ticks(PlotArgs args) {
    int width = args.width;
    int height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    // Autofit number of ticks
    Vector2 tick_label_size = MeasureTextEx(args.font, "estimate", args.axis_font_size, 0);
    int num_x_ticks = 1 + plot_width/tick_label_size.x;
    int num_y_ticks = 1 + plot_height/tick_label_size.y;

    // X ticks
    for (int i=0; i<num_x_ticks; i++) {
        float val = args.x_min + i*(args.x_max - args.x_min)/(num_x_ticks - 1.0f);
        char* label = format_tick_label(val);
        float x_pos = args.left_margin + i*plot_width/(num_x_ticks - 1.0f);
        DrawLine(
            x_pos,
            height - args.bottom_margin - args.tick_length,
            x_pos,
            height - args.bottom_margin + args.tick_length,
            args.axis_color
        );

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                x_pos - this_tick_size.x/2,
                height - args.bottom_margin + args.tick_length + args.tick_margin,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
    }

    // Y ticks
    for (int i=0; i<num_y_ticks; i++) {
        float val = args.y_min + i*(args.y_max - args.y_min)/(num_y_ticks - 1.0f);
        char* label = format_tick_label(val);
        float y_pos = height - args.bottom_margin - i*plot_height/(num_y_ticks - 1.0f);
        DrawLine(
            args.left_margin - args.tick_length,
            y_pos,
            args.left_margin + args.tick_length,
            y_pos,
            args.axis_color
        );
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                args.left_margin - this_tick_size.x - args.tick_length - args.tick_margin,
                y_pos - this_tick_size.y/2,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
 
    }
}

void draw_box_ticks(char* hypers[], int hyper_count, PlotArgs args) {
    float width = args.width;
    float height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    // Autofit number of ticks
    Vector2 tick_label_size = MeasureTextEx(args.font, "estimate", args.axis_font_size, 0);
    int num_x_ticks = 1 + plot_width/tick_label_size.x;
    int num_y_ticks = 1 + plot_height/tick_label_size.y;

    // X ticks
    for (int i=0; i<num_x_ticks; i++) {
        float val = args.x_min + i*(args.x_max - args.x_min)/(num_x_ticks - 1.0f);
        char* label = format_tick_label(val);
        float x_pos = args.left_margin + i*plot_width/(num_x_ticks - 1.0f);
        DrawLine(
            x_pos,
            height - args.bottom_margin - args.tick_length,
            x_pos,
            height - args.bottom_margin + args.tick_length,
            args.axis_color
        );

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                x_pos - this_tick_size.x/2,
                height - args.bottom_margin + args.tick_length + args.tick_margin,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
    }

    // Y ticks
    for (int i=0; i<hyper_count; i++) {
        char* label = hypers[i];
        float y_pos = height - args.bottom_margin - (i + 0.5f)*plot_height/hyper_count;
        DrawLine(
            args.left_margin - args.tick_length,
            y_pos,
            args.left_margin + args.tick_length,
            y_pos,
            args.axis_color
        );
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                args.left_margin - this_tick_size.x - args.tick_length - args.tick_margin,
                y_pos - this_tick_size.y/2,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
 
    }
}


void draw_axes3(PlotArgs args, bool log_x, bool log_y, bool log_z) {
    float extent = 1.0f;
    float dx = log_x ? log10(args.x_max) - log10(args.x_min) : args.x_max - args.x_min;
    float dy = log_y ? log10(args.y_max) - log10(args.y_min) : args.y_max - args.y_min;
    float dz = log_z ? log10(args.z_max) - log10(args.z_min) : args.z_max - args.z_min;
    extent = fmax(extent, dx);
    extent = fmax(extent, dy);
    extent = fmax(extent, dz);

    float x = log_x ? log10(args.x_min) : args.x_min;
    float y = log_y ? log10(args.y_min) : args.y_min;
    float z = log_z ? log10(args.z_min) : args.z_min;

    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
    extent = 1.0f;

    DrawLine3D(
        (Vector3){x, y, z},
        (Vector3){x + extent, y, z},
        RED
    );
    DrawLine3D(
        (Vector3){x, y, z},
        (Vector3){x, y + extent, z},
        GREEN
    );
    DrawLine3D(
        (Vector3){x, y, z},
        (Vector3){x, y, z + extent},
        BLUE
    );
}

float hyper_min(Dataset *data, char* key, int start, int end) {
    float mmin = FLT_MAX;
    for (int env=start; env<end; env++) {
        for (int i=0; i<data->envs[env].n; i++) {
            Hyper* hyper = &data->envs[env].hypers[i];
            if (strcmp(hyper->key, key) != 0) {
                continue;
            }
            for (int j=0; j<hyper->n; j++) {
                float val = hyper->ary[j];
                if (val < mmin){
                    mmin = val;
                }
            }
        }
    }
    return mmin;
}

float hyper_max(Dataset *data, char* key, int start, int end) {
    float mmax = -FLT_MAX;
    for (int i=start; i<end; i++) {
        for (int j=0; j<data->envs[i].n; j++) {
            Hyper* hyper = &data->envs[i].hypers[j];
            if (strcmp(hyper->key, key) != 0) {
                continue;
            }
            for (int k=0; k<hyper->n; k++) {
                float val = hyper->ary[k];
                if (val > mmax){
                    mmax = val;
                }
            }
        }
    }
    return mmax;
}


void boxplot(Hyper* hyper, bool log_x, int i, int hyper_count, PlotArgs args, Color color, bool* filter) {
    int width = args.width;
    int height = args.height;

    float x_min = args.x_min;
    float x_max = args.x_max;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    if (log_x) {
        x_min = x_min<=1e-8 ? -8 : log10(x_min);
        x_max = x_max<=1e-8 ? -8 : log10(x_max);
    }

    float dx = x_max - x_min;
    if (dx == 0) dx = 1.0f;
    x_min -= 0.1f * dx; x_max += 0.1f * dx;
    dx = x_max - x_min;
    float dy = plot_height/((float)hyper_count);

    Color faded = Fade(color, 0.15f);

    float* ary = hyper->ary;
    float mmin = ary[0];
    float mmax = ary[0];
    for (int j=0; j<hyper->n; j++) {
        if (filter != NULL && !filter[j]) {
            continue;
        }
        mmin = fmin(mmin, ary[j]);
        mmax = fmax(mmax, ary[j]);
    }

    if (log_x) {
        mmin = mmin <= 0 ? 0 : log10(mmin);
        mmax = mmax <= 0 ? 0 : log10(mmax);
    }

    float left = args.left_margin + (mmin - x_min)/(x_max - x_min)*plot_width;
    float right = args.left_margin + (mmax - x_min)/(x_max - x_min)*plot_width;

    // TODO - rough patch
    left = fmax(left, args.left_margin);
    right = fmin(right, width - args.right_margin);
    DrawRectangle(left, args.top_margin + i*dy, right - left, dy, faded);
}

// Struct for vertex data (screen-space position and color)
typedef struct {
    Vector2 pos; // Screen-space x, y
    Color color; // RGBA color
} PlotVertex;

void plot_gl(Shader shader, VertexBuffer vertices) {
    Particle* particles = vertices.vertices;
    int n = vertices.n;

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n*sizeof(Particle), particles, GL_STATIC_DRAW);
        glVertexAttribPointer(shader.locs[SHADER_LOC_VERTEX_POSITION], 3, GL_FLOAT, GL_FALSE, sizeof(Particle), 0);
        glEnableVertexAttribArray(shader.locs[SHADER_LOC_VERTEX_POSITION]);
        int vertexColorLoc = shader.locs[SHADER_LOC_VERTEX_COLOR];
        glVertexAttribPointer(vertexColorLoc, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(3*sizeof(float)));
        glEnableVertexAttribArray(vertexColorLoc);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    rlDrawRenderBatchActive();      // Draw iternal buffers data (previous draw calls)
    rlSetBlendMode(RL_BLEND_ADDITIVE);

    int currentTimeLoc = GetShaderLocation(shader, "currentTime");
    //float time = GetTime();
    //SetShaderValue(shader, currentTimeLoc, &time, SHADER_UNIFORM_FLOAT);
    // Switch to plain OpenGL
    //------------------------------------------------------------------------------
    glUseProgram(shader.id);

        glUniform1f(currentTimeLoc, GetTime());

        // Get the current modelview and projection matrix so the particle system is displayed and transformed
        Matrix modelViewProjection = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());

        glUniformMatrix4fv(shader.locs[SHADER_LOC_MATRIX_MVP], 1, false, MatrixToFloat(modelViewProjection));

        glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);

    glUseProgram(0);
    //------------------------------------------------------------------------------
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    rlSetBlendMode(RL_BLEND_ALPHA);
}

void plot(Shader shader, Hyper* x, Hyper* y, bool log_x, bool log_y, PlotArgs args, float* cmap, bool* filter) {
    assert(x->n == y->n);

    int width = args.width;
    int height = args.height;

    float plot_width = width - args.left_margin - args.right_margin;
    float plot_height = height - args.top_margin - args.bottom_margin;

    // Compute ranges and apply log scaling if needed
    //float x_min = log_x ? log10f(args.x_min) : args.x_min;
    //float x_max = log_x ? log10f(args.x_max) : args.x_max;
    //float y_min = log_y ? log10f(args.y_min) : args.y_min;
    //float y_max = log_y ? log10f(args.y_max) : args.y_max;
    float x_min = args.x_min;
    float x_max = args.x_max;
    float y_min = args.y_min;
    float y_max = args.y_max;

    float dx = x_max - x_min;
    float dy = y_max - y_min;

    // Count valid points after filtering
    int valid_count = 0;
    for (int i = 0; i < x->n; i++) {
        if (filter == NULL || filter[i]) valid_count++;
    }

    if (valid_count == 0) return; // Early exit if no points

    // Allocate vertex array
    PlotVertex* vertices = (PlotVertex*)malloc(valid_count * sizeof(PlotVertex));
    int idx = 0;

    // Preprocess points: transform and map to screen space
    Particle particles[MAX_PARTICLES] = { 0 };
    for (int i = 0; i < x->n; i++) {
        if (filter != NULL && !filter[i]) continue;

        // Apply log scaling
        float xi = log_x ? log10f(x->ary[i]) : x->ary[i];
        float yi = log_y ? log10f(y->ary[i]) : y->ary[i];

        // Map to screen coordinates with margins
        xi = args.left_margin + (xi - x_min) / dx * plot_width;
        yi = args.height - args.bottom_margin - (yi - y_min) / dy * plot_height;

        particles[i].x = xi;
        particles[i].y = yi;
        particles[i].i = i;
        Color c = rgb(cmap[i]);
        particles[i].r = c.r/255.0f;
        particles[i].g = c.g/255.0f;
        particles[i].b = c.b/255.0f;
        particles[i].a = c.a/255.0f;
        idx++;
    }

    VertexBuffer buffer = {&particles, MAX_PARTICLES};
    plot_gl(shader, buffer);
}

void plot3(Camera3D camera, Shader shader, Hyper* x, Hyper* y, Hyper* z,
        bool log_x, bool log_y, bool log_z, PlotArgs args, float* cmap, bool* filter) {
    assert(x->n == y->n  && x->n == z->n);
    float x_min = args.x_min;
    float x_max = args.x_max;
    float y_min = args.y_min;
    float y_max = args.y_max;
    float z_min = args.z_min;
    float z_max = args.z_max;

    if (log_x) {
        x_min = signed_log10(x_min);
        x_max = signed_log10(x_max);
    }
    if (log_y) {
        y_min = signed_log10(y_min);
        y_max = signed_log10(y_max);
    }
    if (log_z) {
        z_min = signed_log10(z_min);
        z_max = signed_log10(z_max);
    }

    float dx = x_max - x_min;
    float dy = y_max - y_min;
    float dz = z_max - z_min;

    Particle particles[MAX_PARTICLES] = { 0 };
    int idx = 0;
    // Plot lines
    for (int i = 0; i < x->n; i++) {
        if (filter != NULL && !filter[i]) {
            continue;
        }
        float xi = (log_x) ? signed_log10(x->ary[i]) : x->ary[i];
        float yi = (log_y) ? signed_log10(y->ary[i]) : y->ary[i];
        float zi = (log_z) ? signed_log10(z->ary[i]) : z->ary[i];

        Color c = rgb(cmap[i]);
        Vector3 point = (Vector3){(xi - x_min)/dx, (yi - y_min)/dy, (zi - z_min)/dz};

        /*
        DrawCube(
            (Vector3){(xi - x_min)/dx, (yi - y_min)/dy, (zi - z_min)/dz},
            0.02f, 0.02f, 0.02f, c
        );

        DrawSphere(
            (Vector3){(xi - x_min)/dx, (yi - y_min)/dy, (zi - z_min)/dz},
            0.02f, c
        );
        */

        // Project to screen space
        Vector2 screen_pos = GetWorldToScreenEx(point, camera, 960, 520);
 
        particles[i].x = screen_pos.x;
        particles[i].y = screen_pos.y;
        particles[i].i = i;
        c = rgb(cmap[i]);
        particles[i].r = c.r/255.0f;
        particles[i].g = c.g/255.0f;
        particles[i].b = c.b/255.0f;
        particles[i].a = c.a/255.0f;
        idx++;

        //DrawBillboard(camera, whiteTexture, point, 0.1f, c);

    }
    VertexBuffer buffer = {&particles, idx};
    plot_gl(shader, buffer);


}


int cleanup(Hyper *map, int map_count, cJSON *root, char *json_str) {
    if (map) {
        for (int i=0; i<map_count; i++) {
            if (map[i].key) free(map[i].key);
            if (map[i].ary) free(map[i].ary);
        }
    }
    if (root) cJSON_Delete(root);
    if (json_str) free(json_str);
    return 1;
}

void GuiDropdownCheckbox(int x, int y, char* options, int *selection, bool *active, char *text, bool *checked) {
    Rectangle rect = {x, y, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
    if (GuiDropdownBox(rect, options, selection, *active)) {
        *active = !*active;
    }
    Rectangle check_rect = {x + rect.width , y, SETTINGS_HEIGHT, rect.height};
    GuiCheckBox(check_rect, text, checked);
}

void GuiDropdownFilter(int x, int y, char* options, int *selection, bool *dropdown_active,
        Vector2 focus, char *text1, float *text1_val, char *text2, float *text2_val) {
    Rectangle rect = {x, y, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
    if (GuiDropdownBox(rect, options, selection, *dropdown_active)) {
        *dropdown_active = !*dropdown_active;
    }
    Rectangle text1_rect = {x + rect.width, y, DROPDOWN_WIDTH/2, SETTINGS_HEIGHT};
    bool text1_active = CheckCollisionPointRec(focus, text1_rect);
    if (GuiTextBox(text1_rect, text1, 32, text1_active)) {
        *text1_val = atof(text1);
    }
    Rectangle text2_rect = {x + 1.5*DROPDOWN_WIDTH, y, DROPDOWN_WIDTH/2, SETTINGS_HEIGHT};
    bool text2_active = CheckCollisionPointRec(focus, text2_rect);
    if (GuiTextBox(text2_rect, text2, 32, text2_active)) {
        *text2_val = atof(text2);
    }
}
 


void apply_filter(bool* filter, Hyper* param, float min, float max) {
    for (int i=0; i<param->n; i++) {
        float val = param->ary[i];
        if (val < min || val > max) {
            filter[i] = false;
        }
    }
}

void calc_cmap(float* cmap, Hyper* param, float c_min, float c_max, bool log) {
    if (log) {
        c_min = signed_log10(c_min);
        c_max = signed_log10(c_max);
    }
    for (int i=0; i<param->n; i++) {
        float val = param->ary[i];
        if (log) {
            val = signed_log10(val);
        }
        cmap[i] = (val - c_min)/(c_max - c_min);
    }
}

int main(void) {
    FILE *file = fopen("pufferlib/ocean/constellation/all_cache.json", "r");
    if (!file) {
        printf("Error opening file\n");
        return 1;
    }

    // Read in file
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_str = malloc(file_size + 1);
    fread(json_str, 1, file_size, file);
    json_str[file_size] = '\0';
    fclose(file);
    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("JSON parse error: %s\n", cJSON_GetErrorPtr());
        free(json_str);
        return 1;
    }
    if (!cJSON_IsObject(root)) {
        printf("Error: Root is not an object\n");
        return cleanup(NULL, 0, root, json_str);
    }

    // Load in dataset
    Dataset data = {NULL, 0};
    cJSON *json_env = root->child;
    while (json_env) {
        data.n++;
        json_env = json_env->next;
    }

    Env *envs = calloc(data.n, sizeof(Env));
    data.envs = envs;
    json_env = root->child;
    int max_data_points = 0;
    for (int i=0; i<data.n; i++) {
        json_env = cJSON_GetArrayItem(root, i);
        cJSON *json_hyper = json_env->child;
        int hyper_points = 0;
        while (json_hyper) {
            envs[i].n++;
            envs[i].key = strdup(json_env->string);
            int nxt_hyper_points = cJSON_GetArraySize(json_hyper);
            if (hyper_points == 0) {
                hyper_points = nxt_hyper_points;
            } else {
                assert(hyper_points == nxt_hyper_points);
            }
            if (hyper_points > max_data_points) {
                max_data_points = hyper_points;
            }
            json_hyper = json_hyper->next;
        }
        envs[i].hypers = calloc(envs[i].n, sizeof(Hyper));
        for (int j=0; j<envs[i].n; j++) {
            cJSON *json_hyper = cJSON_GetArrayItem(json_env, j);
            envs[i].hypers[j].key = strdup(json_hyper->string);
            envs[i].hypers[j].ary = calloc(hyper_points, sizeof(float));
            int n = cJSON_GetArraySize(json_hyper);
            envs[i].hypers[j].n = n;
            for (int k = 0; k < n; k++) {
                cJSON *sub = cJSON_GetArrayItem(json_hyper, k);
                if (cJSON_IsNumber(sub)) {
                    envs[i].hypers[j].ary[k] = (float)sub->valuedouble;
                } else {
                    continue;
                    //printf("Error: Non-number in array for key '%s' at index %d\n", map[idx].key, j);
                }
            }
        }
    }

    int hyper_count = 24;
    char *hyper_key[24] = {
        "agent_steps",
        "cost",
        "environment/perf",
        "environment/score",
        "train/learning_rate",
        "train/ent_coef",
        "train/gamma",
        "train/gae_lambda",
        "train/vtrace_rho_clip",
        "train/vtrace_c_clip",
        "train/clip_coef",
        "train/vf_clip_coef",
        "train/vf_coef",
        "train/max_grad_norm",
        "train/adam_beta1",
        "train/adam_beta2",
        "train/adam_eps",
        "train/prio_alpha",
        "train/prio_beta0",
        "train/bptt_horizon",
        "train/num_minibatches",
        "train/minibatch_size",
        "policy/hidden_size",
        "env/num_envs",
    };

    //char* items[] = {"environment/score", "cost", "train/learning_rate", "train/gamma", "train/gae_lambda"};
    //char options[] = "environment/score;cost;train/learning_rate;train/gamma;train/gae_lambda";
          
    // Create options as a semicolon-separated string
    size_t options_len = 0;
    for (int i = 0; i < hyper_count; i++) {
        options_len += strlen(hyper_key[i]) + 1;
    }
    char *options = malloc(options_len);
    options[0] = '\0';
    for (int i = 0; i < hyper_count; i++) {
        if (i > 0) strcat(options, ";");
        strcat(options, hyper_key[i]);
    }

    // Options with extra "env_name;"
    char* extra = "env_name;";
    char *env_hyper_options = malloc(options_len + strlen(extra));
    strcpy(env_hyper_options, extra);
    strcat(env_hyper_options, options);

    // Env names as semi-colon-separated string
    size_t env_options_len = 4;
    for (int i = 0; i < data.n; i++) {
        env_options_len += strlen(data.envs[i].key) + 1;
    }
    char *env_options = malloc(env_options_len);
    strcpy(env_options, "all;");
    env_options[4] = '\0';
    for (int i = 0; i < data.n; i++) {
        if (i > 0) strcat(env_options, ";");
        strcat(env_options, data.envs[i].key);
    }

    // Initialize Raylib
    InitWindow(2*DEFAULT_PLOT_ARGS.width, 2*DEFAULT_PLOT_ARGS.height + 2*SETTINGS_HEIGHT, "Puffer Constellation");

    DEFAULT_PLOT_ARGS.font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 32, NULL, 255);
    DEFAULT_PLOT_ARGS.font_small = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 16, NULL, 255);
    Font gui_font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 14, NULL, 255);

    GuiLoadStyle("pufferlib/ocean/constellation/puffer.rgs");
    GuiSetFont(gui_font);
    ClearBackground(PUFF_BACKGROUND);
    SetTargetFPS(60);

    Shader shader = LoadShader(TextFormat("pufferlib/ocean/constellation/point_particle.vs", GLSL_VERSION),
                               TextFormat("pufferlib/ocean/constellation/point_particle.fs", GLSL_VERSION));

    // Allows the vertex shader to set the point size of each particle individually
    #ifndef GRAPHICS_API_OPENGL_ES2
    glEnable(GL_PROGRAM_POINT_SIZE);
    #endif

    Camera3D camera = (Camera3D){ 0 };
    camera.position = (Vector3){ 1.5f, 1.25f, 1.5f };
    camera.target = (Vector3){ 0.5f, 0.5f, 0.5f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    PlotArgs args1 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig1 = LoadRenderTexture(args1.width, args1.height);
    int fig1_env_idx = 0;
    bool fig1_env_active = false;
    bool fig1_x_active = false;
    int fig1_x_idx = 0;
    bool fig1_x_log = true;
    bool fig1_y_active = false;
    int fig1_y_idx = 2;
    bool fig1_y_log = false;
    bool fig1_z_active = false;
    int fig1_z_idx = 1;
    bool fig1_z_log = true;
    int fig1_color_idx = 0;
    bool fig1_color_active = false;
    bool fig1_log_color = true;

    PlotArgs args2 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig2 = LoadRenderTexture(args2.width, args2.height);
    //SetTextureFilter(fig2.texture, TEXTURE_FILTER_POINT);
    args2.left_margin = 50;
    args2.right_margin = 50;
    int fig2_env_idx = 1;
    bool fig2_env_active = false;
    bool fig2_x_active = false;
    int fig2_x_idx = 1;
    bool fig2_x_log = true;
    bool fig2_y_active = false;
    int fig2_y_idx = 2;
    bool fig2_y_log = false;
    int fig2_color_idx = 1;
    bool fig2_color_active = false;
    bool fig2_log_color = true;

    PlotArgs args3 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig3 = LoadRenderTexture(args3.width, args3.height);
    args3.left_margin = 10;
    args3.right_margin = 10;
    args3.top_margin = 10;
    args3.bottom_margin = 10;
    bool fig3_range1_active = false;
    int fig3_range1_idx = 2;
    char fig3_range1_min[32];
    char fig3_range1_max[32];
    float fig3_range1_min_val = 0;
    float fig3_range1_max_val = 1;
    bool fig3_range2_active = false;
    int fig3_range2_idx = 1;
    char fig3_range2_min[32];
    char fig3_range2_max[32];
    float fig3_range2_min_val = 0;
    float fig3_range2_max_val = 10000;

    PlotArgs args4 = DEFAULT_PLOT_ARGS;
    RenderTexture2D fig4 = LoadRenderTexture(args4.width, args4.height);
    bool fig4_x_log = true;
    bool fig4_range1_active = false;
    int fig4_range1_idx = 2;
    char fig4_range1_min[32];
    char fig4_range1_max[32];
    float fig4_range1_min_val = 0;
    float fig4_range1_max_val = 1;
    bool fig4_range2_active = false;
    int fig4_range2_idx = 1;
    char fig4_range2_min[32];
    char fig4_range2_max[32];
    float fig4_range2_min_val = 0;
    float fig4_range2_max_val = 10000;

    Hyper* x;
    Hyper* y;
    Hyper* z;
    char* x_label;
    char* y_label;
    char* z_label;

    bool *filter = calloc(max_data_points, sizeof(bool));
    float *cmap = calloc(max_data_points, sizeof(float));

    Vector2 focus = {0, 0};

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            focus = GetMousePosition();
        }

        // Figure 1
        x_label = hyper_key[fig1_x_idx];
        y_label = hyper_key[fig1_y_idx];
        z_label = hyper_key[fig1_z_idx];
        args1.x_label = x_label;
        args1.y_label = y_label;
        args1.z_label = z_label;
        int start = 0;
        int end = data.n;
        float c_min = 0.0f;
        float c_max = 1.0f;
        if (fig1_env_idx != 0) {
            start = fig1_env_idx - 1;
            end = fig1_env_idx;
        }
        args1.x_min = hyper_min(&data, hyper_key[fig1_x_idx], start, end);
        args1.x_max = hyper_max(&data, hyper_key[fig1_x_idx], start, end);
        args1.y_min = hyper_min(&data, hyper_key[fig1_y_idx], start, end);
        args1.y_max = hyper_max(&data, hyper_key[fig1_y_idx], start, end);
        args1.z_min = hyper_min(&data, hyper_key[fig1_z_idx], start, end);
        args1.z_max = hyper_max(&data, hyper_key[fig1_z_idx], start, end);
        float x_mid = fig1_x_log ? (log10(args1.x_max) + log10(args1.x_min))/2.0f : (args1.x_max + args1.x_min)/2.0f;
        float y_mid = fig1_y_log ? (log10(args1.y_max) + log10(args1.y_min))/2.0f : (args1.y_max + args1.y_min)/2.0f;
        float z_mid = fig1_z_log ? (log10(args1.z_max) + log10(args1.z_min))/2.0f : (args1.z_max + args1.z_min)/2.0f;
        //camera.target = (Vector3){x_mid, y_mid, z_mid};
        BeginTextureMode(fig1);
        ClearBackground(PUFF_BACKGROUND);

        if (fig1_color_idx != 0) {
            c_min = hyper_min(&data, hyper_key[fig1_color_idx - 1], start, end);
            c_max = hyper_max(&data, hyper_key[fig1_color_idx - 1], start, end);
        }
        memset(cmap, 0.0f, data.n * sizeof(float));
        Hyper* color_param = NULL;
        for (int i=start; i<end; i++) {
            char* env = data.envs[i].key;
            x = get_hyper(&data, env, hyper_key[fig1_x_idx]);
            y = get_hyper(&data, env, hyper_key[fig1_y_idx]);
            z = get_hyper(&data, env, hyper_key[fig1_z_idx]);
            if (fig1_color_idx != 0) {
                color_param = get_hyper(&data, env, hyper_key[fig1_color_idx - 1]);
                calc_cmap(cmap, color_param, c_min, c_max, fig1_log_color);
            } else {
                for (int j=0; j<x->n; j++) {
                    cmap[j] = i/(float)data.n;
                }
            }
            //BeginShaderMode(shader);
            plot3(camera, shader, x, y, z, fig1_x_log, fig1_y_log, fig1_z_log, args1, cmap, NULL);
            //EndShaderMode();
        }
        BeginMode3D(camera);
        CustomUpdateCamera(&camera, CAMERA_ORBITAL);
        draw_axes3(args1, fig1_x_log, fig1_y_log, fig1_z_log);
        EndMode3D();
        EndTextureMode();


        // Figure 2
        x_label = hyper_key[fig2_x_idx];
        y_label = hyper_key[fig2_y_idx];
        args2.x_label = x_label;
        args2.y_label = y_label;
        args2.top_margin = 20;
        BeginTextureMode(fig2);
        ClearBackground(PUFF_BACKGROUND);

        start = 0;
        end = data.n;
        c_min = 0.0f;
        c_max = 1.0f;
        if (fig2_env_idx != 0) {
            start = fig2_env_idx - 1;
            end = fig2_env_idx;
        }

        args2.x_min = hyper_min(&data, hyper_key[fig2_x_idx], start, end);
        args2.x_max = hyper_max(&data, hyper_key[fig2_x_idx], start, end);
        args2.y_min = hyper_min(&data, hyper_key[fig2_y_idx], start, end);
        args2.y_max = hyper_max(&data, hyper_key[fig2_y_idx], start, end);
        args2.x_min = (fig2_x_log) ? log10(args2.x_min) : args2.x_min;
        args2.x_max = (fig2_x_log) ? log10(args2.x_max) : args2.x_max;
        args2.y_min = (fig2_y_log) ? log10(args2.y_min) : args2.y_min;
        args2.y_max = (fig2_y_log) ? log10(args2.y_max) : args2.y_max;
 
        if (fig2_color_idx != 0) {
            c_min = hyper_min(&data, hyper_key[fig2_color_idx - 1], start, end);
            c_max = hyper_max(&data, hyper_key[fig2_color_idx - 1], start, end);
        }
        memset(cmap, 0.0f, data.n * sizeof(float));
        color_param = NULL;

        //rlSetBlendMode(RL_BLEND_ADDITIVE);
        //BeginShaderMode(shader);
        for (int i=start; i<end; i++) {
            char* env = data.envs[i].key;
            x = get_hyper(&data, env, hyper_key[fig2_x_idx]);
            y = get_hyper(&data, env, hyper_key[fig2_y_idx]);
            if (fig2_color_idx != 0) {
                color_param = get_hyper(&data, env, hyper_key[fig2_color_idx - 1]);
                calc_cmap(cmap, color_param, c_min, c_max, fig2_log_color);
            } else {
                for (int j=0; j<x->n; j++) {
                    cmap[j] = i/(float)data.n;
                }
            }
            plot(shader, x, y, fig2_x_log, fig2_y_log, args2, cmap, NULL);
        }
        //EndShaderMode();
        //rlSetBlendMode(RL_BLEND_ALPHA);

        draw_axes(args2);
        draw_ticks(args2);
        EndTextureMode();

        // Figure 3
        args3.x_label = "tsne1";
        args3.y_label = "tsne2";
        args3.x_min = hyper_min(&data, "tsne1", 0, data.n);
        args3.x_max = hyper_max(&data, "tsne1", 0, data.n);
        args3.y_min = hyper_min(&data, "tsne2", 0, data.n);
        args3.y_max = hyper_max(&data, "tsne2", 0, data.n);
        BeginTextureMode(fig3);
        ClearBackground(PUFF_BACKGROUND);

        for (int i=0; i<data.n; i++) {
            char* env = data.envs[i].key;
            x = get_hyper(&data, env, "tsne1");
            y = get_hyper(&data, env, "tsne2");
            for (int j=0; j<x->n; j++) {
                cmap[j] = i/(float)data.n;
            }
            for (int j=0; j<x->n; j++) {
                filter[j] = true;
            }
            Hyper* filter_param_1 = get_hyper(&data, env, hyper_key[fig3_range1_idx]);
            apply_filter(filter, filter_param_1, fig3_range1_min_val, fig3_range1_max_val);
            Hyper* filter_param_2 = get_hyper(&data, env, hyper_key[fig3_range2_idx]);
            apply_filter(filter, filter_param_2, fig3_range2_min_val, fig3_range2_max_val);
            plot(shader, x, y, false, false, args3, cmap, filter);
        }
        //draw_axes(args3);
        EndTextureMode();

        // Figure 4
        args4.x_label = "Value";
        args4.y_label = "Hyperparameter";
        args4.left_margin = 170;
        args4.right_margin = 50;
        args4.top_margin = 10;
        args4.bottom_margin = 50;
        args4.x_min = 1e-8;
        args4.x_max = 1e8;
        BeginTextureMode(fig4);
        ClearBackground(PUFF_BACKGROUND);
        rlSetBlendFactorsSeparate(0x0302, 0x0303, 1, 0x0303, 0x8006, 0x8006);
        BeginBlendMode(BLEND_CUSTOM_SEPARATE);
        for (int i=0; i<data.n; i++) {
            Env* env = &data.envs[i];
            Hyper* filter_param_1 = get_hyper(&data, env->key, hyper_key[fig4_range1_idx]);
            Hyper* filter_param_2 = get_hyper(&data, env->key, hyper_key[fig4_range2_idx]);
            for (int j=0; j<hyper_count; j++) {
                Hyper* hyper = get_hyper(&data, env->key, hyper_key[j]);
                for (int k=0; k<hyper->n; k++) {
                    filter[k] = true;
                }
                apply_filter(filter, filter_param_1, fig4_range1_min_val, fig4_range1_max_val);
                apply_filter(filter, filter_param_2, fig4_range2_min_val, fig4_range2_max_val);
                boxplot(hyper, fig4_x_log, j, hyper_count, args4, PUFF_CYAN, filter);
            }
        }
        EndBlendMode();
        draw_axes(args4);
        draw_box_ticks(hyper_key, hyper_count, args4);
        EndTextureMode();


        // Figure 1-4
        DrawTextureRec(
            fig1.texture,
            (Rectangle){0, 0, fig1.texture.width, -fig1.texture.height },
            (Vector2){ 0, SETTINGS_HEIGHT }, WHITE
        );
        DrawTextureRec(
            fig2.texture,
            (Rectangle){ 0, 0, fig2.texture.width, -fig2.texture.height },
            (Vector2){ fig1.texture.width, 2*SETTINGS_HEIGHT }, WHITE
        );
        DrawTextureRec(
            fig3.texture,
            (Rectangle){ 0, 0, fig3.texture.width, -fig3.texture.height },
            (Vector2){ 0, 2*SETTINGS_HEIGHT + fig1.texture.height }, WHITE
        );
        DrawTextureRec(
            fig4.texture,
            (Rectangle){ 0, 0, fig4.texture.width, -fig4.texture.height },
            (Vector2){ fig1.texture.width, fig1.texture.height + 2*SETTINGS_HEIGHT }, WHITE
        );


        // Figure 3 UI
        GuiDropdownFilter(0, SETTINGS_HEIGHT, options, &fig3_range1_idx, &fig3_range1_active, focus,
            fig3_range1_min, &fig3_range1_min_val, fig3_range1_max, &fig3_range1_max_val);
        GuiDropdownFilter(2*DROPDOWN_WIDTH, SETTINGS_HEIGHT, options, &fig3_range2_idx, &fig3_range2_active, focus,
            fig3_range2_min, &fig3_range2_min_val, fig3_range2_max, &fig3_range2_max_val);


        // Figure 4 UI
        GuiDropdownFilter(fig1.texture.width, SETTINGS_HEIGHT, options, &fig4_range1_idx, &fig4_range1_active, focus,
            fig4_range1_min, &fig4_range1_min_val, fig4_range1_max, &fig4_range1_max_val);
        GuiDropdownFilter(fig1.texture.width + 2*DROPDOWN_WIDTH, SETTINGS_HEIGHT, options, &fig4_range2_idx, &fig4_range2_active, focus,
            fig4_range2_min, &fig4_range2_min_val, fig4_range2_max, &fig4_range2_max_val); 
        

        // Figure 1 UI
        Rectangle fig1_env_rect = {0, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig1_env_rect, env_options, &fig1_env_idx, fig1_env_active)){
            fig1_env_active = !fig1_env_active;
        }
        GuiDropdownCheckbox(DROPDOWN_WIDTH, 0, options, &fig1_x_idx, &fig1_x_active, "Log X", &fig1_x_log);
        GuiDropdownCheckbox(2*DROPDOWN_WIDTH + TOGGLE_WIDTH, 0, options, &fig1_y_idx, &fig1_y_active, "Log Y", &fig1_y_log);
        GuiDropdownCheckbox(3*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, options, &fig1_z_idx, &fig1_z_active, "Log Z", &fig1_z_log);
        GuiDropdownCheckbox(4*DROPDOWN_WIDTH + 3*TOGGLE_WIDTH, 0, env_hyper_options, &fig1_color_idx, &fig1_color_active, "Log Color", &fig1_log_color);

        // Figure 2 UI
        Rectangle fig2_env_rect = {fig1.texture.width, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig2_env_rect, env_options, &fig2_env_idx, fig2_env_active)){
            fig2_env_active = !fig2_env_active;
        }
        GuiDropdownCheckbox(fig1.texture.width + DROPDOWN_WIDTH, 0, options, &fig2_x_idx, &fig2_x_active, "Log X", &fig2_x_log);
        GuiDropdownCheckbox(fig1.texture.width + 2*DROPDOWN_WIDTH + TOGGLE_WIDTH, 0, options, &fig2_y_idx, &fig2_y_active, "Log Y", &fig2_y_log);
        GuiDropdownCheckbox(fig1.texture.width + 3*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, env_hyper_options, &fig2_color_idx, &fig2_color_active, "Log Color", &fig2_log_color);


        //DrawFPS(GetScreenWidth() - 95, 10);
        EndDrawing();
    }

    UnloadShader(shader);
    CloseWindow();
    return 0;
}
