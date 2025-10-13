#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "raylib.h"
#include "rlgl.h"

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "cJSON.h"

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

#define SEP 4
#define SETTINGS_HEIGHT 20
#define TOGGLE_WIDTH 60
#define DROPDOWN_WIDTH 120
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
Color rgb(float h) {
    float r = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f, 6.f) - 3.f) - 1.f));
    float g = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f + 4.f, 6.f) - 3.f) - 1.f));
    float b = fmaxf(0.f, fminf(1.f, fabsf(fmodf(h * 6.f + 2.f, 6.f) - 3.f) - 1.f));
    return (Color){r * 255.f + .5f, g * 255.f + .5f, b * 255.f + .5f, 255};
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
    int x_margin;
    int y_margin;
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
    .x_margin = 100,
    .y_margin = 70,
    .font_color = PUFF_WHITE,
    .background_color = PUFF_BACKGROUND,
    .axis_color = PUFF_WHITE,
    .x_label = "Cost",
    .y_label = "Score",
    .z_label = "Train/Learning Rate",
};

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
    int width = args.width;
    int height = args.height;

    // Draw axes
    DrawLine(args.x_margin, args.y_margin,
        args.x_margin, height - args.y_margin, PUFF_WHITE);
    DrawLine(args.x_margin, height - args.y_margin,
        width - args.x_margin, height - args.y_margin, PUFF_WHITE);

    // X label
    Vector2 x_font_size = MeasureTextEx(args.font, args.x_label, args.axis_font_size, 0);
    DrawTextEx(
        args.font,
        args.x_label,
        (Vector2){
            width/2 - x_font_size.x/2,
            height - x_font_size.y,
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
            height/2 + y_font_size.x/2
        },
        (Vector2){ 0, 0 },
        -90,
        args.axis_font_size,
        0,
        PUFF_WHITE
    );

    // Autofit number of ticks
    Vector2 tick_label_size = MeasureTextEx(args.font, "estimate", args.axis_font_size, 0);
    int num_x_ticks = (width - 2*args.x_margin)/tick_label_size.x;
    int num_y_ticks = (height - 2*args.y_margin)/tick_label_size.x;

    // X ticks
    for (int i=0; i<num_x_ticks; i++) {
        float val = args.x_min + i*(args.x_max - args.x_min)/(float)num_x_ticks;
        char* label = format_tick_label(val);
        float x_pos = args.x_margin + i*(width - 2*args.x_margin)/num_x_ticks;
        DrawLine(
            x_pos,
            height - args.y_margin - args.tick_length,
            x_pos,
            height - args.y_margin + args.tick_length,
            args.axis_color
        );

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                x_pos - this_tick_size.x/2,
                height - args.y_margin + args.tick_length,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
    }

    // Y ticks
    for (int i=0; i<num_y_ticks; i++) {
        float val = args.y_min + i*(args.y_max - args.y_min)/(float)num_y_ticks;
        char* label = format_tick_label(val);
        float y_pos = height - args.y_margin - i*(height - 2*args.y_margin)/num_y_ticks;
        DrawLine(
            args.x_margin - args.tick_length,
            y_pos,
            args.x_margin + args.tick_length,
            y_pos,
            args.axis_color
        );
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                args.x_margin - this_tick_size.x - args.tick_length,
                y_pos,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
 
    }
}

void draw_box_axes(char* hypers[], int hyper_count, PlotArgs args) {
    int width = args.width;
    int height = args.height;

    // Draw axes
    DrawLine(args.x_margin, args.y_margin,
        args.x_margin, height - args.y_margin, PUFF_WHITE);
    DrawLine(args.x_margin, height - args.y_margin,
        width - args.x_margin, height - args.y_margin, PUFF_WHITE);

    // X label
    Vector2 x_font_size = MeasureTextEx(args.font, args.x_label, args.axis_font_size, 0);
    DrawTextEx(
        args.font,
        args.x_label,
        (Vector2){
            width/2 - x_font_size.x/2,
            height - x_font_size.y,
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
            height/2 + y_font_size.x/2
        },
        (Vector2){ 0, 0 },
        -90,
        args.axis_font_size,
        0,
        PUFF_WHITE
    );

    // Autofit number of ticks
    Vector2 tick_label_size = MeasureTextEx(args.font, "estimate", args.axis_font_size, 0);
    int num_x_ticks = (width - 2*args.x_margin)/tick_label_size.x;
    int num_y_ticks = (height - 2*args.y_margin)/tick_label_size.x;

    // X ticks
    for (int i=0; i<num_x_ticks; i++) {
        float val = args.x_min + i*(args.x_max - args.x_min)/(float)num_x_ticks;
        char* label = format_tick_label(val);
        float x_pos = args.x_margin + i*(width - 2*args.x_margin)/num_x_ticks;
        DrawLine(
            x_pos,
            height - args.y_margin - args.tick_length,
            x_pos,
            height - args.y_margin + args.tick_length,
            args.axis_color
        );

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                x_pos - this_tick_size.x/2,
                height - args.y_margin + args.tick_length,
            },
            args.axis_tick_font_size,
            0,
            PUFF_WHITE
        );
    }

    // Y ticks
    for (int i=0; i<hyper_count; i++) {
        char* label = hypers[i];
        float y_pos = height - args.y_margin - i*(height - 2*args.y_margin)/hyper_count;
        DrawLine(
            args.x_margin - args.tick_length,
            y_pos,
            args.x_margin + args.tick_length,
            y_pos,
            args.axis_color
        );
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_tick_font_size, 0);
        DrawTextEx(
            args.font_small,
            label,
            (Vector2){
                args.x_margin - this_tick_size.x - args.tick_length,
                y_pos,
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


/*
float hyper_max(Hyper *hyper) {
    float max = hyper->ary[0];
    for (int i=1; i<hyper->n; i++) {
        if (hyper->ary[i] > max) max = hyper->ary[i];
    }
    return max;
}

float ary_min(float* ary, int num) {
    float min = ary[0];
    for (int i=1; i<num; i++) {
        if (ary[i] < min) min = ary[i];
    }
    return min;
}
float ary_max(float* ary, int num) {
    float max = ary[0];
    for (int i=1; i<num; i++) {
        if (ary[i] > max) max = ary[i];
    }
    return max;
}

*/

void boxplot(Dataset* data, bool log_x, char* env, char* hyper_key[], int hyper_count, PlotArgs args, Color color, float* filter) {
    int width = args.width;
    int height = args.height;

    float x_min = args.x_min;
    float x_max = args.x_max;

    if (log_x) {
        x_min = x_min<=1e-8 ? -8 : log10(x_min);
        x_max = x_max<=1e-8 ? -8 : log10(x_max);
    }

    float dx = x_max - x_min;
    if (dx == 0) dx = 1.0f;
    x_min -= 0.1f * dx; x_max += 0.1f * dx;
    dx = x_max - x_min;
    float dy = (height - 2*args.y_margin)/((float)hyper_count);

    Color faded = Fade(color, 0.15f);

    for (int i=0; i<hyper_count; i++) {
        Hyper* hyper = get_hyper(data, env, hyper_key[i]);
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

        float left = args.x_margin + (mmin - x_min)/(x_max - x_min)*(width - 2*args.x_margin);
        float right = args.x_margin + (mmax - x_min)/(x_max - x_min)*(width - 2*args.x_margin);
        DrawRectangle(left, args.y_margin + i*dy, right - left, dy, faded);
    }
}

void plot(Hyper* x, Hyper* y, bool log_x, bool log_y, PlotArgs args, float* cmap, bool* filter) {
    assert(x->n == y->n);

    int width = args.width;
    int height = args.height;
    float x_min = args.x_min;
    float x_max = args.x_max;
    float y_min = args.y_min;
    float y_max = args.y_max;

    float dx = x_max - x_min;
    float dy = y_max - y_min;

    for (int i=0; i<x->n; i++) {
        if (filter != NULL && !filter[i]) {
            continue;
        }
        float xi = log_x ? log10(x->ary[i]) : x->ary[i];
        float yi = log_y ? log10(y->ary[i]) : y->ary[i];
        xi = args.x_margin + (xi - x_min) / dx * (width - 2*args.x_margin);
        yi = (height - args.y_margin) - (yi - y_min) / dy * (height - 2*args.y_margin);
        if (xi < args.x_margin) {
            int s = 2;
        }
        Color c = rgb(cmap[i]);
        DrawCircle(xi, yi, args.line_width, c);
    }
}

void plot3(Hyper* x, Hyper* y, Hyper* z, bool log_x, bool log_y, bool log_z, PlotArgs args, float* cmap, bool* filter) {
    assert(x->n == y->n  && x->n == z->n);
    int width = args.width;
    int height = args.height;

    float x_min = args.x_min;
    float x_max = args.x_max;
    float y_min = args.y_min;
    float y_max = args.y_max;
    float z_min = args.z_min;
    float z_max = args.z_max;

    float dx = x_max - x_min;
    float dy = y_max - y_min;
    float dz = z_max - z_min;
    if (dx == 0) dx = 1.0f;
    if (dy == 0) dy = 1.0f;
    if (dz == 0) dz = 1.0f;
    x_min -= 0.1f * dx; x_max += 0.1f * dx;
    y_min -= 0.1f * dy; y_max += 0.1f * dy;
    z_min -= 0.1f * dz; z_max += 0.1f * dz;
    dx = x_max - x_min;
    dy = y_max - y_min;
    dz = z_max - z_min;


    // Plot lines
    for (int i = 0; i < x->n; i++) {
        if (filter != NULL && !filter[i]) {
            continue;
        }
        float xi = (log_x) ? log10(x->ary[i]) : x->ary[i];
        float yi = (log_y) ? log10(y->ary[i]) : y->ary[i];
        float zi = (log_z) ? log10(z->ary[i]) : z->ary[i];
        Color c = rgb(cmap[i]);
        DrawCube((Vector3){xi, yi, zi}, 0.02f, 0.02f, 0.02f, c);
    }
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
 


void apply_filter(float* filter, Hyper* param, float min, float max) {
    for (int i=0; i<param->n; i++) {
        float val = param->ary[i];
        if (val < min || val > max) {
            filter[i] = 0.0f;
        }
    }
}

float signed_log10(float x) {
    if (fabs(x) < 1e-8) {
        return -8.0f;
    }
    if (x > 0) {
        return log10(x);
    }
    return -log10(-x);
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

    int hyper_count = 9;
    char *hyper_key[9] = {
        "agent_steps", "cost", "environment/perf", "environment/score",
        "train/learning_rate", "train/gamma", "train/gae_lambda", "train/ent_coef", "train/vf_coef"
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
    GuiLoadStyle("pufferlib/ocean/constellation/puffer.rgs");
    ClearBackground(PUFF_BACKGROUND);
    SetTargetFPS(60);

    DEFAULT_PLOT_ARGS.font = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 32, NULL, 255);
    DEFAULT_PLOT_ARGS.font_small = LoadFontEx("resources/shared/JetBrainsMono-SemiBold.ttf", 16, NULL, 255);

    Camera3D camera = (Camera3D){ 0 };
    camera.position = (Vector3){ 5.0f, 2.0f, 5.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
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
    float *box_mmin = malloc(hyper_count * sizeof(float));
    float *box_mmax = malloc(hyper_count * sizeof(float));
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
    int num_points;
    char* x_label;
    char* y_label;
    char* z_label;

    float *filter = calloc(max_data_points, sizeof(float));
    float *cmap = calloc(max_data_points, sizeof(float));

    Vector2 focus = {0, 0};

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            focus = GetMousePosition();
        }

        x_label = hyper_key[fig1_x_idx];
        y_label = hyper_key[fig1_y_idx];
        z_label = hyper_key[fig1_z_idx];
        args1.x_label = x_label;
        args1.y_label = y_label;
        args1.z_label = z_label;
        args1.x_min = hyper_min(&data, hyper_key[fig1_x_idx], 0, data.n);
        args1.x_max = hyper_max(&data, hyper_key[fig1_x_idx], 0, data.n);
        args1.y_min = hyper_min(&data, hyper_key[fig1_y_idx], 0, data.n);
        args1.y_max = hyper_max(&data, hyper_key[fig1_y_idx], 0, data.n);
        args1.z_min = hyper_min(&data, hyper_key[fig1_z_idx], 0, data.n);
        args1.z_max = hyper_max(&data, hyper_key[fig1_z_idx], 0, data.n);
        float x_mid = fig1_x_log ? (log10(args1.x_max) + log10(args1.x_min))/2.0f : (args1.x_max + args1.x_min)/2.0f;
        float y_mid = fig1_y_log ? (log10(args1.y_max) + log10(args1.y_min))/2.0f : (args1.y_max + args1.y_min)/2.0f;
        float z_mid = fig1_z_log ? (log10(args1.z_max) + log10(args1.z_min))/2.0f : (args1.z_max + args1.z_min)/2.0f;
        camera.target = (Vector3){x_mid, y_mid, z_mid};
        BeginTextureMode(fig1);
        ClearBackground(PUFF_BACKGROUND);
        BeginMode3D(camera);
        UpdateCamera(&camera, CAMERA_ORBITAL);

        int start = 0;
        int end = data.n;
        float c_min = 0.0f;
        float c_max = 1.0f;
        if (fig1_env_idx != 0) {
            start = fig1_env_idx;
            end = fig1_env_idx + 1;
        }
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
            plot3(x, y, z, fig1_x_log, fig1_y_log, fig1_z_log, args1, cmap, NULL);
        }
        draw_axes3(args1, fig1_x_log, fig1_y_log, fig1_z_log);
        EndMode3D();
        EndTextureMode();
        DrawTextureRec(
            fig1.texture,
            (Rectangle){0, 0, fig1.texture.width, -fig1.texture.height },
            (Vector2){ 0, SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig1_env_rect = {0, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig1_env_rect, env_options, &fig1_env_idx, fig1_env_active)){
            fig1_env_active = !fig1_env_active;
        }
        GuiDropdownCheckbox(DROPDOWN_WIDTH, 0, options, &fig1_x_idx, &fig1_x_active, "Log X", &fig1_x_log);
        GuiDropdownCheckbox(2*DROPDOWN_WIDTH + TOGGLE_WIDTH, 0, options, &fig1_y_idx, &fig1_y_active, "Log Y", &fig1_y_log);
        GuiDropdownCheckbox(3*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, options, &fig1_z_idx, &fig1_z_active, "Log Z", &fig1_z_log);
        GuiDropdownCheckbox(4*DROPDOWN_WIDTH + 3*TOGGLE_WIDTH, 0, env_hyper_options, &fig1_color_idx, &fig1_color_active, "Log Color", &fig1_log_color);


        // Figure 2
        x_label = hyper_key[fig2_x_idx];
        y_label = hyper_key[fig2_y_idx];
        args2.x_label = x_label;
        args2.y_label = y_label;
        args2.x_min = hyper_min(&data, hyper_key[fig2_x_idx], 0, data.n);
        args2.x_max = hyper_max(&data, hyper_key[fig2_x_idx], 0, data.n);
        args2.y_min = hyper_min(&data, hyper_key[fig2_y_idx], 0, data.n);
        args2.y_max = hyper_max(&data, hyper_key[fig2_y_idx], 0, data.n);
        args2.x_min = (fig2_x_log) ? log10(args2.x_min) : args2.x_min;
        args2.x_max = (fig2_x_log) ? log10(args2.x_max) : args2.x_max;
        args2.y_min = (fig2_y_log) ? log10(args2.y_min) : args2.y_min;
        args2.y_max = (fig2_y_log) ? log10(args2.y_max) : args2.y_max;
        BeginTextureMode(fig2);
        ClearBackground(PUFF_BACKGROUND);

        start = 0;
        end = data.n;
        c_min = 0.0f;
        c_max = 1.0f;
        if (fig2_env_idx != 0) {
            start = fig2_env_idx;
            end = fig2_env_idx + 1;
        }
        if (fig2_color_idx != 0) {
            c_min = hyper_min(&data, hyper_key[fig2_color_idx - 1], start, end);
            c_max = hyper_max(&data, hyper_key[fig2_color_idx - 1], start, end);
        }
        memset(cmap, 0.0f, data.n * sizeof(float));
        color_param = NULL;
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
            plot(x, y, fig2_x_log, fig2_y_log, args2, cmap, NULL);
        }
        draw_axes(args2);
        EndTextureMode();
        DrawTextureRec(
            fig2.texture,
            (Rectangle){ 0, 0, fig2.texture.width, -fig2.texture.height },
            (Vector2){ fig1.texture.width, SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig2_env_rect = {fig1.texture.width, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig2_env_rect, env_options, &fig2_env_idx, fig2_env_active)){
            fig2_env_active = !fig2_env_active;
        }
        GuiDropdownCheckbox(fig1.texture.width + DROPDOWN_WIDTH, 0, options, &fig2_x_idx, &fig2_x_active, "Log X", &fig2_x_log);
        GuiDropdownCheckbox(fig1.texture.width + 2*DROPDOWN_WIDTH + TOGGLE_WIDTH, 0, options, &fig2_y_idx, &fig2_y_active, "Log Y", &fig2_y_log);
        GuiDropdownCheckbox(fig1.texture.width + 3*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, env_hyper_options, &fig2_color_idx, &fig2_color_active, "Log Color", &fig2_log_color);


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
                filter[j] = 1.0f;
            }
            Hyper* filter_param_1 = get_hyper(&data, env, hyper_key[fig3_range1_idx]);
            apply_filter(filter, filter_param_1, fig3_range1_min_val, fig3_range1_max_val);
            Hyper* filter_param_2 = get_hyper(&data, env, hyper_key[fig3_range2_idx]);
            apply_filter(filter, filter_param_2, fig3_range2_min_val, fig3_range2_max_val);
            plot(x, y, false, false, args3, cmap, filter);
        }
        draw_axes(args3);
        EndTextureMode();
        DrawTextureRec(
            fig3.texture,
            (Rectangle){ 0, 0, fig3.texture.width, -fig3.texture.height },
            (Vector2){ 0, SETTINGS_HEIGHT + fig1.texture.height }, WHITE
        );
        GuiDropdownFilter(0, fig1.texture.height, options, &fig3_range1_idx, &fig3_range1_active, focus,
            fig3_range1_min, &fig3_range1_min_val, fig3_range1_max, &fig3_range1_max_val);
        GuiDropdownFilter(2*DROPDOWN_WIDTH, fig1.texture.height, options, &fig3_range2_idx, &fig3_range2_active, focus,
            fig3_range2_min, &fig3_range2_min_val, fig3_range2_max, &fig3_range2_max_val);

        // Figure 4
        args4.x_label = "Value";
        args4.y_label = "Hyperparameter";
        args4.x_margin = 200;
        args4.x_min = 1e-8;
        args4.x_max = 1e8;
        BeginTextureMode(fig4);
        ClearBackground(PUFF_BACKGROUND);
        rlSetBlendFactorsSeparate(0x0302, 0x0303, 1, 0x0303, 0x8006, 0x8006);
        BeginBlendMode(BLEND_CUSTOM_SEPARATE);
        for (int i=0; i<data.n; i++) {
            //Hyper* filter_param_1 = get_hyper(&data, env, hyper_key[fig3_range1_idx]);
            //apply_filter(filter, filter_param_1, fig3_range1_min_val, fig3_range1_max_val);
            //Hyper* filter_param_2 = get_hyper(&data, env, hyper_key[fig3_range2_idx]);
            //apply_filter(filter, filter_param_2, fig3_range2_min_val, fig3_range2_max_val);
            boxplot(&data, fig4_x_log, data.envs[i].key, hyper_key, hyper_count, args4, PUFF_CYAN, filter);
        }
        EndBlendMode();
        draw_box_axes(hyper_key, hyper_count, args4);
        EndTextureMode();
        DrawTextureRec(
            fig4.texture,
            (Rectangle){ 0, 0, fig4.texture.width, -fig4.texture.height },
            (Vector2){ fig1.texture.width, fig1.texture.height + 2*SETTINGS_HEIGHT }, WHITE
        );
        GuiDropdownFilter(fig1.texture.width, fig1.texture.height + SETTINGS_HEIGHT, options, &fig4_range1_idx, &fig4_range1_active, focus,
            fig4_range1_min, &fig4_range1_min_val, fig4_range1_max, &fig4_range1_max_val);
        GuiDropdownFilter(fig1.texture.width + 2*DROPDOWN_WIDTH, fig1.texture.height + SETTINGS_HEIGHT, options, &fig4_range2_idx, &fig4_range2_active, focus,
            fig4_range2_min, &fig4_range2_min_val, fig4_range2_max, &fig4_range2_max_val); 
        
        DrawFPS(GetScreenWidth() - 95, 10);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
