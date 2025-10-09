#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"


#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include "cJSON.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

const float EMPTY = -4242.0f;

#define SEP 4
#define SETTINGS_HEIGHT 20
#define TOGGLE_WIDTH 60
#define DROPDOWN_WIDTH 200

typedef struct {
    char *key;
    float *values;
    int size;
} KeyValue;

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
    .title_font_size = 24,
    .axis_font_size = 20,
    .axis_tick_font_size = 12,
    .legend_font_size = 12,
    .line_width = 2,
    .tick_length = 8,
    .x_margin = 70,
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
    DrawText(
        args.x_label,
        width/2 - x_font_size.x/2,
        height - x_font_size.y,
        args.axis_font_size,
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

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_font_size, 0);
        DrawText(
            label,
            x_pos - this_tick_size.x/2,
            height - args.y_margin + args.tick_length,
            args.axis_tick_font_size,
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
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_font_size, 0);
        DrawText(
            label,
            args.x_margin - this_tick_size.x - args.tick_length,
            y_pos,
            args.axis_tick_font_size,
            PUFF_WHITE
        );
 
    }
}

void draw_box_axes(KeyValue *hypers, int hyper_count, PlotArgs args) {
    int width = args.width;
    int height = args.height;

    // Draw axes
    DrawLine(args.x_margin, args.y_margin,
        args.x_margin, height - args.y_margin, PUFF_WHITE);
    DrawLine(args.x_margin, height - args.y_margin,
        width - args.x_margin, height - args.y_margin, PUFF_WHITE);

    // X label
    Vector2 x_font_size = MeasureTextEx(args.font, args.x_label, args.axis_font_size, 0);
    DrawText(
        args.x_label,
        width/2 - x_font_size.x/2,
        height - x_font_size.y,
        args.axis_font_size,
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

        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_font_size, 0);
        DrawText(
            label,
            x_pos - this_tick_size.x/2,
            height - args.y_margin + args.tick_length,
            args.axis_tick_font_size,
            PUFF_WHITE
        );
    }

    // Y ticks
    for (int i=0; i<hyper_count; i++) {
        char* label = hypers[i].key;
        float y_pos = height - args.y_margin - i*(height - 2*args.y_margin)/hyper_count;
        DrawLine(
            args.x_margin - args.tick_length,
            y_pos,
            args.x_margin + args.tick_length,
            y_pos,
            args.axis_color
        );
        Vector2 this_tick_size = MeasureTextEx(args.font, label, args.axis_font_size, 0);
        DrawText(
            label,
            args.x_margin - this_tick_size.x - args.tick_length,
            y_pos,
            args.axis_tick_font_size,
            PUFF_WHITE
        );
 
    }
}


void draw_axes3(PlotArgs args) {
    DrawLine3D(
        (Vector3){-10.0f, 0, 0},
        (Vector3){10.0f, 0, 0},
        RED
    );
    DrawLine3D(
        (Vector3){0, -10.0f, 0},
        (Vector3){0, 10.0f, 0},
        GREEN
    );
    DrawLine3D(
        (Vector3){0, 0, -10.0f},
        (Vector3){0, 0, 10.0f},
        BLUE
    );
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

void boxplot(float* mmin, float* mmax, bool log_x, int num_points, PlotArgs args) {
    int width = args.width;
    int height = args.height;

    // Find min/max for scaling
    //float z_min = args.z_min == EMPTY ? ary_min(z, num_points) : args.z_min;
    //float z_max = args.z_max == EMPTY ? ary_max(z, num_points) : args.z_max;

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
    float dy = (height - 2*args.y_margin)/((float)num_points);

    // Plot lines
    for (int j=0; j<num_points; j++) {
        float x1 = mmin[j];
        float x2 = mmax[j];

        if (log_x) {
            x1 = x1 <= 0 ? 0 : log10(x1);
            x2 = x2 <= 0 ? 0 : log10(x2);
        }

        float left = args.x_margin + (x1 - x_min)/(x_max - x_min)*(width - 2*args.x_margin);
        float right = args.x_margin + (x2 - x_min)/(x_max - x_min)*(width - 2*args.x_margin);
        DrawRectangle(left, args.y_margin + j*dy, right - left, dy, PUFF_CYAN);
    }
}


void plot(float* x, float* y, int num_points, PlotArgs args) {
    int width = args.width;
    int height = args.height;

    // Find min/max for scaling
    //float z_min = args.z_min == EMPTY ? ary_min(z, num_points) : args.z_min;
    //float z_max = args.z_max == EMPTY ? ary_max(z, num_points) : args.z_max;

    float x_min = args.x_min;
    float x_max = args.x_max;
    float y_min = args.y_min;
    float y_max = args.y_max;

    float dx = x_max - x_min;
    float dy = y_max - y_min;
    if (dx == 0) dx = 1.0f;
    if (dy == 0) dy = 1.0f;
    x_min -= 0.1f * dx; x_max += 0.1f * dx;
    y_min -= 0.1f * dy; y_max += 0.1f * dy;
    dx = x_max - x_min;
    dy = y_max - y_min;

    // Plot lines
    for (int j = 0; j < num_points - 1; j++) {
        float x1 = args.x_margin + (x[j] - x_min) / dx * (width - 2*args.x_margin);
        float y1 = (height - args.y_margin) - (y[j] - y_min) / dy * (height - 2*args.y_margin);
        /*
        float x2 = args.margin + (x[j + 1] - x_min) / dx * (width - 2*args.margin);
        float y2 = (height - args.margin) - (y[j + 1] - y_min) / dy * (height - 2*args.margin);
        DrawLine(x1, y1, x2, y2, PUFF_CYAN);
        */
        DrawCircle(x1, y1, args.line_width, PUFF_CYAN);
    }
}

void plot3(float* x, float* y, float* z, bool log_x, bool log_y, bool log_z, int num_points, PlotArgs args) {
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
    for (int j = 0; j < num_points - 1; j++) {
        float xj = (log_x) ? log10(x[j]) : x[j];
        float yj = (log_y) ? log10(y[j]) : y[j];
        float zj = (log_z) ? log10(z[j]) : z[j];
        DrawSphere((Vector3){xj, yj, zj}, 0.1f, PUFF_CYAN);
    }
}


float* get_values(KeyValue *map, int map_count, char *search_key, int *out_size) {
    for (int i = 0; i < map_count; i++) {
        if (map[i].key && strcmp(map[i].key, search_key) == 0) {
            *out_size = map[i].size;
            return map[i].values;
        }
    }
    return NULL;
}

int cleanup(KeyValue *map, int map_count, cJSON *root, char *json_str) {
    if (map) {
        for (int i=0; i<map_count; i++) {
            if (map[i].key) free(map[i].key);
            if (map[i].values) free(map[i].values);
        }
    }
    if (root) cJSON_Delete(root);
    if (json_str) free(json_str);
    return 1;
}

void compute_boxplot_data(KeyValue *hypers, float *box_mmin, float *box_mmax, int hyper_count, PlotArgs *args) {
    args->x_min = 1e-8;
    args->x_max = 1e8;

    for (int i=0; i<hyper_count; i++) {
        float* values = hypers[i].values;
        box_mmin[i] = values[0];
        box_mmax[i] = values[0];

        for (int j=0; j<hypers[i].size; j++) {
            box_mmin[i] = fmin(box_mmin[i], values[j]);
            box_mmax[i] = fmax(box_mmax[i], values[j]);
            //args->x_min = fmin(args->x_min, values[j]);
            //args->x_max = fmax(args->x_max, values[j]);
        }
    }
}



int main(void) {
    FILE *file = fopen("pufferlib/ocean/plot/data.json", "r");
    if (!file) {
        printf("Error opening file\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_str = malloc(file_size + 1);
    if (!json_str) {
        printf("Memory allocation error\n");
        fclose(file);
        return 1;
    }

    // Read file into buffer
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

    int map_count = 0;
    cJSON *item = root->child;
    while (item) {
        map_count++;
        item = item->next;
    }
    KeyValue *map = calloc(map_count, sizeof(KeyValue));
    if (!map) {
        printf("Memory allocation error\n");
        return cleanup(NULL, 0, root, json_str);
    }

    // Load all keys and their float arrays
    int hyper_count = 0;
    int idx = 0;
    item = root->child;
    while (item) {
        map[idx].key = strdup(item->string);
        if (strncmp(map[idx].key, "train", 5) == 0) {
            hyper_count++;
        }
        if (!map[idx].key) {
            printf("Memory allocation error for key\n");
            return cleanup(map, map_count, root, json_str);
        }

        if (!cJSON_IsArray(item)) {
            printf("Error: Value for key '%s' is not an array\n", map[idx].key);
            return cleanup(map, map_count, root, json_str);
        }

        int array_size = cJSON_GetArraySize(item);
        map[idx].values = malloc(array_size * sizeof(float));
        if (!map[idx].values) {
            printf("Memory allocation error for values\n");
            return cleanup(map, map_count, root, json_str);
        }

        map[idx].size = array_size;

        for (int j = 0; j < array_size; j++) {
            cJSON *sub = cJSON_GetArrayItem(item, j);
            if (cJSON_IsNumber(sub)) {
                map[idx].values[j] = (float)sub->valuedouble;
            } else {
                continue;
                printf("Error: Non-number in array for key '%s' at index %d\n", map[idx].key, j);
                return cleanup(map, map_count, root, json_str);
            }
        }

        idx++;
        item = item->next;
    }

    // Create items as an array of strings
    //if (map_count > 100) {
    //    map_count = 100;
    //}
    char **items = malloc(map_count * sizeof(char *));
    if (!items) {
        printf("Memory allocation error\n");
        return cleanup(map, map_count, root, json_str);
    }
    for (int i = 0; i < map_count; i++) {
        items[i] = map[i].key;  // Or strdup if you need copies
    }

    // Create options as a semicolon-separated string
    size_t options_len = 0;
    for (int i = 0; i < map_count; i++) {
        options_len += strlen(map[i].key) + 1;  // +1 for semicolon or null
    }
    char *options = malloc(options_len);
    if (!options) {
        printf("Memory allocation error\n");
        free(items);
        return cleanup(map, map_count, root, json_str);
    }
    options[0] = '\0';
    for (int i = 0; i < map_count; i++) {
        if (i > 0) strcat(options, ";");
        strcat(options, map[i].key);
    }

    // Hypers

    hyper_count = 5;
    char *hyper_key[5] = {"train/learning_rate", "train/gamma", "train/gae_lambda", "train/ent_coef", "train/vf_coef"};
    KeyValue hypers[5];
    for (int i=0; i<5; i++) {
        hypers[i].key = hyper_key[i];
        hypers[i].values = get_values(map, map_count, hyper_key[i], &hypers[i].size);
    }
    float *box_mmin = malloc(hyper_count * sizeof(float));
    float *box_mmax = malloc(hyper_count * sizeof(float));

    // Example usage: Print the arrays
    // Cleanup
    //free(cost_array);
    //free(score_array);
    //cJSON_Delete(root);
    //free(json_str);

    //float *x = cost_array;
    //float *y = score_array;
    //float num_points = cost_size;

    //float *x = malloc(num_points * sizeof(float));
    //float *y = malloc(num_points * sizeof(float));
    
  
    // Initialize Raylib
    InitWindow(2*DEFAULT_PLOT_ARGS.width, 2*DEFAULT_PLOT_ARGS.height + 2*SETTINGS_HEIGHT, "Puffer Constellation");
    ClearBackground(PUFF_BACKGROUND);
    SetTargetFPS(60);

    Camera3D camera = (Camera3D){ 0 };
    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    PlotArgs args1 = DEFAULT_PLOT_ARGS;
    args1.font = GetFontDefault();
    RenderTexture2D fig1 = LoadRenderTexture(args1.width, args1.height);
    bool fig1_x_active = false;
    int fig1_x_idx = 2;
    bool fig1_x_log = true;
    bool fig1_y_active = false;
    int fig1_y_idx = 6;
    bool fig1_y_log = false;
    bool fig1_z_active = false;
    int fig1_z_idx = 1;
    bool fig1_z_log = true;

    PlotArgs args2 = DEFAULT_PLOT_ARGS;
    args2.font = GetFontDefault();
    RenderTexture2D fig2 = LoadRenderTexture(args2.width, args2.height);
    bool fig2_x_active = false;
    int fig2_x_idx = 1;
    bool fig2_y_active = false;
    int fig2_y_idx = 0;

    PlotArgs args3 = DEFAULT_PLOT_ARGS;
    args3.x_margin = 250;
    args3.font = GetFontDefault();
    RenderTexture2D fig3 = LoadRenderTexture(args3.width, args3.height);
    bool fig3_x_active = false;
    int fig3_x_idx = 3;
    bool fig3_x_log = true;
    bool fig3_y_active = false;
    int fig3_y_idx = 0;

    PlotArgs args4 = DEFAULT_PLOT_ARGS;
    args4.font = GetFontDefault();
    RenderTexture2D fig4 = LoadRenderTexture(args4.width, args4.height);
    bool fig4_x_active = false;
    int fig4_x_idx = 4;
    bool fig4_y_active = false;
    int fig4_y_idx = 0;

    //char* items[] = {"environment/score", "cost", "train/learning_rate", "train/gamma", "train/gae_lambda"};
    //char options[] = "environment/score;cost;train/learning_rate;train/gamma;train/gae_lambda";

    float* x;
    float* y;
    float* z;
    int num_points;
    char* x_label;
    char* y_label;
    char* z_label;

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);

        x_label = items[fig1_x_idx];
        y_label = items[fig1_y_idx];
        z_label = items[fig1_z_idx];
        args1.x_label = x_label;
        args1.y_label = y_label;
        args1.z_label = z_label;
        x = get_values(map, map_count, x_label, &num_points);
        y = get_values(map, map_count, y_label, &num_points);
        z = get_values(map, map_count, z_label, &num_points);
        args1.x_min = ary_min(x, num_points);
        args1.x_max = ary_max(x, num_points);
        args1.y_min = ary_min(y, num_points);
        args1.y_max = ary_max(y, num_points);
        args1.z_min = ary_min(z, num_points);
        args1.z_max = ary_max(z, num_points);
        float x_mid = fig1_x_log ? (log10(args1.x_max) + log10(args1.x_min))/2.0f : (args1.x_max + args1.x_min)/2.0f;
        float y_mid = fig1_y_log ? (log10(args1.y_max) + log10(args1.y_min))/2.0f : (args1.y_max + args1.y_min)/2.0f;
        float z_mid = fig1_z_log ? (log10(args1.z_max) + log10(args1.z_min))/2.0f : (args1.z_max + args1.z_min)/2.0f;
        camera.target = (Vector3){x_mid, y_mid, z_mid};
        BeginTextureMode(fig1);
        ClearBackground(PUFF_BACKGROUND);
        BeginMode3D(camera);
        UpdateCamera(&camera, CAMERA_ORBITAL);
        plot3(x, y, z, fig1_x_log, fig1_y_log, fig1_z_log, num_points, args1);
        draw_axes3(args1);
        EndMode3D();
        EndTextureMode();
        DrawTextureRec(
            fig1.texture,
            (Rectangle){0, 0, fig1.texture.width, -fig1.texture.height },
            (Vector2){ 0, SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig1_x_rect = {0, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig1_x_rect, options, &fig1_x_idx, fig1_x_active)){
            fig1_x_active = !fig1_x_active;
        }
        Rectangle fig1_x_check_rect = {DROPDOWN_WIDTH, 0, SETTINGS_HEIGHT, SETTINGS_HEIGHT};
        GuiCheckBox(fig1_x_check_rect, "Log X", &fig1_x_log);
        Rectangle fig1_y_rect = {DROPDOWN_WIDTH + TOGGLE_WIDTH, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig1_y_rect, options, &fig1_y_idx, fig1_y_active)){
            fig1_y_active = !fig1_y_active;
        }
        Rectangle fig1_y_check_rect = {2*DROPDOWN_WIDTH+TOGGLE_WIDTH, 0, SETTINGS_HEIGHT, SETTINGS_HEIGHT};
        GuiCheckBox(fig1_y_check_rect, "Log Y", &fig1_y_log);
        Rectangle fig1_z_rect = {2*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig1_z_rect, options, &fig1_z_idx, fig1_z_active)){
            fig1_z_active = !fig1_z_active;
        }
        Rectangle fig1_z_check_rect = {3*DROPDOWN_WIDTH + 2*TOGGLE_WIDTH, 0, SETTINGS_HEIGHT, SETTINGS_HEIGHT};
        GuiCheckBox(fig1_z_check_rect, "Log Z", &fig1_z_log);

        x_label = items[fig2_x_idx];
        y_label = items[fig2_y_idx];
        args2.x_label = x_label;
        args2.y_label = y_label;
        x = get_values(map, map_count, x_label, &num_points);
        y = get_values(map, map_count, y_label, &num_points);
        args2.x_min = ary_min(x, num_points);
        args2.x_max = ary_max(x, num_points);
        args2.y_min = ary_min(y, num_points);
        args2.y_max = ary_max(y, num_points);
        BeginTextureMode(fig2);
        ClearBackground(PUFF_BACKGROUND);
        plot(x, y, num_points, args2);
        draw_axes(args2);
        EndTextureMode();
        DrawTextureRec(
            fig2.texture,
            (Rectangle){ 0, 0, fig2.texture.width, -fig2.texture.height },
            (Vector2){ fig1.texture.width, SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig2_x_rect = {fig1.texture.width, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig2_x_rect, options, &fig2_x_idx, fig2_x_active)){
            fig2_x_active = !fig2_x_active;
        }
        Rectangle fig2_y_rect = {fig1.texture.width + DROPDOWN_WIDTH, 0, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig2_y_rect, options, &fig2_y_idx, fig2_y_active)){
            fig2_y_active = !fig2_y_active;
        }

        compute_boxplot_data(hypers, box_mmin, box_mmax, hyper_count, &args3);
        args3.x_label = "Value";
        args3.y_label = "Hyperparameter";
        BeginTextureMode(fig3);
        ClearBackground(PUFF_BACKGROUND);
        boxplot(box_mmin, box_mmax, fig3_x_log, hyper_count, args3);
        //draw_axes(args3);
        draw_box_axes(hypers, hyper_count, args3);
        EndTextureMode();
        DrawTextureRec(
            fig3.texture,
            (Rectangle){ 0, 0, fig3.texture.width, -fig3.texture.height },
            (Vector2){ 0, fig1.texture.height + 2*SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig3_x_rect = {0, fig1.texture.height + SETTINGS_HEIGHT, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig3_x_rect, options, &fig3_x_idx, fig3_x_active)){
            fig3_x_active = !fig3_x_active;
        }
        Rectangle fig3_y_rect = {DROPDOWN_WIDTH, fig1.texture.height + SETTINGS_HEIGHT, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig3_y_rect, options, &fig3_y_idx, fig3_y_active)){
            fig3_y_active = !fig3_y_active;
        }

        x_label = items[fig4_x_idx];
        y_label = items[fig4_y_idx];
        args4.x_label = x_label;
        args4.y_label = y_label;
        x = get_values(map, map_count, x_label, &num_points);
        y = get_values(map, map_count, y_label, &num_points);
        args4.x_min = ary_min(x, num_points);
        args4.x_max = ary_max(x, num_points);
        args4.y_min = ary_min(y, num_points);
        args4.y_max = ary_max(y, num_points);
        BeginTextureMode(fig4);
        ClearBackground(PUFF_BACKGROUND);
        plot(x, y, num_points, args4);
        draw_axes(args4);
        EndTextureMode();
        DrawTextureRec(
            fig4.texture,
            (Rectangle){ 0, 0, fig4.texture.width, -fig4.texture.height },
            (Vector2){ fig1.texture.width, fig1.texture.height + 2*SETTINGS_HEIGHT }, WHITE
        );
        Rectangle fig4_x_rect = {fig1.texture.width, fig1.texture.height + SETTINGS_HEIGHT, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig4_x_rect, options, &fig4_x_idx, fig4_x_active)){
            fig4_x_active = !fig4_x_active;
        }
        Rectangle fig4_y_rect = {fig1.texture.width + DROPDOWN_WIDTH, fig1.texture.height + SETTINGS_HEIGHT, DROPDOWN_WIDTH, SETTINGS_HEIGHT};
        if (GuiDropdownBox(fig4_y_rect, options, &fig4_y_idx, fig4_y_active)){
            fig4_y_active = !fig4_y_active;
        }

        DrawFPS(GetScreenWidth() - 95, 10);
        EndDrawing();
    }

    //free(x);
    //free(y);
    CloseWindow();
    return 0;
}
