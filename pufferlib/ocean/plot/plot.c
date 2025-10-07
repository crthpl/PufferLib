#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct PlotArgs {
    int title_font_size;
    int axis_font_size;
    int axis_tick_font_size;
    int legend_font_size;
    int line_width;
    int margin;
    Color font_color;
    Color background_color;
    Color axis_color;
} PlotArgs;

PlotArgs DEFAULT_PLOT_ARGS = {
    .title_font_size = 24,
    .axis_font_size = 16,
    .axis_tick_font_size = 12,
    .legend_font_size = 12,
    .line_width = 2,
    .margin = 50,
    .font_color = PUFF_WHITE,
    .background_color = PUFF_BACKGROUND,
    .axis_color = PUFF_WHITE,
};

void plot(float* x, float* y, int num_points, PlotArgs args) {
    int width = GetScreenWidth();
    int height = GetScreenHeight();

    // Draw axes
    DrawLine(args.margin, args.margin,
        args.margin, height - args.margin, PUFF_WHITE);
    DrawLine(args.margin, height - args.margin,
        width - args.margin, height - args.margin, PUFF_WHITE);

    // Find min/max for scaling
    float min_x = x[0], max_x = x[0], min_y = y[0], max_y = y[0];
    for (int j = 1; j < num_points; j++) {
        if (x[j] < min_x) min_x = x[j];
        if (x[j] > max_x) max_x = x[j];
        if (y[j] < min_y) min_y = y[j];
        if (y[j] > max_y) max_y = y[j];
    }
    float dx = max_x - min_x;
    float dy = max_y - min_y;
    if (dx == 0) dx = 1.0f;
    if (dy == 0) dy = 1.0f;
    min_x -= 0.1f * dx; max_x += 0.1f * dx;
    min_y -= 0.1f * dy; max_y += 0.1f * dy;
    dx = max_x - min_x;
    dy = max_y - min_y;

    // Plot lines
    for (int j = 0; j < num_points - 1; j++) {
        float x1 = args.margin + (x[j] - min_x) / dx * (width - 2*args.margin);
        float y1 = (height - args.margin) - (y[j] - min_y) / dy * (height - 2*args.margin);
        float x2 = args.margin + (x[j + 1] - min_x) / dx * (width - 2*args.margin);
        float y2 = (height - args.margin) - (y[j + 1] - min_y) / dy * (height - 2*args.margin);
        DrawLine(x1, y1, x2, y2, PUFF_CYAN);
    }
}


int main(void) {
    // Read CSV file
    FILE *fp = fopen("pufferlib/ocean/plot/data.csv", "r");
    if (!fp) {
        printf("Failed to open data.csv\n");
        return 1;
    }

    // Skip header line
    char line[1024];
    if (!fgets(line, sizeof(line), fp)) {
        printf("Failed to read header\n");
        fclose(fp);
        return 1;
    }

    // Count lines for number of points
    int num_points = 0;
    long file_pos = ftell(fp);
    while (fgets(line, sizeof(line), fp)) num_points++;
    rewind(fp);
    fseek(fp, file_pos, SEEK_SET);  // Reset to after header

    if (num_points == 0) {
        printf("No data points\n");
        fclose(fp);
        return 1;
    }

    float *x = malloc(num_points * sizeof(float));
    float *y = malloc(num_points * sizeof(float));
    int i = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *token = strtok(line, ",");
        if (token) x[i] = atof(token);
        token = strtok(NULL, ",");
        if (token) y[i] = atof(token);
        i++;
    }
    fclose(fp);

    // Initialize Raylib
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int margin = 50;
    InitWindow(screenWidth, screenHeight, "CSV Data Plot");
    SetTargetFPS(60);

    RenderTexture2D fig = LoadRenderTexture(screenWidth, screenHeight);

    while (!WindowShouldClose()) {
        BeginTextureMode(fig);
        ClearBackground(PUFF_BACKGROUND);
        plot(x, y, num_points, DEFAULT_PLOT_ARGS);
        EndTextureMode();
        BeginDrawing();
        DrawTextureRec(
            fig.texture,
            (Rectangle){ 0, 0, fig.texture.width, -fig.texture.height },
            (Vector2){ 0, 0 }, WHITE
        );
        EndDrawing();
    }

    free(x);
    free(y);
    CloseWindow();
    return 0;
}
