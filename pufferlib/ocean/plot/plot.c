#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

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

    // Initialize Raylib
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int margin = 50;
    InitWindow(screenWidth, screenHeight, "CSV Data Plot");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);

        // Draw axes
        DrawLine(margin, margin, margin, screenHeight - margin, PUFF_WHITE);  // Y-axis
        DrawLine(margin, screenHeight - margin, screenWidth - margin, screenHeight - margin, PUFF_WHITE);  // X-axis

        // Plot lines
        for (int j = 0; j < num_points - 1; j++) {
            float px1 = margin + (x[j] - min_x) / dx * (screenWidth - 2 * margin);
            float py1 = (screenHeight - margin) - (y[j] - min_y) / dy * (screenHeight - 2 * margin);
            float px2 = margin + (x[j + 1] - min_x) / dx * (screenWidth - 2 * margin);
            float py2 = (screenHeight - margin) - (y[j + 1] - min_y) / dy * (screenHeight - 2 * margin);
            DrawLine(px1, py1, px2, py2, PUFF_CYAN);
        }

        EndDrawing();
    }

    free(x);
    free(y);
    CloseWindow();
    return 0;
}
