/* Quick rendering demo. You can use Xvfb to make it headless.
 * It is faster than writing individual frames to a file.
 *
 * apt install ffmpeg Xvfb
 * Xvfb :99 -screen 0 1280x720x24 &
 * 
 * Separate terminal:
 * bash scripts/build_ocean video fast
 * export DISPLAY=:99
 * ./video
 */

#include <raylib.h>
#include "rlgl.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/wait.h>

const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct {
    int pipefd[2];
    pid_t pid;
} VideoRecorder;

bool OpenVideo(VideoRecorder *recorder) {
    if (pipe(recorder->pipefd) == -1) {
        TraceLog(LOG_ERROR, "Failed to create pipe");
        return false;
    }

    recorder->pid = fork();
    if (recorder->pid == -1) {
        TraceLog(LOG_ERROR, "Failed to fork");
        return false;
    }

    if (recorder->pid == 0) { // Child process: run ffmpeg
        close(recorder->pipefd[1]);
        dup2(recorder->pipefd[0], STDIN_FILENO);
        close(recorder->pipefd[0]);
        execlp("ffmpeg", "ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-pix_fmt", "rgba",
               "-s", TextFormat("%dx%d", GetScreenWidth(), GetScreenHeight()),
               "-r", "60",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "fast",
               "-crf", "18",
               "-loglevel", "error",
               "output.mp4",
               NULL);
        TraceLog(LOG_ERROR, "Failed to launch ffmpeg");
        return false;
    }

    close(recorder->pipefd[0]); // Close read end in parent
    return true;
}

void WriteFrame(VideoRecorder *recorder) {
    int width = GetScreenWidth();
    int height = GetScreenHeight();
    unsigned char *screen_data = rlReadScreenPixels(width, height);
    write(recorder->pipefd[1], screen_data, width * height * 4 * sizeof(*screen_data));
    RL_FREE(screen_data);
}

void CloseVideo(VideoRecorder *recorder) {
    close(recorder->pipefd[1]);
    wait(NULL);
}

int main(void) {
    const int screenWidth = 800;
    const int screenHeight = 600;
    const int maxFrames = 300;
    float elapsedTime = 0;
    float writeFPS = 0;

    InitWindow(screenWidth, screenHeight, "Headless video saving");
    SetTargetFPS(6000); // High FPS so we can time this test

    VideoRecorder recorder;
    if (!OpenVideo(&recorder)) {
        CloseWindow();
        return -1;
    }

    // Sample program
    Texture2D texture = LoadTexture("resources/shared/puffers.png");
    Rectangle rightRec = {0.0f, 0.0f, 128.0f, 128.0f};
    Rectangle leftRec = {0.0f, 128.0f, 128.0f, 128.0f};
    Vector2 pos = {(float)(screenWidth/2.0f), (float)(screenHeight/2.0f)};
    Vector2 vel= {3.0f, 3.0f};

    int frame = 0;
    double startTime = GetTime();
    while (!WindowShouldClose()) {
        pos.x += vel.x;
        pos.y += vel.y;
        if (pos.x <= 0 || pos.x + 128 >= screenWidth) vel.x = -vel.x;
        if (pos.y <= 0 || pos.y + 128 >= screenHeight) vel.y = -vel.y;

        // Render as normal; no changes required
        BeginDrawing();
        ClearBackground(PUFF_BACKGROUND);
        DrawTextureRec(texture, (vel.x >= 0) ? rightRec : leftRec, pos, WHITE);
        DrawText("Headless video saving demo", 20, 20, 20, WHITE);
        EndDrawing();

        // Write a single frame
        if (frame < maxFrames) {
            WriteFrame(&recorder);
            frame++;
        }

        // Don't forget to close the file
        if (frame >= maxFrames) {
            double endTime = GetTime(); // End timing
            elapsedTime = endTime - startTime;
            writeFPS = (elapsedTime > 0) ? maxFrames / elapsedTime : 0;
            CloseVideo(&recorder);
            break;
        }
    }

    UnloadTexture(texture);
    CloseWindow();

    printf("Wrote %d frames in %.2f seconds (%.2f FPS)\n",
        maxFrames, elapsedTime, writeFPS);

    return 0;
}
