#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

#if defined(__APPLE__)
    #define GL_SILENCE_DEPRECATION
    #include <OpenGL/gl3.h>
    #include <OpenGL/gl3ext.h>
#else
    #include "glad.h"
#endif
#include "rlgl.h"
#include "raymath.h"

#define GLSL_VERSION 330

#define SCREEN_W 1920
#define SCREEN_H 1080

#define FONT_TITLE 72
#define FONT_LABEL 42

static const Color BG     = {4, 8, 20, 255};
static const Color C_WHITE = {220, 230, 255, 255};
static const Color C_CYAN  = {0, 187, 187, 255};

// ─── Vertex layout (x, y, size_scale,  r, g, b, a) ───────────────────────────
typedef struct {
    float x, y, size_scale;
    float r, g, b, a;
} StarVertex;

// ─── GL draw ──────────────────────────────────────────────────────────────────
static void draw_stars(StarVertex *verts, int n, Shader *sh) {
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, n * sizeof(StarVertex), verts, GL_STREAM_DRAW);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_POSITION],
            3, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)0);
        glEnableVertexAttribArray(sh->locs[SHADER_LOC_VERTEX_POSITION]);
        glVertexAttribPointer(sh->locs[SHADER_LOC_VERTEX_COLOR],
            4, GL_FLOAT, GL_FALSE, sizeof(StarVertex), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(sh->locs[SHADER_LOC_VERTEX_COLOR]);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    rlDrawRenderBatchActive();
    rlSetBlendMode(RL_BLEND_ADDITIVE);
    int timeLoc = GetShaderLocation(*sh, "currentTime");
    glUseProgram(sh->id);
        glUniform1f(timeLoc, (float)GetTime());
        Matrix mvp = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection());
        glUniformMatrix4fv(sh->locs[SHADER_LOC_MATRIX_MVP], 1, GL_FALSE, MatrixToFloat(mvp));
        glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, n);
        glBindVertexArray(0);
    glUseProgram(0);
    rlSetBlendMode(RL_BLEND_ALPHA);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

// ─── Shared background star field ─────────────────────────────────────────────
#define NUM_BG_STARS 800
typedef struct { float x, y, brightness, size_scale; } BgStar;
static BgStar bg_stars[NUM_BG_STARS];
static StarVertex bg_verts[NUM_BG_STARS];

static void init_bg_stars(void) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        bg_stars[i].x          = (float)(rand() % SCREEN_W);
        bg_stars[i].y          = (float)(rand() % SCREEN_H);
        bg_stars[i].brightness = 0.2f + (float)(rand() % 80) / 100.0f;
        // Most stars small, a handful large enough to show spikes
        float r = (float)rand() / (float)RAND_MAX;
        bg_stars[i].size_scale = (r < 0.85f)
            ? (0.2f + r * 0.4f) * 0.7f           // 85%: small  (0.14 – 0.38)
            : (0.8f + (r - 0.85f) * 6.0f) * 0.7f; // 15%: larger (0.56 – 1.19)
    }
}

static void build_bg_verts(void) {
    for (int i = 0; i < NUM_BG_STARS; i++) {
        float br = bg_stars[i].brightness;
        bg_verts[i] = (StarVertex){
            .x = bg_stars[i].x, .y = bg_stars[i].y,
            .size_scale = bg_stars[i].size_scale,
            .r = C_WHITE.r / 255.0f * br,
            .g = C_WHITE.g / 255.0f * br,
            .b = C_WHITE.b / 255.0f * br,
            .a = (float)i,  // stable seed for twinkle
        };
    }
}

/* ============================================================
   ANIMATION 1: Speed Comparison (0–3s)
   Four rows, each a star bouncing at a rate proportional to
   the environment's simulation throughput (20M/4M/1M/30k sps).
   The 20M ball completes one full traverse in 1/1.333 s.
   ============================================================ */

#define NUM_BALLS    4
#define LANE_H       195         // 780px / 4 lanes
#define A1_MARGIN_L  160
#define A1_MARGIN_R  160
#define TRACK_W      (SCREEN_W - A1_MARGIN_L - A1_MARGIN_R)
#define TRACK_Y0     298         // 200 top + 780/8 = first lane center

#define SPEED_20M (2.0f * TRACK_W / 0.75f)
static const float SPEEDS[NUM_BALLS] = {
    SPEED_20M,
    SPEED_20M * (4.0f  / 20.0f),
    SPEED_20M * (1.0f  / 20.0f),
    SPEED_20M * (0.03f / 20.0f),
};
static const char *BALL_LABELS[NUM_BALLS] = {"20M", "4M", "1M", "30k"};

#define BALL_STAR_SCALE 5.28f

static float ball_x[NUM_BALLS];
static float ball_dir[NUM_BALLS];
static StarVertex ball_verts[NUM_BALLS];

static void init_anim1(void) {
    for (int i = 0; i < NUM_BALLS; i++) {
        ball_x[i]   = A1_MARGIN_L;
        ball_dir[i] = 1.0f;
    }
}

static void update_anim1(float dt, Font roboto, Shader *sh) {
    for (int i = 0; i < NUM_BALLS; i++) {
        ball_x[i] += SPEEDS[i] * ball_dir[i] * dt;
        if (ball_x[i] >= A1_MARGIN_L + TRACK_W) {
            ball_x[i] = A1_MARGIN_L + TRACK_W; ball_dir[i] = -1.0f;
        } else if (ball_x[i] <= A1_MARGIN_L) {
            ball_x[i] = A1_MARGIN_L; ball_dir[i] = 1.0f;
        }
        float lane_y = TRACK_Y0 + i * LANE_H;
        ball_verts[i] = (StarVertex){
            .x = ball_x[i], .y = lane_y,
            .size_scale = BALL_STAR_SCALE,
            .r = C_CYAN.r / 255.0f, .g = C_CYAN.g / 255.0f,
            .b = C_CYAN.b / 255.0f, .a = (float)(NUM_BG_STARS + i),  // stable seed
        };
    }

    build_bg_verts();
    draw_stars(bg_verts, NUM_BG_STARS, sh);
    draw_stars(ball_verts, NUM_BALLS, sh);

    for (int i = 0; i < NUM_BALLS; i++) {
        int lane_y = TRACK_Y0 + i * LANE_H;
        DrawLine(A1_MARGIN_L, lane_y, A1_MARGIN_L + TRACK_W, lane_y,
            (Color){0, 187, 187, 20});
        DrawTextEx(roboto, BALL_LABELS[i], (Vector2){100, lane_y - FONT_LABEL/2}, FONT_LABEL, 1, C_CYAN);
    }
    DrawTextEx(roboto, "Steps Per Second", (Vector2){SCREEN_W/2 - 280, 80}, FONT_TITLE, 1, C_WHITE);
}

/* ============================================================
   ANIMATION 2: Horizontal Bar Chart – Solve Times (3–6s)
   Bars are starry regions. Each bar lerps from a large start
   width down to a smaller end width (showing speedup).
   Stars are pre-scattered across the start width; the right
   edge shrinks by culling stars beyond the current width.
   Phase:  0–1s static at start widths
           1–2s lerp start → end
           2–3s static at end widths
   ============================================================ */

#define NUM_ENVS      5
#define CHART_LEFT    440
#define CHART_MAX_W   (SCREEN_W/2 - CHART_LEFT - 40)
#define CHART_TOP     238        // 200 top + 156/2 - BAR_H/2; first bar center at 278
#define BAR_H         80
#define BAR_GAP       76        // 780px / 5 bars = 156 per slot; 156 - 80 = 76

#define BAR_STAR_SCALE  0.20f
#define STAR_DENSITY    0.05f   // stars per square pixel
#define MAX_BAR_STARS   30000   // upper bound for static allocation

typedef struct {
    const char *name;
    float start_val;
    float end_val;
    float start_px;   // set by init
    float end_ratio;  // end_val / start_val, set by init
    int   star_off;   // offset into flat arrays, set by init
    int   star_count; // set by init
} EnvBar;

static EnvBar envs[NUM_ENVS] = {
    {"Environment A", 27.0f,  3.0f,  0, 0, 0, 0},
    {"Environment B",  3.0f,  0.24f, 0, 0, 0, 0},
    {"Environment C",  1.0f,  0.14f, 0, 0, 0, 0},
    {"Environment D",  0.5f,  0.08f, 0, 0, 0, 0},
    {"Environment E",  0.2f,  0.04f, 0, 0, 0, 0},
};

static float bar_nx[MAX_BAR_STARS];
static float bar_sy[MAX_BAR_STARS];
static StarVertex bar_verts[MAX_BAR_STARS];

static void init_anim2(void) {
    float max_val = 0.0f;
    for (int e = 0; e < NUM_ENVS; e++)
        if (envs[e].start_val > max_val) max_val = envs[e].start_val;

    int offset = 0;
    for (int e = 0; e < NUM_ENVS; e++) {
        envs[e].start_px  = (envs[e].start_val / max_val) * CHART_MAX_W;
        envs[e].end_ratio = envs[e].end_val / envs[e].start_val;
        envs[e].star_count = (int)(STAR_DENSITY * envs[e].start_px * BAR_H);
        envs[e].star_off   = offset;
        float cy = CHART_TOP + e * (BAR_H + BAR_GAP) + BAR_H * 0.5f;
        for (int i = 0; i < envs[e].star_count; i++) {
            bar_nx[offset + i] = (float)rand() / (float)RAND_MAX;
            float fy = ((float)rand() / (float)RAND_MAX) * BAR_H - BAR_H * 0.5f;
            bar_sy[offset + i] = cy + fy;
        }
        offset += envs[e].star_count;
    }
}

static void draw_anim2(float anim_t, Font roboto, Shader *sh) {
    float lerp_t;
    if      (anim_t < 1.0f) lerp_t = 0.0f;
    else if (anim_t < 2.0f) lerp_t = anim_t - 1.0f;
    else                    lerp_t = 1.0f;

    int n = 0;
    for (int e = 0; e < NUM_ENVS; e++) {
        float cull = 1.0f - lerp_t * (1.0f - envs[e].end_ratio);
        int off = envs[e].star_off;
        for (int i = 0; i < envs[e].star_count; i++) {
            if (bar_nx[off + i] > cull) continue;
            bar_verts[n++] = (StarVertex){
                .x = CHART_LEFT + bar_nx[off + i] * envs[e].start_px,
                .y = bar_sy[off + i],
                .size_scale = BAR_STAR_SCALE,
                .r = C_CYAN.r / 255.0f, .g = C_CYAN.g / 255.0f,
                .b = C_CYAN.b / 255.0f, .a = (float)(off + i),  // stable seed
            };
        }
    }

    build_bg_verts();
    draw_stars(bg_verts, NUM_BG_STARS, sh);
    draw_stars(bar_verts, n, sh);

    for (int e = 0; e < NUM_ENVS; e++) {
        float cy = CHART_TOP + e * (BAR_H + BAR_GAP) + BAR_H * 0.5f;
        DrawTextEx(roboto, envs[e].name, (Vector2){100, cy - FONT_LABEL/2}, FONT_LABEL, 1, C_WHITE);
    }
    DrawTextEx(roboto, "Solve Time", (Vector2){SCREEN_W/4 - 120, 80}, FONT_TITLE, 1, C_WHITE);
}

/* ============================================================
   ANIMATION 3: PufferNet logo with pulsing glow (6–9s)
   Image is wide black outlines on transparent BG. A Sobel
   edge + blurred halo is driven by cos(time) for the pulse.
   ============================================================ */

static void draw_anim3(float anim_t, Shader *glow_sh, Texture2D tex, Shader *star_sh) {
    float glow = 0.65f + 0.15f * cosf(anim_t * 2.5f);  // 0..1 drives the oscillating portion

    float fade = fminf(anim_t / 0.5f, 1.0f);

    int glow_loc     = GetShaderLocation(*glow_sh, "glowStrength");
    int texel_loc    = GetShaderLocation(*glow_sh, "texelSize");
    int fade_loc     = GetShaderLocation(*glow_sh, "fadeAlpha");
    float texel[2]   = {1.0f / tex.width, 1.0f / tex.height};

    SetShaderValue(*glow_sh, glow_loc,  &glow,  SHADER_UNIFORM_FLOAT);
    SetShaderValue(*glow_sh, texel_loc, texel,  SHADER_UNIFORM_VEC2);
    SetShaderValue(*glow_sh, fade_loc,  &fade,  SHADER_UNIFORM_FLOAT);

    // Center image in the right half of the screen
    float x = SCREEN_W/2 + (SCREEN_W/2 - tex.width)  * 0.5f;
    float y = (SCREEN_H - tex.height) * 0.5f;

    BeginShaderMode(*glow_sh);
        DrawTexture(tex, (int)x, (int)y, WHITE);
    EndShaderMode();
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(void) {
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(SCREEN_W, SCREEN_H, "PufferLib Trailer");
    SetTargetFPS(60);
    glEnable(GL_PROGRAM_POINT_SIZE);

    Shader star_shader = LoadShader(
        TextFormat("resources/trailer/star_%i.vs", GLSL_VERSION),
        TextFormat("resources/trailer/star_%i.fs", GLSL_VERSION)
    );
    Shader glow_shader = LoadShader(
        TextFormat("resources/trailer/glow_%i.vs", GLSL_VERSION),
        TextFormat("resources/trailer/glow_%i.fs", GLSL_VERSION)
    );
    Texture2D puffernet = LoadTexture("resources/trailer/PufferNet.png");

    Font roboto = LoadFontEx("resources/shared/Roboto-Regular.ttf", FONT_TITLE, NULL, 0);
    SetTextureFilter(roboto.texture, TEXTURE_FILTER_BILINEAR);

    init_bg_stars();
    init_anim1();
    init_anim2();

    while (!WindowShouldClose()) {
        float t  = GetTime();
        float dt = GetFrameTime();

        BeginDrawing();
        ClearBackground(BG);

        /* ============================================================
           ANIMATION 1: Speed Comparison (0–3s)
           ============================================================ */
        if (t < 3.0f) {
            update_anim1(dt, roboto, &star_shader);
        }

        /* ============================================================
           ANIMATION 2: Bar chart (0–3s), then held final state (3–6s)
           ANIMATION 3: PufferNet glow on right half, overlaps 3–6s
           ============================================================ */
        else if (t < 9.0f) {
            float anim2_t = (t < 6.0f) ? t : 6.0f;
            draw_anim2(anim2_t - 3.0f, roboto, &star_shader);

            if (t >= 6.0f) {
                draw_anim3(t - 6.0f, &glow_shader, puffernet, &star_shader);
            }
        }

        else {
            break;
        }


        DrawFPS(SCREEN_W - 80, 8);
        EndDrawing();
    }

    UnloadFont(roboto);
    UnloadShader(star_shader);
    UnloadShader(glow_shader);
    UnloadTexture(puffernet);
    CloseWindow();
    return 0;
}
