#ifndef MCENV_MATH_CUH
#define MCENV_MATH_CUH

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Vec3 (f64) — mirrors Rust math::Vec3
// ---------------------------------------------------------------------------

struct Vec3 {
    double x, y, z;
};

__host__ __device__ inline Vec3 vec3_new(double x, double y, double z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}

__host__ __device__ inline Vec3 vec3_zero() {
    return vec3_new(0.0, 0.0, 0.0);
}

__host__ __device__ inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return vec3_new(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return vec3_new(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vec3 vec3_scale(Vec3 v, double s) {
    return vec3_new(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline double vec3_length_squared(Vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ inline double vec3_horizontal_length_squared(Vec3 v) {
    return v.x * v.x + v.z * v.z;
}

__host__ __device__ inline double vec3_length(Vec3 v) {
    return sqrt(vec3_length_squared(v));
}

// Must use division (not multiply by reciprocal) to match Rust's Vec3 / f64.
__host__ __device__ inline Vec3 vec3_div(Vec3 v, double d) {
    return vec3_new(v.x / d, v.y / d, v.z / d);
}

__host__ __device__ inline Vec3 vec3_normalize_or_zero(Vec3 v) {
    double len = vec3_length(v);
    if (len > 1.0e-4)
        return vec3_div(v, len);
    return vec3_zero();
}

// ---------------------------------------------------------------------------
// MathHelper sin/cos — 65536-entry f32 lookup table (Minecraft-faithful)
// ---------------------------------------------------------------------------

__host__ __device__ inline float MathHelper_sin(float angle, const float* sine_table) {
    int idx = ((int)(angle * 10430.378f)) & 0xFFFF;
    return sine_table[idx];
}

__host__ __device__ inline float MathHelper_cos(float angle, const float* sine_table) {
    int idx = ((int)(angle * 10430.378f + 16384.0f)) & 0xFFFF;
    return sine_table[idx];
}

// ---------------------------------------------------------------------------
// lerp / fractional_part — mirrors Rust MathHelper
// ---------------------------------------------------------------------------

__host__ __device__ inline double mcenv_lerp(double delta, double start, double end) {
    return start + delta * (end - start);
}

__host__ __device__ inline double mcenv_fractional_part(double value) {
    return value - floor(value);
}

// ---------------------------------------------------------------------------
// ULP nudge — mirrors Rust f64::from_bits(to_bits().wrapping_sub/add(1))
// ---------------------------------------------------------------------------

__host__ __device__ inline double ulp_nudge_toward_zero(double v) {
    if (v == 0.0) return v;
    unsigned long long bits;
    memcpy(&bits, &v, sizeof(bits));
    if (v > 0.0)
        bits -= 1ULL;
    else
        bits += 1ULL;
    double result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// Host-side sine table generation
// ---------------------------------------------------------------------------

inline void generate_sine_table(float* table) {
    const double TWO_PI = 3.14159265358979323846 * 2.0;
    for (int i = 0; i < 65536; i++) {
        table[i] = (float)sin(((double)i * TWO_PI) / 65536.0);
    }
}

#endif // MCENV_MATH_CUH
