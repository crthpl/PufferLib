#ifndef MCENV_RAYCAST_CUH
#define MCENV_RAYCAST_CUH

#include "world.cuh"

// ---------------------------------------------------------------------------
// Bresenham-style DDA raycast — mirrors Rust Player::raycast_first_solid_between
//
// Returns true if a solid block was found. Writes the solid block position
// into `out_solid` and the air block just before it into `out_air`.
// ---------------------------------------------------------------------------

__host__ __device__ inline bool raycast_first_solid_between(
    const uint8_t* grid, const WorldDims* w,
    Vec3 start, Vec3 end,
    BlockPos* out_solid, BlockPos* out_air)
{
    if (start.x == end.x && start.y == end.y && start.z == end.z)
        return false;

    double d = mcenv_lerp(-1.0e-7, end.x, start.x);
    double e = mcenv_lerp(-1.0e-7, end.y, start.y);
    double f = mcenv_lerp(-1.0e-7, end.z, start.z);
    double g = mcenv_lerp(-1.0e-7, start.x, end.x);
    double h = mcenv_lerp(-1.0e-7, start.y, end.y);
    double i = mcenv_lerp(-1.0e-7, start.z, end.z);

    int j = (int)floor(g);
    int k = (int)floor(h);
    int l = (int)floor(i);

    // Check starting cell
    bool has_last_air = false;
    BlockPos last_air;
    if (is_solid(grid, w, j, k, l)) {
        // Solid at start, no air predecessor
        return false;
    }
    has_last_air = true;
    last_air = blockpos_new(j, k, l);

    double m = d - g;
    double n = e - h;
    double o = f - i;

    // signum
    int p = (m > 0.0) ? 1 : ((m < 0.0) ? -1 : 0);
    int q = (n > 0.0) ? 1 : ((n < 0.0) ? -1 : 0);
    int r = (o > 0.0) ? 1 : ((o < 0.0) ? -1 : 0);

    double s = (p == 0) ? 1.0e300 : (double)p / m;  // INFINITY-like
    double t = (q == 0) ? 1.0e300 : (double)q / n;
    double u = (r == 0) ? 1.0e300 : (double)r / o;

    double v = s * ((p > 0) ? (1.0 - mcenv_fractional_part(g)) : mcenv_fractional_part(g));
    double ww = t * ((q > 0) ? (1.0 - mcenv_fractional_part(h)) : mcenv_fractional_part(h));
    double x = u * ((r > 0) ? (1.0 - mcenv_fractional_part(i)) : mcenv_fractional_part(i));

    while (v <= 1.0 || ww <= 1.0 || x <= 1.0) {
        if (v < ww) {
            if (v < x) {
                j += p;
                v += s;
            } else {
                l += r;
                x += u;
            }
        } else if (ww < x) {
            k += q;
            ww += t;
        } else {
            l += r;
            x += u;
        }

        if (is_solid(grid, w, j, k, l)) {
            if (has_last_air) {
                *out_solid = blockpos_new(j, k, l);
                *out_air = last_air;
                return true;
            }
            return false;
        }
        has_last_air = true;
        last_air = blockpos_new(j, k, l);
    }

    return false;
}

// ---------------------------------------------------------------------------
// Player raycast (eye → look direction * reach)
// ---------------------------------------------------------------------------

__host__ __device__ inline bool player_raycast_first_solid(
    const McEnvState* state, const uint8_t* grid, const WorldDims* w,
    const float* sine_table,
    BlockPos* out_solid, BlockPos* out_air)
{
    // Eye position
    double eye_height = state->sneaking ? 1.27 : 1.62;
    Vec3 eye = vec3_new(state->pos_x, state->bb_min_y + eye_height, state->pos_z);

    // Look direction
    float yaw_rad = state->yaw * (3.14159265358979323846f / 180.0f);
    float pitch_rad = state->pitch * (3.14159265358979323846f / 180.0f);
    double cos_pitch = (double)MathHelper_cos(pitch_rad, sine_table);
    double sin_pitch = (double)MathHelper_sin(pitch_rad, sine_table);
    double sin_yaw   = (double)MathHelper_sin(yaw_rad, sine_table);
    double cos_yaw   = (double)MathHelper_cos(yaw_rad, sine_table);
    Vec3 look = vec3_new(-cos_pitch * sin_yaw, -sin_pitch, cos_pitch * cos_yaw);

    Vec3 end = vec3_add(eye, vec3_scale(look, SURVIVAL_REACH));
    return raycast_first_solid_between(grid, w, eye, end, out_solid, out_air);
}

#endif // MCENV_RAYCAST_CUH
