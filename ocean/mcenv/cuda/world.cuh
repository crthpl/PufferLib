#ifndef MCENV_WORLD_CUH
#define MCENV_WORLD_CUH

#include "mcenv_cuda.cuh"

// ---------------------------------------------------------------------------
// Dense block grid helpers
//
// Grid layout: blocks[x * WORLD_Y * WORLD_Z + y * WORLD_Z + z]
// World coordinates are offset: grid(x,y,z) = world(x - off_x, y - off_y, z - off_z)
// ---------------------------------------------------------------------------

struct WorldDims {
    int size_x, size_y, size_z;
    int off_x,  off_y,  off_z;
};

__host__ __device__ inline int block_idx(const WorldDims* w, int x, int y, int z) {
    int gx = x + w->off_x;
    int gy = y + w->off_y;
    int gz = z + w->off_z;
    return gx * (w->size_y * w->size_z) + gy * w->size_z + gz;
}

__host__ __device__ inline bool in_bounds(const WorldDims* w, int x, int y, int z) {
    int gx = x + w->off_x;
    int gy = y + w->off_y;
    int gz = z + w->off_z;
    return gx >= 0 && gx < w->size_x
        && gy >= 0 && gy < w->size_y
        && gz >= 0 && gz < w->size_z;
}

__host__ __device__ inline bool is_solid(const uint8_t* grid, const WorldDims* w,
                                         int x, int y, int z) {
    if (!in_bounds(w, x, y, z)) return false;
    return grid[block_idx(w, x, y, z)] != 0;
}

__host__ __device__ inline void set_block(uint8_t* grid, const WorldDims* w,
                                          int x, int y, int z, uint8_t val) {
    if (in_bounds(w, x, y, z))
        grid[block_idx(w, x, y, z)] = val;
}

__host__ __device__ inline uint8_t get_block(const uint8_t* grid, const WorldDims* w,
                                             int x, int y, int z) {
    if (!in_bounds(w, x, y, z)) return 0;
    return grid[block_idx(w, x, y, z)];
}

// ---------------------------------------------------------------------------
// Block properties (all blocks are Air=0 or FullCube=1)
// Slipperiness, jump/velocity multiplier are constants for our block types.
// ---------------------------------------------------------------------------

__host__ __device__ inline float block_slipperiness(uint8_t /*block*/) {
    return 0.6f;   // DEFAULT_SLIPPERINESS for both Air and FullCube
}

__host__ __device__ inline float block_jump_multiplier(uint8_t /*block*/) {
    return 1.0f;
}

__host__ __device__ inline float block_velocity_multiplier(uint8_t /*block*/) {
    return 1.0f;
}

// ---------------------------------------------------------------------------
// Collision box gathering
// Fills `out` with AABBs for all solid blocks overlapping `region`.
// Returns number of colliders written (capped at max_colliders).
// ---------------------------------------------------------------------------

__host__ __device__ inline int gather_collision_boxes(
    const uint8_t* grid, const WorldDims* w,
    Aabb region, Aabb* out, int max_colliders)
{
    int min_x = (int)floor(region.min.x);
    int max_x = (int)ceil(region.max.x);
    int min_y = (int)floor(region.min.y);
    int max_y = (int)ceil(region.max.y);
    int min_z = (int)floor(region.min.z);
    int max_z = (int)ceil(region.max.z);

    int count = 0;
    for (int bx = min_x; bx < max_x; bx++) {
        for (int by = min_y; by < max_y; by++) {
            for (int bz = min_z; bz < max_z; bz++) {
                if (is_solid(grid, w, bx, by, bz) && count < max_colliders) {
                    out[count++] = aabb_new(
                        vec3_new((double)bx, (double)by, (double)bz),
                        vec3_new((double)bx + 1.0, (double)by + 1.0, (double)bz + 1.0)
                    );
                }
            }
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// does_not_collide — true if no solid block overlaps region
// ---------------------------------------------------------------------------

__host__ __device__ inline bool does_not_collide(
    const uint8_t* grid, const WorldDims* w, Aabb region)
{
    int min_x = (int)floor(region.min.x);
    int max_x = (int)ceil(region.max.x);
    int min_y = (int)floor(region.min.y);
    int max_y = (int)ceil(region.max.y);
    int min_z = (int)floor(region.min.z);
    int max_z = (int)ceil(region.max.z);

    for (int bx = min_x; bx < max_x; bx++)
        for (int by = min_y; by < max_y; by++)
            for (int bz = min_z; bz < max_z; bz++)
                if (is_solid(grid, w, bx, by, bz))
                    return false;
    return true;
}

// ---------------------------------------------------------------------------
// World query helpers matching Rust World methods
// ---------------------------------------------------------------------------

__host__ __device__ inline float slipperiness_at(
    const uint8_t* grid, const WorldDims* w, BlockPos pos)
{
    return block_slipperiness(get_block(grid, w, pos.x, pos.y, pos.z));
}

__host__ __device__ inline float jump_multiplier_at(
    const uint8_t* grid, const WorldDims* w, BlockPos pos)
{
    return block_jump_multiplier(get_block(grid, w, pos.x, pos.y, pos.z));
}

__host__ __device__ inline float velocity_multiplier_at(
    const uint8_t* grid, const WorldDims* w, BlockPos pos)
{
    return block_velocity_multiplier(get_block(grid, w, pos.x, pos.y, pos.z));
}

// ---------------------------------------------------------------------------
// place_block: place a solid block at pos if currently air. Returns true on success.
// ---------------------------------------------------------------------------

__host__ __device__ inline bool place_block(
    uint8_t* grid, const WorldDims* w, BlockPos pos)
{
    if (is_solid(grid, w, pos.x, pos.y, pos.z))
        return false;
    set_block(grid, w, pos.x, pos.y, pos.z, 1);
    return true;
}

#endif // MCENV_WORLD_CUH
