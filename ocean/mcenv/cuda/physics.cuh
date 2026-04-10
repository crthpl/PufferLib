#ifndef MCENV_PHYSICS_CUH
#define MCENV_PHYSICS_CUH

#include "world.cuh"
#include "raycast.cuh"

// ---------------------------------------------------------------------------
// movement_input_to_velocity — mirrors Rust physics::movement_input_to_velocity
// ---------------------------------------------------------------------------

__host__ __device__ inline Vec3 movement_input_to_velocity(
    Vec3 movement_input, double speed, float yaw, const float* sine_table)
{
    double len_sq = vec3_length_squared(movement_input);
    if (len_sq < 1.0e-7)
        return vec3_zero();

    Vec3 base;
    if (len_sq > 1.0)
        base = vec3_scale(vec3_normalize_or_zero(movement_input), speed);
    else
        base = vec3_scale(movement_input, speed);

    float yaw_rad = yaw * (3.14159265358979323846f / 180.0f);
    double sin_yaw = (double)MathHelper_sin(yaw_rad, sine_table);
    double cos_yaw = (double)MathHelper_cos(yaw_rad, sine_table);

    double x = base.x * cos_yaw - base.z * sin_yaw;
    double z = base.z * cos_yaw + base.x * sin_yaw;

    // ULP nudge toward zero — mirrors JVM float rounding during rotation
    if (sin_yaw != 0.0 && cos_yaw != 0.0) {
        x = ulp_nudge_toward_zero(x);
        z = ulp_nudge_toward_zero(z);
    }

    return vec3_new(x, base.y, z);
}

// ---------------------------------------------------------------------------
// Collision axis clamping — mirrors Player::calculate_max_offset_{x,y,z}
// ---------------------------------------------------------------------------

__host__ __device__ inline double calculate_max_offset_x(
    Aabb bb, double dx, const Aabb* colliders, int n)
{
    double result = dx;
    for (int i = 0; i < n; i++) {
        Aabb col = colliders[i];
        if (col.max.y <= bb.min.y || col.min.y >= bb.max.y ||
            col.max.z <= bb.min.z || col.min.z >= bb.max.z)
            continue;
        if (dx > 0.0) {
            double offset = col.min.x - bb.max.x;
            if (offset < result && offset > -COLLISION_EPS)
                result = (offset > 0.0) ? offset : 0.0;
        } else if (dx < 0.0) {
            double offset = col.max.x - bb.min.x;
            if (offset > result && offset < COLLISION_EPS)
                result = (offset < 0.0) ? offset : 0.0;
        }
    }
    return result;
}

__host__ __device__ inline double calculate_max_offset_y(
    Aabb bb, double dy, const Aabb* colliders, int n)
{
    double result = dy;
    for (int i = 0; i < n; i++) {
        Aabb col = colliders[i];
        if (col.max.x <= bb.min.x || col.min.x >= bb.max.x ||
            col.max.z <= bb.min.z || col.min.z >= bb.max.z)
            continue;
        if (dy > 0.0) {
            double offset = col.min.y - bb.max.y;
            if (offset < result && offset > -COLLISION_EPS)
                result = (offset > 0.0) ? offset : 0.0;
        } else if (dy < 0.0) {
            double offset = col.max.y - bb.min.y;
            if (offset > result && offset < COLLISION_EPS)
                result = (offset < 0.0) ? offset : 0.0;
        }
    }
    return result;
}

__host__ __device__ inline double calculate_max_offset_z(
    Aabb bb, double dz, const Aabb* colliders, int n)
{
    double result = dz;
    for (int i = 0; i < n; i++) {
        Aabb col = colliders[i];
        if (col.max.x <= bb.min.x || col.min.x >= bb.max.x ||
            col.max.y <= bb.min.y || col.min.y >= bb.max.y)
            continue;
        if (dz > 0.0) {
            double offset = col.min.z - bb.max.z;
            if (offset < result && offset > -COLLISION_EPS)
                result = (offset > 0.0) ? offset : 0.0;
        } else if (dz < 0.0) {
            double offset = col.max.z - bb.min.z;
            if (offset > result && offset < COLLISION_EPS)
                result = (offset < 0.0) ? offset : 0.0;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// adjust_axes — mirrors Player::adjust_axes
// ---------------------------------------------------------------------------

__host__ __device__ inline Vec3 adjust_axes(
    Vec3 movement, Aabb bb, const Aabb* colliders, int n)
{
    double dx = movement.x;
    double dy = movement.y;
    double dz = movement.z;

    if (dy != 0.0) {
        dy = calculate_max_offset_y(bb, dy, colliders, n);
        if (dy != 0.0)
            bb = aabb_offset(bb, vec3_new(0.0, dy, 0.0));
    }

    bool prioritize_z = fabs(dx) < fabs(dz);
    if (prioritize_z && dz != 0.0) {
        dz = calculate_max_offset_z(bb, dz, colliders, n);
        if (dz != 0.0)
            bb = aabb_offset(bb, vec3_new(0.0, 0.0, dz));
    }

    if (dx != 0.0) {
        dx = calculate_max_offset_x(bb, dx, colliders, n);
        if (!prioritize_z && dx != 0.0)
            bb = aabb_offset(bb, vec3_new(dx, 0.0, 0.0));
    }

    if (!prioritize_z && dz != 0.0) {
        dz = calculate_max_offset_z(bb, dz, colliders, n);
    }

    return vec3_new(dx, dy, dz);
}

// ---------------------------------------------------------------------------
// adjust_movement_for_collisions — mirrors Player::adjust_movement_for_collisions
// ---------------------------------------------------------------------------

__host__ __device__ inline Vec3 adjust_movement_for_collisions(
    McEnvState* s, Vec3 movement,
    const uint8_t* grid, const WorldDims* w)
{
    Aabb bb = state_bb(s);
    if (aabb_height(bb) <= 0.0 || aabb_width(bb) <= 0.0)
        return movement;

    Aabb region = aabb_expand(aabb_stretch(bb, movement), COLLISION_EPS);
    Aabb colliders[MAX_COLLIDERS];
    int n = gather_collision_boxes(grid, w, region, colliders, MAX_COLLIDERS);

    Vec3 base = adjust_axes(movement, bb, colliders, n);

    bool horiz_blocked = (movement.x != base.x) || (movement.z != base.z);
    bool vert_blocked  = (movement.y != base.y);
    bool on_ground_or_blocked = s->on_ground || (vert_blocked && movement.y < 0.0);

    double sh = s->step_height;
    if (sh < STEP_HEIGHT) sh = STEP_HEIGHT; // min
    if (sh > STEP_HEIGHT) sh = STEP_HEIGHT;

    if (sh > 0.0 && on_ground_or_blocked && (horiz_blocked || vert_blocked)) {
        Vec3 step_up = adjust_axes(
            vec3_new(movement.x, sh, movement.z),
            bb, colliders, n);

        Vec3 step_offset = adjust_axes(
            vec3_new(0.0, sh, 0.0),
            aabb_stretch(bb, vec3_new(movement.x, 0.0, movement.z)),
            colliders, n);

        Vec3 candidate = step_up;
        if (step_offset.y < sh) {
            Vec3 combo_move = adjust_axes(
                vec3_new(movement.x, 0.0, movement.z),
                aabb_offset(bb, step_offset),
                colliders, n);
            Vec3 combo = vec3_add(combo_move, step_offset);
            if (vec3_horizontal_length_squared(combo) >
                vec3_horizontal_length_squared(candidate))
                candidate = combo;
        }

        if (vec3_horizontal_length_squared(candidate) >
            vec3_horizontal_length_squared(base)) {
            Vec3 down = adjust_axes(
                vec3_new(0.0, movement.y - candidate.y, 0.0),
                aabb_offset(bb, candidate),
                colliders, n);
            return vec3_add(candidate, down);
        }
    }

    return base;
}

// ---------------------------------------------------------------------------
// adjust_movement_for_sneaking — mirrors Player::adjust_movement_for_sneaking
// ---------------------------------------------------------------------------

__host__ __device__ inline Vec3 adjust_movement_for_sneaking(
    McEnvState* s, Vec3 movement,
    const uint8_t* grid, const WorldDims* w)
{
    double dx = movement.x;
    double dz = movement.z;
    double step = 0.05;
    double sh = s->step_height;
    if (sh > STEP_HEIGHT) sh = STEP_HEIGHT;
    double drop = -sh;
    Aabb bb = state_bb(s);

    while (dx != 0.0 && does_not_collide(grid, w,
           aabb_offset(bb, vec3_new(dx, drop, 0.0)))) {
        if (fabs(dx) < step)
            dx = 0.0;
        else if (dx > 0.0)
            dx -= step;
        else
            dx += step;
    }

    while (dz != 0.0 && does_not_collide(grid, w,
           aabb_offset(bb, vec3_new(0.0, drop, dz)))) {
        if (fabs(dz) < step)
            dz = 0.0;
        else if (dz > 0.0)
            dz -= step;
        else
            dz += step;
    }

    while (dx != 0.0 && dz != 0.0 && does_not_collide(grid, w,
           aabb_offset(bb, vec3_new(dx, drop, dz)))) {
        if (fabs(dx) < step)
            dx = 0.0;
        else if (dx > 0.0)
            dx -= step;
        else
            dx += step;

        if (fabs(dz) < step)
            dz = 0.0;
        else if (dz > 0.0)
            dz -= step;
        else
            dz += step;
    }

    return vec3_new(dx, movement.y, dz);
}

// ---------------------------------------------------------------------------
// move_self — mirrors Player::move_self
// ---------------------------------------------------------------------------

__host__ __device__ inline void move_self(
    McEnvState* s, const uint8_t* grid, const WorldDims* w)
{
    Vec3 desired = state_vel(s);

    if (s->on_ground && s->sneaking) {
        desired = adjust_movement_for_sneaking(s, desired, grid, w);
    }

    Vec3 adjusted = adjust_movement_for_collisions(s, desired, grid, w);

    if (vec3_length_squared(adjusted) > 1.0e-7) {
        Aabb bb = aabb_offset(state_bb(s), adjusted);
        state_set_bb(s, bb);
        sync_to_bounding_box(s);
    }

    double eps = 2.2204460492503131e-16; // f64::EPSILON
    s->horizontal_collision =
        (fabs(desired.x - adjusted.x) > eps) || (fabs(desired.z - adjusted.z) > eps);
    s->vertical_collision = (fabs(desired.y - adjusted.y) > eps);
    s->on_ground = s->vertical_collision && (desired.y < 0.0);

    if (desired.x != adjusted.x) s->vel_x = 0.0;
    if (desired.z != adjusted.z) s->vel_z = 0.0;
    if (desired.y != adjusted.y) s->vel_y = 0.0;
}

// ---------------------------------------------------------------------------
// Sprint logic — mirrors Player::compute_sprinting
// ---------------------------------------------------------------------------

__host__ __device__ inline int compute_sprinting(
    McEnvState* s, const McEnvInput* input)
{
    if (s->sprint_toggle_timer > 0)
        s->sprint_toggle_timer--;

    int sprint = input->sprint || s->sprinting;
    int forward_pressed = (input->forward > 0.0f);
    int can_double_tap = s->on_ground && !input->sneak;

    if (!s->sprinting && can_double_tap && forward_pressed && !s->prev_forward_positive) {
        if (s->sprint_toggle_timer == 0)
            s->sprint_toggle_timer = 7;
        else
            sprint = 1;
    }

    if (sprint) {
        if (!forward_pressed || input->sneak)
            sprint = 0;
    }

    s->prev_forward_positive = forward_pressed;
    return sprint;
}

// ---------------------------------------------------------------------------
// velocity_affecting_pos — mirrors Player::velocity_affecting_pos
// ---------------------------------------------------------------------------

__host__ __device__ inline BlockPos velocity_affecting_pos(const McEnvState* s) {
    return blockpos_new(
        (int)floor(s->pos_x),
        (int)floor(s->bb_min_y - 0.5000001),
        (int)floor(s->pos_z)
    );
}

// ---------------------------------------------------------------------------
// Movement speed — mirrors Player::movement_speed / base_movement_speed / flying_speed
// ---------------------------------------------------------------------------

__host__ __device__ inline double base_movement_speed(const McEnvState* s) {
    double attr = (double)s->movement_speed_attr;
    double mult = s->sprinting ? (1.0 + 0.30000001192092896) : 1.0;
    return attr * mult;
}

__host__ __device__ inline double flying_speed(const McEnvState* s) {
    float base = 0.02f;
    if (s->sprinting) {
        float boosted = (float)((double)base + 0.005999999865889549);
        return (double)boosted;
    }
    return (double)base;
}

__host__ __device__ inline double movement_speed(
    const McEnvState* s, float slipperiness)
{
    if (s->on_ground) {
        float attr = (float)base_movement_speed(s);
        float slip = slipperiness;
        float speed = attr * (0.21600002f / (slip * slip * slip));
        return (double)speed;
    }
    return flying_speed(s);
}

// ---------------------------------------------------------------------------
// travel — mirrors Player::travel
// ---------------------------------------------------------------------------

__host__ __device__ inline void travel(
    McEnvState* s, Vec3 movement_input,
    const uint8_t* grid, const WorldDims* w, const float* sine_table)
{
    BlockPos vap = velocity_affecting_pos(s);
    float slip = slipperiness_at(grid, w, vap);

    // Friction: f32 multiply then widen
    double friction;
    if (s->on_ground)
        friction = (double)(slip * (float)FRICTION);
    else
        friction = (double)(float)FRICTION;

    double speed = movement_speed(s, slip);
    Vec3 delta = movement_input_to_velocity(movement_input, speed, s->yaw, sine_table);
    s->vel_x += delta.x;
    s->vel_y += delta.y;
    s->vel_z += delta.z;

    move_self(s, grid, w);

    // Post-collision: apply friction and gravity in one assignment
    // vel is the post-collision velocity from move_self
    double vx = s->vel_x * friction;
    double vy = (s->vel_y - GRAVITY) * (double)(float)DRAG;
    double vz = s->vel_z * friction;
    s->vel_x = vx;
    s->vel_y = vy;
    s->vel_z = vz;
}

// ---------------------------------------------------------------------------
// Block placement — mirrors Player::place_block + can_place_without_self_collision
// ---------------------------------------------------------------------------

__host__ __device__ inline bool can_place_without_self_collision(
    const McEnvState* s, BlockPos pos)
{
    // A full cube block at pos; check it doesn't intersect player bb
    Aabb block_bb = aabb_new(
        vec3_new((double)pos.x, (double)pos.y, (double)pos.z),
        vec3_new((double)pos.x + 1.0, (double)pos.y + 1.0, (double)pos.z + 1.0)
    );
    return !aabb_intersects(block_bb, state_bb(s));
}

__host__ __device__ inline bool do_place_block(
    McEnvState* s, uint8_t* grid, const WorldDims* w, const float* sine_table,
    BlockPos* out_pos)
{
    BlockPos solid, air;
    if (!player_raycast_first_solid(s, grid, w, sine_table, &solid, &air))
        return false;
    if (!can_place_without_self_collision(s, air))
        return false;
    if (place_block(grid, w, air)) {
        *out_pos = air;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Block breaking — mirrors Environment::break_target_block
// ---------------------------------------------------------------------------

__host__ __device__ inline void do_break_block(
    McEnvState* s, uint8_t* grid, const WorldDims* w, const float* sine_table)
{
    BlockPos solid, air;
    if (player_raycast_first_solid(s, grid, w, sine_table, &solid, &air)) {
        set_block(grid, w, solid.x, solid.y, solid.z, 0);
    }
}

// ---------------------------------------------------------------------------
// Full environment tick — mirrors Environment::input + Environment::tick
// ---------------------------------------------------------------------------

__host__ __device__ inline void mcenv_input(McEnvState* s, const McEnvInput* input) {
    if (input->place_repeat || (input->place && !s->prev_place_pressed))
        s->latched_place = 1;
    if (input->break_block && !s->prev_break_pressed)
        s->latched_break = 1;
    s->prev_place_pressed = input->place;
    s->prev_break_pressed = input->break_block;
}

__host__ __device__ inline void mcenv_tick(
    McEnvState* s, const McEnvInput* input,
    uint8_t* grid, const WorldDims* w, const float* sine_table)
{
    // --- Player::tick ---

    // 1. Look update
    if (input->has_look) {
        s->yaw = input->look_yaw;
        s->pitch = input->look_pitch;
    }

    // Store prev_pos
    s->prev_pos_x = s->pos_x;
    s->prev_pos_y = s->pos_y;
    s->prev_pos_z = s->pos_z;

    // 2. Compute sprint state
    s->sprinting = compute_sprinting(s, input);
    s->sneaking = input->sneak;

    // 3. Zero small velocities
    if (fabs(s->vel_x) < TINY_VEL_THRESH) s->vel_x = 0.0;
    if (fabs(s->vel_y) < TINY_VEL_THRESH) s->vel_y = 0.0;
    if (fabs(s->vel_z) < TINY_VEL_THRESH) s->vel_z = 0.0;

    // 4. Grounded velocity fixup
    if (s->on_ground && s->vel_y == 0.0) {
        s->vel_y = -(double)((float)GRAVITY * (float)DRAG);
    }

    // 5. Jump
    if (input->jump && s->on_ground) {
        // get_jump_velocity_multiplier: always 1.0 for our block types
        BlockPos jpos = blockpos_from_vec3(state_pos(s));
        BlockPos jvap = velocity_affecting_pos(s);
        float jm1 = jump_multiplier_at(grid, w, jpos);
        float jm2 = jump_multiplier_at(grid, w, jvap);
        float jm = (fabsf(jm1 - 1.0f) < 1.0e-7f) ? jm2 : jm1;

        float base_jump = 0.42f * jm;
        s->vel_y = (double)base_jump;
        if (s->sprinting) {
            double boost = (double)0.2f;
            float yaw_rad = s->yaw * (3.14159265358979323846f / 180.0f);
            double yaw_sin = (double)MathHelper_sin(yaw_rad, sine_table);
            double yaw_cos = (double)MathHelper_cos(yaw_rad, sine_table);
            s->vel_x -= yaw_sin * boost;
            s->vel_z += yaw_cos * boost;
        }
    }

    // 6. Compute movement input
    double forward = (double)(float)(input->forward);
    double strafe  = (double)(float)(input->strafe);
    if (input->sneak) {
        forward *= 0.3;
        strafe  *= 0.3;
    }
    // Multiply in f32 to match vanilla rounding
    double mx = (double)((float)strafe  * 0.98f);
    double mz = (double)((float)forward * 0.98f);
    Vec3 movement_input = vec3_new(mx, 0.0, mz);

    // 7. Travel
    travel(s, movement_input, grid, w, sine_table);

    // 8. Velocity multiplier
    BlockPos vmpos = blockpos_from_vec3(state_pos(s));
    BlockPos vmvap = velocity_affecting_pos(s);
    float vm1 = velocity_multiplier_at(grid, w, vmpos);
    float vm2 = velocity_multiplier_at(grid, w, vmvap);
    float vm = (fabsf(vm1 - 1.0f) < 1.0e-7f) ? vm2 : vm1;
    s->vel_x *= (double)vm;
    s->vel_z *= (double)vm;

    // --- Environment::tick (break then place) ---

    if (s->latched_break) {
        do_break_block(s, grid, w, sine_table);
    }
    s->last_placed_valid = 0;
    if (s->latched_place) {
        BlockPos placed;
        if (do_place_block(s, grid, w, sine_table, &placed)) {
            s->last_placed_x = placed.x;
            s->last_placed_y = placed.y;
            s->last_placed_z = placed.z;
            s->last_placed_valid = 1;
        }
    }
    s->latched_place = 0;
    s->latched_break = 0;
}

#endif // MCENV_PHYSICS_CUH
