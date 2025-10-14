#version 330

in vec4 fragColor;
out vec4 finalColor;

void main()
{
vec2 uv = gl_PointCoord - vec2(0.5); // center to edge
float dist = length(uv); // distance from center (0.0–0.707)

// Optional: discard hard edge for exact circular point
if (dist > 0.5)
    discard;

// Smooth exponential falloff
float glow = exp(-24.0 * dist * dist); // steeper falloff = tighter glow

// Bright, saturated core color
vec3 color = fragColor.rgb * glow * 5.0;

// Only output color if visible — remove black halo
if (glow < 0.01)
    discard;

finalColor = vec4(color, 1.0);

}
