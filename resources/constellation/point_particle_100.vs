#version 100
attribute vec3 vertexPosition;
attribute vec4 vertexColor;
uniform mat4 mvp;
uniform float currentTime;
varying vec4 fragColor;

float twinkle(float idx, float t) {
    float base_size = 14.0;
    float phase = mod(idx * 137.5, 360.0);
    float frequency = 0.6 + mod(idx, 3.0) * 0.2;
    float amplitude = (mod(idx, 30.0) == 0.0) ? 8.0 : 1.0;
    float size_variation = amplitude * sin(frequency * t + radians(phase));
    return max(0.5, base_size + size_variation);
}

void main() {
    vec2 pos = vertexPosition.xy;
    float idx = vertexPosition.z;
    gl_Position = mvp * vec4(pos, 0.0, 1.0);
    gl_PointSize = twinkle(idx, currentTime);
    fragColor = vertexColor;
}
