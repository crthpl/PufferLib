#version 100
precision mediump float;
varying vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord - vec2(0.5);
    float dist = length(uv);
    float falloff = exp(-20.0 * dist * dist);
    if (falloff < 0.01) discard;
    vec3 color = fragColor.rgb * falloff * 5.0;
    gl_FragColor = vec4(color, falloff);
}
