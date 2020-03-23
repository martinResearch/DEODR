vertex_shader_source = """
#version 140
uniform mat4 intrinsic;
uniform mat4 extrinsic;
uniform float k1;
uniform float k2;
uniform float k3;
uniform float p1;
uniform float p2;

in vec3 in_vert;
in vec3 in_norm;
in vec3 in_text;

out vec3 v_vert;
out vec3 v_norm;
out vec3 v_text;

void main() {
        v_vert = in_vert;
        v_norm = in_norm;
        v_text = in_text;
        vec4 pcam = extrinsic* vec4(v_vert, 1.0);
        vec2 projected = pcam.xy/pcam.z;
        float x = projected.x;
        float y = projected.y;
        float r2 = x * x + y * y;
        float radial_distortion = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 *r2;
        float tangential_distortionx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        float tangential_distortiony = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        float distortedx = x * radial_distortion + tangential_distortionx;
        float distortedy = y * radial_distortion + tangential_distortiony;
        pcam.xy = vec2(distortedx, distortedy)*pcam.z;
        gl_Position=intrinsic*pcam;
}
"""

fragment_shader_rgb_source = """
#version 130
uniform sampler2D Texture;
uniform vec4 Color;
uniform vec3 light_directional;
uniform float ligth_ambient;
in vec3 v_vert;
in vec3 v_norm;
in vec3 v_text;

out vec4 f_color;

void main() {
        float lum = ligth_ambient + max(dot(normalize(v_norm),- light_directional),0.0);
        vec3 color = texture(Texture, v_text.xy).rgb;
        color = color * (1.0 - Color.a) + Color.rgb * Color.a;
        f_color = vec4(color * lum, 1.0);
}"""
