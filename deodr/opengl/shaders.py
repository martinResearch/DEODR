"""OpenGL sharders used to render scenes with camera distortion."""

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
        vec4 p_camera = extrinsic* vec4(v_vert, 1.0);
        vec2 projected = p_camera.xy/p_camera.z;
        float x = projected.x;
        float y = projected.y;
        float r2 = x * x + y * y;
        float radial_distortion = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 *r2;
        float tangential_distortion_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        float tangential_distortion_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        float distorted_x = x * radial_distortion + tangential_distortion_x;
        float distorted_y = y * radial_distortion + tangential_distortion_y;
        p_camera.xy = vec2(distorted_x, distorted_y)*p_camera.z;
        gl_Position=intrinsic*p_camera;
}
"""

fragment_shader_rgb_source = """
#version 130
uniform sampler2D Texture;
uniform vec4 Color;
uniform vec3 light_directional;
uniform float light_ambient;
in vec3 v_vert;
in vec3 v_norm;
in vec3 v_text;

out vec4 f_color;

void main() {
        float lum = light_ambient + max(dot(normalize(v_norm),- light_directional),0.0);
        vec3 color = texture(Texture, v_text.xy).rgb;
        color = color * (1.0 - Color.a) + Color.rgb * Color.a;
        f_color = vec4(color * lum, 1.0);
}"""
