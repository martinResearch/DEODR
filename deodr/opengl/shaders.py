vertex_shader_source = """
                        uniform mat4 Mvp;

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
                                gl_Position = Mvp * vec4(v_vert, 1.0);
                        }
                        """

fragment_shader_RGB_source = """uniform sampler2D Texture;
                                uniform vec4 Color;
                                uniform vec3 ligth_directional;
                                uniform float ligth_ambiant;
                                in vec3 v_vert;
                                in vec3 v_norm;
                                in vec3 v_text;

                                out vec4 f_color;

                                void main() {
                                        float lum = ligth_ambiant + max(dot(normalize(v_norm),- ligth_directional),0);
                                        vec3 color = texture(Texture, v_text.xy).rgb;
                                        color = color * (1.0 - Color.a) + Color.rgb * Color.a;
                                        f_color = vec4(color * lum, 1.0);
                                }"""
