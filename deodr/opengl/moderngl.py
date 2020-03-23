import moderngl
from pyrr import Matrix44
import numpy as np
from . import shaders as opengl_shaders


def opencv_to_opengl_perspective(camera, znear, zfar):
    # https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    fx = camera.intrinsic[0, 0]
    fy = camera.intrinsic[1, 1]
    cx = camera.intrinsic[0, 2]
    cy = camera.intrinsic[1, 2]
    cx2 = cx + 0.5  # half a pixel offset to be consistent with deodr convention
    cy2 = cy + 0.5  # half a pixel offset to be consistent with deodr convention
    width = camera.width
    height = camera.height
    np.testing.assert_array_equal(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], camera.intrinsic
    )
    m = np.array(
        [
            [2.0 * fx / width, 0, 0, 0],
            [0, -2.0 * fy / height, 0, 0],
            [
                1.0 - 2.0 * cx2 / width,
                1.0 - 2.0 * cy2 / height,
                (zfar + znear) / (znear - zfar),
                -1,
            ],
            [0, 0, 2.0 * zfar * znear / (znear - zfar), 0.0],
        ]
    )
    return m


class OffscreenRenderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()

        # Shaders
        self.shader_program = self.ctx.program(
            vertex_shader=opengl_shaders.vertex_shader_source,
            fragment_shader=opengl_shaders.fragment_shader_rgb_source,
        )
        self.fbo = None
        self.texture = None

    def render(self, deodr_scene, camera):
        ctx = self.ctx
        shader_program = self.shader_program
        bg_color = deodr_scene.background[0, 0]
        if False and not (np.all(deodr_scene.background == bg_color[None, None, :])):
            raise (
                BaseException(
                    "does not support background image yet, please provide a backround\
                     image that correspond to a uniform color"
                )
            )
        # Context creation

        # Setting up camera

        extrinsic = np.row_stack((camera.extrinsic, [0, 0, 0, 1]))

        zfar = 1000
        znear = 0.1
        intrinsic = Matrix44(
            np.diag([1, -1, -1, 1]).dot(
                opencv_to_opengl_perspective(camera, znear, zfar)
            )
        )
        intrinsic

        #
        shader_program["light_directional"].value = tuple(deodr_scene.light_directional)
        shader_program["ligth_ambient"].value = deodr_scene.light_ambient
        shader_program["intrinsic"].write(intrinsic.astype("f4").tobytes())
        shader_program["extrinsic"].write(extrinsic.T.astype("f4").tobytes())
        if camera.distortion is None:
            k1, k2, p1, p2, k3 = (0, 0, 0, 0, 0)
        else:
            k1, k2, p1, p2, k3, = camera.distortion
        shader_program["k1"].value = k1
        shader_program["k2"].value = k2
        shader_program["k3"].value = k3
        shader_program["p1"].value = p1
        shader_program["p2"].value = p2

        # Texture
        assert not deodr_scene.mesh.texture.flags["WRITEABLE"]
        texture_id = id(deodr_scene.mesh.texture)
        if self.texture is None or self.texture_id != texture_id:
            self.texture_id = id(deodr_scene.mesh.texture)
            self.texture = ctx.texture(
                (deodr_scene.mesh.texture.shape[0], deodr_scene.mesh.texture.shape[1]),
                deodr_scene.mesh.texture.shape[2],
                (deodr_scene.mesh.texture * 255).astype(np.uint8).tobytes(),
            )
        # texture.build_mipmaps()

        # compputing the box around the displaced mesh to get maximum accuracy
        # of the xyz point cloud using unit8 opengl type

        # creat triangles soup
        mesh = deodr_scene.mesh
        vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
        normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
        uv = mesh.uv[mesh.faces_uv].reshape(-1, 2)
        moderngl_uv = np.column_stack(
            ((uv[:, 0]) / mesh.texture.shape[0], (uv[:, 1] / mesh.texture.shape[1]))
        )

        vbo_vert = ctx.buffer(vertices.astype("f4").tobytes())
        vbo_norm = ctx.buffer(normals.astype("f4").tobytes())
        vbo_uv = ctx.buffer(moderngl_uv.astype("f4").tobytes())

        vao = ctx.vertex_array(
            shader_program,
            [
                (vbo_vert, "3f", "in_vert"),
                (vbo_norm, "3f", "in_norm"),
                (vbo_uv, "2f", "in_text"),
            ],
        )

        # Framebuffers
        if self.fbo is None:
            self.fbo = ctx.framebuffer(
                ctx.renderbuffer((camera.width, camera.height)),
                ctx.depth_renderbuffer((camera.width, camera.height)),
            )

        # Rendering the RGB image
        self.fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(bg_color[0], bg_color[1], bg_color[2])
        self.texture.use()
        vao.render()
        data = self.fbo.read(components=3, alignment=1)
        array_rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            camera.height, camera.width, 3
        )

        return array_rgb
