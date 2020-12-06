"""Module to render deodr scenes using OpenGL through moderngl.

This is used to have a reference implementation using OpenGL shaders that produce identical images to deodr
"""

import moderngl

import numpy as np

from pyrr import Matrix44

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
    """Class to perform offscreen rendering of deodr scenes using moderngl."""

    def __init__(self):
        self.ctx = moderngl.create_standalone_context()

        # Shaders
        self.shader_program = self.ctx.program(
            vertex_shader=opengl_shaders.vertex_shader_source,
            fragment_shader=opengl_shaders.fragment_shader_rgb_source,
        )
        self.fbo = None
        self.texture = None

    def set_scene(self, deodr_scene):

        self.bg_color = deodr_scene.background[0, 0]
        if False and not (np.all(deodr_scene.background == self.bg_color[None, None, :])):
            raise (
                BaseException(
                    "does not support background image yet, please provide a backround\
                     image that correspond to a uniform color"
                )
            )

        self.shader_program["light_directional"].value = tuple(
            deodr_scene.light_directional
        )
        self.shader_program["light_ambient"].value = deodr_scene.light_ambient

        self.set_mesh(deodr_scene.mesh)

    def set_mesh(self, mesh):

        self.set_texture(mesh.texture)
        # create triangles soup

        vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
        min_max = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        self.bounding_box_corners = np.stack(
            np.meshgrid(min_max[:, 0], min_max[:, 1], min_max[:, 2]), axis=-1
        ).reshape(-1, 3)
        normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
        uv = mesh.uv[mesh.faces_uv].reshape(-1, 2)
        moderngl_uv = np.column_stack(
            (
                (uv[:, 0] + 0.5) / mesh.texture.shape[1],
                ((uv[:, 1] + 0.5) / mesh.texture.shape[0]),
            )
        )

        vbo_vert = self.ctx.buffer(vertices.astype("f4").tobytes())
        vbo_norm = self.ctx.buffer(normals.astype("f4").tobytes())
        vbo_uv = self.ctx.buffer(moderngl_uv.astype("f4").tobytes())

        self.vao = self.ctx.vertex_array(
            self.shader_program,
            [
                (vbo_vert, "3f", "in_vert"),
                (vbo_norm, "3f", "in_norm"),
                (vbo_uv, "2f", "in_text"),
            ],
        )

    def set_texture(self, texture):
        # Texture
        assert not texture.flags["WRITEABLE"]
        texture_id = id(texture)
        if self.texture is None or self.texture_id != texture_id:
            self.texture_id = id(texture)
            self.texture = self.ctx.texture(
                (texture.shape[1], texture.shape[0]),
                texture.shape[2],
                (texture * 255).astype(np.uint8).tobytes(),
            )
        # texture.build_mipmaps()

    def set_camera(self, camera):
        extrinsic = np.row_stack((camera.extrinsic, [0, 0, 0, 1]))

        intrinsic = Matrix44(
            np.diag([1, -1, -1, 1]).dot(
                opencv_to_opengl_perspective(camera, self.znear, self.zfar)
            )
        )

        #

        self.shader_program["intrinsic"].write(intrinsic.astype("f4").tobytes())
        self.shader_program["extrinsic"].write(extrinsic.T.astype("f4").tobytes())
        if camera.distortion is None:
            k1, k2, p1, p2, k3 = (0, 0, 0, 0, 0)
        else:
            k1, k2, p1, p2, k3, = camera.distortion
        self.shader_program["k1"].value = k1
        self.shader_program["k2"].value = k2
        self.shader_program["k3"].value = k3
        self.shader_program["p1"].value = p1
        self.shader_program["p2"].value = p2

    def render(self, camera):
        ctx = self.ctx
        self.zfar = camera.world_to_camera(self.bounding_box_corners)[:, 2].max()
        self.znear = 1e-3 * self.zfar
        # Setting up camera
        self.set_camera(camera)

        # compputing the box around the displaced mesh to get maximum accuracy
        # of the xyz point cloud using unit8 opengl type

        # Framebuffers
        if self.fbo is None:
            self.fbo = ctx.framebuffer(
                ctx.renderbuffer((camera.width, camera.height)),
                ctx.depth_renderbuffer((camera.width, camera.height)),
            )

        # Rendering the RGB image
        self.fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2])
        self.texture.use()
        self.vao.render()
        data = self.fbo.read(components=3, alignment=1)
        array_rgb = np.frombuffer(data, dtype=np.uint8).reshape(
            camera.height, camera.width, 3
        )

        return array_rgb
