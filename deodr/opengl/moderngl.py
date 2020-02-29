import moderngl
from PIL import Image
from pyrr import Matrix44
import numpy as np
from . import shaders as OpenGLShaders


def opencv_to_opengl_perspective(intrinsic):
    pass


def render(deodr_scene, camera):
    bg_color = deodr_scene.background[0, 0]
    if not (np.all(deodr_scene.background == bg_color[None, None, :])):
        raise (
            BaseException(
                "does not support background image yet, please provide a backround image that correspond to a uniform color"
            )
        )
    # Context creation
    ctx = moderngl.create_standalone_context()

    # Shaders
    shader_program = ctx.program(
        vertex_shader=OpenGLShaders.vertex_shader_source,
        fragment_shader=OpenGLShaders.fragment_shader_RGB_source,
    )

    # Setting up camera

    focal_length = 100
    fov = 2 * np.arctan(camera.width / (focal_length * 2)) * 180 / np.pi
    perspective = Matrix44.perspective_projection(fov, 1.0, 0.1, 1000.0)

    # perspective = Matrix44(np.array(
    #     [
    #         [0.5958768, 0.0, 0.49921753, 0.0],
    #         [0.0, 0.7945024, -0.49895616, 0.0],
    #         [0.0, 0.0, -1.0010005, -0.10005003],
    #         [0.0, 0.0, -1.0, 0.0],
    #     ]).T
    # )
    modelview = Matrix44(
        np.row_stack((camera.extrinsic, [0, 0, 0, 1])).T.dot(np.diag([1, -1, -1, 1]))
    )

    mvp = perspective * modelview

    proj = Matrix44.perspective_projection(
        65.0, camera.width / camera.height, 0.1, 1000.0
    )
    # lookat = Matrix44.look_at(
    #     (0, 0, 2), deodr_scene.mesh.vertices.mean(axis=0), (0.0, 1.0, 0.0)
    # )
    intrinsic=camera.intrinsic
    zfar=1000
    znear=0.1
    proj2 =Matrix44 (np.array([  [2*intrinsic[0,0]/camera.width,0,0,0],[0, 2*intrinsic[0,0]/camera.height,0,0],[0,0, -1*(zfar+2*znear)/zfar,-1],[0,0,-2*znear,0]]))

    mvp = proj2 * modelview
    #
    shader_program["ligth_directional"].value = tuple(deodr_scene.ligth_directional)
    shader_program["ligth_ambiant"].value = deodr_scene.ambiant_light
    shader_program["Mvp"].write(mvp.astype("f4").tobytes())

    # Texture
    texture = ctx.texture(
        (deodr_scene.texture.shape[0], deodr_scene.texture.shape[1]),
        deodr_scene.texture.shape[2],
        (deodr_scene.texture * 255).astype(np.uint8).tobytes(),
    )
    texture.build_mipmaps()

    # compputing the box around the displaced mesh to get maximum accuracy of the xyz point cloud using unit8 opengl type

    # creat triangles soup
    mesh = deodr_scene.mesh
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
    uv = mesh.uv[mesh.faces_uv].reshape(-1, 2)
    moderngl_uv = np.column_stack(
        ((uv[:, 0]) / mesh.texture.shape[0], ( uv[:, 1] / mesh.texture.shape[1]))
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
    fbo = ctx.framebuffer(
        ctx.renderbuffer((camera.width, camera.height)),
        ctx.depth_renderbuffer((camera.width, camera.height)),
    )

    # Rendering the RGB image
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.clear(bg_color[0],bg_color[1],bg_color[2])
    texture.use()
    vao.render()
    data = fbo.read(components=3, alignment=1)
    img = Image.frombytes("RGB", fbo.size, data, "raw", "RGB", 0, -1)
    array_rgb = np.array(img)

    return array_rgb
