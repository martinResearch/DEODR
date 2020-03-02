import moderngl
from PIL import Image
from pyrr import Matrix44
import numpy as np
from . import shaders as OpenGLShaders
from hashlib import sha1


def opencv_to_opengl_perspective(camera, znear, zfar):
    # https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
    fx = camera.intrinsic[0, 0]
    fy = camera.intrinsic[1, 1]
    cx = camera.intrinsic[0, 2]
    cy = camera.intrinsic[1, 2]
    cx2 = cx + 0.5  # half a pixel offset to be consistent with deodr convention
    cy2 = cy - 0.5  # half a pixel offset to be consistent with deodr convention
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
                2.0 * cy2 / height - 1.0,
                (zfar + znear) / (znear - zfar),
                -1,
            ],
            [0, 0, 2.0 * zfar * znear / (znear - zfar), 0.0],
        ]
    )
    return m

    # viewport = np.array((0, 0, camera.width, camera.height))
    # M = np.array(
    #     [
    #         [0.5 * viewport[2], 0, 0, 0.5 * viewport[2] + viewport[0]],
    #         [0, 0.5 * viewport[3], 0, 0.5 * viewport[3] + viewport[1]],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # we want to find gl_projection such that camera.intrinsic == M * gl_projection
    # gl_projection*[0,0,0,znear,1]=[0,0,0,0]
    # gl_projection*[0,0,0,zfar,1]=[0,0,0,1]

    # M is not square , so we need some addition information on
    #  % GL_PROJECTION to guess it from CV_INTERN
    #  % we choose the depth used for depth cropping is not rescaled i.e
    #  % third and forth line in GL_PROJECTION are identical
    #  % this lead to the system :
    #  %  camera.CV_INTERN=M* camera.GL_PROJECTION
    #  %  [0,0,0,0]=[0,0,1,1]*camera.GL_PROJECTION

    #  camera.GL_PROJECTION = (([[1,0,0;0,-1,viewport(4);0,0,1]*M;[0,0,1,1]]^-1)*[camera.CV_INTERN;[0,0,0,0]])* diag([1,-1,-1,1]) ;
    # % warning('we should add a field to the class camera that contain the missing information on depth cropping')
    #  camera.GL_MODELVIEW  = diag([1,-1,-1,1])*camera.CV_EXTERN;


#   function camera=camera_get_CV_from_OPENGL(camera)
#  % get camera matrices in Computeur vision standart from opengl standart
#  % depth obtained in CV is not rescaled as with Opengl

#      viewport=camera.viewport;
#      M=[0.5*viewport(3),        0           ,     0      ,   0.5*viewport(3)+viewport(1);...
#            0          ,    0.5*viewport(4) ,     0     ,     0.5*viewport(4)+viewport(2);...
#            0          ,         0          ,     0    ,               1             ];
#      camera.CV_INTERN  = [1,0,0;0,-1,viewport(4);0,0,1]*M*camera.GL_PROJECTION* diag([1,-1,-1,1]);
#      camera.CV_EXTERN  = diag([1,-1,-1,1])*camera.GL_MODELVIEW;
#      camera.CV_TOTAL   = camera.CV_INTERN*camera.CV_EXTERN;


#       camera.CV_TOTAL_B=zeros(size( camera.CV_TOTAL ));
#       camera.CV_EXTERN_B=zeros(size(camera.CV_EXTERN ));
#       camera.CV_INTERN_B=zeros(size( camera.CV_INTERN ));


class OffscreenRenderer:
    def __init__(self):
        self.ctx = moderngl.create_standalone_context()

        # Shaders
        self.shader_program = self.ctx.program(
            vertex_shader=OpenGLShaders.vertex_shader_source,
            fragment_shader=OpenGLShaders.fragment_shader_RGB_source,
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
                    "does not support background image yet, please provide a backround image that correspond to a uniform color"
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
        shader_program["ligth_directional"].value = tuple(deodr_scene.ligth_directional)
        shader_program["ligth_ambiant"].value = deodr_scene.ambiant_light
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
        assert deodr_scene.mesh.texture.flags["WRITEABLE"] == False
        texture_id = id(deodr_scene.mesh.texture)
        if self.texture is None or self.texture_id != texture_id:
            self.texture_id = id(deodr_scene.mesh.texture)
            self.texture = ctx.texture(
                (deodr_scene.mesh.texture.shape[0], deodr_scene.mesh.texture.shape[1]),
                deodr_scene.mesh.texture.shape[2],
                (deodr_scene.mesh.texture * 255).astype(np.uint8).tobytes(),
            )
        # texture.build_mipmaps()

        # compputing the box around the displaced mesh to get maximum accuracy of the xyz point cloud using unit8 opengl type

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
