import numpy as np
from . import differentiable_renderer_cython
import copy


class Camera:
    def __init__(
        self, extrinsic, intrinsic, resolution, distortion=None, checks=True, tol=1e-6
    ):
        """camera with same distortion paramterization as opencv"""
        if checks:
            assert extrinsic.shape == (3, 4)
            assert intrinsic.shape == (3, 3)
            assert np.all(intrinsic[2, :] == [0, 0, 1])
            assert (
                np.linalg.norm(extrinsic[:3, :3].T.dot(extrinsic[:3, :3]) - np.eye(3))
                < tol
            )
            if distortion is not None:
                assert len(distortion) == 5

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.resolution = resolution

    @property
    def width(self):
        return self.resolution[0]

    @property
    def height(self):
        return self.resolution[1]

    def world_to_camera(self, points_3d):
        return points_3d.dot(self.extrinsic[:3, :3].T) + self.extrinsic[:3, 3]

    def camera_to_world_mtx_4x4(self):
        return np.row_stack(
            (
                np.column_stack((self.extrinsic[:, :3].T, self.get_center())),
                np.array((0, 0, 0, 1)),
            )
        )

    def left_mul_intrinsic(self, projected):
        return projected.dot(self.intrinsic[:2, :2].T) + self.intrinsic[:2, 2]

    def column_stack(self, values):
        return np.column_stack(values)

    def project_points(
        self, points_3d, get_jacobians=False, store_backward=None, return_depths=True
    ):  # similar to cv2.project_points
        pcam = self.world_to_camera(points_3d)
        depths = pcam[:, 2]
        projected = pcam[:, :2] / depths[:, None]

        if self.distortion is None:
            projected_image_coordinates = self.left_mul_intrinsic(projected)
            if store_backward is not None:
                store_backward["project_points"] = (pcam, depths, projected)
        else:
            k1, k2, p1, p2, k3, = self.distortion
            x = projected[:, 0]
            y = projected[:, 1]
            r2 = x ** 2 + y ** 2
            radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            tangential_distortionx = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
            tangential_distortiony = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
            distortedx = x * radial_distortion + tangential_distortionx
            distortedy = y * radial_distortion + tangential_distortiony
            distorted = self.column_stack((distortedx, distortedy))
            projected_image_coordinates = self.left_mul_intrinsic(distorted)
            if store_backward is not None:
                store_backward["project_points"] = (
                    pcam,
                    depths,
                    projected,
                    r2,
                    radial_distortion,
                )

        if return_depths:
            return projected_image_coordinates, depths
        else:
            return projected_image_coordinates

    def project_points_backward(
        self, projected_image_coordinates_b, store_backward, depths_b=None
    ):

        if self.distortion is None:
            pcam, depths, projected = store_backward["project_points"]
            projected_b = projected_image_coordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose

        else:
            pcam, depths, projected, r2, radial_distortion = store_backward[
                "project_points"
            ]
            k1, k2, p1, p2, k3, = self.distortion
            x = projected[:, 0]
            y = projected[:, 1]
            distorted_b = projected_image_coordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose
            distortedx_b = distorted_b[:, 0]
            distortedy_b = distorted_b[:, 1]
            x_b = distortedx_b * radial_distortion
            y_b = distortedy_b * radial_distortion
            radial_distortion_b = distortedx_b * x + distortedy_b * y
            tangential_distortionx_b = distortedx_b
            tangential_distortiony_b = distortedy_b
            x_b += tangential_distortionx_b * (2 * p1 * y + p2 * 4 * x)
            y_b += tangential_distortionx_b * 2 * p1 * x
            x_b += tangential_distortiony_b * 2 * p2 * y
            y_b += tangential_distortiony_b * (2 * p2 * x + p1 * 4 * y)
            r2_b = tangential_distortionx_b * p2 + tangential_distortiony_b * p1
            r2_b += radial_distortion_b * (k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2)
            x_b += r2_b * 2 * x
            y_b += r2_b * 2 * y
            projected_b = np.column_stack((x_b, y_b))

        pcam_b = np.column_stack(
            (
                projected_b / depths[:, None],
                -np.sum(projected_b * pcam[:, :2], axis=1) / (depths ** 2),
            )
        )
        if depths_b is not None:
            pcam_b[:, 2] += depths_b
        points_3d_b = pcam_b.dot(self.extrinsic[:3, :3].T)

        return points_3d_b

    def get_center(self):
        return -self.extrinsic[:3, :3].T.dot(self.extrinsic[:, 3])


class PerspectiveCamera(Camera):
    def __init__(
        self, width, height, fov, camera_center, rot=np.eye(3), distortion=None
    ):
        """"
        - width: width of the camera in pixels
        - height: eight of the camera in pixels
        - fov: horizontal field of view in degrees
        - camera_center: center of the camera in world coordinate system
        - rot: 3x3 rotation matrix word to camera (x_cam = rot.dot(x_world))\
            default to identity
        - distortion: distortion parameters
        """

        focal = 0.5 * width / np.tan(0.5 * fov * np.pi / 180)
        focal_x = focal
        pixel_aspect_ratio = 1
        focal_y = focal * pixel_aspect_ratio
        trans = -rot.T.dot(camera_center)
        cx = width / 2
        cy = height / 2
        intrinsic = np.array([[focal_x, 0,cx], [0, focal_y, cy], [0, 0, 1]])
        extrinsic = np.column_stack((rot, trans))
        super().__init__(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            resolution=(width, height),
        )


def default_camera(width, height, fov, vertices, rot=None, distortion=None):
    """computes the position of the camera center so that the entire mesh is visible
     and covers most or the image"""
    cam_vertices = vertices.dot(rot.T)
    box_min = cam_vertices.min(axis=0)
    box_max = cam_vertices.max(axis=0)
    box_center = 0.5 * (box_max + box_min)
    box_size = box_max - box_min
    camera_distance_x = (
        0.5 * box_size[0] / np.tan(0.5 * fov * np.pi / 180) + 0.5 * box_size[2]
    )
    camera_distance_y = (
        0.5 * box_size[1] * (width / height) / np.tan(0.5 * fov * np.pi / 180)
        + 0.5 * box_size[2]
    )
    camera_distance = max(camera_distance_x, camera_distance_y)
    camera_center = rot.T.dot(box_center + np.array([0, 0, -camera_distance]))
    return PerspectiveCamera(width, height, fov, camera_center, rot, distortion)


class Scene2DBase:
    """this class represents the structure representing the 2.5
    scene expect by the C++ code"""

    def __init__(
        self,
        faces,
        faces_uv,
        ij,
        depths,
        textured,
        uv,
        shade,
        colors,
        shaded,
        edgeflags,
        height,
        width,
        nb_colors,
        texture,
        background,
        clockwise=False,
    ):
        self.faces = faces
        self.faces_uv = faces_uv
        self.ij = ij
        self.depths = depths
        self.textured = textured
        self.uv = uv
        self.shade = shade
        self.colors = colors
        self.shaded = shaded
        self.edgeflags = edgeflags
        self.height = height
        self.width = width
        self.nb_colors = nb_colors
        self.texture = texture
        self.background = background
        self.clockwise = clockwise


class Scene2D(Scene2DBase):
    """this class represents a 2.5D scene. It contains a set of 2D vertices with
    associated depths and a list of faces that are triplets of vertices indexes"""

    def __init__(
        self,
        faces,
        faces_uv,
        ij,
        depths,
        textured,
        uv,
        shade,
        colors,
        shaded,
        edgeflags,
        height,
        width,
        nb_colors,
        texture,
        background,
        clockwise=False,
    ):
        self.faces = faces
        self.faces_uv = faces_uv
        self.ij = ij
        self.depths = depths
        self.textured = textured
        self.uv = uv
        self.shade = shade
        self.colors = colors
        self.shaded = shaded
        self.edgeflags = edgeflags
        self.height = height
        self.width = width
        self.nb_colors = nb_colors
        self.texture = texture
        self.background = background
        self.clockwise = clockwise

        # fields to store gradients
        self.uv_b = np.zeros(self.uv.shape)
        self.ij_b = np.zeros(self.ij.shape)
        self.shade_b = np.zeros(self.shade.shape)
        self.colors_b = np.zeros(self.colors.shape)
        self.texture_b = np.zeros(self.texture.shape)

    def clear_gradients(self):
        self.uv_b.fill(0)
        self.ij_b.fill(0)
        self.shade_b.fill(0)
        self.colors_b.fill(0)
        self.texture_b.fill(0)

    def render_error(self, obs, sigma=1):
        image = np.zeros((self.height, self.width, self.nb_colors))
        z_buffer = np.zeros((self.height, self.width))
        err_buffer = np.empty((self.height, self.width))
        antialiase_error = True
        differentiable_renderer_cython.renderScene(
            self, sigma, image, z_buffer, antialiase_error, obs, err_buffer
        )
        self.store_backward = (sigma, obs, image, z_buffer, err_buffer)
        return image, z_buffer, err_buffer

    def render(self, sigma=1):
        image = np.zeros((self.height, self.width, self.nb_colors))
        z_buffer = np.zeros((self.height, self.width))
        antialiase_error = False
        differentiable_renderer_cython.renderScene(
            self, sigma, image, z_buffer, antialiase_error, None, None
        )
        self.store_backward = (sigma, image, z_buffer)
        return image, z_buffer

    def render_error_backward(self, err_buffer_b, make_copies=True):
        sigma, obs, image, z_buffer, err_buffer = self.store_backward
        antialiase_error = True
        if make_copies:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                image,
                z_buffer,
                None,
                antialiase_error,
                obs,
                err_buffer.copy(),
                err_buffer_b,
            )
        else:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                image,
                z_buffer,
                None,
                antialiase_error,
                obs,
                err_buffer,
                err_buffer_b,
            )

    def render_backward(self, image_b, make_copies=True):
        sigma, image, z_buffer = self.store_backward
        antialiase_error = False
        if (
            make_copies
        ):  # if we make copies we keep the antialized image unchanged image
            # along the occlusion boundaries
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                image.copy(),
                z_buffer,
                image_b,
                antialiase_error,
                None,
                None,
                None,
            )
        else:
            differentiable_renderer_cython.renderSceneB(
                self,
                sigma,
                image,
                z_buffer,
                image_b,
                antialiase_error,
                None,
                None,
                None,
            )

    def render_compare_and_backward(
        self,
        sigma=1,
        antialiase_error=False,
        obs=None,
        mask=None,
        clear_gradients=True,
        make_copies=True,
    ):
        if mask is None:
            mask = np.ones((obs.shape[0], obs.shape[1]))
        if antialiase_error:
            image, z_buffer, err_buffer = self.render_error(obs, sigma)
        else:
            image, z_buffer = self.render(sigma)

        if clear_gradients:
            self.clear_gradients()

        if antialiase_error:
            err_buffer = err_buffer * mask
            err = np.sum(err_buffer)
            err_buffer_b = copy.copy(mask)
            self.render_error_backward(err_buffer_b, make_copies=make_copies)
        else:
            diff_image = (image - obs) * mask[:, :, None]
            err_buffer = (diff_image) ** 2
            err = np.sum(err_buffer)
            image_b = 2 * diff_image
            self.render_backward(image_b, make_copies=make_copies)

        return image, z_buffer, err_buffer, err


class Scene3D:
    """this class represents a 3D scene containing a single mesh, a directional light
    and an ambient light. The parameter sigma control the width of
    antialiasing edge overdraw"""

    def __init__(self, sigma=1):
        self.mesh = None
        self.light_directional = None
        self.light_ambient = None
        self.sigma = sigma

    def clear_gradients(self):
        # fields to store gradients
        self.uv_b = np.zeros((self.mesh.nb_vertices, 2))
        self.ij_b = np.zeros((self.mesh.nb_vertices, 2))
        self.shade_b = np.zeros((self.mesh.nb_vertices))
        self.colors_b = np.zeros(self.colors.shape)
        self.texture_b = np.zeros((0, 0))

    def set_light(self, light_directional, light_ambient):
        self.light_directional = np.array(light_directional)
        self.light_ambient = light_ambient

    def set_mesh(self, mesh):
        self.mesh = mesh

    def set_background(self, background_image):
        assert background_image.dtype == np.double
        self.background = background_image

    def compute_vertices_luminosity(self):
        directional = np.maximum(
            0, -np.sum(self.mesh.vertex_normals * self.light_directional, axis=1)
        )
        if self.store_backward_current is not None:
            self.store_backward_current["compute_vertices_luminosity"] = directional
        return directional + self.light_ambient

    def _compute_vertices_colors_with_illumination(self):

        vertices_luminosity = self.compute_vertices_luminosity()
        colors = self.mesh.vertices_colors * vertices_luminosity[:, None]
        if self.store_backward_current is not None:
            self.store_backward_current[
                "_compute_vertices_colors_with_illumination"
            ] = vertices_luminosity
        return colors

    def _compute_vertices_colors_with_illumination_backward(self, colors_b):
        vertices_luminosity = self.store_backward_current[
            "_compute_vertices_colors_with_illumination"
        ]
        vertices_luminosity_b = np.sum(self.mesh.vertices_colors * colors_b, axis=1)
        self.mesh.vertices_colors_b = colors_b * vertices_luminosity[:, None]
        self.light_ambient_b = np.sum(vertices_luminosity_b)
        directional_b = vertices_luminosity_b
        self.compute_vertices_luminosity_backward(directional_b)

    def compute_vertices_luminosity_backward(self, directional_b):
        directional = self.store_backward_current["compute_vertices_luminosity"]
        self.light_directional_b = -np.sum(
            ((directional_b * (directional > 0))[:, None]) * self.mesh.vertex_normals,
            axis=0,
        )
        self.vertex_normals_b = (
            -((directional_b * (directional > 0))[:, None]) * self.light_directional
        )

    def _render_2d(self, ij, colors):
        nb_color_chanels = colors.shape[1]
        image = np.empty((self.height, self.width, nb_color_chanels))
        z_buffer = np.empty((self.height, self.width))
        self.ij = np.array(ij)
        self.colors = np.array(colors)
        differentiable_renderer_cython.renderScene(self, self.sigma, image, z_buffer)

        if self.store_backward_current is not None:
            self.store_backward_current["render_2d"] = (ij, colors, image, z_buffer)

        return image, z_buffer

    def _render_2d_backward(self, image_b):
        ij, colors, image, z_buffer = self.store_backward_current["render_2d"]
        self.ij = np.array(ij)
        self.colors = np.array(colors)
        differentiable_renderer_cython.renderSceneB(
            self, self.sigma, image.copy(), z_buffer, image_b
        )
        return self.ij_b, self.colors_b

    def render(self, camera, return_zbuffer=False):
        self.store_backward_current = {}
        self.mesh.compute_vertex_normals()

        ij, depths = camera.project_points(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        self.edgeflags = self.mesh.edge_on_silhouette(ij)
        # construct 2D scene
        self.faces = self.mesh.faces.astype(np.uint32)

        self.depths = depths
        if self.mesh.uv is not None:
            self.uv = self.mesh.uv
            self.faces_uv = self.mesh.faces_uv
            self.textured = np.ones((self.mesh.nb_faces), dtype=np.bool)
            self.shade = self.compute_vertices_luminosity()
            self.shaded = np.ones(
                (self.mesh.nb_faces), dtype=np.bool
            )  # could eventually be non zero if we were using texture
            self.texture = self.mesh.texture
            colors = np.zeros((self.mesh.nb_vertices, self.texture.shape[2]))
        else:
            colors = self._compute_vertices_colors_with_illumination()
            self.faces_uv = self.faces
            self.uv = np.zeros((self.mesh.nb_vertices, 2))
            self.textured = np.zeros((self.mesh.nb_faces), dtype=np.bool)
            self.shade = np.zeros(
                (self.mesh.nb_vertices), dtype=np.float
            )  # could eventually be non zero if we were using texture
            self.shaded = np.zeros(
                (self.mesh.nb_faces), dtype=np.bool
            )  # could eventually be non zero if we were using texture
            self.texture = np.zeros((0, 0))

        self.height = camera.resolution[1]
        self.width = camera.resolution[0]

        self.clockwise = self.mesh.clockwise
        image, z_buffer = self._render_2d(ij, colors)
        if self.store_backward_current is not None:
            self.store_backward_current["render"] = (
                camera,
                self.edgeflags,
            )  # store this field as it could be overwritten when
            # rendering several views
        if return_zbuffer:
            return image, z_buffer
        else:
            return image

    def render_backward(self, image_b):

        camera, self.edgeflags = self.store_backward_current["render"]
        ij_b, colors_b = self._render_2d_backward(image_b)
        self._compute_vertices_colors_with_illumination_backward(colors_b)
        self.mesh.vertices_b = camera.project_points_backward(
            ij_b, store_backward=self.store_backward_current
        )
        self.mesh.compute_vertex_normals_backward(self.vertex_normals_b)

    def render_depth(self, camera, resolution, depth_scale=1):
        self.store_backward_current = {}
        points_2d, depths = camera.project_points(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        self.mesh.compute_face_normals()
        edge_bool = self.mesh.edge_on_silhouette(points_2d)

        self.faces = self.mesh.faces.astype(np.uint32)
        self.faces_uv = self.faces
        ij = points_2d
        colors = depths[:, None] * depth_scale
        self.depths = depths
        self.edgeflags = edge_bool
        self.uv = np.zeros((self.mesh.nb_vertices, 2))
        self.textured = np.zeros((self.mesh.nb_faces), dtype=np.bool)
        self.shade = np.zeros(
            (self.mesh.nb_vertices), dtype=np.bool
        )  # eventually used when using texture
        self.height = resolution[1]
        self.width = resolution[0]
        self.shaded = np.zeros(
            (self.mesh.nb_faces), dtype=np.bool
        )  # eventually used when using texture
        self.texture = np.zeros((0, 0))
        self.clockwise = self.mesh.clockwise
        image, _ = self._render_2d(ij, colors)
        if self.store_backward_current is not None:
            self.store_backward_current["render_depth"] = (camera, depth_scale)
        return image

    def render_depth_backward(self, depth_b):
        camera, depth_scale = self.store_backward_current["render_depth"]
        ij_b, colors_b = self._render_2d_backward(depth_b)
        depths_b = np.squeeze(colors_b * depth_scale, axis=1)
        self.mesh.vertices_b = camera.project_points_backward(
            ij_b, depths_b=depths_b, store_backward=self.store_backward_current
        )

    def render_deffered(
        self,
        camera,
        depth_scale=1,
        color=True,
        depth=True,
        faceid=True,
        normal=True,
        luminosity=True,
        uv=True,
        xyz=True,
    ):

        points_2d, depths = camera.project_points(self.mesh.vertices)

        # compute silhouette edges
        self.store_backward_current = None
        edgeflags = self.mesh.edge_on_silhouette(points_2d)
        self.mesh.compute_vertex_normals()
        vertices_luminosity = self.compute_vertices_luminosity()

        # construct triangle soup (loosing connectivity), needed to render
        # discontinuous uv maps and face ids
        soup_nb_faces = self.mesh.nb_faces
        soup_nb_vertices = 3 * self.mesh.nb_faces
        soup_faces = np.arange(0, soup_nb_vertices, dtype=np.uint32).reshape(
            self.mesh.nb_faces, 3
        )
        soup_faces_uv = soup_faces
        soup_ij = points_2d[self.mesh.faces].reshape(soup_nb_vertices, 2)
        soup_xyz = self.mesh.vertices[self.mesh.faces].reshape(soup_nb_vertices, 3)
        soup_faceids = np.tile(
            np.arange(0, self.mesh.nb_faces)[:, None], (1, 3)
        ).reshape(soup_nb_vertices, 1)
        soup_depths = depths[self.mesh.faces].reshape(soup_nb_vertices, 1)
        soup_normals = self.mesh.vertex_normals[self.mesh.faces].reshape(
            soup_nb_vertices, 3
        )
        soup_luminosity = vertices_luminosity[self.mesh.faces].reshape(
            soup_nb_vertices, 1
        )
        channels = {}
        if depth:
            channels["depth"] = soup_depths * depth_scale
        if faceid:
            channels["faceid"] = soup_faceids
        if normal:
            channels["normal"] = soup_normals
        if luminosity:
            channels["luminosity"] = soup_luminosity
        if xyz:
            channels["xyz"] = soup_xyz

        if self.mesh.uv is None:
            if color:
                soup_vcolors = self.mesh.vertices_color[self.mesh.faces]
                channels["color"] = soup_vcolors
        else:
            soup_uv = self.mesh.uv[self.mesh.faces_uv].reshape(soup_nb_vertices, 2)
            channels["uv"] = soup_uv

        offset = 0
        ranges = {}
        for k, v in channels.items():
            size = v.shape[1]
            ranges[k] = (offset , offset + size)
            offset += size

        colors = np.column_stack(channels.values())

        nb_colors = colors.shape[1]
        uv = np.zeros((soup_nb_vertices, 2))
        textured = np.zeros((soup_nb_faces), dtype=np.bool)
        shade = np.zeros((soup_nb_vertices), dtype=np.bool)

        height = camera.resolution[1]
        width = camera.resolution[0]
        shaded = np.zeros(
            (soup_nb_faces), dtype=np.bool
        )  # eventually used when using texture
        texture = np.zeros((0, 0))

        background = np.zeros((height, width, nb_colors))
        if 'depth' in channels:
            background[:, :, ranges['depth'][0]:ranges['depth'][1]] = depths.max()

        scene_2d = Scene2DBase(
            faces=soup_faces,
            faces_uv=soup_faces_uv,
            ij=soup_ij,
            depths=soup_depths,
            textured=textured,
            uv=uv,
            shade=shade,
            colors=colors,
            shaded=shaded,
            edgeflags=edgeflags,
            height=height,
            width=width,
            nb_colors=nb_colors,
            texture=texture,
            background=background,
        )
        buffers = np.empty((camera.height, camera.width, nb_colors))
        z_buffer = np.empty((camera.height, camera.width))
        differentiable_renderer_cython.renderScene(scene_2d, 0, buffers, z_buffer)

        output = {}
        for k, v in channels.items():
            output[k]=buffers[:, :, ranges[k][0]:ranges[k][1]]           

        return output
