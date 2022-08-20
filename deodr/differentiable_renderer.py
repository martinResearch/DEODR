"""Module to do differentiable rendering of 2D and 3D scenes."""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
import warnings

import numpy as np

from deodr.triangulated_mesh import ColoredTriMesh

from . import differentiable_renderer_cython  # type: ignore


@dataclass
class Scene2DBase:
    """Class representing the structure representing the 2.5
    scene expected by the C++ code
    """

    faces: np.ndarray
    faces_uv: np.ndarray
    ij: np.ndarray
    depths: np.ndarray
    textured: np.ndarray
    uv: np.ndarray
    shade: np.ndarray
    colors: np.ndarray
    shaded: np.ndarray
    edgeflags: np.ndarray
    height: int
    width: int
    nb_colors: int
    texture: np.ndarray
    background_image: Optional[np.ndarray] = None
    background_color: Optional[np.ndarray] = None
    uv_b: Optional[np.ndarray] = None
    ij_b: Optional[np.ndarray] = None
    shade_b: Optional[np.ndarray] = None
    colors_b: Optional[np.ndarray] = None
    texture_b: Optional[np.ndarray] = None
    clockwise: bool = False
    backface_culling: bool = True
    strict_edge: bool = True
    perspective_correct: bool = False
    integer_pixel_centers: bool = True


def renderScene(
    scene: Scene2DBase,
    sigma: float,
    image: np.ndarray,
    z_buffer: np.ndarray,
    antialiase_error: bool = False,
    obs: Optional[np.ndarray] = None,
    err_buffer: Optional[np.ndarray] = None,
    check_valid: bool = True,
) -> None:

    if check_valid:
        # doing checks here as it seems the debugger in not able to find the pyx file
        # when installed from a wheel. this also make interactive debugging easier
        # for the library user

        assert image is not None
        assert z_buffer is not None
        height = image.shape[0]
        width = image.shape[1]
        nb_colors = image.shape[2]

        nb_triangles = scene.faces.shape[0]
        assert nb_triangles == scene.faces_uv.shape[0]
        nb_vertices = scene.depths.shape[0]
        nb_vertices_uv = scene.uv.shape[0]

        assert scene.faces.dtype == np.uint32
        assert np.all(scene.faces < nb_vertices)
        assert np.all(scene.faces_uv < nb_vertices_uv)

        assert scene.colors.ndim == 2
        assert scene.uv.ndim == 2
        assert scene.ij.ndim == 2
        assert scene.shade.ndim == 1
        assert scene.edgeflags.ndim == 2
        assert scene.textured.ndim == 1
        assert scene.shaded.ndim == 1
        assert scene.uv.shape[1] == 2
        assert scene.ij.shape[0] == nb_vertices
        assert scene.ij.shape[1] == 2
        assert scene.shade.shape[0] == nb_vertices
        assert scene.colors.shape[0] == nb_vertices
        assert scene.colors.shape[1] == nb_colors
        assert scene.edgeflags.shape[0] == nb_triangles
        assert scene.edgeflags.shape[1] == 3
        assert scene.textured.shape[0] == nb_triangles
        assert scene.shaded.shape[0] == nb_triangles

        assert (scene.background_image is not None) != (
            scene.background_color is not None
        )

        if scene.background_image is not None:
            assert scene.background_image.ndim == 3
            assert scene.background_image.shape[0] == height
            assert scene.background_image.shape[1] == width
            assert scene.background_image.shape[2] == nb_colors
            assert (
                scene.background_color is None
            ), "You need to provide either background_image or background_color"
        else:
            assert (
                scene.background_color is not None
            ), "You need to provide background_image or background_color"
            assert scene.background_color.shape[0] == nb_colors

        if scene.texture.size > 0:
            assert scene.texture.ndim == 3
            assert scene.texture.shape[0] > 0
            assert scene.texture.shape[1] > 0
            assert scene.texture.shape[2] == nb_colors

        assert z_buffer.shape[0] == height
        assert z_buffer.shape[1] == width

        if antialiase_error:
            assert err_buffer is not None, "You need to provide err_buffer"
            assert obs is not None, "You need to provide obs"
            assert err_buffer.shape[0] == height
            assert err_buffer.shape[1] == width
            assert obs.shape[0] == height
            assert obs.shape[1] == width
            assert obs.shape[2] == nb_colors

    differentiable_renderer_cython.renderScene(
        scene, sigma, image, z_buffer, antialiase_error, obs, err_buffer
    )


def renderSceneB(
    scene: Scene2DBase,
    sigma: float,
    image: np.ndarray,
    z_buffer: np.ndarray,
    image_b: Optional[np.ndarray] = None,
    antialiase_error: bool = False,
    obs: Optional[np.ndarray] = None,
    err_buffer: Optional[np.ndarray] = None,
    err_buffer_b: Optional[np.ndarray] = None,
    check_valid: bool = True,
) -> None:

    if check_valid:
        # doing checks here as it seems the debugger in not able to find the pyx file
        # when installed from a wheel. this also make interactive debugging easier
        # for the library user

        assert image is not None
        assert z_buffer is not None

        height = image.shape[0]
        width = image.shape[1]
        nb_colors = image.shape[2]
        nb_triangles = scene.faces.shape[0]

        assert nb_colors == scene.colors.shape[1]
        assert z_buffer.shape[0] == height
        assert z_buffer.shape[1] == width
        assert nb_triangles == scene.faces_uv.shape[0]

        nb_vertices = scene.depths.shape[0]
        nb_vertices_uv = scene.uv.shape[0]

        assert scene.faces.dtype == np.uint32
        assert np.all(scene.faces < nb_vertices)
        assert np.all(scene.faces_uv < nb_vertices_uv)

        assert scene.colors.ndim == 2
        assert scene.uv.ndim == 2
        assert scene.ij.ndim == 2
        assert scene.shade.ndim == 1
        assert scene.edgeflags.ndim == 2
        assert scene.textured.ndim == 1
        assert scene.shaded.ndim == 1
        assert scene.uv.shape[1] == 2
        assert scene.ij.shape[0] == nb_vertices
        assert scene.ij.shape[1] == 2
        assert scene.shade.shape[0] == nb_vertices
        assert scene.colors.shape[0] == nb_vertices
        assert scene.colors.shape[1] == nb_colors
        assert scene.edgeflags.shape[0] == nb_triangles
        assert scene.edgeflags.shape[1] == 3
        assert scene.textured.shape[0] == nb_triangles
        assert scene.shaded.shape[0] == nb_triangles

        assert (scene.background_image is not None) != (
            scene.background_color is not None
        )

        if scene.background_image is not None:
            assert scene.background_image.ndim == 3
            assert scene.background_image.shape[0] == height
            assert scene.background_image.shape[1] == width
            assert scene.background_image.shape[2] == nb_colors
            assert (
                scene.background_color is None
            ), "You need to provide either background_image or background_color"
        else:
            assert (
                scene.background_color is not None
            ), "You need to provide background_image or background_color"
            assert scene.background_color.shape[0] == nb_colors

        assert scene.uv_b is not None
        assert scene.ij_b is not None
        assert scene.shade_b is not None
        assert scene.uv_b is not None
        assert scene.colors_b is not None

        assert scene.uv_b.ndim == 2
        assert scene.ij_b.ndim == 2
        assert scene.shade_b.ndim == 1
        assert scene.edgeflags.ndim == 2
        assert scene.textured.ndim == 1
        assert scene.shaded.ndim == 1
        assert scene.uv_b.shape[0] == nb_vertices_uv
        assert scene.uv_b.shape[1] == 2
        assert scene.ij_b.shape[0] == nb_vertices
        assert scene.ij_b.shape[1] == 2
        assert scene.shade_b.shape[0] == nb_vertices
        assert scene.colors_b.shape[0] == nb_vertices
        assert scene.colors_b.shape[1] == nb_colors

        if scene.texture.size > 0:
            assert scene.texture_b is not None
            assert scene.texture.ndim == 3
            assert scene.texture_b.ndim == 3
            assert scene.texture.shape[0] > 0
            assert scene.texture.shape[1] > 0
            assert scene.texture.shape[0] == scene.texture_b.shape[0]
            assert scene.texture.shape[1] == scene.texture_b.shape[1]
            assert scene.texture.shape[2] == nb_colors
            assert scene.texture_b.shape[2] == nb_colors

        if antialiase_error:
            assert err_buffer is not None
            assert obs is not None
            assert err_buffer.shape[0] == height
            assert err_buffer.shape[1] == width
            assert obs.shape[0] == height
            assert obs.shape[1] == width
        else:
            assert image_b is not None
            assert image_b.shape[0] == height
            assert image_b.shape[1] == width

    differentiable_renderer_cython.renderSceneB(
        scene,
        sigma,
        image,
        z_buffer,
        image_b,
        antialiase_error,
        obs,
        err_buffer,
        err_buffer_b,
    )


class Camera:
    """Camera class with the same distortion parameterization as OpenCV."""

    def __init__(
        self,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        height: int,
        width: int,
        distortion: Union[Optional[np.ndarray], Iterable[float]] = None,
        checks: bool = True,
        tol: float = 1e-6,
    ):

        if checks:
            assert extrinsic.shape == (3, 4)
            assert intrinsic.shape == (3, 3)
            assert np.all(intrinsic[2, :] == [0, 0, 1])
            assert (
                np.linalg.norm(extrinsic[:3, :3].T.dot(extrinsic[:3, :3]) - np.eye(3))
                < tol
            )

            if distortion is not None:
                distortion = np.array(distortion)
                assert distortion.shape == (5,)

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.height = height
        self.width = width

    def world_to_camera(self, points_3d: np.ndarray) -> np.ndarray:
        assert points_3d.shape[-1] == 3
        return points_3d.dot(self.extrinsic[:3, :3].T) + self.extrinsic[:3, 3]

    def camera_to_world_mtx_4x4(self) -> np.ndarray:
        return np.row_stack(
            (
                np.column_stack((self.extrinsic[:, :3].T, self.get_center())),
                np.array((0, 0, 0, 1)),
            )
        )

    def left_mul_intrinsic(self, projected: np.ndarray) -> np.ndarray:
        assert projected.ndim == 2
        assert projected.shape[-1] == 2
        return projected.dot(self.intrinsic[:2, :2].T) + self.intrinsic[:2, 2]

    def column_stack(
        self, values: Union[List[np.ndarray], Tuple[np.ndarray, ...]]
    ) -> np.ndarray:
        return np.column_stack(values)

    @overload
    def project_points(
        self,
        points_3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def project_points(
        self,
        points_3d: np.ndarray,
        *,
        store_backward: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def project_points(
        self,
        points_3d: np.ndarray,
        return_depths: Literal[True],
        store_backward: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def project_points(
        self,
        points_3d: np.ndarray,
        return_depths: Literal[False],
        store_backward: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        ...

    def project_points(
        self,
        points_3d: np.ndarray,
        return_depths: bool = True,
        store_backward: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], np.ndarray
    ]:  # similar to cv2.project_points
        p_camera = self.world_to_camera(points_3d)
        depths = p_camera[:, 2]
        projected = p_camera[:, :2] / depths[:, None]

        if self.distortion is None:
            projected_image_coordinates = self.left_mul_intrinsic(projected)
            if store_backward is not None:
                store_backward["project_points"] = (p_camera, depths, projected)
        else:
            (
                k1,
                k2,
                p1,
                p2,
                k3,
            ) = self.distortion
            x = projected[:, 0]
            y = projected[:, 1]
            x2 = x**2
            y2 = y**2
            r2 = x2 + y2
            r4 = r2 * r2
            r6 = r2 * r4
            radial_distortion = 1 + k1 * r2 + k2 * r4 + k3 * r6
            tangential_distortion_x = 2 * p1 * x * y + p2 * (r2 + 2 * x2)
            tangential_distortion_y = p1 * (r2 + 2 * y2) + 2 * p2 * x * y
            distorted_x = x * radial_distortion + tangential_distortion_x
            distorted_y = y * radial_distortion + tangential_distortion_y
            distorted = self.column_stack((distorted_x, distorted_y))
            projected_image_coordinates = self.left_mul_intrinsic(distorted)
            if store_backward is not None:
                store_backward["project_points"] = (
                    p_camera,
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
        self,
        projected_image_coordinates_b: np.ndarray,
        store_backward: Dict[str, Any],
        depths_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if self.distortion is None:
            p_camera, depths, projected = store_backward["project_points"]
            projected_b = projected_image_coordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose

        else:
            p_camera, depths, projected, r2, radial_distortion = store_backward[
                "project_points"
            ]
            (
                k1,
                k2,
                p1,
                p2,
                k3,
            ) = self.distortion
            x = projected[:, 0]
            y = projected[:, 1]
            distorted_b = projected_image_coordinates_b.dot(
                self.intrinsic[:2, :2].T
            )  # not sure about transpose
            distorted_x_b = distorted_b[:, 0]
            distorted_y_b = distorted_b[:, 1]
            x_b = distorted_x_b * radial_distortion
            y_b = distorted_y_b * radial_distortion
            radial_distortion_b = distorted_x_b * x + distorted_y_b * y
            tangential_distortion_x_b = distorted_x_b
            tangential_distortion_y_b = distorted_y_b
            x_b += tangential_distortion_x_b * (2 * p1 * y + p2 * 4 * x)
            y_b += tangential_distortion_x_b * 2 * p1 * x
            x_b += tangential_distortion_y_b * 2 * p2 * y
            y_b += tangential_distortion_y_b * (2 * p2 * x + p1 * 4 * y)
            r2_b = tangential_distortion_x_b * p2 + tangential_distortion_y_b * p1
            r2_b += radial_distortion_b * (k1 + 2 * k2 * r2 + 3 * k3 * r2**2)
            x_b += r2_b * 2 * x
            y_b += r2_b * 2 * y
            projected_b = np.column_stack((x_b, y_b))

        p_camera_b = np.column_stack(
            (
                projected_b / depths[:, None],
                -np.sum(projected_b * p_camera[:, :2], axis=1) / (depths**2),
            )
        )
        if depths_b is not None:
            p_camera_b[:, 2] += depths_b
        return p_camera_b.dot(self.extrinsic[:3, :3].T)

    def get_center(self) -> np.ndarray:
        return -self.extrinsic[:3, :3].T.dot(self.extrinsic[:, 3])

    def __repr__(self) -> str:
        return (
            f"<Camera>\n"
            f"width:\n{str(self.width)}\n"
            f"height:\n{str(self.height)}\n"
            f"extrinsic:\n{str(self.extrinsic)}\n"
            f"intrinsic:\n{str(self.intrinsic)}\n"
            f"distortion:\n{str(self.distortion)}\n"
        )


class PerspectiveCamera(Camera):
    """Camera with perspective projection."""

    def __init__(
        self,
        width: int,
        height: int,
        fov: float,
        camera_center: np.ndarray,
        rot: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
    ):
        """Perspective camera constructor.

        - width: width of the camera in pixels
        - height: eight of the camera in pixels
        - fov: horizontal field of view in degrees
        - camera_center: center of the camera in world coordinate system
        - rot: 3x3 rotation matrix word to camera (x_cam = rot.dot(x_world))\
            default to identity
        - distortion: distortion parameters
        """
        assert camera_center.shape == (3,)
        if rot is None:
            rot = np.eye(3)
        else:
            # check it is indeed a rotation matrix
            assert rot.shape == (3, 3)
            assert np.allclose(rot.T.dot(rot), np.eye(3), 1e-6)
            assert np.linalg.det(rot) > 0
        focal = 0.5 * width / np.tan(0.5 * fov * np.pi / 180)
        focal_x = focal
        pixel_aspect_ratio = 1
        focal_y = focal * pixel_aspect_ratio
        trans = -rot.T.dot(camera_center)
        cx = width / 2
        cy = height / 2
        intrinsic = np.array([[focal_x, 0, cx], [0, focal_y, cy], [0, 0, 1]])
        extrinsic = np.column_stack((rot, trans))
        super().__init__(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            distortion=distortion,
            width=width,
            height=height,
        )


def default_camera(
    width: int,
    height: int,
    fov: float,
    vertices: np.ndarray,
    rot: np.ndarray,
    distortion: Optional[np.ndarray] = None,
) -> Camera:
    """Compute the position of the camera center so that the entire mesh is visible
    and covers most or the image.
    """
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


class Scene2D(Scene2DBase):
    """Class representing a 2.5D scene. It contains a set of 2D vertices with
    associated depths and a list of faces that are triplets of vertices indexes.
    """

    def __init__(
        self,
        faces: np.ndarray,
        faces_uv: np.ndarray,
        ij: np.ndarray,
        depths: np.ndarray,
        textured: np.ndarray,
        uv: np.ndarray,
        shade: np.ndarray,
        colors: np.ndarray,
        shaded: np.ndarray,
        edgeflags: np.ndarray,
        height: int,
        width: int,
        nb_colors: int,
        texture: np.ndarray,
        background_image: Optional[np.ndarray] = None,
        background_color: Optional[np.ndarray] = None,
        clockwise: bool = False,
        backface_culling: bool = False,
        strict_edge: bool = True,
        perspective_correct: bool = False,
        integer_pixel_centers: bool = True,
    ):
        """
        Conventions:
        If integer_pixel_centers is True (default) then pixel centers are at integer coordinates with
            upper left at (0, 0)
            upper right at (width - 1, 0)
            lower left at (0, height - 1)
            lower right at  (width - 1, height - 1)
        if integer_pixel_centers is False, then pixel centers are at half integer coordinates with
            upper left at (0.5, 0.5)
            upper right at (width - 0.5, 0.5)
            lower left at (0.5, height - 0.5)
            lower right at  (width -0.5, height - 0.5)
        """

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
        self.background_image = background_image
        self.background_color = background_color
        self.clockwise = clockwise
        self.backface_culling = backface_culling
        self.strict_edge = strict_edge
        self.perspective_correct = perspective_correct
        self.integer_pixel_centers = integer_pixel_centers

        # fields to store gradients
        self.uv_b = np.zeros(self.uv.shape)
        self.ij_b = np.zeros(self.ij.shape)
        self.shade_b = np.zeros(self.shade.shape)
        self.colors_b = np.zeros(self.colors.shape)
        self.texture_b = np.zeros(self.texture.shape)
        self.store_backward: Tuple

    def clear_gradients(self) -> None:
        assert self.uv_b is not None
        assert self.ij_b is not None
        assert self.shade_b is not None
        assert self.colors_b is not None
        assert self.texture_b is not None

        self.uv_b.fill(0)
        self.ij_b.fill(0)
        self.shade_b.fill(0)
        self.colors_b.fill(0)
        self.texture_b.fill(0)

    def render_error(
        self, obs: np.ndarray, sigma: float = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.zeros((self.height, self.width, self.nb_colors))
        z_buffer = np.zeros((self.height, self.width))
        err_buffer = np.empty((self.height, self.width))
        antialiase_error = True
        renderScene(self, sigma, image, z_buffer, antialiase_error, obs, err_buffer)
        self.store_backward = (sigma, obs, image, z_buffer, err_buffer)
        return image, z_buffer, err_buffer

    def render(self, sigma: float = 1) -> Tuple[np.ndarray, np.ndarray]:
        image = np.zeros((self.height, self.width, self.nb_colors))
        z_buffer = np.zeros((self.height, self.width))
        antialiase_error = False
        renderScene(self, sigma, image, z_buffer, antialiase_error, None, None)
        self.store_backward = (sigma, image, z_buffer)
        return image, z_buffer

    def render_error_backward(
        self, err_buffer_b: np.ndarray, make_copies: bool = True
    ) -> None:
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )

        if not self.backface_culling:
            raise BaseException(
                "use backface_culling=True if you use gradient backpropagation to get valid gradient through edge anti-aliasing."
            )
        sigma, obs, image, z_buffer, err_buffer = self.store_backward
        antialiase_error = True
        if make_copies:
            renderSceneB(
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
            renderSceneB(
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

    def render_backward(self, image_b: np.ndarray, make_copies: bool = True) -> None:
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )
        if not self.backface_culling:
            raise BaseException(
                "use backface_culling=True if you use gradient backpropagation to get valid gradient through edge anti-aliasing."
            )
        sigma, image, z_buffer = self.store_backward
        antialiase_error = False
        if (
            make_copies
        ):  # if we make copies we keep the antialized image unchanged image
            # along the occlusion boundaries
            renderSceneB(
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
            renderSceneB(
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
        obs: np.ndarray,
        sigma: float = 1,
        antialiase_error: bool = False,
        mask: Optional[np.ndarray] = None,
        clear_gradients: bool = True,
        make_copies: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )
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
            err = float(np.sum(err_buffer))
            err_buffer_b = copy.copy(mask)
            self.render_error_backward(err_buffer_b, make_copies=make_copies)
        else:
            diff_image = (image - obs) * mask[:, :, None]
            err_buffer = (diff_image) ** 2
            err = float(np.sum(err_buffer))
            image_b = 2 * diff_image
            self.render_backward(image_b, make_copies=make_copies)

        return image, z_buffer, err_buffer, err


class Scene3D:
    """Class representing a 3D scene containing a single mesh, a directional light
    and an ambient light. The parameter sigma control the width of
    antialiasing edge overdraw.
    """

    def __init__(
        self,
        sigma: float = 1,
        perspective_correct: bool = False,
        integer_pixel_centers: bool = True,
    ):
        self.mesh: Optional[ColoredTriMesh] = None

        self.light_directional: Optional[np.ndarray] = None
        self.light_ambient: float = 0
        self.sigma = sigma
        self.perspective_correct = perspective_correct
        self.background_image: Optional[np.ndarray] = None
        self.background_color: Optional[np.ndarray] = None
        self.integer_pixel_centers = integer_pixel_centers
        self.colors: Optional[np.ndarray] = None

        self.store_backward_current: Optional[Dict[str, Any]] = None
        self.vertices_b: Optional[np.ndarray] = None
        self.scene2D: Optional[Scene2D] = None

    def clear_gradients(self) -> None:
        # fields to store gradients
        assert self.mesh is not None
        self.scene_2d.clear_gradients()

    def set_light(
        self,
        light_directional: Union[Tuple[float, float, float], np.ndarray],
        light_ambient: float,
    ) -> None:
        """
        light_ambient : scalar. Intensity of the ambient light
        light_directional : 3d vector. Directional light are at an infinite distance and thus
        there is no position.  The light_directional vector corresponds to the  direction
        multiplied by the intensity (instead of a normalized direction and a scalar intensity).
        This  parameterization has been chosen because it makes it easier to do gradient
        descent  as there is not normalization constraint. However it does not support colored lights.
        """
        if light_directional is not None:
            self.light_directional = np.array(light_directional)
        else:
            self.light_directional = None
        self.light_ambient = light_ambient

    def set_mesh(self, mesh: ColoredTriMesh) -> None:
        self.mesh = mesh

    def set_background(self, background_image: np.ndarray) -> None:
        warnings.warn(
            "This will be deprecated, please use set_background_image or set_background_color",
            UserWarning,
        )
        self.set_background_image(background_image)

    def set_background_image(self, background_image: np.ndarray) -> None:
        if self.background_color is not None:
            raise BaseException(
                "you cannot provide both background image and background color"
            )
        background_image = np.asanyarray(background_image)
        assert background_image.dtype == np.double
        assert background_image.ndim == 3
        self.background_image = background_image

    def set_background_color(
        self, background_color: Union[Iterable[float], np.ndarray]
    ) -> None:
        if self.background_image is not None:
            raise BaseException(
                "you cannot provide both background image and background color"
            )
        background_color = np.asanyarray(background_color, dtype=np.float64)
        assert background_color.dtype == np.double
        assert background_color.ndim == 1
        self.background_color = background_color

    def compute_vertices_luminosity(self) -> np.ndarray:
        assert self.mesh is not None
        if self.light_directional is not None:
            directional = np.maximum(
                0, -np.sum(self.mesh.vertex_normals * self.light_directional, axis=1)
            )
        else:
            directional = np.zeros((self.mesh.nb_vertices))
        if self.store_backward_current is not None:
            self.store_backward_current["compute_vertices_luminosity"] = directional
        return directional + self.light_ambient

    def _compute_vertices_colors_with_illumination(self) -> np.ndarray:
        assert self.mesh is not None
        vertices_luminosity = self.compute_vertices_luminosity()
        colors = self.mesh.vertices_colors * vertices_luminosity[:, None]
        if self.store_backward_current is not None:
            self.store_backward_current[
                "_compute_vertices_colors_with_illumination"
            ] = vertices_luminosity
        return colors

    def _compute_vertices_colors_with_illumination_backward(
        self, colors_b: np.ndarray
    ) -> None:
        assert self.mesh is not None
        assert self.store_backward_current is not None
        vertices_luminosity: np.ndarray = self.store_backward_current[
            "_compute_vertices_colors_with_illumination"
        ]
        vertices_luminosity_b = np.sum(self.mesh.vertices_colors * colors_b, axis=1)
        self.mesh.vertices_colors_b = colors_b * vertices_luminosity[:, None]

        self.compute_vertices_luminosity_backward(vertices_luminosity_b)

    def compute_vertices_luminosity_backward(
        self, vertices_luminosity_b: np.ndarray
    ) -> None:
        assert self.mesh is not None
        assert self.store_backward_current is not None
        directional: np.ndarray = self.store_backward_current[
            "compute_vertices_luminosity"
        ]
        if self.light_directional is not None:
            self.light_directional_b = -np.sum(
                ((vertices_luminosity_b * (directional > 0))[:, None])
                * self.mesh.vertex_normals,
                axis=0,
            )
            self.vertex_normals_b = (
                -((vertices_luminosity_b * (directional > 0))[:, None])
                * self.light_directional
            )
        self.light_ambient_b = np.sum(vertices_luminosity_b)

    def _render_2d(self) -> Tuple[np.ndarray, np.ndarray]:
        nb_color_channels = self.scene_2d.nb_colors
        image = np.empty((self.scene_2d.height, self.scene_2d.width, nb_color_channels))
        z_buffer = np.empty((self.scene_2d.height, self.scene_2d.width))

        renderScene(self.scene_2d, self.sigma, image, z_buffer)

        if self.store_backward_current is not None:
            self.store_backward_current["render_2d"] = (image, z_buffer)

        return image, z_buffer

    def _render_2d_backward(self, image_b: np.ndarray) -> None:
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )
        assert self.store_backward_current is not None
        image, z_buffer = self.store_backward_current["render_2d"]
        # self.ij = np.array(ij)
        # self.colors = np.array(colors)
        renderSceneB(self.scene_2d, self.sigma, image.copy(), z_buffer, image_b)
        self.scene_2D = None

    @overload
    def render(
        self,
        camera: Camera,
    ) -> np.ndarray:
        ...

    @overload
    def render(
        self,
        camera: Camera,
        return_z_buffer: Literal[True],
        backface_culling: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def render(
        self,
        camera: Camera,
        return_z_buffer: Literal[False],
        backface_culling: bool = True,
    ) -> np.ndarray:
        ...

    def render(
        self,
        camera: Camera,
        return_z_buffer: bool = False,
        backface_culling: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        assert self.mesh is not None, "You need to provide a mesh first."
        self.store_backward_current = {}

        if self.light_directional is not None:
            self.mesh.compute_vertex_normals()

        if (self.background_image is None) == (self.background_color is None):
            raise BaseException(
                "You need to provide either a background image or background color"
            )

        points_2d, depths = camera.project_points(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        if self.sigma > 0:
            edgeflags = self.mesh.edge_on_silhouette(points_2d)
        else:
            edgeflags = np.zeros((self.mesh.nb_faces, 3), dtype=bool)
        # construct 2D scene
        faces = self.mesh.faces.astype(np.uint32)

        if self.mesh.uv is not None:
            assert self.mesh.texture is not None
            uv = self.mesh.uv
            faces_uv = self.mesh.faces_uv
            textured = np.ones((self.mesh.nb_faces), dtype=bool)
            shade = self.compute_vertices_luminosity()
            shaded = np.ones(
                (self.mesh.nb_faces), dtype=bool
            )  # could eventually be non zero if we were using texture
            texture = self.mesh.texture
            nb_colors = texture.shape[2]
            colors = np.zeros((self.mesh.nb_vertices, nb_colors))
        else:
            colors = self._compute_vertices_colors_with_illumination()
            nb_colors = colors.shape[1]
            faces_uv = faces
            uv = np.zeros((self.mesh.nb_vertices, 2))
            textured = np.zeros((self.mesh.nb_faces), dtype=bool)
            shade = np.zeros(
                (self.mesh.nb_vertices), dtype=np.float64
            )  # could eventually be non zero if we were using texture
            shaded = np.zeros(
                (self.mesh.nb_faces), dtype=bool
            )  # could eventually be non zero if we were using texture
            texture = np.zeros((0, 0))

        height = camera.height
        width = camera.width
        strict_edge = True

        clockwise = self.mesh.clockwise

        assert faces_uv is not None  # helping mypy

        scene_2d = Scene2D(
            faces=faces,
            faces_uv=faces_uv,
            ij=points_2d,
            depths=depths,
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
            background_image=self.background_image,
            background_color=self.background_color,
            clockwise=clockwise,
            backface_culling=backface_culling,
            strict_edge=strict_edge,
            perspective_correct=self.perspective_correct,
            integer_pixel_centers=self.integer_pixel_centers,
        )
        self.scene_2d = scene_2d

        image, z_buffer = self._render_2d()
        if self.store_backward_current is not None:
            self.store_backward_current["render"] = (
                camera,
                edgeflags,
            )  # store this field as it could be overwritten when
            # rendering several views
        return (image, z_buffer) if return_z_buffer else image

    def render_backward(self, image_b: np.ndarray) -> None:
        assert self.scene_2d is not None
        assert self.scene_2d.colors_b is not None
        assert self.mesh is not None
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )
        assert self.store_backward_current is not None
        camera, self.edgeflags = self.store_backward_current["render"]
        self._render_2d_backward(image_b)
        self._compute_vertices_colors_with_illumination_backward(self.scene_2d.colors_b)
        self.mesh._vertices_b = camera.project_points_backward(
            self.scene_2d.ij_b, store_backward=self.store_backward_current
        )
        if self.light_directional is not None:
            self.mesh.compute_vertex_normals_backward(self.vertex_normals_b)

    def render_depth(
        self, camera: Camera, depth_scale: float = 1, backface_culling: bool = True
    ) -> np.ndarray:
        assert self.mesh is not None
        self.store_backward_current = {}
        points_2d, depths = camera.project_points(
            self.mesh.vertices, store_backward=self.store_backward_current
        )

        # compute silhouette edges
        if self.sigma > 0:
            edgeflags = self.mesh.edge_on_silhouette(points_2d)
        else:
            edgeflags = np.zeros((self.mesh.nb_faces, 3), dtype=bool)

        scene_2d = Scene2D(
            ij=points_2d,
            nb_colors=1,
            faces=self.mesh.faces.astype(np.uint32),
            faces_uv=self.mesh.faces,
            colors=depths[:, None] * depth_scale,
            depths=depths,
            edgeflags=edgeflags,
            uv=np.zeros((self.mesh.nb_vertices, 2)),
            textured=np.zeros((self.mesh.nb_faces), dtype=bool),
            shade=np.zeros(
                (self.mesh.nb_vertices), dtype=bool
            ),  # eventually used when using texture
            height=camera.height,
            width=camera.width,
            shaded=np.zeros(
                (self.mesh.nb_faces), dtype=bool
            ),  # eventually used when using texture
            texture=np.zeros((0, 0)),
            clockwise=self.mesh.clockwise,
            backface_culling=backface_culling,
            strict_edge=True,
            background_color=self.background_color,
        )
        self.scene_2d = scene_2d
        image, _ = self._render_2d()
        if self.store_backward_current is not None:
            self.store_backward_current["render_depth"] = (camera, depth_scale)
        return image

    def render_depth_backward(self, depth_b: np.ndarray) -> None:
        assert self.store_backward_current is not None
        assert self.mesh is not None
        if self.perspective_correct:
            raise BaseException(
                "perspective_correct not supported yet for gradient back propagation"
            )
        camera, depth_scale = self.store_backward_current["render_depth"]
        self._render_2d_backward(depth_b)
        depths_b = np.squeeze(self.scene_2d.colors_b * depth_scale, axis=1)
        self.mesh._vertices_b = camera.project_points_backward(
            self.scene_2d.ij_b,
            depths_b=depths_b,
            store_backward=self.store_backward_current,
        )

    def render_deferred(
        self,
        camera: Camera,
        depth_scale: float = 1,
        color: bool = True,
        depth: bool = True,
        face_id: bool = True,
        barycentric: bool = True,
        normal: bool = True,
        luminosity: bool = True,
        uv: bool = True,
        xyz: bool = True,
        backface_culling: bool = True,
    ) -> Dict[str, np.ndarray]:
        assert self.mesh is not None, "You need to provide a mesh first"
        points_2d, depths = camera.project_points(self.mesh.vertices)

        # compute silhouette edges
        self.store_backward_current = None

        if self.sigma > 0:
            raise BaseException(
                "Antialiasing is not supposed to be used when using deferred rendering, please use sigma==0"
            )

        edgeflags = np.zeros((self.mesh.nb_faces, 3), dtype=bool)

        if luminosity or normal:
            self.mesh.compute_vertex_normals()
        if luminosity:
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
        soup_depths = depths[self.mesh.faces].reshape(soup_nb_vertices, 1)

        channels = {}
        if depth:
            channels["depth"] = soup_depths * depth_scale
        if face_id:
            soup_face_ids = np.tile(
                np.arange(0, self.mesh.nb_faces)[:, None], (1, 3)
            ).reshape(soup_nb_vertices, 1)
            channels["face_id"] = soup_face_ids
        if barycentric:
            soup_barycentric = np.tile(
                np.eye(3, 3)[None, :, :], (self.mesh.nb_faces, 1, 1)
            ).reshape(soup_nb_vertices, 3)
            channels["barycentric"] = soup_barycentric
        if normal:
            soup_normals = self.mesh.vertex_normals[self.mesh.faces].reshape(
                soup_nb_vertices, 3
            )
            channels["normal"] = soup_normals
        if luminosity:
            soup_luminosity = vertices_luminosity[self.mesh.faces].reshape(
                soup_nb_vertices, 1
            )
            channels["luminosity"] = soup_luminosity
        if xyz:
            soup_xyz = self.mesh.vertices[self.mesh.faces].reshape(soup_nb_vertices, 3)
            channels["xyz"] = soup_xyz

        if self.mesh.uv is None:
            if color:
                assert self.mesh.vertices_colors is not None
                soup_vertices_colors = self.mesh.vertices_colors[
                    self.mesh.faces
                ].reshape(soup_nb_vertices, 3)
                channels["color"] = soup_vertices_colors
        elif uv:
            soup_uv = self.mesh.uv[self.mesh.faces_uv].reshape(soup_nb_vertices, 2)
            channels["uv"] = soup_uv

        offset = 0
        ranges = {}
        for k, v in channels.items():
            size = v.shape[1]
            ranges[k] = (offset, offset + size)
            offset += size

        colors = np.column_stack(list(channels.values()))

        nb_colors = colors.shape[1]
        uv_zeros = np.zeros((soup_nb_vertices, 2))
        textured = np.zeros((soup_nb_faces), dtype=bool)
        shade = np.zeros((soup_nb_vertices), dtype=bool)

        height = camera.height
        width = camera.width
        shaded = np.zeros(
            (soup_nb_faces), dtype=bool
        )  # eventually used when using texture
        texture = np.zeros((0, 0))

        background_image = np.zeros((height, width, nb_colors))
        if "depth" in channels:
            background_image[
                :, :, ranges["depth"][0] : ranges["depth"][1]
            ] = depths.max()

        scene_2d = Scene2DBase(
            faces=soup_faces,
            faces_uv=soup_faces_uv,
            ij=soup_ij,
            depths=soup_depths,
            textured=textured,
            uv=uv_zeros,
            shade=shade,
            colors=colors,
            shaded=shaded,
            edgeflags=edgeflags,
            height=height,
            width=width,
            nb_colors=nb_colors,
            texture=texture,
            background_image=background_image,
            backface_culling=backface_culling,
            background_color=None,
        )
        buffers = np.empty((camera.height, camera.width, nb_colors))
        z_buffer = np.empty((camera.height, camera.width))
        renderScene(scene_2d, 0, buffers, z_buffer)

        output: Dict[str, np.ndarray] = {}
        for k in channels.keys():
            output[k] = buffers[:, :, ranges[k][0] : ranges[k][1]]

        return output
