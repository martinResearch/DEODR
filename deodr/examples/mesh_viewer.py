"""Example of interactive 3D mesh visualization using DEODR and OpenCV."""

from typing import Any

import argparse
import os
import time
import pickle
from typing import Callable, Dict, Optional, Tuple, Union
from typing_extensions import Literal
import cv2

import deodr
from deodr import differentiable_renderer
from deodr.differentiable_renderer import Camera
from deodr.triangulated_mesh import ColoredTriMesh

import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

import trimesh


InteractorModeType = Literal["camera_centered", "object_centered_trackball"]


class Interactor:
    """Class that implements various mouse interaction with the 3D scene."""

    def __init__(
        self,
        camera: Camera,
        mode: InteractorModeType = "object_centered_trackball",
        object_center: Optional[np.ndarray] = None,
        rotation_speed: float = 0.003,
        z_translation_speed: float = 0.05,
        xy_translation_speed: float = 0.01,
    ):
        self.left_is_down = False
        self.right_is_down = False
        self.middle_is_down = False
        self.mode = mode
        if mode == "object_centered_trackball":
            assert object_center is not None
            assert object_center.shape == (3,)
        elif object_center is None:
            object_center = np.array([0, 0, 0])

        self.object_center = object_center
        self.rotation_speed = rotation_speed
        self.z_translation_speed = z_translation_speed
        self.xy_translation_speed = xy_translation_speed
        self.camera = camera

    def toggle_mode(self) -> None:
        if self.mode == "object_centered_trackball":
            self.mode = "camera_centered"
        else:
            self.mode = "object_centered_trackball"
        print(f"trackball mode = {self.mode}")

    def rotate(
        self,
        rot_vec: np.ndarray,
    ) -> None:
        assert np.array(rot_vec).shape == (3,)
        rotation = Rotation.from_rotvec(np.array(rot_vec))
        if self.mode == "camera_centered":
            self.camera.extrinsic = rotation.as_matrix().dot(self.camera.extrinsic)
        else:
            n_rotation = rotation.as_matrix().dot(self.camera.extrinsic[:, :3])
            nt = (
                self.camera.extrinsic[:, :3].dot(self.object_center)
                + self.camera.extrinsic[:, 3]
                - n_rotation.dot(self.object_center)
            )
            self.camera.extrinsic = np.column_stack((n_rotation, nt))

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        if event == 0 and flags == 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_is_down = True
            self.x_last = x
            self.y_last = y
        # check to see if the left mouse button was released
        if event == cv2.EVENT_LBUTTONUP:
            self.left_is_down = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self.right_is_down = True
            self.x_last = x
            self.y_last = y
            # check to see if the left mouse button was released
        if event == cv2.EVENT_RBUTTONUP:
            self.right_is_down = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.middle_is_down = True
            self.x_last = x
            self.y_last = y
        # check to see if the left mouse button was released
        if event == cv2.EVENT_MBUTTONUP:
            self.middle_is_down = False

        self.ctrl_is_down = flags & cv2.EVENT_FLAG_CTRLKEY
        self.shift_is_down = flags & cv2.EVENT_FLAG_SHIFTKEY

        if self.left_is_down and not (self.ctrl_is_down):

            if self.mode == "camera_centered":
                rot_vec = np.array(
                    [
                        -0.3 * self.rotation_speed * (y - self.y_last),
                        0.3 * self.rotation_speed * (x - self.x_last),
                        0,
                    ]
                )
                self.rotate(rot_vec)

                # assert np.allclose(center_in_camera, self.camera.world_to_camera(self.object_center))
                self.x_last = x
                self.y_last = y

            elif self.mode == "object_centered_trackball":

                self.rotate(
                    np.array(
                        [
                            self.rotation_speed * (y - self.y_last),
                            -self.rotation_speed * (x - self.x_last),
                            0,
                        ]
                    )
                )

                self.x_last = x
                self.y_last = y
            else:
                raise (BaseException(f"unknown camera mode {self.mode}"))

        if self.right_is_down and self.shift_is_down:
            delta_y = self.y_last - y
            ratio = np.power(2, delta_y / 20)
            self.camera.intrinsic[0, 0] = self.camera.intrinsic[0, 0] * ratio
            self.camera.intrinsic[1, 1] = self.camera.intrinsic[1, 1] * ratio
            self.x_last = x
            self.y_last = y

        if self.right_is_down and not (self.ctrl_is_down):
            if self.mode in ["camera_centered", "object_centered_trackball"]:
                if np.abs(self.y_last - y) >= np.abs(self.x_last - x):
                    self.camera.extrinsic[2, 3] += self.z_translation_speed * (
                        self.y_last - y
                    )
                else:
                    self.rotate(
                        np.array(
                            [
                                0,
                                0,
                                -self.rotation_speed * (self.x_last - x),
                            ]
                        )
                    )
                self.x_last = x
                self.y_last = y

            else:
                raise (BaseException(f"unknown camera mode {self.mode}"))

        if self.middle_is_down or (self.left_is_down and self.ctrl_is_down):
            # translation

            object_depth = (
                self.camera.extrinsic[2, :3].dot(self.object_center)
                + self.camera.extrinsic[2, 3]
            )

            tx = self.xy_translation_speed * object_depth * (x - self.x_last)
            ty = self.xy_translation_speed * object_depth * (y - self.y_last)

            self.object_center -= (
                self.camera.extrinsic[0, :3] * tx + self.camera.extrinsic[1, :3] * ty
            )
            self.camera.extrinsic[0, 3] += tx
            self.camera.extrinsic[1, 3] += ty
            self.x_last = x
            self.y_last = y

    def print_help(self) -> None:
        help_str = "" + "Mouse:\n"
        if self.mode == "object_centered_trackball":

            help_str += (
                "mouse left + vertical motion: rotate object along camera x axis\n"
            )
            help_str += (
                "mouse left + horizontal motion: rotate object along camera y axis\n"
            )
            help_str += (
                "mouse right + vertical motion: translate object along camera z axis\n"
            )
            help_str += (
                "mouse right + horizontal motion: rotate object along camera z axis\n"
            )
            help_str += "CTRL + mouse left + vertical motion: translate object along camera y axis\n"
            help_str += "CTRL + mouse left + horizontal motion: translate object along camera x axis\n"

        else:
            help_str += (
                "mouse right + vertical motion: translate camera along its z axis\n"
            )
            help_str += (
                "mouse right + horizontal motion: rotate camera along its z axis\n"
            )
            help_str += "mouse left + vertical motion: rotate camera along its x axis\n"
            help_str += (
                "mouse left + horizontal motion: rotate camera along its y axis\n"
            )
            help_str += "CTRL + mouse left + vertical motion: translate camera along its y axis\n"
            help_str += "CTRL + mouse left + horizontal motion: translate camera along its x axis\n"
        help_str += (
            "SHIFT + mouse left + vertical motion: change the camera field of view\n"
        )
        print(help_str)


class Viewer:
    def __init__(
        self,
        file_or_mesh: Union[str, ColoredTriMesh],
        display_texture_map: bool = True,
        width: int = 640,
        height: int = 480,
        display_fps: bool = True,
        title: Optional[str] = None,
        use_moderngl: bool = False,
        light_directional: Tuple[float, float, float] = (0, 0, -0.5),
        light_ambient: float = 0.5,
        background_color: Tuple[float, float, float] = (1, 1, 1),
        use_antialiasing: bool = True,
        use_light: bool = True,
        fps_exp_average_decay: float = 0.1,
        horizontal_fov: float = 60,
        video_pattern: str = "deodr_viewer_recording{id}.avi",
        video_format: str = "MJPG",
    ):
        self.title = title
        self.scene = differentiable_renderer.Scene3D(sigma=1)
        self.set_mesh(file_or_mesh)
        self.windowname = f"DEODR mesh viewer:{self.title}"

        self.width = width
        self.height = height
        self.display_fps = display_fps
        self.use_moderngl = use_moderngl
        self.use_antialiasing = use_antialiasing
        self.use_light = use_light
        self.fps_exp_average_decay = fps_exp_average_decay
        self.last_time: Optional[float] = None
        self.horizontal_fov = horizontal_fov
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.video_pattern = video_pattern
        self.video_format = video_format
        self.fps: float = 0

        if display_texture_map:
            self.display_texture_map()

        self.set_background_color(background_color)
        self.set_light(light_directional, light_ambient)
        self.recenter_camera()

        self.offscreen_renderer: Optional[
            "deodr.opengl.moderngl.OffscreenRenderer"
        ] = None
        if use_moderngl:
            self.setup_moderngl()

        self.register_keys()

    def set_light(
        self,
        light_directional: Union[Tuple[float, float, float], np.ndarray],
        light_ambient: float,
    ) -> None:
        self.light_directional = np.array(light_directional)
        self.light_ambient = light_ambient
        self.scene.set_light(
            light_directional=self.light_directional, light_ambient=light_ambient
        )

    def setup_moderngl(self) -> None:
        import deodr.opengl.moderngl

        self.offscreen_renderer = deodr.opengl.moderngl.OffscreenRenderer()
        assert self.scene.mesh is not None
        self.scene.mesh.compute_vertex_normals()
        self.offscreen_renderer.set_scene(self.scene)

    def set_background_color(
        self, background_color: Tuple[float, float, float]
    ) -> None:
        self.scene.set_background_color(background_color)

    def display_texture_map(self) -> None:
        if self.mesh.textured:
            ax = plt.subplot(111)
            self.mesh.plot_uv_map(ax)

    def set_mesh(self, file_or_mesh: Union[str, ColoredTriMesh]) -> None:
        if isinstance(file_or_mesh, str):
            if self.title is None:
                self.title = file_or_mesh
            mesh_trimesh = trimesh.load(file_or_mesh)
            self.mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)
        elif isinstance(file_or_mesh, trimesh.base.Trimesh):
            mesh_trimesh = file_or_mesh
            self.mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)
            if self.title is None:
                self.title = "unknown"
        elif isinstance(file_or_mesh, ColoredTriMesh):
            self.mesh = file_or_mesh
            if self.title is None:
                self.title = "unknown"
        else:
            raise (
                TypeError(
                    f"unknown type {type(file_or_mesh)} for input obj_file_or_trimesh,"
                    " can be string or trimesh.base.Trimesh"
                )
            )
        self.object_center = 0.5 * (
            self.mesh.vertices.max(axis=0) + self.mesh.vertices.min(axis=0)
        )
        self.object_radius = np.max(
            self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
        )
        self.scene.set_mesh(self.mesh)

    def recenter_camera(self) -> None:
        camera_center = self.object_center + np.array([0, 0, 3]) * self.object_radius
        rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        translation = -rotation.T.dot(camera_center)
        extrinsic = np.column_stack((rotation, translation))
        focal = 0.5 * self.width / np.tan(0.5 * self.horizontal_fov * np.pi / 180)
        intrinsic = np.array(
            [[focal, 0, self.width / 2], [0, focal, self.height / 2], [0, 0, 1]]
        )

        distortion = [0, 0, 0, 0, 0]
        self.camera = differentiable_renderer.Camera(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            width=self.width,
            height=self.height,
            distortion=distortion,
        )

        self.interactor = Interactor(
            camera=self.camera,
            object_center=self.object_center,
            z_translation_speed=0.01 * self.object_radius,
            xy_translation_speed=3e-4,
        )

    def start(self, print_help: bool = True, loop: bool = True) -> None:
        """Open the window and start the loop if loop true."""
        if print_help:
            self.print_help()
        self.fps = 0
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowname, self.width, self.height)
        cv2.setMouseCallback(self.windowname, self.interactor.mouse_callback)
        if loop:
            while cv2.getWindowProperty(self.windowname, 0) >= 0:
                self.refresh()

    def update_fps(self) -> None:
        new_time = time.perf_counter()
        if self.last_time is None:
            self.fps = 0
        elif self.fps == 0:
            self.fps = 1 / (new_time - self.last_time)
        else:
            new_fps = 1 / (new_time - self.last_time)
            self.fps = (
                1 - self.fps_exp_average_decay
            ) * self.fps + self.fps_exp_average_decay * new_fps
        self.last_time = new_time

    def refresh(self) -> None:
        self.width, self.height = cv2.getWindowImageRect(self.windowname)[2:]
        self.resize_camera()

        if self.use_moderngl:
            assert self.offscreen_renderer is not None
            image = self.offscreen_renderer.render(self.camera)
        else:
            image = (self.scene.render(self.interactor.camera) * 255).astype(np.uint8)

        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.recording:
            assert self.video_writer is not None
            self.video_writer.write(bgr_image.astype(np.uint8))

        self.update_fps()
        if self.recording:
            cv2.circle(
                bgr_image,
                (image.shape[1] - 20, image.shape[0] - 20),
                8,
                (0, 0, 255),
                cv2.FILLED,
            )
        if self.display_fps:
            self.print_fps(bgr_image, self.fps)

        cv2.imshow(self.windowname, bgr_image)

        key = cv2.waitKey(1)
        if key > 0:
            self.process_key(key)

    def resize_camera(self) -> None:
        ratio = self.width / self.camera.width

        intrinsic = np.array(
            [
                [self.camera.intrinsic[0, 0] * ratio, 0, self.width / 2],
                [0, self.camera.intrinsic[1, 1] * ratio, self.height / 2],
                [0, 0, 1],
            ]
        )
        self.camera.intrinsic = intrinsic
        self.camera.width = self.width
        self.camera.height = self.height

    def print_fps(self, image: np.ndarray, fps: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (20, image.shape[0] - 20)
        font_scale = 1
        font_color = (0, 0, 255)
        thickness = 2
        cv2.putText(
            image,
            "fps:%0.1f" % fps,
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            thickness,
        )

    def print_help(self) -> None:
        """Print the help message."""
        help_str = "" + "-----------------\n"
        help_str += "DEODR Mesh Viewer\n"
        help_str += "-----------------\n"
        help_str += "Keys:\n"
        for key, func in self.keys_map.items():
            help_str += f"{key}: {func.__doc__}\n"
        print(help_str)
        self.interactor.print_help()

    def toggle_renderer(self) -> None:
        """Toggle the renderer between DEODR cpu rendering and moderngl."""
        self.use_moderngl = not (self.use_moderngl)
        print(f"use_moderngl = { self.use_moderngl}")

        if self.use_moderngl and self.offscreen_renderer is None:
            self.setup_moderngl()

    def toggle_perspective_texture_mapping(self) -> None:
        """Toggle between linear texture mapping and perspective correct texture mapping."""
        if self.use_moderngl:
            print("can only use perspective correct mapping  when using moderngl")
        else:
            self.scene.perspective_correct = not (self.scene.perspective_correct)
            print(f"perspective_correct = {self.scene.perspective_correct}")

    def toggle_lights(self) -> None:
        """Toggle between uniform lighting vs directional + ambient."""
        self.use_light = not (self.use_light)
        print(f"use_light = { self.use_light}")

        if self.use_light:
            if self.use_moderngl:
                assert self.offscreen_renderer is not None
                self.offscreen_renderer.set_light(
                    light_directional=np.array(self.light_directional),
                    light_ambient=self.light_ambient,
                )
            else:
                self.scene.set_light(
                    light_directional=np.array(self.light_directional),
                    light_ambient=self.light_ambient,
                )
        elif self.use_moderngl:
            assert self.offscreen_renderer is not None
            self.offscreen_renderer.set_light(
                light_directional=np.ndarray((0, 0, 0)),
                light_ambient=1.0,
            )
        else:
            self.scene.set_light(light_directional=(0, 0, 0), light_ambient=1.0)

    def toggle_edge_overdraw_antialiasing(self) -> None:
        """Toggle edge overdraw anti-aliasing (DEODR rendering only)."""
        if self.use_moderngl:
            print("no anti-aliasing available when using moderngl")
        else:
            self.use_antialiasing = not (self.use_antialiasing)
            print(f"use_antialiasing = {self.use_antialiasing}")
            self.scene.sigma = 1.0 if self.use_antialiasing else 0.0

    def pickle_scene_and_cameras(self) -> None:
        """Save scene and camera in a pickle file."""
        filename = os.path.abspath("scene.pickle")
        # save scene and camera in pickle file
        with open(filename, "wb") as file:
            # dump information to the file
            pickle.dump(self.scene, file)
        print(f"saved scene in {filename}")

        filename = os.path.abspath("camera.pickle")
        print(f"save scene in {filename}")
        with open(filename, "wb") as file:
            # dump information to the file
            pickle.dump(self.camera, file)
        print(f"saved camera in {filename}")

    def toggle_interactor_mode(self) -> None:
        """Change the camera interactor mode."""
        self.interactor.toggle_mode()
        self.interactor.print_help()

    def toggle_video_recording(self) -> None:
        """Start and stop video recording."""
        if not self.recording:
            id = 0
            while os.path.exists(self.video_pattern.format(**dict(id=id))):
                id += 1
            filename = self.video_pattern.format(**dict(id=id))

            self.video_writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*self.video_format),
                30,
                (self.width, self.height),
            )
            self.recording = True
        else:
            assert self.video_writer is not None
            self.video_writer.release()
            self.recording = False

    def register_keys(self) -> None:
        self.keys_map: Dict[str, Callable[[], None]] = {}
        self.register_key("h", self.print_help)
        self.register_key("r", self.toggle_renderer)
        self.register_key("p", self.toggle_perspective_texture_mapping)
        self.register_key("l", self.toggle_lights)
        self.register_key("a", self.toggle_edge_overdraw_antialiasing)
        self.register_key("d", self.pickle_scene_and_cameras)
        self.register_key("s", self.toggle_video_recording)
        self.register_key("t", self.toggle_interactor_mode)

    def register_key(self, key: str, func: Callable[[], None]) -> None:
        self.keys_map[key] = func

    def process_key(self, key: int) -> None:
        chr_key = chr(key)
        if chr_key in self.keys_map:
            self.keys_map[chr(key)]()
        else:
            print(f"no function registered for key {chr_key}")


def run() -> None:
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    Viewer(obj_file, use_moderngl=False).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mesh_viewer", usage="%(prog)s [options]")
    duck_file = os.path.join(deodr.data_path, "duck.obj")
    parser.add_argument("mesh_file", type=str, nargs="?", default=duck_file)
    args = parser.parse_args()
    mesh_file = args.mesh_file
    Viewer(mesh_file, use_moderngl=True).start()
