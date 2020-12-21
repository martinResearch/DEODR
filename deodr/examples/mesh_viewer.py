"""Example of interactive 3D mesh visualization using deodr and opencv."""

import argparse
import os
import time
import pickle

import cv2

import deodr
from deodr import differentiable_renderer
from deodr.triangulated_mesh import ColoredTriMesh

import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

import trimesh


class Interactor:
    """Class that implements various mouse interaction with the 3D scene."""

    def __init__(
        self,
        camera,
        mode="object_centered_trackball",
        object_center=None,
        rotation_speed=0.003,
        z_translation_speed=0.05,
        xy_translation_speed=0.01,
    ):
        self.left_is_down = False
        self.right_is_down = False
        self.middle_is_down = False
        self.mode = mode
        self.object_center = object_center
        self.rotation_speed = rotation_speed
        self.z_translation_speed = z_translation_speed
        self.xy_translation_speed = xy_translation_speed
        self.camera = camera

    def mouse_callback(self, event, x, y, flags, param):
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

        if self.left_is_down and not (self.ctrl_is_down):

            if self.mode == "camera_centered":

                center_in_camera = self.camera.world_to_camera(self.object_center)
                rotation = Rotation.from_rotvec(
                    np.array(
                        [
                            -self.rotation_speed * (y - self.y_last),
                            self.rotation_speed * (x - self.x_last),
                            0,
                        ]
                    )
                )
                self.camera.extrinsic = rotation.as_dcm().dot(self.camera.extrinsic)
                # assert np.allclose(center_in_camera, self.camera.world_to_camera(self.object_center))
                self.x_last = x
                self.y_last = y

            if self.mode == "object_centered_trackball":

                rotation = Rotation.from_rotvec(
                    np.array(
                        [
                            self.rotation_speed * (y - self.y_last),
                            -self.rotation_speed * (x - self.x_last),
                            0,
                        ]
                    )
                )
                n_rotation = rotation.as_dcm().dot(self.camera.extrinsic[:, :3])
                nt = (
                    self.camera.extrinsic[:, :3].dot(self.object_center)
                    + self.camera.extrinsic[:, 3]
                    - n_rotation.dot(self.object_center)
                )
                self.camera.extrinsic = np.column_stack((n_rotation, nt))
                self.x_last = x
                self.y_last = y
            else:
                raise (BaseException(f"unknown camera mode {self.mode}"))

        if self.right_is_down and not (self.ctrl_is_down):
            if self.mode == "camera_centered":
                self.camera.extrinsic[2, 3] += self.z_translation_speed * (
                    self.y_last - y
                )
                self.x_last = x
                self.y_last = y
            if self.mode == "object_centered_trackball":
                self.camera.extrinsic[2, 3] += self.z_translation_speed * (
                    self.y_last - y
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
            center_in_camera = self.camera.world_to_camera(self.object_center)
            self.x_last = x
            self.y_last = y
            assert np.max(np.abs(center_in_camera[:2])) < 1e-3


def mesh_viewer(
    file_or_mesh,
    display_texture_map=True,
    width=640,
    height=480,
    display_fps=True,
    title=None,
    use_moderngl=False,
    light_directional=(0, 0, -0.5),
    light_ambient=0.5,
):
    if isinstance(file_or_mesh, str):
        if title is None:
            title = file_or_mesh
        mesh_trimesh = trimesh.load(file_or_mesh)
        mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)
    elif isinstance(file_or_mesh, trimesh.base.Trimesh):
        mesh_trimesh = file_or_mesh
        mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)
        if title is None:
            title = "unknown"
    elif isinstance(file_or_mesh, ColoredTriMesh):
        mesh = file_or_mesh
        if title is None:
            title = "unknown"
    else:
        raise (
            BaseException(
                f"unknown type {type(file_or_mesh)}for input obj_file_or_trimesh,"
                " can be string or trimesh.base.Trimesh"
            )
        )

    if display_texture_map:
        ax = plt.subplot(111)
        if mesh.textured:
            mesh.plot_uv_map(ax)

    object_center = 0.5 * (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0))
    object_radius = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))

    camera_center = object_center + np.array([0, 0, 3]) * object_radius

    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    translation = -rotation.T.dot(camera_center)
    extrinsic = np.column_stack((rotation, translation))

    focal = 2 * width
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    distortion = [0, 0, 0, 0, 0]
    camera = differentiable_renderer.Camera(
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        width=width,
        height=height,
        distortion=distortion,
    )
    use_antiliazing = True
    use_light = True

    scene = differentiable_renderer.Scene3D(sigma=1)

    scene.set_light(
        light_directional=np.array(light_directional), light_ambient=light_ambient
    )
    scene.set_mesh(mesh)

    scene.set_background_color([1, 1, 1])

    if mesh.texture is not None:
        mesh.texture = mesh.texture[
            :, :, ::-1
        ]  # convert texture to GBR to avoid future conversion when ploting in Opencv

    fps = 0
    fps_decay = 0.1
    windowname = f"DEODR mesh viewer:{title}"

    interactor = Interactor(
        camera=camera,
        object_center=object_center,
        z_translation_speed=0.01 * object_radius,
        xy_translation_speed=3e-4,
    )

    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, width, height)
    cv2.setMouseCallback(windowname, interactor.mouse_callback)

    if use_moderngl:
        import deodr.opengl.moderngl

        offscreen_renderer = deodr.opengl.moderngl.OffscreenRenderer()
        scene.mesh.compute_vertex_normals()
        offscreen_renderer.set_scene(scene)
    else:
        offscreen_renderer = None
    while cv2.getWindowProperty(windowname, 0) >= 0:

        # mesh.set_vertices(mesh.vertices+np.random.randn(*mesh.vertices.shape)*0.001)
        width, height = cv2.getWindowImageRect(windowname)[2:]
        focal = 2 * width
        intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        camera.intrinsic = intrinsic
        camera.width = width
        camera.height = height

        start = time.clock()
        if use_moderngl:
            image = offscreen_renderer.render(camera)
        else:
            image = scene.render(interactor.camera)

        if display_fps:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner_of_text = (20, height - 20)
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
        cv2.imshow(windowname, image)
        stop = time.clock()
        fps = (1 - fps_decay) * fps + fps_decay * (1 / (stop - start))
        key = cv2.waitKey(1)
        if key >= 0:
            if key == ord("r"):
                # change renderer between DEODR cpu rendering and moderngl
                use_moderngl = not (use_moderngl)
                print(f"use_moderngl = {use_moderngl}")

                if offscreen_renderer is None:
                    offscreen_renderer = deodr.opengl.moderngl.OffscreenRenderer()
                    scene.mesh.compute_vertex_normals()
                    offscreen_renderer.set_scene(scene)

            if key == ord("p"):
                if use_moderngl:
                    print(
                        "can only use perspective corect mapping  when using moderngl"
                    )
                else:
                    # toggle perspective correct mapping (texture or interpolation)
                    scene.perspective_correct = not (scene.perspective_correct)
                    print(f"perspective_correct = {scene.perspective_correct}")

            if key == ord("l"):
                # toggle directional light + ambient vs ambient = 1
                use_light = not (use_light)
                print(f"use_light = {use_light}")
                if use_light:
                    if use_moderngl:
                        offscreen_renderer.set_light(
                            light_directional=np.array(light_directional),
                            light_ambient=light_ambient,
                        )
                    else:
                        scene.set_light(
                            light_directional=np.array(light_directional),
                            light_ambient=light_ambient,
                        )
                else:
                    if use_moderngl:
                        offscreen_renderer.set_light(
                            light_directional=(0, 0, 0), light_ambient=1.0,
                        )
                    else:
                        scene.set_light(light_directional=None, light_ambient=1.0)

            if key == ord("a"):
                # toggle edge overdraw anti-aliasing
                if use_moderngl:
                    print("no anti-alizaing available when using moderngl")
                else:
                    use_antiliazing = not (use_antiliazing)
                    print(f"use_antialiazing = {use_antiliazing}")
                    if use_antiliazing:
                        scene.sigma = 1.0
                    else:
                        scene.sigma = 0.0

            if key == ord("s"):
                filename = os.path.abspath("scene.pickle")
                # save scene and camera in pickle file
                with open(filename, "wb") as file:
                    # dump information to the file
                    pickle.dump(scene, file)
                print(f"saved scene in {filename}")

                filename = os.path.abspath("camera.pickle")
                print(f"save scene in {filename}")
                with open(filename, "wb") as file:
                    # dump information to the file
                    pickle.dump(camera, file)
                print(f"saved camera in {filename}")


def run():
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    mesh_viewer(obj_file, use_moderngl=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mesh_viewer", usage="%(prog)s [options]")
    duck_file = os.path.join(deodr.data_path, "duck.obj")
    parser.add_argument("mesh_file", type=str, nargs="?", default=duck_file)
    args = parser.parse_args()
    mesh_file = args.mesh_file
    mesh_viewer(mesh_file, use_moderngl=True)
