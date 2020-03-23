from deodr.triangulated_mesh import ColoredTriMesh
from deodr import differentiable_renderer
import deodr
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
import time
from scipy.spatial.transform import Rotation
import os


class Interactor:
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

        if self.left_is_down:
            if self.mode == "camera_centered":
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
                raise (BaseException(f"unkown camera mode {self.mode}"))

        if self.right_is_down:
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
                raise (BaseException(f"unkown camera mode {self.mode}"))

        if self.middle_is_down:
            object_depth = (
                self.camera.extrinsic[2, :3].dot(self.object_center)
                + self.camera.extrinsic[2, 3]
            )

            self.camera.extrinsic[0, 3] += (
                self.xy_translation_speed * object_depth * (x - self.x_last)
            )
            self.camera.extrinsic[1, 3] += (
                self.xy_translation_speed * object_depth * (y - self.y_last)
            )
            self.x_last = x
            self.y_last = y


def mesh_viewer(
    obj_file_or_trimesh,
    display_texture_map=True,
    width=640,
    height=480,
    display_fps=True,
    title=None,
    use_moderngl=False,
):
    if type(obj_file_or_trimesh) == str:
        if title is None:
            title = obj_file_or_trimesh
        mesh_trimesh = trimesh.load(obj_file_or_trimesh)
    elif type(obj_file_or_trimesh) == trimesh.base.Trimesh:
        mesh_trimesh = obj_file_or_trimesh
        if title is None:
            title = "unknown"
    else:
        raise (
            BaseException(
                f"unkown type {type(obj_file_or_trimesh)}for input obj_file_or_trimesh,"
                " can be string or trimesh.base.Trimesh"
            )
        )

    mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)
    if display_texture_map:
        ax = plt.subplot(111)
        if mesh.textured:
            mesh.plot_uv_map(ax)

    object_center = 0.5 * (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0))
    object_radius = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))

    camera_center = object_center + np.array([0, 0, 3]) * object_radius
    focal = 2 * width

    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    translation = -rotation.T.dot(camera_center)
    extrinsic = np.column_stack((rotation, translation))
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    distortion = [0, 0, 0, 0, 0]
    camera = differentiable_renderer.Camera(
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        resolution=(width, height),
        distortion=distortion,
    )

    scene = differentiable_renderer.Scene3D()
    scene.set_light(light_directional=np.array([-0.5, 0, -0.5]), light_ambient=0.3)
    scene.set_mesh(mesh)
    background_image = np.ones((height, width, 3))
    scene.set_background(background_image)

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
        xy_translation_speed=1e-7 * object_radius,
    )

    cv2.namedWindow(windowname)
    cv2.setMouseCallback(windowname, interactor.mouse_callback)

    if use_moderngl:
        import deodr.opengl.moderngl

        offscreen_renderer = deodr.opengl.moderngl.OffscreenRenderer()
        scene.mesh.compute_vertex_normals()
    while cv2.getWindowProperty(windowname, 0) >= 0:
        # mesh.set_vertices(mesh.vertices+np.random.randn(*mesh.vertices.shape)*0.001)
        start = time.clock()
        if use_moderngl:
            image = offscreen_renderer.render(scene, camera)
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
        cv2.waitKey(1)


def run():
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    mesh_viewer(obj_file, use_moderngl=False)


if __name__ == "__main__":
    run()
