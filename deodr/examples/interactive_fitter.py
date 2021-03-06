"""Example of interactive GUI to fit model to an image"""

import argparse
import os


import cv2
import deodr
from deodr.examples.mesh_viewer import Viewer
from deodr.mesh_fitter import MeshRGBFitterWithPose
from deodr.triangulated_mesh import ColoredTriMesh
import numpy as np
import trimesh


def segment_foreground_grabcut(image):
    height, width = image.shape[:2]
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (5, 5, width - 10, height - 10)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # first grabcut using rectangle
    cv2.grabCut(
        image,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    # second grabcut to try to retrieve foreground on the borders of the image
    mask[mask == cv2.GC_BGD] = cv2.GC_PR_BGD
    cv2.grabCut(
        image,
        mask,
        None,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    foreground_mask = mask == 3
    return foreground_mask


class InteractiveFitter(Viewer):
    def __init__(
        self,
        mesh_file,
        image_file,
        light_directional=(0, 0, -0.5),
        light_ambient=0.5,
        horizontal_fov=60,
        n_subdivision=0,
        use_grabcut_segmentation=True,
    ):

        self.target_image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_RGB2BGR)
        width = self.target_image.shape[1]
        height = self.target_image.shape[0]

        mesh_trimesh = trimesh.load(mesh_file)
        self.mesh = ColoredTriMesh.from_trimesh(mesh_trimesh).subdivise(n_subdivision)

        # centering vertices
        self.mesh.vertices = self.mesh.vertices - np.mean(self.mesh.vertices, axis=0)

        default_color = np.array([0.4, 0.3, 0.25])
        default_light = {
            "directional": -np.array([0.1, 0.5, 0.4]),
            "ambient": np.array([0.6]),
        }
        # background_color = (0.8, 0.8, 0.8)
        background_color = np.array([0.5, 0.6, 0.7])

        self.mesh.set_vertices_colors(
            np.tile(default_color, (self.mesh.nb_vertices, 1))
        )
        self.windowname = "DEODR Interactive Fitter"
        super(InteractiveFitter, self).__init__(
            width=width,
            height=height,
            file_or_mesh=self.mesh,
            display_fps=False,
            windowname=self.windowname,
            use_moderngl=False,
            background_color=background_color,
        )
        self.fitting = False
        self.register_key("f", self.toggle_fitting)

        euler_init = np.array([0, 0, 0])
        translation_init = np.array([0, 0, 0])

        self.fitter = MeshRGBFitterWithPose(
            self.mesh.vertices,
            self.mesh.faces,
            default_color=default_color,
            default_light=default_light,
            update_lights=True,
            update_color=True,
            euler_init=euler_init,
            translation_init=translation_init,
            cregu=1000,
        )
        if use_grabcut_segmentation:

            foreground_mask = segment_foreground_grabcut(self.target_image)
            self.target_image = self.target_image * foreground_mask[:, :, None] + (
                ~foreground_mask
            )[:, :, None] * (np.array(background_color) * 255).astype(np.uint8)

            self.fitter.set_image(
                self.target_image.astype(np.float) / 255, focal=None, distortion=None
            )
        self.fitter.set_background_color(background_color)
        cv2.createTrackbar("slider", "sliders", 0, 100, self.change_damping)

    def change_damping(value):
        print(value)

    def toggle_fitting(self):
        """Toggle mesh fitting."""
        self.fitting = not self.fitting
        if self.fitting:
            self.fitter.set_camera(self.camera)
            print("fitting")

    def resize_camera(self):
        pass

    def refresh(self):
        if self.fitting:
            energy, image, diff_image = self.fitter.step()
            cv2.imshow(
                "fitting error",
                diff_image,
            )
        self.mesh.vertices = self.fitter.mesh.vertices

        super(InteractiveFitter, self).refresh()

    def get_image(self):
        rendered_image = (self.scene.render(self.interactor.camera) * 255).astype(
            np.uint8
        )
        blended_image = (0.5 * rendered_image + 0.5 * self.target_image).astype(
            np.uint8
        )

        cv2.imshow(
            "images",
            cv2.cvtColor(
                np.column_stack((self.target_image, rendered_image)), cv2.COLOR_RGB2BGR
            ),
        )
        return blended_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="interactive_fitter", usage="%(prog)s [options]"
    )
    # image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
    # mesh_file = os.path.join(deodr.data_path, "duck.obj")

    image_file = os.path.join(deodr.data_path, "hand.png")
    mesh_file = os.path.join(deodr.data_path, "hand.obj")

    parser.add_argument("image_file", type=str, nargs="?", default=image_file)
    parser.add_argument("mesh_file", type=str, nargs="?", default=mesh_file)
    args = parser.parse_args()
    image_file = args.image_file
    mesh_file = args.mesh_file
    InteractiveFitter(mesh_file, image_file, n_subdivision=1).start()
