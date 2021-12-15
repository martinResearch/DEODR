"""Test using rgb mesh rendering."""

import os

import deodr
from deodr.examples.render_mesh import example_moderngl, example_rgb
from deodr.examples.triangle_soup_fitting import create_example_scene
from deodr import differentiable_renderer_cython

import imageio

import numpy as np
import hashlib


def test_render_mesh_moderngl():
    example_moderngl(display=False)


def test_render_mesh_duck(update=False):
    image = example_rgb(display=False, save_image=False, width=320, height=240)
    image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
    image_uint8 = (image * 255).astype(np.uint8)
    if update:
        imageio.imwrite(image_file, image_uint8)
    image_prev = imageio.imread(image_file)
    assert np.max(np.abs(image_prev - image_uint8)) == 0

    assert (
        hashlib.sha256(image.tobytes()).hexdigest()
        == "2f22e5402cd5a396bb09b4378ff5d619b47d8886209869111c148e4f97a8778e"
    )


def test_render_mesh_triangle_soup():

    np.random.seed(2)
    scene_gt = create_example_scene(clockwise=True)
    sigma = 1
    image = np.zeros((scene_gt.height, scene_gt.width, scene_gt.nb_colors))
    z_buffer = np.zeros((scene_gt.height, scene_gt.width))
    differentiable_renderer_cython.renderScene(scene_gt, sigma, image, z_buffer)

    assert (
        hashlib.sha256(image.tobytes()).hexdigest()
        == "4de52cc3e902f92ff64324b261ddc45cd6d148ec7e670cf2942532d515af62d8"
    )
    assert (
        hashlib.sha256(z_buffer.tobytes()).hexdigest()
        == "b6f87e03c60bd820efa09d0536495b25d5852f67ecbecd2622f8bf1910d6052a"
    )


if __name__ == "__main__":
    test_render_mesh_duck()
    test_render_mesh_triangle_soup()
    test_render_mesh_moderngl()
