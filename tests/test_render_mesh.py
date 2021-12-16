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
    if not os.name == "nt":  # did not manage to install mesa on windows github action
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

    hashlib.sha256(
        scene_gt.ij.tobytes()
    ).hexdigest() == "56a498bf243bd514c9ab4a3bfd90f8105aa2c168023fa288dc39ad82e2d36a20"
    hashlib.sha256(
        scene_gt.depths.tobytes()
    ).hexdigest() == "e25eed6310fef37e401aef594c4c95e1b3cccf962a3646976cf546c58ddfac0a"
    hashlib.sha256(
        scene_gt.uv.tobytes()
    ).hexdigest() == "f436623445124ecff7139efa57cce21c2768e23727bac974e236ea33651cc7c9"
    hashlib.sha256(
        scene_gt.shade.tobytes()
    ).hexdigest() == "4b796b925c4349245e52a3e6311e99d536dc71e8aa8dc43cbd67cbe35d48892f"
    hashlib.sha256(
        scene_gt.colors.tobytes()
    ).hexdigest() == "76dbff728be3eb0860bd27adf493e935dbd81cd7232ec732ba30c4f73ea35c94"

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
