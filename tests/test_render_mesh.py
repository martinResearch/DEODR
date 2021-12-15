"""Test using rgb mesh rendering."""

import os

import deodr
from deodr.examples.render_mesh import example_moderngl, example_rgb


import imageio

import numpy as np
import hashlib


def test_render_mesh_moderngl():
    example_moderngl(display=False)


def test_render_mesh(update=False):
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


if __name__ == "__main__":
    test_render_mesh()
    test_render_mesh_moderngl()
