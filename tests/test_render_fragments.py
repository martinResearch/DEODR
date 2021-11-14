"""Test using rgb mesh rendering."""

import os

import deodr
from deodr.examples.render_mesh import example_moderngl, example_channels, example_rgb, default_scene


import imageio

import numpy as np





def test_render_mesh(update=False):
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=320, height=240)
    image = scene.render(camera)

    if update:
        image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        image_uint8 = (image * 255).astype(np.uint8)
        imageio.imwrite(image_file, image_uint8)
    if display:
        plt.figure()
        plt.title("deodr rendering")
        plt.imshow(image)




if __name__ == "__main__":
    test_render_mesh()

