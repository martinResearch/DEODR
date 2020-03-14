import os
import imageio
import numpy as np
import deodr
from deodr.examples.render_mesh import example_rgb, example_moderngl


def test_render_mesh_moderngl():
    example_moderngl(display=False)


def test_render_mesh():
    image = example_rgb(display=False, save_image=False, width=320, height=240)
    image_file = os.path.abspath(os.path.join(deodr.data_path, "test/duck.png"))
    image_uint8 = (image * 255).astype(np.uint8)
    image_prev = imageio.imread(image_file)
    assert np.max(np.abs(image_prev - image_uint8)) == 0
