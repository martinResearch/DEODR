from deodr.examples.render_mesh import render_mesh
import os
import imageio
import numpy as np

def test_render_mesh():
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    Abuffer, channels = render_mesh(obj_file, display=False, SizeW=320, SizeH=240)
    image_file = os.path.abspath(deodr.data_path, "test/duck.png"))
    Abuffer_uint8 = (Abuffer*255).astype(np.uint8)
    Abuffer_prev = imageio.imread( image_file)
    assert (np.max(np.abs(Abuffer_prev-Abuffer_uint8)) == 0)

