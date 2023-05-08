"""Test using rgb mesh rendering."""

import os

import deodr
from deodr.examples.render_mesh import (
    example_moderngl,
    example_channels,
    example_rgb,
    default_scene,
)
import imageio
import matplotlib.pyplot as plt
import numpy as np


def test_render_fragments(update: bool = False, display: bool = True) -> None:
    obj_file = os.path.join(deodr.data_path, "duck.obj")
    scene, camera = default_scene(obj_file, width=320, height=240)
    image = scene.render(camera)
    fragments = scene.render_fragments(camera)
    x , y, alpha, values = fragments


if __name__ == "__main__":
    test_render_fragments()
