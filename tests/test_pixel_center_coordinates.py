"""Test using rgb mesh rendering."""

import os

from matplotlib import pyplot as plt

import deodr
from deodr.examples.render_mesh import example_moderngl, example_rgb
from deodr.differentiable_renderer import Scene2D
import numpy as np


def test_upper_left_pixel_center_coordinates():
    """testing  that pixel centers are at integer coordinates with
        upper left at (0, 0)
        upper right at (width - 1, 0)
         lower left at (0, height - 1)
        lower right at  (width - 1, height - 1)
    """
    height = 4
    width = 3

    point_coordinates = [
        (0, 0),  # upper left
        (width - 1, 0),  # upper right,
        (0, height - 1),  # lower left,
        (width - 1, height - 1),  # lower right
    ]

    eps = 0.001

    depths = np.array([1, 1, 1])
    shade = np.array([0, 0, 0])
    shade = np.array([1, 1, 1])
    faces_uv = np.array([[0, 2, 1]], dtype=np.uint32)
    uv = np.zeros((3, 2), dtype=np.bool)
    textured = np.array([0], dtype=np.bool)
    shaded = np.array([0], dtype=np.bool)
    colors = np.array([[1], [1], [1]])
    edgeflags = np.zeros((1, 3), dtype=np.bool)
    clockwise = True
    faces = np.array([[0, 2, 1]], dtype=np.uint32)

    texture = np.ones((2, 2, 1))
    background_color = np.array([0])

    for point_coordinate in point_coordinates:
        ij = np.array([[-eps, -eps], [-eps, eps], [eps, -eps]]) + np.array(
            point_coordinate
        )

        scene2D = Scene2D(
            ij=ij,
            faces=faces,
            faces_uv=faces_uv,
            uv=uv,
            texture=texture,
            height=height,
            width=width,
            nb_colors=1,
            background_image=None,
            background_color=background_color,
            depths=depths,
            textured=textured,
            shade=shade,
            colors=colors,
            shaded=shaded,
            edgeflags=edgeflags,
            strict_edge=False,
            perspective_correct=True,
            clockwise=clockwise,
        )

        image, _ = scene2D.render(sigma=0)

        expected_image = np.zeros((height, width, 1))
        expected_image[point_coordinate[1], point_coordinate[0], 0] = 1
        assert np.allclose(expected_image, image)


if __name__ == "__main__":
    test_upper_left_pixel_center_coordinates()
