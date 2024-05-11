"""Test using rgb mesh rendering."""

import numpy as np

from deodr.differentiable_renderer import Scene2D


def test_upper_left_pixel_center_coordinates() -> None:
    """Testing pixel center coordinates.

    pixel centers are at integer coordinates when integer_pixel_centers=True with
        upper left at (0, 0)
        upper right at (width - 1, 0)
        lower left at (0, height - 1)
        lower right at  (width - 1, height - 1)

    pixel centers are at half integer coordinates when integer_pixel_centers=False with
        upper left at (0.5, 0.5)
        upper right at (width - 0.5, 0.5)
        lower left at (0.5, height - 0.5)
        lower right at  (width - 0.5, height - 0.5)
    """
    height = 4
    width = 3
    integer_points_coordinates = [
        (0, 0),  # upper left
        (width - 1, 0),  # upper right,
        (0, height - 1),  # lower left,
        (width - 1, height - 1),  # lower right
    ]
    eps = 0.001

    clockwise = True
    for integer_pixel_centers in [False, True]:
        if integer_pixel_centers:
            points_coordinates = [
                (0.0, 0.0),  # upper left
                (width - 1.0, 0.0),  # upper right,
                (0, height - 1.0),  # lower left,
                (width - 1.0, height - 1.0),  # lower right
            ]
        else:
            points_coordinates = [
                (0.5, 0.5),  # upper left
                (width - 0.5, 0.5),  # upper right,
                (0.5, height - 0.5),  # lower left,
                (width - 0.5, height - 0.5),  # lower right
            ]

        depths = np.array([1, 1, 1])
        shade = np.array([0, 0, 0])
        shade = np.array([1, 1, 1])
        faces_uv = np.array([[0, 2, 1]], dtype=np.uint32)
        uv = np.zeros((3, 2), dtype=bool)
        textured = np.array([0], dtype=bool)
        shaded = np.array([0], dtype=bool)
        colors = np.array([[1], [1], [1]])
        edgeflags = np.zeros((1, 3), dtype=bool)
        faces = np.array([[0, 2, 1]], dtype=np.uint32)

        texture = np.ones((2, 2, 1))
        background_color = np.array([0])

        for integer_point_coordinates, point_coordinates in zip(integer_points_coordinates, points_coordinates):
            ij = np.array([[-eps, -eps], [-eps, eps], [eps, -eps]]) + np.array(point_coordinates)

            scene_2d = Scene2D(
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
                integer_pixel_centers=integer_pixel_centers,
            )

            image, _ = scene_2d.render(sigma=0)

            expected_image = np.zeros((height, width, 1))
            expected_image[integer_point_coordinates[1], integer_point_coordinates[0], 0] = 1
            assert np.allclose(expected_image, image)


if __name__ == "__main__":
    test_upper_left_pixel_center_coordinates()
