"""Test texture coordinates regarding 0.5 texel shift."""

import numpy as np

from deodr.differentiable_renderer import Scene2D


def test_texture_coordinates() -> None:
    texture = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], dtype=np.float64)

    # coordinate of upper left pixel center is (0,0)
    # coordinate of the upper left texture pixel center (texel) is (0, 0)
    # the color of the texture bilinearly sampled as (0,0) is texture[0, 0]
    # this contrasts with OpenGL when the texture at position (0.5/texture_width , 0.5/texture_width) is texture[0, 0]

    uv = np.array([[0, 0], [1, 0], [0, 1]])
    ij = np.array([[1, 1], [1, 15], [15, 1]])
    depths = np.array([1, 1, 1])
    shade = np.array([0, 0, 0])
    shade = np.array([1, 1, 1])
    textured = np.array([1], dtype=bool)
    shaded = np.array([1], dtype=bool)
    colors = np.eye(3)
    edgeflags = np.zeros((1, 3), dtype=bool)

    for clockwise in [False, True]:
        if clockwise:
            faces = np.array([[0, 2, 1]], dtype=np.uint32)
            faces_uv = np.array([[0, 2, 1]], dtype=np.uint32)
        else:
            faces = np.array([[0, 1, 2]], dtype=np.uint32)
            faces_uv = np.array([[0, 1, 2]], dtype=np.uint32)

        height = 40
        width = 60

        background_color = np.array([0, 0, 0])

        # using strict_edge == True so that the pixel that are exactly on the edge
        # are drawn
        scene_2d = Scene2D(
            ij=ij,
            faces=faces,
            faces_uv=faces_uv,
            uv=uv,
            texture=texture,
            height=height,
            width=width,
            nb_colors=3,
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

        image, _ = scene_2d.render(sigma=0)
        assert np.allclose(image[0, :, :], [0, 0, 0])
        assert np.allclose(image[:, 0, :], [0, 0, 0])
        assert np.allclose(image[1, 1, :], [1, 0, 0])
        assert np.allclose(image[15, 1, :], [0, 1, 0])
        assert np.allclose(image[1, 15, :], [0, 0, 1])


if __name__ == "__main__":
    test_texture_coordinates()
